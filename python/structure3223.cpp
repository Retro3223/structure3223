#include <Python.h>
#include <iostream>
#include <string>
#include <sstream>
#include <string.h>
#include <ni2/OpenNI.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_common.h>
#define NPY_NO_DEPRECATED_API

using namespace openni;

static VideoFrameRef depthFrame, irFrame;
static Device device;
static bool structure_initialized;
static VideoStream depthStream, irStream;
const char *modeToString(VideoMode mode);

const char *errorMessage(const char *msg);
PyObject *read_frame_into_array(PyObject *dst, VideoFrameRef frame);

PyObject *read_a_frame(
        PyObject *dst,
        VideoStream& stream, int timeout, 
        PixelFormat expectedPixelFormat);

#define ERR_N_DIE(p) {PyErr_SetString(PyExc_RuntimeError, errorMessage(p)); return NULL;}
#define ERR_N_DIE_NO_NI(p) {PyErr_SetString(PyExc_RuntimeError, (p)); return NULL;}

PyObject *structure_init(PyObject *self, PyObject *args) {
    import_array(); // required for to use numpy C-API
    Status rc = OpenNI::initialize();
    if (rc != STATUS_OK) ERR_N_DIE("OpenNI initialize failed");
    rc = device.open(ANY_DEVICE);
    if (rc != STATUS_OK) ERR_N_DIE("OpenNI device open failed");
    if (device.getSensorInfo(SENSOR_DEPTH) == NULL) {
        ERR_N_DIE("Device does not support depth sensing");
    }
    rc = depthStream.create(device, SENSOR_DEPTH);
    if (rc != STATUS_OK) ERR_N_DIE("depth stream create failed");
    rc = irStream.create(device, SENSOR_IR);
    if (rc != STATUS_OK) ERR_N_DIE("ir stream create failed");
    rc = device.setDepthColorSyncEnabled(true);
    if (rc != STATUS_OK) ERR_N_DIE("device color sync failed");
    rc = depthStream.start();
    if (rc != STATUS_OK) ERR_N_DIE("depth stream start failed");
    rc = irStream.start();
    if (rc != STATUS_OK) ERR_N_DIE("ir stream start failed");


    structure_initialized = true;
    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *structure_destroy(PyObject *self, PyObject *args) {
    depthStream.stop();
    depthStream.destroy();
    irStream.stop();
    irStream.destroy();
    device.close();
    OpenNI::shutdown();
    Py_INCREF(Py_None);
    return Py_None;
}

const char *errorMessage(const char *msg) {
    std::string result(msg);
    result += ": ";
    result += OpenNI::getExtendedError();
    std::cout << result << std::endl;
    return result.c_str();
}

PyObject* structure_read_frame(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *depth = NULL;
    PyObject *ir = NULL;
    int timeout = TIMEOUT_FOREVER;
    int result = -1;
    static char *argnoms[] = {"depth", "ir", "timeout", NULL};
    PyArg_ParseTupleAndKeywords(args, kwargs, "OO|i", argnoms, &depth, &ir, &timeout);

    if(depth == NULL) {
        ERR_N_DIE_NO_NI("parameter depth not provided");
    }
    if(ir == NULL) {
        ERR_N_DIE_NO_NI("parameter ir not provided");
    }

    depth = read_a_frame(depth, depthStream, timeout, PIXEL_FORMAT_DEPTH_1_MM);
    // error 
    if(depth == NULL) return NULL;
    // timeout
    if(depth == Py_None) return Py_None;
    ir = read_a_frame(ir, irStream, timeout, PIXEL_FORMAT_GRAY16);
    // error 
    if(ir == NULL) return NULL;
    // timeout
    if(ir == Py_None) return Py_None;

    PyObject *arrayTuple = PyTuple_New(2);
    PyTuple_SetItem(arrayTuple, 0, depth);
    PyTuple_SetItem(arrayTuple, 1, ir);

    return arrayTuple;
}

PyObject *read_a_frame(
        PyObject *dst,
        VideoStream& stream, 
        int timeout, 
        PixelFormat expectedPixelFormat) 
{
    VideoStream *pStream = &stream;
    Status rc;
    int somedum;
    rc = OpenNI::waitForAnyStream(&pStream, 1, &somedum, timeout /*ms*/);
    if (rc != STATUS_OK) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    VideoFrameRef frame;
    rc = stream.readFrame(&frame);
    if(rc != STATUS_OK) ERR_N_DIE("read depth frame failed");
    VideoMode mode = frame.getVideoMode();
    if(mode.getPixelFormat() != expectedPixelFormat) {
        std::stringstream strm;
        strm << "unexpected pixel format: " << modeToString(mode);
        PyErr_SetString(PyExc_RuntimeError, strm.str().c_str()); 
        return NULL;
    }
    PyObject *array = read_frame_into_array(dst, frame);
    return array;
}

PyObject *check_buffer_against_frame(
        PyObject *dst, Py_buffer *buffer, VideoFrameRef frame) {
    if(!PyObject_CheckBuffer(dst)) {
        ERR_N_DIE_NO_NI("read_frame was passed a non-buffer");
    }
    if(PyObject_GetBuffer(dst, buffer, PyBUF_WRITABLE | PyBUF_ND) != 0) {
        ERR_N_DIE_NO_NI("read_frame was passed a read only buffer (?)");
    }
    if(!PyBuffer_IsContiguous(buffer, 'C')) {
        ERR_N_DIE_NO_NI("read_frame was passed a noncontiguous buffer");
    }
    std::stringstream str;
    if(buffer->ndim != 2) {
        str << "expected ndim=2, got ndim=" << buffer->ndim;
        ERR_N_DIE_NO_NI(str.str().c_str());
    }
    if(buffer->shape[0] != frame.getHeight() || 
            buffer->shape[1] != frame.getWidth()) {
        str << "expected shape=(" << frame.getHeight();
        str << ", " << frame.getWidth() << "), got shape=(";
        str << buffer->shape[0] << ", " << buffer->shape[1] << ")";
        ERR_N_DIE_NO_NI(str.str().c_str());
    }
    int expected_item_size = frame.getDataSize() / frame.getHeight();
    expected_item_size /= frame.getWidth();
    if(buffer->itemsize != expected_item_size) {
        str << "expected item size=" << expected_item_size;
        str << ", got item size=" << buffer->itemsize;
        ERR_N_DIE_NO_NI(str.str().c_str());
    }
    return dst;
}

PyObject *read_frame_into_array(PyObject *dst, VideoFrameRef frame) {
    Py_buffer buffer;
    if(!check_buffer_against_frame(dst, &buffer, frame)) {
        return NULL;
    }
    memcpy(buffer.buf, frame.getData(), frame.getDataSize());
    return dst;
}

#define STRENUM(p) case (p): return #p; 
const char *modeToString(VideoMode mode) {
        switch(mode.getPixelFormat()){
            STRENUM(PIXEL_FORMAT_DEPTH_1_MM)
            STRENUM(PIXEL_FORMAT_DEPTH_100_UM)
            STRENUM(PIXEL_FORMAT_SHIFT_9_2)
            STRENUM(PIXEL_FORMAT_RGB888)
            STRENUM(PIXEL_FORMAT_YUV422)
            STRENUM(PIXEL_FORMAT_GRAY8)
            STRENUM(PIXEL_FORMAT_GRAY16)
            STRENUM(PIXEL_FORMAT_JPEG)
            STRENUM(PIXEL_FORMAT_YUYV)
            default:
                std::stringstream str;
                str << mode.getPixelFormat() << " - ??";
                return str.str().c_str();
        }
}

static PyMethodDef methods[] = {
    {"init", &structure_init, METH_VARARGS, 
        "Initializer function. Call this before trying to read the structure sensor."},
    {"read_frame", (PyCFunction) &structure_read_frame, METH_VARARGS | METH_KEYWORDS, 
        "reads a frame from the depth sensor and infrared camera and returns (depthFrame, irFrame). Can be given a timeout. will return None if timeout is exceeded."},
    {"destroy", &structure_destroy, METH_VARARGS, 
        "close function. Call this before stopping your program."},
    {NULL} /*sentinal*/
};

#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif


static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "structure3223",
    "provides basic functionality to read depth and IR data from structure sensor.",
    -1,
    methods
};

PyMODINIT_FUNC
PyInit_structure3223(void)
{
    structure_initialized = false;
    PyObject* m;

    m = PyModule_Create(&moduledef);
    return m;
}
