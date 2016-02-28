#include "structure3223.h"
#include <iostream>
#include <string>
#include <sstream>
#include <string.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_common.h>


static VideoFrameRef depthFrame, irFrame;
static Device device;
static bool structure_initialized;
static VideoStream depthStream, irStream;

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
    if(chooseDepthVideoMode() == NULL) return NULL;
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

PyObject *chooseDepthVideoMode() {
	const SensorInfo *sensorInfo = device.getSensorInfo(SENSOR_DEPTH);
	const Array<VideoMode> &videoModes = sensorInfo->getSupportedVideoModes();
	int gotit = 0;
	for(int i = 0; i < videoModes.getSize(); i++) {
		std::cout << "larl " << i << std::endl;
		if(videoModes[i].getFps() == 30 &&
		   videoModes[i].getResolutionX() == 320 &&
		   videoModes[i].getResolutionY() == 240 &&
		   videoModes[i].getPixelFormat() == PIXEL_FORMAT_DEPTH_1_MM) {
			depthStream.setVideoMode(videoModes[i]);
			gotit = 1;
			break;
		}
	}
	if(gotit == 0) {
		ERR_N_DIE_NO_NI("Couldn't find suitable video mode for depth sensor");
	}
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

PyObject *structure_read_frame(
        PyObject *self, PyObject *args, PyObject *kwargs) {
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

    depth = read_a_frame(depth, depthStream, timeout, PIXEL_FORMAT_DEPTH_1_MM, "depth");
    // error 
    if(depth == NULL) return NULL;
    // timeout
    if(depth == Py_None) return Py_None;
    ir = read_a_frame(ir, irStream, timeout, PIXEL_FORMAT_GRAY16, "ir");
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
        PixelFormat expectedPixelFormat,
        const char *nom) 
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
    PyObject *array = read_frame_into_array(dst, frame, "read_frame", nom);
    return array;
}

PyObject *check_buffer(
        PyObject *dst, Py_buffer *buffer, 
        const char *funcnom, const char *nom, int write,
        int height, int width, int size) {
    std::stringstream str;
    if(!PyObject_CheckBuffer(dst)) {
        str << funcnom << " was passed a non-buffer for " << nom;
        ERR_N_DIE_NO_NI(str.str().c_str());
    }
    if(write) {
        if(PyObject_GetBuffer(dst, buffer, PyBUF_WRITABLE | PyBUF_ND) != 0) {
            str << funcnom << " was passed a read only buffer (?) for " << nom;
            ERR_N_DIE_NO_NI(str.str().c_str());
        }
    }else{
        if(PyObject_GetBuffer(dst, buffer, PyBUF_ND) != 0) {
            str << funcnom << " was passed a ??not good enough?? buffer for ";
            str << nom;
            ERR_N_DIE_NO_NI(str.str().c_str());
        }
    }
    if(!PyBuffer_IsContiguous(buffer, 'C')) {
        str << funcnom << " was passed a noncontiguous buffer for " << nom;
        ERR_N_DIE_NO_NI(str.str().c_str());
    }
    if(buffer->ndim != 2) {
        str << funcnom << " expected ndim=2, got ndim=" << buffer->ndim; 
        str << " for " << nom;
        ERR_N_DIE_NO_NI(str.str().c_str());
    }
    if(height != -1 && width != -1) {
        if(buffer->shape[0] != height || 
                buffer->shape[1] != width) {
            str << funcnom << " expected shape=(" << height;
            str << ", " << width << "), got shape=(";
            str << buffer->shape[0] << ", " << buffer->shape[1] << ")";
            str << " for " << nom;
            ERR_N_DIE_NO_NI(str.str().c_str());
        }
        int expected_item_size = size / width / height;
        if(buffer->itemsize != expected_item_size) {
            str << funcnom << " expected item size=" << expected_item_size;
            str << ", got item size=" << buffer->itemsize;
            str << " for " << nom;
            ERR_N_DIE_NO_NI(str.str().c_str());
        }
    }
    return dst;
}

PyObject *read_frame_into_array(PyObject *dst, VideoFrameRef frame, const char *funcnom, const char *bufnom) {
    Py_buffer buffer;
    if(!check_buffer(dst, &buffer, funcnom, bufnom, 1,
                frame.getHeight(), frame.getWidth(), frame.getDataSize())) {
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

PyObject *check_xyz(
        PyObject *xyz, Py_buffer *buffer, 
        const char *funcnom, const char *bufnom, 
        int write, int expected_y_shape, int expected_x_shape) {
    std::stringstream str;
    if(!PyObject_CheckBuffer(xyz)) {
        str << funcnom << " was passed a non-buffer for " << bufnom;
        ERR_N_DIE_NO_NI(str.str().c_str());
    }
    if(write) {
        if(PyObject_GetBuffer(xyz, buffer, PyBUF_WRITABLE | PyBUF_ND) != 0) {
            str << funcnom << " was passed a read-only buffer for " << bufnom;
            ERR_N_DIE_NO_NI(str.str().c_str());
        }
    }else{
        if(PyObject_GetBuffer(xyz, buffer, PyBUF_ND) != 0) {
            str << funcnom << " was passed a ??not good enough?? buffer for ";
            str << bufnom;
            ERR_N_DIE_NO_NI(str.str().c_str());
        }
    }
    if(!PyBuffer_IsContiguous(buffer, 'C')) {
        str << funcnom << " was passed a noncontiguousbuffer for " << bufnom;
        ERR_N_DIE_NO_NI(str.str().c_str());
    }
    if(buffer->ndim != 3) {
        str << funcnom << " expected ndim=3, got ndim=" << buffer->ndim; 
        str << " for " << bufnom;
        ERR_N_DIE_NO_NI(str.str().c_str());
    }
    if(expected_y_shape != -1 && expected_x_shape != -1) {
        if(buffer->shape[0] != 3 || 
                buffer->shape[1] != expected_y_shape || 
                buffer->shape[2] != expected_x_shape) {
            str << funcnom << " expected shape=(3, " << expected_y_shape;
            str << ", " << expected_x_shape << "), got shape=(";
            str << buffer->shape[0] << ", " << buffer->shape[1]; 
            str << ", " << buffer->shape[2] << ") for " << bufnom;
            ERR_N_DIE_NO_NI(str.str().c_str());
        }
    }
    if(buffer->itemsize != 4) {
        str << "expected item size=" << 4;
        str << ", got item size=" << buffer->itemsize;
        str << " for " << bufnom << " (item should be float)";
        ERR_N_DIE_NO_NI(str.str().c_str());
    }
    if(buffer->strides != NULL) {
        str << "bugger, there are strides: [";
        for (int i = 0; i < buffer->ndim; i++) {
            str << buffer->strides[i];
            if (i < buffer->ndim-1) {
                str << ", ";
            }
        }
        str << "]";
        ERR_N_DIE_NO_NI(str.str().c_str());
    }
    return xyz;
}

PyObject *check_theta(
        PyObject *theta, Py_buffer *buffer, 
        const char *funcnom, const char *bufnom, 
        int write, int expected_y_shape, int expected_x_shape) {
    std::stringstream str;
    if(!PyObject_CheckBuffer(theta)) {
        str << funcnom << " was passed a non-buffer for " << bufnom;
        ERR_N_DIE_NO_NI(str.str().c_str());
    }
    if(write) {
        if(PyObject_GetBuffer(theta, buffer, PyBUF_WRITABLE | PyBUF_ND) != 0) {
            str << funcnom << " was passed a read-only buffer for " << bufnom;
            ERR_N_DIE_NO_NI(str.str().c_str());
        }
    }else{
        if(PyObject_GetBuffer(theta, buffer, PyBUF_ND) != 0) {
            str << funcnom << " was passed a ??not good enough?? buffer for ";
            str << bufnom;
            ERR_N_DIE_NO_NI(str.str().c_str());
        }
    }
    if(!PyBuffer_IsContiguous(buffer, 'C')) {
        str << funcnom << " was passed a noncontiguousbuffer for " << bufnom;
        ERR_N_DIE_NO_NI(str.str().c_str());
    }
    if(buffer->ndim != 2) {
        str << funcnom << " expected ndim=2, got ndim=" << buffer->ndim; 
        str << " for " << bufnom;
        ERR_N_DIE_NO_NI(str.str().c_str());
    }
    if(expected_y_shape != -1 && expected_x_shape != -1) {
        if( 
                buffer->shape[0] != expected_y_shape || 
                buffer->shape[1] != expected_x_shape) {
            str << funcnom << " expected shape=(3, " << expected_y_shape;
            str << ", " << expected_x_shape << "), got shape=(";
            str << buffer->shape[0] << ", " << buffer->shape[1]; 
            str << ", " << buffer->shape[2] << ") for " << bufnom;
            ERR_N_DIE_NO_NI(str.str().c_str());
        }
    }
    if(buffer->itemsize != 4) {
        str << "expected item size=" << 4;
        str << ", got item size=" << buffer->itemsize;
        str << " for " << bufnom << " (item should be float)";
        ERR_N_DIE_NO_NI(str.str().c_str());
    }
    if(buffer->strides != NULL) {
        str << "bugger, there are strides: [";
        for (int i = 0; i < buffer->ndim; i++) {
            str << buffer->strides[i];
            if (i < buffer->ndim-1) {
                str << ", ";
            }
        }
        str << "]";
        ERR_N_DIE_NO_NI(str.str().c_str());
    }
    return theta;
}

PyObject *structure_depth_to_xyz(
        PyObject *self, PyObject *args, PyObject *kwargs) {
    Py_buffer depth_buffer;
    Py_buffer xyz_buffer;
    std::stringstream str;
    PyObject* depth = NULL;
    PyObject* xyz = NULL;
    static char *argnoms[] = {"depth", "xyz", NULL};
    PyArg_ParseTupleAndKeywords(args, kwargs, "OO", argnoms, &depth, &xyz);

    depth = check_buffer(depth, &depth_buffer, "depth_to_xyz", "depth", 0, -1, -1, -1);
    if(depth == NULL) return NULL;
    xyz = check_xyz(xyz, &xyz_buffer, "depth_to_xyz", "xyz", 1, 
            depth_buffer.shape[0],
            depth_buffer.shape[1]);
    if(xyz == NULL) return NULL;
    const unsigned short *depth_data = (const unsigned short *) depth_buffer.buf;
    float *xyz_data = (float *) xyz_buffer.buf;
    size_t height = depth_buffer.shape[0];
    size_t width = depth_buffer.shape[1];
    size_t i = 0;

    for(size_t y = 0; y < height; y++) {
        for(size_t x = 0; x < width; x++, i++) {
            float *xptr = xyz_data + 0*width*height + i;
            float *yptr = xyz_data + 1*width*height + i;
            float *zptr = xyz_data + 2*width*height + i;
            if(depth_data[i] == 0) {
                *xptr = 0;
                *yptr = 0;
                *zptr = 0;
            }else{
                CoordinateConverter::convertDepthToWorld(
                    depthStream, x, y, depth_data[i], 
                    xptr, yptr, zptr);
            }
        }
    }

    return xyz;
}

PyObject *structure_xyz_to_theta(PyObject *self, PyObject *args, PyObject *kwargs) 
{
    Py_buffer xyz_buffer;
    Py_buffer theta_buffer;
    std::stringstream str;
    PyObject* xyz = NULL;
    PyObject* theta = NULL;
    static char *argnoms[] = {"xyz", "theta", NULL};
    PyArg_ParseTupleAndKeywords(args, kwargs, "OO", argnoms, &xyz, &theta);

    xyz = check_xyz(xyz, &xyz_buffer, "xyz_to_theta", "xyz", 0, -1, -1);
    if(xyz == NULL) return NULL;
    theta = check_theta(theta, &theta_buffer, "xyz_to_theta", "theta", 1, 
            xyz_buffer.shape[1], xyz_buffer.shape[2]);
    if(theta == NULL) return NULL;
        
    const float *xyz_data = (const float *) xyz_buffer.buf;
    float *theta_data = (float *) theta_buffer.buf;
    size_t height = xyz_buffer.shape[1];
    size_t width = xyz_buffer.shape[2];

    std::cout << "begin theta calc" << std::endl;
    for(size_t y = 0; y < height; y++) {
        for(size_t x = 0; x < width; x++) {
            if(y == 0 || x == 0) {
                theta_data[y*width+x] = 1000; // invalid?
                continue;
            }
            const float *xptr = (const float *)(xyz_data + 0*width*height); 
            const float *yptr = (const float *)(xyz_data + 1*width*height);
            const float *zptr = (const float *)(xyz_data + 2*width*height);
            if(zptr[y*width+x] == 0) {
                theta_data[y*width+x] = 1000; // invalid?
            }else{
                // pt1: [y-1][x]
                // pt2: [y][x]
                // pt3: [y][x-1]
                // u = pt2-pt1
                // v = pt3-pt1
                // compute angle between x-z plane projection of 
                // normal (pt1, pt2, pt3) and view vector ((0,0,1), i think) 
                float ux = xptr[y*width+x] - xptr[(y-1)*width+x];
                float uy = yptr[y*width+x] - yptr[(y-1)*width+x];
                float uz = zptr[y*width+x] - zptr[(y-1)*width+x];

                float vx = xptr[y*width+x-1] - xptr[(y-1)*width+x];
                float vy = yptr[y*width+x-1] - yptr[(y-1)*width+x];
                float vz = zptr[y*width+x-1] - zptr[(y-1)*width+x];

                float nx = uy*vz - uz*vy;
                float ny = uz*vx - ux*vz;
                float nz = ux*vy - uy-vx;

                float theta_rad = atan(nx / nz);
                // atan will return -pi/2 to pi/2. 
                // nz == 0 => nx/nz == inf => atan returns pi/2, yay!
                theta_data[y*width+x] = theta_rad / M_PI * 180.0f;
                std::cout << "theta[121, 161]=" << theta_data[y*width+x] << std::endl;
                //std::cout << " depends on x[" << (x-1
            }
        }
    }
    std::cout << "end theta calc" << std::endl;
    return theta;
}

static PyMethodDef methods[] = {
    {"init", &structure_init, METH_VARARGS, 
        "Initializer function. Call this before trying to read the structure sensor."},
    {"read_frame", (PyCFunction) &structure_read_frame, METH_VARARGS | METH_KEYWORDS, 
        "reads a frame from the depth sensor and infrared camera and returns (depthFrame, irFrame). Can be given a timeout. will return None if timeout is exceeded."},
    {"depth_to_xyz", (PyCFunction) &structure_depth_to_xyz, METH_VARARGS | METH_KEYWORDS, "twiddle?"},
    {"xyz_to_theta", (PyCFunction) &structure_xyz_to_theta, METH_VARARGS | METH_KEYWORDS, "twaddle?"},
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
