#include <ni2/OpenNI.h>
#include <Python.h>

#define ERR_N_DIE(p) {PyErr_SetString(PyExc_RuntimeError, errorMessage(p)); return NULL;}
#define ERR_N_DIE_NO_NI(p) {PyErr_SetString(PyExc_RuntimeError, (p)); return NULL;}

using namespace openni;

PyObject *structure_init(PyObject *self, PyObject *args);

PyObject *structure_destroy(PyObject *self, PyObject *args);

PyObject *structure_read_frame(
    PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *structure_depth_to_xyz(
        PyObject *self, PyObject *args, PyObject *kwargs);

PyObject *structure_xyz_to_theta(
        PyObject *self, PyObject *args, PyObject *kwargs);

const char *modeToString(VideoMode mode);

const char *errorMessage(const char *msg);

PyObject *read_frame_into_array(
        PyObject *dst, VideoFrameRef frame, 
        const char *funcnom, const char *bufnom);

PyObject *read_a_frame(
        PyObject *dst,
        VideoStream& stream, int timeout, 
        PixelFormat expectedPixelFormat,
        const char *nom);

PyObject *chooseDepthVideoMode();

PyObject *check_buffer(
        PyObject *dst, Py_buffer *buffer, 
        const char *funcnom, const char *nom, int write,
        int height, int width, int size);

PyObject *check_xyz(
        PyObject *xyz, Py_buffer *buffer, 
        const char *funcnom, const char *bufnom, int write, 
        int expected_y_shape, int expected_x_shape);

PyObject *check_theta(
        PyObject *theta, Py_buffer *buffer, 
        const char *funcnom, const char *bufnom, int write, 
        int expected_y_shape, int expected_x_shape);

struct XYZConversionFactor {
    int used;
    int xDim;
    int yDim;

    float *xFactors;
    float *yFactors;
};

void initFactor(struct XYZConversionFactor *conversionFactor, int yDim, int xDim);
