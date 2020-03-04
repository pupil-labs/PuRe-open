import typing as T

import numpy as np
cimport numpy as np

from libcpp cimport bool

cdef extern from '<opencv2/core.hpp>':
  int CV_8UC1
  int CV_8UC3


cdef extern from '<opencv2/core.hpp>' namespace 'cv':
    cdef cppclass Mat :
        Mat() except +
        Mat(int height, int width, int type, void* data) except +
        Mat(int height, int width, int type) except +
        unsigned char* data

    cdef cppclass Point_[T]:
        Point_() except +
        T x
        T y

    cdef cppclass Size_[T]:
        Size_() except +
        T width
        T height
    
    ctypedef Point_[float] Point2f
    ctypedef Size_[float] Size2f


cdef extern from "pure.hpp" namespace "pure":

    cdef struct Parameters:
        bool auto_pupil_diameter
        double min_pupil_diameter
        double max_pupil_diameter

    cdef struct Confidence:
        double value
        double aspect_ratio
        double angular_spread
        double outline_contrast

    cdef struct Result:
        Point2f center
        Size2f axes
        double angle
        Confidence confidence
    
    cdef cppclass Detector:
        Parameters params
        Detector()
        Result detect(const Mat& gray_img)
        Result detect(const Mat& gray_img, Mat* debug_color_img)


cdef class ParametersWrapper:
    cdef Parameters c_params

    def __cinit__(self):
        # need to mimic default values from c++
        self.c_params.auto_pupil_diameter = True
        self.c_params.min_pupil_diameter = 0.0
        self.c_params.max_pupil_diameter = 0.0

    property auto_pupil_diameter:
        def __get__(self):
            return self.c_params.auto_pupil_diameter
        def __set__(self, bool value):
            self.c_params.auto_pupil_diameter = value

    property min_pupil_diameter:
        def __get__(self):
            return self.c_params.min_pupil_diameter
        def __set__(self, double value):
            self.c_params.min_pupil_diameter = value

    property max_pupil_diameter:
        def __get__(self):
            return self.c_params.max_pupil_diameter
        def __set__(self, double value):
            self.c_params.max_pupil_diameter = value


cdef class PuReDetector:

    cdef public ParametersWrapper params

    cdef Detector* c_detector_ptr

    def __cinit__(self, *args, **kwargs):
        self.c_detector_ptr = new Detector()
        self.params = ParametersWrapper()

    def __dealloc__(self):
        del self.c_detector_ptr

    def detect(self, gray_img: np.ndarray) -> T.Dict[str, T.Any]:
        result, _ = self.c_detect(gray_img, debug=False)
        return result

    def detect_debug(self, gray_img: np.ndarray) -> T.Tuple[T.Dict[str, T.Any], np.ndarray]:
        result, debug_img = self.c_detect(gray_img, debug=True)
        return result, debug_img

    cdef tuple c_detect(self, gray_img: np.ndarray, debug: bool=False):
        # NOTE: tuple unpacking does not work with cimport numpy
        image_height, image_width = (gray_img.shape[0], gray_img.shape[1])

        # cython memory views for accessing the raw data (does not copy)
        # NOTE: [:, ::1] marks the view as c-contiguous
        cdef unsigned char[:, ::1] gray_img_data = gray_img
        cdef unsigned char[:, :, ::1] debug_img_data

        cdef Mat gray_mat = Mat(image_height, image_width, CV_8UC1, <void *> &gray_img_data[0, 0])
        cdef Mat debug_mat

        debug_img = None

        self.c_detector_ptr.params = self.params.c_params

        if debug:
            debug_mat = Mat()
            c_result = self.c_detector_ptr.detect(gray_mat, &debug_mat)
            debug_img = np.empty((image_height, image_width, 3), dtype=np.uint8)
            debug_img[...] = <unsigned char[:image_height, :image_width, :3]>debug_mat.data
        else:
            c_result = self.c_detector_ptr.detect(gray_mat, NULL)

        self.params.c_params = self.c_detector_ptr.params

        # convert c struct to python dict
        result = {}
        result["ellipse"] = {
            "center": (c_result.center.x, c_result.center.y),
            "axes": (2 * c_result.axes.width, 2 * c_result.axes.height),
            "angle": c_result.angle,
        }
        result["diameter"] = max(result["ellipse"]["axes"])
        result["location"] = result["ellipse"]["center"]
        result["confidence"] = c_result.confidence.outline_contrast
        
        return result, debug_img
