import typing as T

import numpy as np


cdef extern from '<opencv2/core.hpp>':
  int CV_8UC1
  int CV_8UC3


cdef extern from '<opencv2/core.hpp>' namespace 'cv':
    cdef cppclass Mat :
        Mat() except +
        Mat(int height, int width, int type, void* data) except +
        Mat(int height, int width, int type) except +

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
        Detector()
        Result detect(const Mat& gray_img)
        Result detect(const Mat& gray_img, Mat* debug_color_img)




cdef class PuReDetector:

    cdef Detector* c_detector_ptr

    def __cinit__(self, *args, **kwargs):
        self.c_detector_ptr = new Detector()

    def __dealloc__(self):
        del self.c_detector_ptr

    def detect(
        self,
        gray_img: np.ndarray,
        debug_img: T.Optional[np.ndarray]=None,
    ):
        # TODO: This returns semi-axes! The other detectors expect full-axes!
        # This needs to be adjusted before finalizing!
        c_result = self.c_detect(gray_img, debug_img)

        # convert c struct to python dict
        result = {}
        result["ellipse"] = {
            "center": (c_result.center.x, c_result.center.y),
            "axes": (2 * c_result.axes.width, 2 * c_result.axes.height),
            "angle": c_result.angle,
        }
        result["diameter"] = max(result["ellipse"]["axes"])
        result["location"] = result["ellipse"]["center"]
        result["confidence"] = c_result.confidence.value
        return result

    cdef Result c_detect(
        self,
        gray_img: np.ndarray,
        debug_img: T.Optional[np.ndarray]=None,
    ):
        image_height, image_width = gray_img.shape

        # cython memory views for accessing the raw data (does not copy)
        # NOTE: [:, ::1] marks the view as c-contiguous
        cdef unsigned char[:, ::1] gray_img_data = gray_img
        cdef unsigned char[:, :, ::1] debug_img_data

        cdef Mat gray_mat = Mat(image_height, image_width, CV_8UC1, <void *> &gray_img_data[0, 0])
        cdef Mat debug_mat

        if debug_img is not None:
            debug_img_data = debug_img
            debug_mat = Mat(image_height, image_width, CV_8UC3, <void *> &debug_img_data[0, 0, 0])
            result = self.c_detector_ptr.detect(gray_mat, &debug_mat)
        else:
            result = self.c_detector_ptr.detect(gray_mat, NULL)

        return result
