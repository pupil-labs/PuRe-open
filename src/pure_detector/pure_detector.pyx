import typing as T

import numpy as np


cdef extern from '<opencv2/core.hpp>':
  int CV_8UC1
  int CV_8UC3


cdef extern from '<opencv2/core.hpp>' namespace 'cv':
    cdef cppclass Mat :
        Mat() except +
        Mat( int height, int width, int type, void* data  ) except+
        Mat( int height, int width, int type ) except+


cdef extern from "pure.hpp" namespace "pure":

    cdef struct OutResult:
        float center_x
        float center_y
        float first_ax
        float second_ax
        double angle
        double confidence
    
    cdef cppclass Detector:
        Detector()
        OutResult detect(const Mat& gray_img)
        OutResult detect(const Mat& gray_img, Mat* debug_color_img)




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
        return self.c_detect(gray_img, debug_img)

    cdef OutResult c_detect(
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
