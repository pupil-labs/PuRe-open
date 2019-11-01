#pragma once

#include <opencv2/core/mat.hpp>

using namespace cv;

namespace pure {

    void thin_edges(const Mat& edge_img, Mat& out_img);

    void straighten_edges(const Mat& edge_img, Mat& out_img);

    void break_orthogonals(const Mat& edge_img, Mat& out_img);

}
