#pragma once

#include <opencv2/core/mat.hpp>

using namespace cv;

namespace pure {

    void thin_edges(const Mat& edge_img, Mat& out_img);

    void break_crossings(const Mat& edge_img, Mat& out_img);

    void straighten_edges(const Mat& edge_img, Mat& out_img);

    void break_orthogonals(const Mat& edge_img, Mat& out_img);





    struct Result {
        Point center;
        Size axes;
        double angle;
        double confidence;
        Result() : center(0, 0), axes(0, 0), angle(0), confidence(0) {}
    };
    class Detector
    {
    public:
        Result detect(const Mat& img, Mat& debug_img);

    private:
        bool debug = false;

    };

}
