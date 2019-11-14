#pragma once

#include <vector>

#include <opencv2/core/mat.hpp>

using namespace std;
using namespace cv;

namespace pure {

    



    struct Parameters {
        double canny_lower_threshold = 160;
        double canny_upper_threshold = 160*2;
        double min_pupil_diameter_ratio = 0.07 * 2/3;
        double max_pupil_diameter_ratio = 0.29;
        double axes_ratio_threshold = 0.2;
    };

    struct Result {
        Point2f center = {0, 0};
        Size2f axes = {0, 0};
        double angle = 0;
        double confidence = 0;
        bool operator<(const Result& other) const
        {
            return confidence < other.confidence;
        }
    };
    
    typedef vector<Point> Segment;
    class Detector
    {
    public:
        Parameters params;
    public:
        Detector() = default;
        Detector(Parameters params) : params(params) {}
        Result detect(const Mat& gray_img, Mat* debug_color_img = nullptr);

    private:
        Mat img;
        // Note: the pointers are not owned, but just cached here.
        const Mat* orig_img;
        Mat* debug_img;
        bool debug = true;

    private:
        // 3.1. Preprocessing
        void preprocess();

    private:
        // 3.2. Edge Detection and Morphological Manipulation
        void detect_edges();
        void thin_edges();
        void break_crossings();
        void straighten_edges();
        void break_orthogonals();

    private:
        // 3.3. Edge Segment Selection and 3.4. Confidence Measure
        vector<Segment> segments;
        vector<Result> candidates;
        double MIN_PUPIL_DIAMETER, MAX_PUPIL_DIAMETER;
        void select_edge_segments();
        
        bool segment_large_enough(const Segment& segment);
        bool segment_diameter_valid(const Segment& segment);
        bool segment_curvature_valid(const Segment& segment);
        bool axes_ratio_is_invalid(double ratio);
        bool fit_ellipse(const Segment& segment, Result& result);
        bool segment_mean_in_ellipse(const Segment& segment, const Result& result);

        double calculate_confidence(const Segment& segment, const Result& result);
        double angular_edge_spread(const Segment& segment, const Result& result);
        double ellipse_outline_constrast(const Result& result);


    };

}
