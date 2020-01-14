#pragma once

#include <memory>
#include <vector>

#include <opencv2/core/mat.hpp>

using namespace std;
using namespace cv;

namespace pure {

    struct Parameters {
        double min_pupil_diameter_ratio = 0.07 * 2/3;
        double max_pupil_diameter_ratio = 0.29;
    };

    struct Confidence
    {
        double value = 0;
        double aspect_ratio = 0;
        double angular_spread = 0;
        double outline_contrast = 0;
    };

    struct Result {
        Point2f center = {0, 0};
        Size2f axes = {0, 0};
        double angle = 0;
        Confidence confidence = {0, 0, 0, 0};
        bool operator<(const Result& other) const
        {
            return confidence.value < other.confidence.value;
        }
    };

    
    struct OutResult
    {
        float center_x = 0;
        float center_y = 0;
        float first_ax = 0;
        float second_ax = 0;
        double angle = 0;
        double confidence = 0;
        OutResult() = default;
        
        OutResult(const Result& other) :
            center_x(other.center.x),
            center_y(other.center.y),
            first_ax(other.axes.width),
            second_ax(other.axes.height),
            angle(other.angle),
            confidence(other.confidence.value)
        {}
    };
    
    typedef vector<Point> Segment;
    
    class Detector
    {
    public:
        Parameters params;
    public:
        Detector() = default;
        Detector(Parameters params) : params(params) {}
        OutResult detect(const Mat& gray_img, Mat* debug_color_img = nullptr);

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
        Mat dx_img, dy_img, mag_img, bin_img;
        void calculate_canny();
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

        void evaluate_segment(const Segment& segment, Result& result, Mat* tmp=nullptr) const;
        bool segment_large_enough(const Segment& segment) const;
        bool segment_diameter_valid(const Segment& segment) const;
        bool segment_curvature_valid(const Segment& segment) const;
        bool axes_ratio_is_invalid(double ratio) const;
        bool fit_ellipse(const Segment& segment, Result& result) const;
        bool segment_mean_in_ellipse(const Segment& segment, const Result& result) const;

        Confidence calculate_confidence(const Segment& segment, const Result& result) const;
        double angular_edge_spread(const Segment& segment, const Result& result) const;
        double ellipse_outline_constrast(const Result& result, Mat* tmp=nullptr) const;

    private:
        // 3.5. Conditional Segment Combination
        void combine_segments();
        bool proper_intersection(const Rect& r1, const Rect& r2) const;
        Segment merge_segments(const Segment& s1, const Segment& s2) const;

    private:
        Result select_final_segment();

    };

}
