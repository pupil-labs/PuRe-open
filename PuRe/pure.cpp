#include <algorithm>
#include <cmath>
#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "pure.hpp"

using namespace std;
using namespace cv;


namespace pure {

    OutResult Detector::detect(const Mat& input_img, Mat* debug_color_img)
    {
        orig_img = &input_img;
        input_img.copyTo(img);
        debug_img = debug_color_img;
        debug = debug_img != nullptr;

        preprocess();
        detect_edges();

        select_edge_segments();
        combine_segments();

        return select_final_segment();
    }

    void Detector::preprocess()
    {
        // NOTE: we assume the resizing to take place outside, which makes it easier for
        // users to create the debug and output imaged
        normalize(img, img, 0, 255, NORM_MINMAX);
    }

    
    void Detector::detect_edges()
    {   
        calculate_canny();

        if (debug)
        {
            cvtColor(img, *debug_img, CV_GRAY2BGR);
            *debug_img *= 0.5;
        }

        // TODO: is canny already thresholded?
		threshold(img, img, 127, 255, THRESH_BINARY);
        thin_edges();
        break_crossings();
        straighten_edges();
        // NOTE: The straightening resulst in segments that would have been removed by
        // previous edge-thinning! Maybe we should thin again?
        // Example:
        //   X
        //  X X
        // X
        //  X
        // will become:
        //  XXX
        //  X
        //  X
        // which would hit the thinning filter with the top-left corner!
        // TODO: Investigate the effect of this.
        break_orthogonals();

        // if (debug)
        // {
        //     vector<Mat> channels(3);
        //     split(*debug_img, channels);
        //     channels[0] = 0.5 * img;
        //     channels[1] = 0.5 * img;
        //     merge(channels, *debug_img);
        // }
    }
    
    void Detector::calculate_canny()
    {
        constexpr double target_edgepx_ratio = 0.06;
        Canny(*orig_img, img, 0.3*params.canny_upper_threshold, params.canny_upper_threshold, 3, true);
        const double ratio = ((double)countNonZero(img)) / (img.size[0] * img.size[1]);
        params.canny_upper_threshold += (ratio - target_edgepx_ratio) * 255;
        if (debug)
        {
            cout << "T: " << params.canny_upper_threshold << " -> Ratio: " << ratio << endl;
        }
        
    }

    void Detector::thin_edges()
    {
        // Thinning

        // This is an efficient implementation of the orginal thinning algorithm described
        // in the ExCuSe paper (Fuhl et al., 2015). While the PuRe paper links to the ElSe
        // paper (Fuhl et al. 2016), the description there is ambiguous so I looked up the
        // previous paper.

        // The idea of this implementation is to move a 3x3 filter across the image, which
        // removes the central pixel from the edge images, if it would have been matched by
        // any of the 4 described thinning masks. This way we can apply all 4 masks
        // efficiently in one operation.
        
        // If the pixels with E have an edge, pixel X is removed:
        // |_|E|_| |_|E|_| |_|_|_| |_|_|_|
        // |E|X|_| |_|X|E| |E|X|_| |_|X|E|
        // |_|_|_| |_|_|_| |_|E|_| |_|E|_|

        // TODO: See other (more elaborate) thinning algorithms in opencv:
        // https://github.com/opencv/opencv_contrib/blob/master/modules/ximgproc/src/thinning.cpp

        // TODO: Handle borders of the image.
        const uchar *above, *below;
        uchar *current;
        const int rows = img.rows - 2;
        const int cols = img.cols - 2;
        int r, c;
        for (r = 0; r < rows; ++r)
        {
            above = img.ptr(r);
            current = img.ptr(r + 1);
            below = img.ptr(r + 2);
            for (c = 0; c < cols; ++c)
            {
                if (above[c + 1] && current[c] ||
                    above[c + 1] && current[c + 2] ||
                    below[c + 1] && current[c] ||
                    below[c + 1] && current[c + 2])
                {
                    current[c + 1] = 0;
                }
            }
        }
    }



    void Detector::break_crossings()
    {
        // Break connections of more than 2 lines
        // As described in in the ElSe paper (Fuhl et al. 2016)

        const uchar *above, *below;
        const int rows = img.rows - 2;
        const int cols = img.cols - 2;
        uchar *current;
        int r, c;
        for (r = 0; r < rows; ++r)
        {
            above = img.ptr(r);
            current = img.ptr(r + 1);
            below = img.ptr(r + 2);
            for (c = 0; c < cols; ++c)
            {
                int neighbors = 0;
                neighbors += (above[c] > 0) ? 1 : 0;
                neighbors += (above[c + 1] > 0) ? 1 : 0;
                neighbors += (above[c + 2] > 0) ? 1 : 0;
                neighbors += (current[c] > 0) ? 1 : 0;
                neighbors += (current[c + 2] > 0) ? 1 : 0;
                neighbors += (below[c] > 0) ? 1 : 0;
                neighbors += (below[c + 1] > 0) ? 1 : 0;
                neighbors += (below[c + 2] > 0) ? 1 : 0;
                if (neighbors > 2)
                {
                    current[c + 1] = 0;
                }
            }
        }
    }


    void Detector::straighten_edges()
    {
        // Straightening
        // As described in in the ElSe paper (Fuhl et al. 2016)

        // NOTE: The order of operation does make a difference here as some of those filters
        // overlap. Since there was no specific information about the order of application,
        // I implemented then basically in the order of presentation in the paper.

        // TODO: This could be optimized even more with something like karnaugh maps.
        // TODO: Handle border of the image.
        uchar *row0, *row1, *row2, *row3;
        const int rows = img.rows - 3;
        const int cols = img.cols - 3;
        int r, c;
        for (r = 0; r < rows; ++r)
        {
            row0 = img.ptr(r);
            row1 = img.ptr(r + 1);
            row2 = img.ptr(r + 2);
            row3 = img.ptr(r + 3);
            for (c = 0; c < cols; ++c)
            {
                if (row1[c] && row0[c + 1] && row1[c + 2])
                {
                    //  X
                    // XXX
                    row0[c + 1] = 0;
                    row1[c + 1] = 255;
                }
                if (row1[c] && row0[c + 1] && row0[c + 2] && row1[c + 3])
                {
                    //  XX
                    // XXXX
                    row0[c + 1] = 0;
                    row0[c + 2] = 0;
                    row1[c + 1] = 255;
                    row1[c + 2] = 255;
                }
                if (row0[c + 1] && row1[c] && row2[c + 1])
                {
                    //  X
                    // XX
                    //  X
                    row1[c] = 0;
                    row1[c + 1] = 255;
                }
                if (row0[c + 1] && row1[c] && row2[c] && row3[c + 1])
                {
                    //  X
                    // XX
                    // XX
                    //  X
                    row1[c] = 0;
                    row2[c] = 0;
                    row1[c + 1] = 255;
                    row2[c + 1] = 255;
                }
                if (row0[c] && row1[c + 1] && row2[c])
                {
                    // X
                    // XX
                    // X
                    row1[c + 1] = 0;
                    row1[c] = 255;
                }
                if (row0[c] && row1[c + 1] && row2[c + 1] && row3[c])
                {
                    // X
                    // XX
                    // XX
                    // X
                    row1[c + 1] = 0;
                    row2[c + 1] = 0;
                    row1[c] = 255;
                    row2[c] = 255;
                }
                if (row0[c] && row1[c + 1] && row0[c + 2])
                {
                    // XXX
                    //  X
                    row1[c + 1] = 0;
                    row0[c + 1] = 255;
                }
                if (row0[c] && row1[c + 1] && row1[c + 2] && row0[c + 3])
                {
                    // XXXX
                    //  XX
                    row1[c + 1] = 0;
                    row1[c + 2] = 0;
                    row0[c + 1] = 255;
                    row0[c + 2] = 255;
                }
            }
        }
    }

    void Detector::break_orthogonals()
    {
        // Removal of orthogonal connections
        // As described in in the ElSe paper (Fuhl et al. 2016)

        uchar *row0, *row1, *row2, *row3;
        const uchar *row4, *row5;
        const int rows = img.rows - 5;
        const int cols = img.cols - 5;
        int r, c;
        for (r = 0; r < rows; ++r)
        {
            row0 = img.ptr(r);
            row1 = img.ptr(r + 1);
            row2 = img.ptr(r + 2);
            row3 = img.ptr(r + 3);
            row4 = img.ptr(r + 4);
            row5 = img.ptr(r + 5);
            for (c = 0; c < cols; ++c)
            {
                // Every pattern affects only 1 pixel. Here the patterns are grouped by the
                // pixel they affect. First pixel read is the one potentially turned off.
                // Following are all patterns that can turn this pixel off. This allows for
                // using pattern overlapping, e.g. between f2 and g2 (see below). Also it
                // results in every pixel being written only once.
                
                // See here which pixels are affected by which pattern:
                // +----+----+----+----+
                // |    |d1d3|f1  |g1  |
                // |    |    |    |    |
                // +----+----+----+----+
                // |    |e3  |    |e1  |
                // |    |    |    |    |
                // +----+----+----+----+
                // |f2g2|d2d4|f3f4|    |
                // |    |    |g4  |    |
                // +----+----+----+----+
                // |    |e2  |g3  |e4  |
                // |    |    |    |    |
                // +----+----+----+----+
                
                if (row0[c + 1] && (
                    row0[c] && row1[c + 2] && row2[c +2] || // d1
                    row0[c + 2] && row1[c] && row2[c] // d3
                )) row0[c + 1] = 0;

                if (row0[c + 2] && row1[c + 1] && row1[c + 3] && row2[c] && row2[c + 4]) row0[c + 2] = 0; // f1

                if (row0[c + 3] && row0[c + 2] && row1[c + 1] && row1[c + 4] && row2[c] && row2[c + 5]) row0[c + 3] = 0; // g1

                if (row1[c + 1] && row0[c + 2] && row0[c + 3] && row0[c + 4] && row2[c] && row3[c] && row4[c]) row1[c + 1] = 0; // e3

                if (row1[c + 3] && row0[c] && row0[c + 1] && row0[c + 2] && row2[c + 4] && row3[c + 4] && row4[c + 4]) row1[c + 3] = 0; //e1

                if (row2[c] && row1[c + 1] && row0[c + 2] && (
                    row3[c + 1] && row4[c + 2] || // f2
                    row3[c] && row4[c + 1] && row5[c + 2] // g2
                )) row2[c] = 0;

                if (row2[c + 1] && (
                    row0[c] && row1[c] && row2[c + 2] || // d2
                    row0[c + 2] && row1[c + 2] && row2[c] // d4
                )) row2[c + 1] = 0;

                if (row2[c + 2] && row0[c] && row1[c + 1] && (
                    row3[c + 1] && row4[c] || // f3
                    row1[c + 3] && row0[c + 4] || // f4
                    row2[c + 3] && row1[c + 4] && row0[c + 5] // g4
                )) row2[c + 2] = 0;

                if (row3[c + 1] && row0[c] && row1[c] && row2[c] && row4[c + 2] && row4[c + 3] && row4[c + 4]) row3[c + 1] = 0; //e2

                if (row3[c + 2] && row0[c] && row1[c + 1] && row2[c + 2] && row4[c + 1] && row5[c]) row3[c + 2] = 0; //g3

                if (row3[c + 3] && row0[c + 4] && row1[c + 4] && row2[c + 4] && row4[c] && row4[c + 1] && row4[c + 2]) row3[c + 3] = 0; // e4
            }
        }
    }

    void Detector::select_edge_segments()
    {
        findContours(img, segments, RETR_LIST, CHAIN_APPROX_TC89_KCOS);

        // NOTE: We are essentially re-using the result from previous runs. Need to make
        // sure that either all values will be overwritten or confidence will be set to
        // 0 for every result!
        candidates.resize(segments.size());

        const int W = img.cols;
        const int H = img.rows;
        const double diagonal = sqrt(W * W + H * H);
        MIN_PUPIL_DIAMETER = params.min_pupil_diameter_ratio * diagonal;
        MAX_PUPIL_DIAMETER = params.max_pupil_diameter_ratio * diagonal;

        for (size_t segment_i = 0; segment_i < segments.size(); ++segment_i)
        {
        	evaluate_segment(segments[segment_i], candidates[segment_i]);
        }
    }

    void Detector::evaluate_segment(const Segment& segment, Result& result) const
    {
        // 3.3.1 Filter small segments
        if (!segment_large_enough(segment))
        {
            result.confidence.value = 0;
            return;
        }

        // 3.3.2 Filter segments based on approximate diameter
        if (!segment_diameter_valid(segment))
        {
            result.confidence.value = 0;
            return;
        }

        // 3.3.3 Filter segments based on curvature approximation
        if (!segment_curvature_valid(segment))
        {
            result.confidence.value = 0;
            return;
        }

        // 3.3.4 Ellipse fitting
        if (!fit_ellipse(segment, result))
        {
            result.confidence.value = 0;
            return;
        }

        // 3.3.5 Additional filter
        if (!segment_mean_in_ellipse(segment, result))
        {
            result.confidence.value = 0;
            return;
        }
        
        // 3.4  Calculate confidence
        result.confidence = calculate_confidence(segment, result);
    }

    inline bool Detector::segment_large_enough(const Segment& segment) const
    {
        return segment.size() >= 5;
    }

    bool Detector::segment_diameter_valid(const Segment& segment) const
    {
        double approx_diameter = 0;
        const auto end = segment.end();
        for (auto p1 = segment.begin(); p1 != end; ++p1)
        {
            for (auto p2 = p1 + 1; p2 != end; ++p2)
            {
                approx_diameter = max(approx_diameter, norm(*p1 - *p2));
                // we can early exit, because we will only get bigger
                if (approx_diameter > MAX_PUPIL_DIAMETER)
                {
                    break;
                }
            }
            // we can early exit, because we will only get bigger
            if (approx_diameter > MAX_PUPIL_DIAMETER)
            {
                break;
            }
        }
        return MIN_PUPIL_DIAMETER < approx_diameter && approx_diameter < MAX_PUPIL_DIAMETER;
    }
    bool Detector::segment_curvature_valid(const Segment& segment) const
    {
        auto rect = minAreaRect(segment);
        double ratio = rect.size.width / rect.size.height;
        return !axes_ratio_is_invalid(ratio);
    }

    inline bool Detector::axes_ratio_is_invalid(double ratio) const
    {
        return (
            ratio < params.axes_ratio_threshold ||
            ratio > 1 && (1.0 / ratio) < params.axes_ratio_threshold
        );
    }

    bool Detector::fit_ellipse(const Segment& segment, Result& result) const
    {
        // NOTE: This is a cv::RotatedRect, see
        // https://stackoverflow.com/a/32798273 for conversion to ellipse
        // parameters. Also below.
        auto fit = fitEllipse(segment);

        // 	(I) discard if center outside image boundaries
        if (fit.center.x < 0 || fit.center.y < 0 || fit.center.x > img.cols || fit.center.y > img.rows)
        {
            return false;
        }

        // 	(II) discard if ellipse is too skewed
        auto ratio = fit.size.width / fit.size.height;
        if (axes_ratio_is_invalid(ratio))
        {
            return false;
        }
        
        result.center = fit.center;
        result.angle = fit.angle;
        // NOTE: width always provides the first axis, which corresponds to the
        // angle. Height provides the second axis, which corresponds to angle +
        // 90deg. This is NOT related to major/minor axes! But we also don't
        // need the information of which is the major and which is the minor
        // axis.
        result.axes = {
            fit.size.width / 2.0f,
            fit.size.height / 2.0f
        };
        return true;
    }

    bool Detector::segment_mean_in_ellipse(const Segment& segment, const Result& result) const
    {
        Point2f segment_mean(0, 0);
        for (const auto& p : segment)
        {
            segment_mean += Point2f(p);
        }
        // NOTE: cv::Point operator /= does not work with size_t scalar
        segment_mean.x /= segment.size();
        segment_mean.y /= segment.size();

        // We need to test if the mean lies in the rhombus defined by the
        // rotated rect of the ellipse. Essentially each vertex of the
        // rhombus corresponds to a midpoint of the sides of the rect.
        // Testing is easiest if we don't rotate all points of the rect, but
        // rotate the segment_mean backwards, because then we can test
        // against the axis-aligned rhombus.

        // See the following rhombus for reference. Note that we only need
        // to test for Q1, since the we can center at (0,0) and the rest is
        // symmetry. (not in image coordinates, but y-up)
        //   /|\
        //  / | \ Q1
        // /  |  \
        //---------
        // \  |  /
        //  \ | /
        //   \|/

        // Shift rotation to origin to center at (0,0).
        segment_mean -= result.center; 
        // Rotate backwards with negative angle
        const auto angle_rad = - result.angle * M_PI / 180.0f;
        const float angle_cos = static_cast<float>(cos(angle_rad));
        const float angle_sin = static_cast<float>(sin(angle_rad));
        // We take the abs values to utilize symmetries. This way can do the
        // entire testing in Q1 of the rhombus.
        Point2f unrotated(
            abs(segment_mean.x * angle_cos - segment_mean.y * angle_sin),
            abs(segment_mean.x * angle_sin + segment_mean.y * angle_cos)
        );
        
        // Discard based on testing first rhombus quadrant Q1. This tests
        // for containment in the axis-aligned triangle.
        return (
            (unrotated.x < result.axes.width) &&
            (unrotated.y < result.axes.height) &&
            ((unrotated.x / result.axes.width) + (unrotated.y / result.axes.height) < 1)
        );
    }

    
    Confidence Detector::calculate_confidence(const Segment& segment, const Result& result) const
    {
        Confidence conf;
        conf.aspect_ratio = result.axes.width / result.axes.height;
        if (conf.aspect_ratio > 1.0) conf.aspect_ratio = 1.0 / conf.aspect_ratio;

        conf.angular_spread = angular_edge_spread(segment, result);
        conf.outline_contrast = ellipse_outline_constrast(result);

        // compute value
        conf.value = (conf.aspect_ratio + conf.angular_spread + conf.outline_contrast) / 3.0;

        return conf;
    }

    double Detector::angular_edge_spread(const Segment& segment, const Result& result) const
    {
        // Q2 | Q1
        // -------
        // Q3 | Q4
        // (not in image coordinates, but y-up)
        bool points_in_q1 = false;
        bool points_in_q2 = false;
        bool points_in_q3 = false;
        bool points_in_q4 = false;

        for (const auto& p : segment)
        {
            if (p.x > result.center.x)
            {
                if (p.y > result.center.y) points_in_q1 = true;
                else points_in_q4 = true;
            }
            else
            {
                if (p.y > result.center.y) points_in_q2 = true;
                else points_in_q3 = true;
            }
            // early exit
            if (points_in_q1 && points_in_q2 && points_in_q3 && points_in_q4) break;
        }
        
        double spread = 0.0;
        if (points_in_q1) spread += 0.25;
        if (points_in_q2) spread += 0.25;
        if (points_in_q3) spread += 0.25;
        if (points_in_q4) spread += 0.25;
        return spread;
    }

    double Detector::ellipse_outline_constrast(const Result& result) const
    {
        double contrast = 0;
        // Iterate circle with stride of 10 degrees (all in radians)
        constexpr double stride = 10 * M_PI / 180.0;
        double angle = 0;
        // NOTE: A for-loop: for(angle=0; angle < 2*PI; ...) will result
        // in 37 iterations because of rounding errors. This will result
        // in one line being counted twice.
        constexpr int n_iterations = 36;
        constexpr int NEIGHBORHOOD_4 = 4;
        const double minor = min(result.axes.width, result.axes.height);
        for (int i = 0; i < n_iterations; ++i)
        {
            Point2f offset(
                static_cast<float>(minor * cos(angle)),
                static_cast<float>(minor * sin(angle))
            );
            Point2f outline_point = result.center + offset;

            LineIterator inner_line(*orig_img, result.center, outline_point, NEIGHBORHOOD_4);

            double inner_avg = 0;
            for (int j = 0; j < inner_line.count; j++, ++inner_line)
            {
                inner_avg += *(*inner_line);
            }
            inner_avg /= inner_line.count;


            LineIterator outer_line(*orig_img, outline_point, outline_point + offset, NEIGHBORHOOD_4);
            double outer_avg = 0;
            for (int j = 0; j < outer_line.count; j++, ++outer_line)
            {
                outer_avg += *(*outer_line);
            }
            outer_avg /= outer_line.count;

            // TODO: How is this actually supposed to be calculated!?
            if (inner_avg < outer_avg) contrast += 1;

            angle += stride;
        }
        return contrast / n_iterations;
    }


    void Detector::combine_segments()
    {
        vector<Segment> combined_segments;
        vector<Result> combined_results;
        if (segments.size() == 0) return; 
        size_t end1 = segments.size() - 1;
        size_t end2 = segments.size();
        for (size_t idx1 = 0; idx1 < end1; ++idx1)
        {
            auto& result1 = candidates[idx1];
            if (result1.confidence.value == 0) continue;
            auto& segment1 = segments[idx1]; 
            const auto rect1 = boundingRect(segment1);
            for (size_t idx2 = idx1 + 1; idx2 < end2; ++idx2)
            {
                auto& result2 = candidates[idx1];
                if (result2.confidence.value == 0) continue;
                auto& segment2 = segments[idx2];
                const auto rect2 = boundingRect(segment2);

                if (proper_intersection(rect1, rect2))
                {
                    auto new_segment = merge_segments(segment1, segment2);
                    Result new_result;
                    evaluate_segment(new_segment, new_result);
                    if (new_result.confidence.value == 0) continue;
                    const auto previous_contrast = max(
                        result1.confidence.outline_contrast,
                        result2.confidence.outline_contrast
                    ); 
                    if (new_result.confidence.outline_contrast <= previous_contrast) continue;
                    combined_segments.push_back(new_segment);
                    combined_results.push_back(new_result);
                }
            }
        }
        segments.insert(segments.end(), combined_segments.begin(), combined_segments.end());
        candidates.insert(candidates.end(), combined_results.begin(), combined_results.end());
    }

    bool Detector::proper_intersection(const Rect& r1, const Rect& r2) const
    {
        const int x11 = r1.x;
        const int x12 = r1.x + r1.width;
        const int x21 = r2.x;
        const int x22 = r2.x + r2.width;
        const int y11 = r1.y;
        const int y12 = r1.y + r1.height;
        const int y21 = r2.y;
        const int y22 = r2.y + r2.height;

        return (
            (x11 < x21 && x21 < x12 && x12 < x22) // x: r1 < r2 and intersecting
            || (x21 < x11 && x11 < x22 && x22 < x12) // x: r2 < r1 and intersecting
            || (y11 < y21 && y21 < y12 && y12 < y22) // y: r1 < r2 and intersecting
            || (y21 < y11 && y11 < y22 && y22 < y12) // y: r2 < r1 and intersecting
        );
    }
    
    Segment Detector::merge_segments(const Segment& s1, const Segment& s2) const
    {
        // TODO: How does this work? Naive approach is to just take the convex hull of
        // the union of both segments. But there is no documentation.
        Segment combined;
        combined.insert(combined.end(), s1.begin(), s1.end());
        combined.insert(combined.end(), s2.begin(), s2.end());
        Segment hull;
        // NOTE: convexHull does not support in-place computation.
        convexHull(combined, hull);
        return hull;
    }

    Result Detector::select_final_segment()
    {
        if (candidates.size() == 0)
        {
            return Result();
        }
        Result initial_pupil = *std::max_element(candidates.begin(), candidates.end());
        double semi_major = max(initial_pupil.axes.width, initial_pupil.axes.height);
        Result *candidate = nullptr;
        for (auto& result : candidates)
        {
            if (max(result.axes.width, result.axes.height) > semi_major) continue;
            if (result.confidence.outline_contrast < 2/3 * initial_pupil.confidence.outline_contrast) continue;
            if (norm(initial_pupil.center - result.center) > semi_major) continue;
            if (candidate && result.confidence.value <= candidate->confidence.value) continue;
            candidate = &result;
        }
        return (candidate) ? *candidate : initial_pupil;
    }

}

