#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <queue>
#include <tuple>
#include <array>
#include <bitset>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/ocl.hpp>

#include "pure.hpp"

using namespace std;
using namespace cv;


namespace pure {

    Result Detector::detect(const Mat& input_img, Mat* debug_color_img)
    {
        // setup debugging
        debug = debug_color_img != nullptr;
        // NOTE: when debugging, the debug_color_img pointer will get filled in the end.
        // The intermediate debug image will potentially be downscaled in order to match
        // the working image. It will be upscaled to input image size in the end.

        // preprocessing
        if (!preprocess(input_img))
        {
            // preprocessing can fail for invalid parameters
            // TODO: add debug information?
            // still postprocess in order to return correct debug image
            Result dummy_result;
            postprocess(dummy_result, input_img, debug_color_img);
            return dummy_result;
        }
        
        detect_edges();

        if (debug)
        {
            // draw edges onto debug view
            Mat edge_color;
            cvtColor(edge_img, edge_color, COLOR_GRAY2BGR);
            debug_img = max(debug_img, 0.5 * edge_color);
        }

        select_edge_segments();
        combine_segments();

        if (debug)
        {
            // Draw all non-zero-confidence segments onto debug view with color coding
            // for confidence. Red: confidence == 0, green: confidence == 1 
            for (size_t i = 0; i < segments.size(); ++i)
            {
                const auto& segment = segments[i];
                const auto& result = candidates[i];
                const double c = result.confidence.value;
                if (c == 0) continue;
                const Scalar color(0, 255 * min(1.0, 2.0 * c), 255 * min(1.0, 2.0 * (1 - c)));
                
                Mat blend = debug_img.clone();
                ellipse(
                    blend,
                    Point(result.center),
                    Size(result.axes),
                    result.angle,
                    0, 360,
                    color,
                    FILLED
                );
                debug_img = 0.9 * debug_img + 0.1 * blend;
                polylines(debug_img, segment, false, 0.8 * color);
            }

            // Draw ellipse min/max indicators onto debug view.
            Point center(orig_img.cols / 2, orig_img.rows / 2);
            Size size(orig_img.cols, orig_img.rows);
            const Scalar white(255, 255, 255);
            const Scalar black(0, 0, 0);
            const Scalar blue(255, 150, 0);
            Mat mask = Mat::zeros(size, CV_8UC3);
            const int min_pupil_radius = static_cast<int>(round(min_pupil_diameter / 2));
            const int max_pupil_radius = static_cast<int>(round(max_pupil_diameter / 2));
            circle(mask, center, max_pupil_radius, white, FILLED);
            circle(mask, center, min_pupil_radius, black, FILLED);
            Mat colored(size, CV_8UC3, blue);
            colored = min(mask, colored);
            debug_img = debug_img * 0.9 + colored * 0.1;
            circle(debug_img, center, max_pupil_radius, blue);
            circle(debug_img, center, min_pupil_radius, blue);
        }

        Result final_result = select_final_segment();

        // draw confidence and pupil diameter info
        if (debug)
        {
            // confidence indicator
            const double c = final_result.confidence.value;
            const Scalar color(0, 255 * min(1.0, 2.0 * c), 255 * min(1.0, 2.0 * (1 - c)));
            int decimal = static_cast<int>(round(c * 10));
            string confidence_string = decimal >= 10 ? "1.0" : "0." + to_string(decimal);
            float font_scale = 0.4f;
            const Scalar white(255, 255, 255);
            int pos = static_cast<int>(round(c * debug_img.cols));
            line(debug_img, Point(pos, debug_img.rows), Point(pos, debug_img.rows - 20), color, 2);
            int baseline = 0;
            Size text_size = getTextSize(confidence_string, FONT_HERSHEY_SIMPLEX, font_scale, 1, &baseline);
            putText(debug_img, confidence_string, Point(c < 0.5 ? pos : pos - text_size.width, debug_img.rows - 20), FONT_HERSHEY_SIMPLEX, font_scale, white);

            // if conf > 0, visualize pupil diameter
            if (c > 0)
            {
                Point center(orig_img.cols / 2, orig_img.rows / 2);
                const Scalar green(0, 255, 0);
                int diameter = static_cast<int>(round(max(final_result.axes.width, final_result.axes.height)));
                circle(debug_img, center, diameter, green);

                const float inverse_factor = scaling_factor != 0.0f ? static_cast<float>(1.0f / scaling_factor) : 1.0f;
                string diameter_text = to_string(diameter);
                text_size = getTextSize(diameter_text, FONT_HERSHEY_SIMPLEX, font_scale, 1, &baseline);
                Point text_offset = Point(text_size.width, -text_size.height) / 2;
                putText(debug_img, diameter_text, center - text_offset, FONT_HERSHEY_SIMPLEX, font_scale, white);
            }
        }

        postprocess(final_result, input_img, debug_color_img);

        return final_result;
    }

    bool Detector::preprocess(const Mat& input_img)
    {
        constexpr int target_width = 192;
        constexpr int target_height = 192;
        constexpr int target_area = target_width * target_height;
        int input_area = input_img.cols * input_img.rows;

        // check if we need to shrink input image
        if (false && input_area > target_area)
        {
            scaling_factor = sqrt(target_area / (double)input_area);
            // OpenCV docs recommend INTER_AREA interpolation for shrinking images
#if CV_MAJOR_VERSION == 3
            resize(input_img, orig_img, Size(0, 0), scaling_factor, scaling_factor, CV_INTER_AREA);
#elif CV_MAJOR_VERSION == 4
            resize(input_img, orig_img, Size(0, 0), scaling_factor, scaling_factor, InterpolationFlags::INTER_AREA);
#endif
        }
        else
        {
            scaling_factor = 0.0;
            orig_img = input_img.clone();
        }

        normalize(orig_img, orig_img, 0, 255, NORM_MINMAX);

        if (debug)
        {
            // init debug view with preprocessed image
            cvtColor(orig_img, debug_img, COLOR_GRAY2BGR);
            debug_img *= 0.4;
        }

        const double diameter_scaling_factor = scaling_factor == 0 ? 1.0 : scaling_factor;
        if (params.auto_pupil_diameter)
        {
            // compute automatic pupil radius bounds
            constexpr double min_pupil_diameter_ratio = 0.07 * 2 / 3;
            constexpr double max_pupil_diameter_ratio = 0.29;
            const double diagonal = sqrt(orig_img.cols * orig_img.cols + orig_img.rows * orig_img.rows);

            min_pupil_diameter = min_pupil_diameter_ratio * diagonal;
            max_pupil_diameter = max_pupil_diameter_ratio * diagonal;

            // report computed parameters back (scaled back)
            params.min_pupil_diameter = min_pupil_diameter / diameter_scaling_factor;
            params.max_pupil_diameter = max_pupil_diameter / diameter_scaling_factor;
        }
        else
        {
            // scale input parameters
            min_pupil_diameter = params.min_pupil_diameter * diameter_scaling_factor;
            max_pupil_diameter = params.max_pupil_diameter * diameter_scaling_factor;
        }

        // ensure valid values
        bool success = (
            0 <= min_pupil_diameter &&
            0 <= max_pupil_diameter &&
            min_pupil_diameter <= max_pupil_diameter
        ); 

        if (!success && debug)
        {
            putText(debug_img, "Invalid pupil size!", Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
        }
        return success;
    }

    void Detector::postprocess(Result& final_result, const Mat& input_img, Mat* debug_color_img)
    {
        // If we shrank the image, we need to enlarge the result.
        if (scaling_factor != 0.0)
        {
            const float inverse_factor = static_cast<float>(1.0f / scaling_factor);
            final_result.axes *= inverse_factor;
            final_result.center *= inverse_factor;
        }

        // same with debug image
        if (debug)
        {
            if (scaling_factor != 0.0)
            {
                Size input_size(input_img.cols, input_img.rows);
                // OpenCV docs recommend INTER_CUBIC interpolation for enlarging images
#if CV_MAJOR_VERSION == 3
                resize(debug_img, *debug_color_img, input_size, 0.0, 0.0, CV_INTER_CUBIC);
#elif CV_MAJOR_VERSION == 4
                resize(debug_img, *debug_color_img, input_size, 0.0, 0.0, InterpolationFlags::INTER_CUBIC);
#endif
            }
            else
            {
                *debug_color_img = debug_img;
            }
        }
    }
    
    void Detector::detect_edges()
    {   
        calculate_canny();
        thin_edges();
        break_crossings();
        straighten_edges();
        break_orthogonals();
    }

    struct Coords
    {
        // Storage of row/column coordinate for Canny filter hysteresis step
        int r, c;
        Coords(int r = 0, int c = 0): r(r), c(c) {}
    };

    void Detector::calculate_canny()
    {
        // NOTE: The canny implementation of OpenCV works different to MATLAB.
        // Experience proved that the MATLAB implementation seems better suited for
        // pupil edge detection. This is a very naive implementation of a MATLAB-like
        // edge detector, based mostly on the hints given in:
        // https://de.mathworks.com/matlabcentral/answers/458235-why-is-the-canny-edge-detection-in-matlab-different-to-opencv#answer_372038
        
        // (1) Image Smoothing
        // NOTE: Matlab appears to be using a blur-size of 15x15. This blur-size is
        // certainly resolution dependent. For a resolution of 320x240 we found 5x5 to
        // perform much better.
        GaussianBlur(orig_img, edge_img, Size(5, 5), 2, 2, BORDER_REPLICATE);

        // (2) Gradient Computation
        // NOTE: although the description recommends DoG gradient computation, we found
        // that Sobel still works fine enough and is ready-to-use from OpenCV
        Sobel(edge_img, dx_img, CV_32F, 1, 0, 7, 1, 0, BORDER_REPLICATE);
        Sobel(edge_img, dy_img, CV_32F, 0, 1, 7, 1, 0, BORDER_REPLICATE);
        magnitude(dx_img, dy_img, mag_img);

        constexpr uchar NO_EDGE = 0;
        constexpr uchar POTENTIAL_EDGE = 127;
        constexpr uchar EDGE = 255;

        // (3) Non-maximum Suppression
        {
            // Idea: Look at the gradient direction at every edge pixel. Compare the
            // magnitude with both neighbors along the gradient direction. Set to
            // NO_EDGE, if pixel does not have maximum magnitude.
            edge_img.setTo(POTENTIAL_EDGE);

            // Constants for direction comparison
            constexpr float tan_pi_8 = 0.4142135623f;  // tan(PI/8)
            constexpr float tan_3pi_8 = 2.4142135623f;  // tan(3PI/8)

            const int rows = mag_img.rows - 1;
            const int cols = mag_img.cols - 1;
            uchar *out_row;
            const float *mag_above, *mag_current, *mag_below, *dx_row, *dy_row;
            int r, c;
            for (r = 1; r < rows; ++r)
            {
                out_row = edge_img.ptr(r);
                mag_above = mag_img.ptr<float>(r - 1);
                mag_current = mag_img.ptr<float>(r);
                mag_below = mag_img.ptr<float>(r + 1);
                dx_row = dx_img.ptr<float>(r);
                dy_row = dy_img.ptr<float>(r);

                for (c = 1; c < cols; ++c)
                {
                    const float mag = mag_current[c];

                    const float dx = dx_row[c];
                    const float dy = dy_row[c];

                    // By taking the absolute values we can just look at the first
                    // quadrant. The quadrant is devided in 3 areas:
                    //   1. angle < PI/8:           horizontal
                    //   2. PI/8 < angle < 3PI/8:   diagonal
                    //   3. angle < 3PI/8:          vertical
                    const float x = abs(dx);
                    const float y = abs(dy);

                    const float tan_angle = y / x;
                    bool is_max = false;
                    if (tan_angle < tan_pi_8)
                    {
                        // horizontal
                        is_max = (mag_current[c - 1] < mag && mag_current[c + 1] <= mag);
                    }
                    else if(tan_angle > tan_3pi_8)
                    {
                        // vertical
                        is_max = (mag_above[c] < mag && mag_below[c] <= mag);
                    }
                    else
                    {
                        // diagonal, look at signs
                        if (signbit(dx) == signbit(dy))
                        {
                            // diagonal (\) 
                            // NOTE: this is not cartesian, but image coordinates! Both
                            // positive means bottom-right direction!
                            is_max = (mag_above[c - 1] < mag && mag_below[c + 1] <= mag);
                        }
                        else
                        {
                            // diagonal (/)
                            is_max = (mag_above[c + 1] < mag && mag_below[c - 1] <= mag);
                        }
                    }
                    
                    // Suppression:
                    if (!is_max)
                    {
                        out_row[c] = NO_EDGE;
                    }
                }
            }
        }

        // (4) Determining Hysteresis Threshold Limits
        int thresh_1 = 0;
        int thresh_2 = 0;
        const int n_pixels = bin_img.rows * bin_img.cols;
        {
            constexpr int n_bins = 64;

            // calculate bin for every pixel
            double max_mag = 0;
            minMaxLoc(mag_img, nullptr, &max_mag);
            double rescaling_factor = (n_bins - 1) / max_mag;
            mag_img.convertTo(bin_img, CV_8U, rescaling_factor);
            array<int, n_bins> bins {};
            uchar* row;
            for (int r = 0; r < bin_img.rows; ++r)
            {
                row = bin_img.ptr(r);
                for (int c = 0; c < bin_img.cols; ++c)
                {
                    ++bins[row[c]];
                }
            }

            // thresholds from histogram
            constexpr float t1_percentile = 0.28f;
            constexpr float t2_percentile = 0.7f;
            const int t1_lower_bound = static_cast<int>(ceil(t1_percentile * n_pixels));
            const int t2_lower_bound = static_cast<int>(ceil(t2_percentile * n_pixels));
            int sum = 0;
            for (int i = 0; i < n_bins; ++i)
            {
                sum += bins[i];
                if (sum >= t1_lower_bound && thresh_1 == 0)
                {
                    thresh_1 = i;
                }
                if (sum >= t2_lower_bound)
                {
                    thresh_2 = i;
                    break;
                }
            }
        }

        // (5) Hysteresis Thresholding
        {
            queue<Coords> growing_edges;
            const uchar* bin_row;
            uchar* out_row;
            int rows = edge_img.rows - 1;
            int cols = edge_img.cols - 1;
            for (int r = 1; r < rows; ++r)
            {
                bin_row = bin_img.ptr(r);
                out_row = edge_img.ptr(r);
                for (int c = 1; c < cols; ++c)
                {
                    if (out_row[c] != POTENTIAL_EDGE)
                    {
                        // potentially filled by hysteresis or non-maximum suppression,
                        // dont't check thresholds again!
                        continue;
                    }

                    if (bin_row[c] < thresh_1)
                    {
                        out_row[c] = NO_EDGE;
                        continue;
                    }
                    if (bin_row[c] < thresh_2)
                    {
                        // still only potential edge, might be filled by hysteresis
                        continue;
                    }

                    // now we know we are >= thresh_2
                    out_row[c] = EDGE;
                    
                    growing_edges.emplace(r, c);
                    while (!growing_edges.empty())
                    {
                        Coords candidate = growing_edges.front();
                        growing_edges.pop();

                        if (candidate.r < 1 || candidate.r > rows - 1 || candidate.c < 1 || candidate.c > cols - 1)
                        {
                            continue;
                        }

                        for (int dr = -1; dr <= 1; ++dr)
                        {
                            for (int dc = -1; dc <= 1; ++dc)
                            {
                                const int r2 = candidate.r + dr;
                                const int c2 = candidate.c + dc;
                                if (edge_img.ptr(r2)[c2] != POTENTIAL_EDGE) continue;
                                edge_img.ptr(r2)[c2] = EDGE;
                                growing_edges.emplace(r2, c2);
                            }
                        }
                    }
                }
            }
            // Threshold away potential edges that are left
            threshold(edge_img, edge_img, POTENTIAL_EDGE + 1, EDGE, CV_8U);
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

        const uchar *above, *below;
        uchar *current;
        const int rows = edge_img.rows - 2;
        const int cols = edge_img.cols - 2;
        int r, c;
        for (r = 0; r < rows; ++r)
        {
            above = edge_img.ptr(r);
            current = edge_img.ptr(r + 1);
            below = edge_img.ptr(r + 2);
            for (c = 0; c < cols; ++c)
            {
                if (
                    (above[c + 1] && current[c])        ||
                    (above[c + 1] && current[c + 2])    ||
                    (below[c + 1] && current[c])        ||
                    (below[c + 1] && current[c + 2])
                )
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
        const int rows = edge_img.rows - 2;
        const int cols = edge_img.cols - 2;
        uchar *current;
        int r, c;
        for (r = 0; r < rows; ++r)
        {
            above = edge_img.ptr(r);
            current = edge_img.ptr(r + 1);
            below = edge_img.ptr(r + 2);
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

        uchar *row0, *row1, *row2, *row3;
        const int rows = edge_img.rows - 3;
        const int cols = edge_img.cols - 3;
        int r, c;
        for (r = 0; r < rows; ++r)
        {
            row0 = edge_img.ptr(r);
            row1 = edge_img.ptr(r + 1);
            row2 = edge_img.ptr(r + 2);
            row3 = edge_img.ptr(r + 3);
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
        const int rows = edge_img.rows - 5;
        const int cols = edge_img.cols - 5;
        int r, c;
        for (r = 0; r < rows; ++r)
        {
            row0 = edge_img.ptr(r);
            row1 = edge_img.ptr(r + 1);
            row2 = edge_img.ptr(r + 2);
            row3 = edge_img.ptr(r + 3);
            row4 = edge_img.ptr(r + 4);
            row5 = edge_img.ptr(r + 5);
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
                    (row0[c] && row1[c + 2] && row2[c +2]) || // d1
                    (row0[c + 2] && row1[c] && row2[c]) // d3
                )) row0[c + 1] = 0;

                if (row0[c + 2] && row1[c + 1] && row1[c + 3] && row2[c] && row2[c + 4]) row0[c + 2] = 0; // f1

                if (row0[c + 3] && row0[c + 2] && row1[c + 1] && row1[c + 4] && row2[c] && row2[c + 5]) row0[c + 3] = 0; // g1

                if (row1[c + 1] && row0[c + 2] && row0[c + 3] && row0[c + 4] && row2[c] && row3[c] && row4[c]) row1[c + 1] = 0; // e3

                if (row1[c + 3] && row0[c] && row0[c + 1] && row0[c + 2] && row2[c + 4] && row3[c + 4] && row4[c + 4]) row1[c + 3] = 0; //e1

                if (row2[c] && row1[c + 1] && row0[c + 2] && (
                    (row3[c + 1] && row4[c + 2]) || // f2
                    (row3[c] && row4[c + 1] && row5[c + 2]) // g2
                )) row2[c] = 0;

                if (row2[c + 1] && (
                    (row0[c] && row1[c] && row2[c + 2]) || // d2
                    (row0[c + 2] && row1[c + 2] && row2[c]) // d4
                )) row2[c + 1] = 0;

                if (row2[c + 2] && row0[c] && row1[c + 1] && (
                    (row3[c + 1] && row4[c]) || // f3
                    (row1[c + 3] && row0[c + 4]) || // f4
                    (row2[c + 3] && row1[c + 4] && row0[c + 5]) // g4
                )) row2[c + 2] = 0;

                if (row3[c + 1] && row0[c] && row1[c] && row2[c] && row4[c + 2] && row4[c + 3] && row4[c + 4]) row3[c + 1] = 0; //e2

                if (row3[c + 2] && row0[c] && row1[c + 1] && row2[c + 2] && row4[c + 1] && row5[c]) row3[c + 2] = 0; //g3

                if (row3[c + 3] && row0[c + 4] && row1[c + 4] && row2[c + 4] && row4[c] && row4[c + 1] && row4[c + 2]) row3[c + 3] = 0; // e4
            }
        }
    }

    void Detector::select_edge_segments()
    {
        findContours(edge_img, segments, RETR_LIST, CHAIN_APPROX_TC89_KCOS);

        // NOTE: We are essentially re-using the result from previous runs. Need to make
        // sure that either all values will be overwritten or confidence will be set to
        // 0 for every result!
        candidates.resize(segments.size());

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
                if (approx_diameter > max_pupil_diameter)
                {
                    break;
                }
            }
            // we can early exit, because we will only get bigger
            if (approx_diameter > max_pupil_diameter)
            {
                break;
            }
        }
        return min_pupil_diameter < approx_diameter && approx_diameter < max_pupil_diameter;
    }
    bool Detector::segment_curvature_valid(const Segment& segment) const
    {
        auto rect = minAreaRect(segment);
        double ratio = rect.size.width / rect.size.height;
        return !axes_ratio_is_invalid(ratio);
    }

    inline bool Detector::axes_ratio_is_invalid(double ratio) const
    {
        constexpr double axes_ratio_threshold = 0.2;
        constexpr double inverse_threshold = 1.0 / axes_ratio_threshold;
        
        return ratio < axes_ratio_threshold || ratio > inverse_threshold;
    }

    bool Detector::fit_ellipse(const Segment& segment, Result& result) const
    {
        // NOTE: This is a cv::RotatedRect, see https://stackoverflow.com/a/32798273 for
        // conversion to ellipse parameters. Also below.
        auto fit = fitEllipse(segment);

        // 	(I) discard if center outside image boundaries
        if (fit.center.x < 0 || fit.center.y < 0 || fit.center.x > edge_img.cols || fit.center.y > edge_img.rows)
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
        //    /|\      |
        //   / | \  Q1 |
        //  /  |  \    |
        // ---------
        //  \  |  /
        //   \ | /
        //    \|/

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

        bitset<8> bins;

        for (const auto& p : segment)
        {
            const auto v = Point2f(p.x - result.center.x, p.y - result.center.y);
            if (v.x > 0)
            {
                if (v.y > 0)
                {
                    if (v.x > v.y) bins[1] = true;
                    else bins[0] = true;
                }
                else
                {
                    if (v.x > v.y) bins[2] = true;
                    else bins[3] = true;
                }
            }
            else
            {
                if (v.y > 0)
                {
                    if (v.x > v.y) bins[7] = true;
                    else bins[6] = true;
                }
                else
                {
                    if (v.x > v.y) bins[4] = true;
                    else bins[5] = true;
                }
            }
            // early exit
            if (bins.count() == 8) break;
        }

        return bins.count() / 8.0;
    }

    double Detector::ellipse_outline_constrast(const Result& result) const
    {
        double contrast = 0;
        constexpr double radian_per_degree = M_PI / 180.0;
        // Iterate circle with stride of 10 degrees (all in radians)
        constexpr double stride = 10 * radian_per_degree;
        double theta = 0;
        // NOTE: A for-loop: for(theta=0; theta < 2*PI; ...) will result
        // in 37 iterations because of rounding errors. This will result
        // in one line being counted twice.
        constexpr int n_iterations = 36;
        const double minor = min(result.axes.width, result.axes.height);
        const double cos_angle = cos(result.angle * radian_per_degree);
        const double sin_angle = sin(result.angle * radian_per_degree);
        const Rect bounds = Rect(0, 0, orig_img.cols, orig_img.rows);
        constexpr int bias = 5;
        // Mat tmp;
        // cvtColor(orig_img, tmp, COLOR_GRAY2BGR);
        for (int i = 0; i < n_iterations; ++i)
        {
            const double x = result.axes.width * cos(theta);
            const double y = result.axes.height * sin(theta);
            Point2f offset(
                static_cast<float>(x * cos_angle - y * sin_angle),
                static_cast<float>(y * cos_angle + x * sin_angle)
            );
            Point2f outline_point = result.center + offset;

            Point2f offset_norm = offset / cv::norm(offset);
            Point2f inner_pt = outline_point - (0.3 * minor) * offset_norm;
            Point2f outer_pt = outline_point + (0.3 * minor) * offset_norm;

            if (!bounds.contains(inner_pt) || !bounds.contains(outer_pt)) 
            {
                theta += stride;
                continue;
            }

            double inner_avg = 0;
            LineIterator inner_line(orig_img, inner_pt, outline_point);
            for (int j = 0; j < inner_line.count; j++, ++inner_line)
            {
                inner_avg += *(*inner_line);
            }
            inner_avg /= inner_line.count;

            double outer_avg = 0;
            LineIterator outer_line(orig_img, outline_point, outer_pt);
            for (int j = 0; j < outer_line.count; j++, ++outer_line)
            {
                outer_avg += *(*outer_line);
            }
            outer_avg /= outer_line.count;

            if (inner_avg + bias < outer_avg) contrast += 1;

            // if (inner_avg + bias < outer_avg)
            //     line(tmp, inner_pt, outer_pt, Scalar(0, 255, 0));
            // else
            //     line(tmp, inner_pt, outer_pt, Scalar(0, 0, 255));

            theta += stride;
        }

        // imshow("pfa", tmp);
        // waitKey(-1);
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
        const Rect r = r1 & r2; // intersection
        return r.area() > 0 && r != r1 && r != r2;
    }
    
    Segment Detector::merge_segments(const Segment& s1, const Segment& s2) const
    {
        // Naive approach is to just take the convex hull of the union of both segments.
        // But there is no documentation.
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
        Result *initial_pupil = &*std::max_element(candidates.begin(), candidates.end());
        double semi_major = max(initial_pupil->axes.width, initial_pupil->axes.height);
        Result *candidate = nullptr;
        for (auto& result : candidates)
        {
            if (result.confidence.value == 0) continue;
            if (result.confidence.outline_contrast < 0.75) continue;
            if (&result == initial_pupil) continue;
            // NOTE: The initial paper mentions to discard candidates with a diameter
            // larger than the initial pupil's semi major, i.e. only candidates that are
            // half the size are considered. In dark environments this leads to bad
            // results though, as the pupil can be up to 80% of the iris size. Therefore
            // we use 0.8 as threshold.
            if (max(result.axes.width, result.axes.height) > 0.8 * semi_major) continue;
            if (norm(initial_pupil->center - result.center) > semi_major) continue;
            if (candidate && result.confidence.value <= candidate->confidence.value) continue;
            candidate = &result;
        }
        return (candidate) ? *candidate : *initial_pupil;
    }

}

