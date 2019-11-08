#include <cmath>
#include <iostream>
#include <vector>

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "pure.hpp"

using namespace std;
using namespace cv;

int main()
{
	cout << "Hello CMake!" << endl;

	VideoCapture cap(R"(D:\pfa\datasets\LPW\1\1.avi)");
	if (!cap.isOpened())
	{
		cerr << "could not open video file!" << endl;
		exit(-1);
	}

	const int W = 320;
	const int H = 240;
	const double MIN_PUPIL_DIAMETER = 0.0467 * sqrt(W * W + H * H);
	const double MAX_PUPIL_DIAMETER = 0.1933 * sqrt(W * W + H * H);

	// Cutoff threshold for ellipse axes ratio, see PuRe 3.3.3
	const double R_th = 0.2;
	const double R_th_inv = 1.0 / R_th;

	{
		Mat color;
		Mat gray;
		Mat edges;
		Mat thinned;
		Mat straightened;
		Mat broken;
		bool running = true;
		vector<vector<Point>> contours;
		while (running)
		{
			cap.read(color);
			if (color.empty())
			{
				cerr << "Empty frame!" << endl;
				break;
			}
			// TODO: Consider aspect ratio!
			resize(color, color, Size(W, H));

			cvtColor(color, gray, COLOR_BGR2GRAY);

			// TODO: resizing to working size (?)
			normalize(gray, gray, 0, 255, NORM_MINMAX);

			// convert back to color for visualization
			cvtColor(gray, color, COLOR_GRAY2BGR);

			// TODO: what to choose as parameters?
			Canny(gray, edges, 160, 160 * 2);
			// TODO: is canny already thresholded?
			threshold(edges, edges, 127, 255, THRESH_BINARY);

			thinned = edges.clone();

			// NOTE: Thinning won't work as expected, if input and output image are
			// different. The algorithm apparently assumes that the filter is applied to
			// the input image sequentially.
			pure::thin_edges(thinned, thinned);
			// TODO: Avoid copying all the time and except use masks and visualization
			pure::break_crossings(thinned, thinned);

			straightened = thinned.clone();
			pure::straighten_edges(thinned, straightened);
			// NOTE: The straightening result in segments that would have been removed
			// by previous edge-thinning! Maybe we should thin again?
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

			broken = straightened.clone();
			pure::break_orthogonals(straightened, broken);

			findContours(broken, contours, RETR_LIST, CHAIN_APPROX_TC89_KCOS);

			double approx_diameter = 0;
			for (auto &segment : contours)
			{
				// 3.3.1 Filter segments with < 5 points
				if (segment.size() < 5)
				{
					continue;
				}

				// 3.3.2 Filter segments based on approximate diameter
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
				if (approx_diameter > MAX_PUPIL_DIAMETER)
				{
					// diameter too large
					continue;
				}
				if (approx_diameter < MIN_PUPIL_DIAMETER)
				{
					// diameter too small
					continue;
				}

				// 3.3.3 Filter segments based on curvature approximation
				{
					auto rect = minAreaRect(segment);
					double ratio = rect.size.width / rect.size.height;
					if (ratio < R_th || ratio > R_th_inv)
					{
						continue;
					}
				}

				// 3.3.4 Ellipse fitting
				// NOTE: This is a cv::RotatedRect, see
				// https://stackoverflow.com/a/32798273 for conversion to ellipse
				// parameters
				auto ellipse = fitEllipse(segment);
				{
					// 	(I) discard if center outside image boundaries
					if (ellipse.center.x < 0 || ellipse.center.y < 0 || ellipse.center.x > W || ellipse.center.y > H)
					{
						continue;
					}

					// 	(II) discard if ellipse is too skewed
					auto ratio = ellipse.size.width / ellipse.size.height;
					if (ratio < R_th || ratio > R_th_inv)
					{
						continue;
					}
				}
				



				// 3.3.5 Additional filter
				{	

					// NOTE: width always provides the first axis, which corresponds to
					// the angle. Height provides the second axis, which corresponds to
					// angle + 90deg. This is NOT related to major/minor axes! But we
					// also don't need the information of which is the major and which
					// is the minor axis.
					auto first_ax = ellipse.size.width / 2;
					auto second_ax = ellipse.size.height / 2;

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
					// symmetry.
					//   /|\
					//  / | \ Q1
					// /  |  \
					//---------
					// \  |  /
					//  \ | /
					//   \|/

					// Shift rotation to origin to center at (0,0).
					segment_mean -= ellipse.center; 
					// Rotate backwards with negative angle
					const auto angle_rad = - ellipse.angle * M_PI / 180.0f;
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
					if (unrotated.x > first_ax) continue;
					if (unrotated.y > second_ax) continue;
					if (unrotated.x / first_ax + unrotated.y / second_ax > 1) continue;

				}
				
				polylines(color, segment, false, Scalar(0, 0, 255));
			}


			imshow("Color", color);
			imshow("Canny", edges);
			imshow("Thinned", thinned);
			imshow("Straightened", straightened);
			imshow("Broken", broken);

			if (waitKey(1) >= 0)
			{
				break;
			}
		}
	}
	return 0;
}