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
	const double MIN_PUPIL_DIAMETER = 0.0467 * sqrt(W*W + H*H);
	const double MAX_PUPIL_DIAMETER = 0.1933 * sqrt(W*W + H*H);

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
			Canny(gray, edges, 160, 160*2);
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
			for (auto& segment : contours)
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
						if (approx_diameter > MAX_PUPIL_DIAMETER) break;
					}
					// we can early exit, because we will only get bigger
					if (approx_diameter > MAX_PUPIL_DIAMETER) break;
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
				auto rect = minAreaRect(segment);
				double ratio = rect.size.width / rect.size.height;
				if (ratio < 0.2 || ratio > 5.0) continue;
				


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