#include <iostream>
#include <array>

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

	{
		Mat color;
		Mat gray;
		Mat edges;
		Mat thinned;
		Mat straightened;
		Mat broken;
		bool running = true;
		while (running)
		{
			cap.read(color);
			if (color.empty())
			{
				cerr << "Empty frame!" << endl;
				break;
			}
			cvtColor(color, gray, COLOR_BGR2GRAY);

			// TODO: resizing to working size (?)
			normalize(gray, gray, 0, 255, NORM_MINMAX);

			// convert back to color for visualization
			cvtColor(gray, color, COLOR_GRAY2BGR);

			// TODO: what to choose as parameters?
			Canny(gray, edges, 160, 160*2);
			threshold(edges, edges, 127, 255, THRESH_BINARY);

			thinned = edges.clone();

			pure::thin_edges(edges, thinned);
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