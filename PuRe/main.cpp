#include <iostream>
#include <array>

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

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
		Mat edges_filtered;
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
			cvtColor(gray, color, COLOR_GRAY2BGR);

			Canny(gray, edges, 160, 160*2);
			threshold(edges, edges, 127, 255, THRESH_BINARY);

			edges_filtered = edges.clone();

			{
				// Thinning

				// Efficient implementation by a combination of all 4 original thinning masks.
				// TODO: Handle borders of the image
				uchar *above, *current, *below, *dest;
				const int rows = edges_filtered.rows - 1;
				const int cols = edges_filtered.cols - 1;
				int r, c;
				for (r = 1; r < rows; ++r)
				{
					above = edges.ptr(r - 1);
					current = edges.ptr(r);
					below = edges.ptr(r + 1);
					dest = edges_filtered.ptr(r);
					for (c = 1; c < cols; ++c)
					{
						if (above[c] && current[c - 1]
							|| above[c] && current[c + 1]
							|| below[c] && current[c - 1]
							|| below[c] && current[c + 1])
						{
							dest[c] = 0;
						}
					}
				}
			}
			



			for (int r = 0; r < gray.rows; ++r)
			{
				for (int c = 0; c < gray.cols; ++c)
				{
					if (edges.at<uchar>(r, c) > 127)
					{
						color.at<Vec3b>(r, c)[0] /= 2;
						color.at<Vec3b>(r, c)[1] /= 2;
						color.at<Vec3b>(r, c)[2] = 255;
					}
				}
			}

			imshow("Video", color);
			imshow("Canny", edges);
			imshow("Filtered", edges_filtered);

			if (waitKey(1) >= 0)
			{
				break;
			}
		}
	}
	return 0;
}