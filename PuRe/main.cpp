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

			pure::thin_edges(edges, edges_filtered);

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