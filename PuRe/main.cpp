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


	VideoCapture cap(R"(..\LPW\10\1.avi)");
	if (!cap.isOpened())
	{
		cerr << "could not open video file!" << endl;
		exit(-1);
	}

	{
		pure::Detector detector;
		Mat color;
		Mat debug;
		Mat gray;
		bool running = true;
		while (running)
		{
			cap.read(color);
			if (color.empty())
			{
				cerr << "Empty frame!" << endl;
				break;
			}
			resize(color, color, Size(320, 240));

        	cvtColor(color, gray, COLOR_BGR2GRAY);
			debug = color.clone();
			auto result = detector.detect(gray, &debug);

			ellipse(color, Point((int)result.center_x, (int)result.center_y), Size((int)result.first_ax, (int)result.second_ax), result.angle, 0, 360, Scalar(0, 0, 255));
			circle(color, Point((int)result.center_x, (int)result.center_y), 2, Scalar(0, 0, 255), 2);
			imshow("Color", color);
			imshow("Debug", debug);

			if (waitKey(1) >= 0)
			{
				break;
			}
		}
	}
	return 0;
}