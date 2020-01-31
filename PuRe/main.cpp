#include <cmath>
#include <iostream>
#include <vector>
#include <string>

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "pure.hpp"

using namespace std;
using namespace cv;

int main()
{
	// VideoCapture cap("../eye_segmentation_500K/p1_image.mp4");
	// VideoCapture cap("/media/pfa/P L/pfa/datasets/LPW/1/1.avi");
	VideoCapture cap("../eye0.mp4");

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
		int n = 0;
		int FRAME = 0;
		while (running)
		{
			cap.read(color);
			if (color.empty())
			{
				cerr << "Empty frame!" << endl;
				break;
			}

        	cvtColor(color, gray, COLOR_BGR2GRAY);

			if (FRAME >= 0 && n < FRAME) {
				cout << n << " --> " << FRAME << endl;
				detector.detect(gray);
				n++;
				continue;
			}


			// debug = color.clone();
			Mat debug;
			auto result = detector.detect(gray, &debug);
			// auto result = detector.detect(gray);
			

			
			

			ellipse(color, Point(result.center), Size(result.axes), result.angle, 0, 360, Scalar(0, 0, 255));
			circle(color, Point(result.center), 2, Scalar(0, 0, 255), 2);
			imshow("Color", color);

			// putText(debug, to_string(n), Point(0, 240), cv::FONT_HERSHEY_PLAIN, 1, Scalar(255, 255, 255));
			imshow("Debug", debug);

			moveWindow("Color", 200, 200);
			moveWindow("Debug", 1000, 200);

			int KEY_ESC = 27;
			if(waitKey(-1) == KEY_ESC) running = false;
            destroyAllWindows();
			n++;
		}
	}
	return 0;
}