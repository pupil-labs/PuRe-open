#include <cmath>
#include <iostream>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "pure.hpp"

using namespace std;
using namespace cv;

int main()
{
	Mat img, gray, debug;

	// read image as color and grayscale
	img = imread("../eye.png", IMREAD_COLOR);
	if (img.empty())
	{
		cerr << "could not open or find the image!" << endl;
		exit(-1);
	}
	cvtColor(img, gray, COLOR_BGR2GRAY);

	// run detector
	pure::Detector detector;
	auto result = detector.detect(gray, &debug);

	// draw result on color image
	ellipse(img, Point(result.center), Size(result.axes), result.angle, 0, 360, Scalar(0, 0, 255));

	// show images, press any key to continue (and exit)
	imshow("img", img);
	imshow("debug", debug);
	waitKey(-1);
	return 0;
}