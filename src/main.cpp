#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>

using namespace cv;
using namespace std;


int thresh = 220;
int max_thresh = 255;

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    if  ( event == EVENT_LBUTTONDOWN )
    {
        cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
    }
//    else if  ( event == EVENT_RBUTTONDOWN )
//    {
//        cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
//    }
//    else if  ( event == EVENT_MBUTTONDOWN )
//    {
//        cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
//    }
//    else if ( event == EVENT_MOUSEMOVE )
//    {
//        cout << "Mouse move over the window - position (" << x << ", " << y << ")" << endl;
//    }
}

void cornerHarris_demo(Mat &gray) {
	cout << "called harris demo" << endl;
	int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    vector<Point2f> corners;
    Mat dst = Mat::zeros( gray.size(), CV_32FC1 );
    cornerHarris( gray, dst, blockSize, apertureSize, k );
    Mat dst_norm, dst_norm_scaled;
    normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
    convertScaleAbs( dst_norm, dst_norm_scaled );
    int countCorners =0;
    for( int i = 0; i < dst_norm.rows ; i++ )
    {
        for( int j = 0; j < dst_norm.cols; j++ )
        {
            if( (int) dst_norm.at<float>(i,j) > thresh )
            {
                circle( dst_norm_scaled, Point(j,i), 5,  Scalar(0), 2, 8, 0 );
                corners.push_back(Point(j,i));
                cout<<Point(j,i)<<endl;
                countCorners++;
            }
        }
    }
    cout<<"total corners found : "<<countCorners<<endl;
    namedWindow("ImageDisplay", 1);

    setMouseCallback("ImageDisplay", CallBackFunc, NULL);
    imshow( "ImageDisplay", dst_norm_scaled );
    imwrite("corners.png", dst_norm_scaled);

    waitKey(0);

    //waitKey(0);

    cornerSubPix(gray, corners, Size(5, 5), Size(-1, -1),
			TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
    for( size_t i = 0; i < corners.size(); i++ )
    {
        cout << " -- Refined Corner [" << i << "]  (" << corners[i].x << "," << corners[i].y << ")" << endl;
    }
}

int main(){

	Size patternsize(16,10); //interior number of corners
	Mat img = imread("4.jpg"); //source image
	Mat gray;

	cvtColor( img, gray, CV_BGR2GRAY );
//	Mat temp ;
//	GaussianBlur(gray, gray, Size(0, 0), 105); //hardcoded filter size, to be tested on 50 mm lens
//	addWeighted(gray, 1.8, gray, -0.8,0,gray) ;
	vector<Point2f> corners; //this will be filled by the detected corners
//	imshow("src", img);
//	waitKey(0);
	cornerHarris_demo(gray);
//	imshow("gray", gray);
//	waitKey(0);

	//CALIB_CB_FAST_CHECK saves a lot of time on images
	//that do not contain any chessboard corners
//	bool patternfound = findChessboardCorners(gray, patternsize, corners,
//	        CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
//	        + CALIB_CB_FAST_CHECK);
//
//	if (patternfound) {
//		cout<<"pattern found"<<endl;
//		cornerSubPix(gray, corners, Size(5, 5), Size(-1, -1),
//				TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
//	}
//
//
//	for(size_t i =0; i<corners.size(); i++){
//		cout<<corners[i]<<endl;
//	}
//
//	drawChessboardCorners(img, patternsize, Mat(corners), patternfound);
//	imshow("src", img);
//	waitKey(0);

	return 0;
}
