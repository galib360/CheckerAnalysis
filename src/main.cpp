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


typedef struct {
	vector<Point2f> pnts2d;
} campnts;


int thresh = 220;
int max_thresh = 255;

vector<Mat> srcs;
vector<Mat> grays;
vector<Mat> dsts;
vector<Mat> P(4);
vector<Mat> Ks;
vector<Mat> Rs;
vector<Mat> Rts;
vector<Mat> ts;
vector<Mat> quats;


vector<campnts> pnts(4);
vector<Mat> points3D;
vector<Mat> points3Dnorm;

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	int whichcam = *((int*)userdata);
    if  ( event == EVENT_LBUTTONDOWN )
    {
        cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ") in "<<whichcam << endl;
		Point2f point = ((Point_<float>)x,(Point_<float>)y);
        pnts[whichcam].pnts2d.push_back(point);
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

void cornerHarris_demo(Mat &gray, int whichcam) {
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
    dsts.push_back(dst_norm_scaled);
    int countCorners =0;
    for( int i = 0; i < dst_norm.rows ; i++ )
    {
        for( int j = 0; j < dst_norm.cols; j++ )
        {
            if( (int) dst_norm.at<float>(i,j) > thresh )
            {
                circle( dst_norm_scaled, Point(j,i), 5,  Scalar(0), 2, 8, 0 );
                corners.push_back(Point(j,i));
                //cout<<Point(j,i)<<endl;
                countCorners++;
            }
        }
    }
    cout<<"total corners found : "<<countCorners<<endl;

    namedWindow("ImageDisplay", 1);
    setMouseCallback("ImageDisplay", CallBackFunc, &whichcam);
    imshow( "ImageDisplay", dst_norm_scaled );
    imwrite("corners.png", dst_norm_scaled);

    waitKey(0);

    //waitKey(0);

    cornerSubPix(gray, corners, Size(5, 5), Size(-1, -1),
			TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
//    for( size_t i = 0; i < corners.size(); i++ )
//    {
//        cout << " -- Refined Corner [" << i << "]  (" << corners[i].x << "," << corners[i].y << ")" << endl;
//    }
}

int main(){

	string inputdir = "data/";
//	Size patternsize(16,10); //interior number of corners
	for(int i = 0; i<4; i++){
		string inputfile = inputdir + to_string(i)+ ".jpg";
		Mat img = imread(inputfile);
		Mat gray;
		cvtColor( img, gray, CV_BGR2GRAY );
		srcs.push_back(img);
		grays.push_back(gray);
//		cornerHarris_demo(gray, i);

		//////////read camera matrices---------->
		ifstream txtfile = ifstream(inputdir + to_string(i) + ".txt");
		vector<string> fid;
		std::string line;
		vector<string> linedata;
		int c = 0;

		while (std::getline(txtfile, line)) {
			std::stringstream linestream(line);
			string val;
			while (linestream >> val) {
				linedata.push_back(val);
				//cout<<val<<endl;
			}
		}

		while (c < linedata.size()) {
			fid.push_back(linedata[c]);
			c++;
			//Put data into K
			Mat kk(3, 3, cv::DataType<float>::type, Scalar(1));
			Mat rotm(3, 3, cv::DataType<float>::type, Scalar(1));
			Mat Rt(3, 4, cv::DataType<float>::type, Scalar(1));
			Mat tvec(3, 1, cv::DataType<float>::type, Scalar(1));
			Mat quat(4, 1, cv::DataType<float>::type, Scalar(1));
			for (int j = 0; j < 3; j++) {
				for (int k = 0; k < 3; k++) {
					float temp = strtof((linedata[c]).c_str(), 0);

					kk.at<float>(j, k) = temp;
					c++;
				}
			}
			//kk = kk.t();
			Ks.push_back(kk);

			for (int j = 0; j < 4; j++) {
				float temp = strtof((linedata[c]).c_str(), 0);
				quat.at<float>(j, 0) = temp;
				c++;
			}
			quats.push_back(quat);

			//cout<<"quat : " << quat<<endl;

			for (int j = 0; j < 3; j++) {
				float temp = strtof((linedata[c]).c_str(), 0);
				tvec.at<float>(j, 0) = temp;
				c++;
			}
			ts.push_back(tvec);

			float qw = quat.at<float>(0,0);
			float qx = quat.at<float>(1,0);
			float qy = quat.at<float>(2,0);
			float qz = quat.at<float>(3,0);

			const float n = 1.0f/sqrt(qx*qx+qy*qy+qz*qz+qw*qw);
			qx *= n;
			qy *= n;
			qz *= n;
			qw *= n;

			rotm.at<float>(0,0) = 1.0f - 2.0f*qy*qy - 2.0f*qz*qz;
			rotm.at<float>(0,1) = 2.0f*qx*qy - 2.0f*qz*qw;
			rotm.at<float>(0,2) = 2.0f*qx*qz + 2.0f*qy*qw;
			rotm.at<float>(1,0) = 2.0f*qx*qy + 2.0f*qz*qw;
			rotm.at<float>(1,1) = 1.0f - 2.0f*qx*qx - 2.0f*qz*qz;
			rotm.at<float>(1,2) = 2.0f*qy*qz - 2.0f*qx*qw;
			rotm.at<float>(2,0) = 2.0f*qx*qz - 2.0f*qy*qw;
			rotm.at<float>(2,1) = 2.0f*qy*qz + 2.0f*qx*qw;
			rotm.at<float>(2,2) = 1.0f - 2.0f*qx*qx - 2.0f*qy*qy;

			//rotm = rotm.t();
			Rs.push_back(rotm);

			Rt.at<float>(0,0) = rotm.at<float>(0,0);
			Rt.at<float>(0,1) = rotm.at<float>(0,1);
			Rt.at<float>(0,2) = rotm.at<float>(0,2);
			Rt.at<float>(1,0) = rotm.at<float>(1,0);
			Rt.at<float>(1,1) = rotm.at<float>(1,1);
			Rt.at<float>(1,2) = rotm.at<float>(1,2);
			Rt.at<float>(2,0) = rotm.at<float>(2,0);
			Rt.at<float>(2,1) = rotm.at<float>(2,1);
			Rt.at<float>(2,2) = rotm.at<float>(2,2);
			Rt.at<float>(0,3) = tvec.at<float>(0,0);
			Rt.at<float>(1,3) = tvec.at<float>(1,0);
			Rt.at<float>(2,3) = tvec.at<float>(2,0);

			Rts.push_back(Rt);

			Mat Ptemp = kk * Rt;
			P[i] = (Ptemp);

			//cout<<"Projection Matrix: "<< P[i]<<endl;

			cornerHarris_demo(gray, i);

		}



	}


	//triagulate points here--------------------->
	Mat temp0(4, pnts[0].pnts2d.size(), CV_32F);
	Mat temp1(4, pnts[1].pnts2d.size(), CV_32F);
	Mat temp2(4, pnts[2].pnts2d.size(), CV_32F);
	Mat temp3(4, pnts[3].pnts2d.size(), CV_32F);

	triangulatePoints(P[0], P[1], pnts[0].pnts2d, pnts[1].pnts2d, temp0);
	cout<<"3D points for cam00 : "<< temp0<<endl;
	triangulatePoints(P[1], P[2], pnts[1].pnts2d, pnts[2].pnts2d, temp1);
	triangulatePoints(P[2], P[3], pnts[2].pnts2d, pnts[3].pnts2d, temp2);
	triangulatePoints(P[3], P[0], pnts[3].pnts2d, pnts[0].pnts2d, temp3);

	points3D.push_back(temp0);
	points3D.push_back(temp1);
	points3D.push_back(temp2);
	points3D.push_back(temp3);

	temp0.release();
	temp1.release();
	temp2.release();
	temp3.release();

	cout<<"Size of points 3D: "<<points3D.size()<<endl;

	for (int c = 0; c<points3D.size(); c++){
		Mat temp;
		points3D[c].copyTo(temp);
		for (int k = 0; k < temp.cols; k++) {
			for (int j = 0; j < 4; j++) {
				temp.at<float>(j, k) = temp.at<float>(j, k)
						/ temp.at<float>(3, k);
			}
		}
		points3Dnorm.push_back(temp);
		cout<<"3D points for cam0"<<to_string(c)<<" : "<< temp<<endl;
	}




	//Calculate eucledian distance here----------------------->




//	Mat img = imread("4.jpg"); //source image
//	Mat gray;
//
//	cvtColor( img, gray, CV_BGR2GRAY );
//	Mat temp ;
//	GaussianBlur(gray, gray, Size(0, 0), 105); //hardcoded filter size, to be tested on 50 mm lens
//	addWeighted(gray, 1.8, gray, -0.8,0,gray) ;
//	vector<Point2f> corners; //this will be filled by the detected corners
//	imshow("src", img);
//	waitKey(0);
//	cornerHarris_demo(gray);
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
