//人脸检测
#include "FaceClassifier.h"
#include <opencv2/videoio.hpp>

int main()
{
    auto a = FaceClassifier::createFaceClassifier("../haarcascade_frontalface_alt.xml", "../haarcascade_eye_tree_eyeglasses.xml");
    cv::VideoCapture cap(0);
    Mat frames(Size(480,640), CV_16UC4);
    if(cap.isOpened())
        for(;;)
        {
            cap >> frames;
            a.DrawCatEar(frames);
            if(waitKey(10) == 27) break;
        }
}

