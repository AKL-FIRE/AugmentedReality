//
// Created by Apple  on 2018/6/19.
//

#ifndef AR3_FACECLASSIFIER_H
#define AR3_FACECLASSIFIER_H

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <string>
#include <vector>
using namespace cv;

class FaceClassifier {

public:
    static FaceClassifier& createFaceClassifier(const std::string &ace_cascade, const std::string &eye_cascade);
    static void FindAndDraw(Mat &src);
    static void DrawCatEar(Mat &src);
    static void DrawGrass(Mat &src);
private:
    static CascadeClassifier face,eye;
    FaceClassifier() = default;
};


#endif //AR3_FACECLASSIFIER_H
