//
// Created by Apple  on 2018/6/19.
//

#include "FaceClassifier.h"

CascadeClassifier FaceClassifier::face;
CascadeClassifier FaceClassifier::eye;

FaceClassifier& FaceClassifier::createFaceClassifier(const std::string &face_cascade, const std::string &eye_cascade)
{
    if(!face.load(face_cascade))
    {
        std::cerr << "Can't not load face_cascade file";
        exit(-1);
    }
    if(!eye.load(eye_cascade))
    {
        std::cerr << "Can't not load eye_cascade file";
        exit(-1);
    }
    static FaceClassifier a;
    return a;
}

void FaceClassifier::FindAndDraw(Mat &src)
{
    Mat frame, faceROI;
    frame = src.clone();
    std::vector<Rect> faces, eyes;
    cvtColor(src, src, COLOR_BGRA2GRAY);
    equalizeHist(src, src);
    face.detectMultiScale(src, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30,30));
    for(size_t i = 0;i < faces.size(); ++i)
    {
        //绘制矩形 BGR。
        rectangle(frame,faces[i],Scalar(0,0,255),1);
        //获取矩形中心点。
        Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        //绘制圆形。
        ellipse(frame, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
        //获取脸部矩形区域。
        faceROI = src(faces[i]);
        //存储找到的眼睛矩形。
        std::vector<Rect> eyes;
        eye.detectMultiScale(faceROI, eyes, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(30,30));
        for(size_t j = 0;j < eyes.size(); ++j)
        {
            Point eye_center(faces[i].x + eyes[j].x + eyes[j].width/2,faces[i].y + eyes[j].y + eyes[j].height/2);
            int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
            circle(frame,eye_center,radius,Scalar( 255, 0, 0 ),4,8,0);
        }
    }
    imshow("Face", frame);
    //waitKey(30);
}

void FaceClassifier::DrawCatEar(Mat &src)
{
    cvtColor(src, src, COLOR_BGR2BGRA);
    Mat catEar_left = imread("../image/cat/left1.png", CV_LOAD_IMAGE_UNCHANGED);
    Mat catEar_right = imread("../image/cat/right1.png", CV_LOAD_IMAGE_UNCHANGED);
    Mat frame, faceROI;
    frame = src.clone();
    std::vector<Rect> faces, eyes;
    cvtColor(src, src, COLOR_BGRA2GRAY);
    equalizeHist(src, src);
    face.detectMultiScale(src, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30,30));

    if(!faces.empty()) {
        std::vector<Mat> source_left, source_right;

        Size sizeofface = faces[0].size();
        sizeofface.width /= 2;
        sizeofface.height /= 2;
        resize(catEar_left, catEar_left, sizeofface);
        resize(catEar_right, catEar_right, sizeofface);
        //rectangle(frame, faces[0], Scalar(0, 0, 225), 1);
        split(catEar_left, source_left);
        split(catEar_right, source_right);

        cv::Rect left_ear = faces[0];
        cv::Rect right_ear = faces[0];

        left_ear.width = sizeofface.width;
        left_ear.height = sizeofface.height;
        left_ear.x -= sizeofface.width * 0.2;
        left_ear.y -= sizeofface.height * 0.4;
        right_ear.width = sizeofface.width;
        right_ear.height = sizeofface.height;
        right_ear.x += sizeofface.width * 1.2;
        right_ear.y -= sizeofface.height * 0.4;

        if(left_ear.x < 0)
            left_ear.x = 0;
        if(left_ear.y < 0)
            left_ear.y = 0;
        if(right_ear.x < 0)
            right_ear.x = 0;
        if(right_ear.y < 0)
            right_ear.y = 0;

        Mat CatRoi_left = frame(left_ear);
        Mat CatRoi_right = frame(right_ear);
        std::vector<Mat> Roi_lefts, Roi_rights;
        split(CatRoi_left, Roi_lefts);
        split(CatRoi_right, Roi_rights);
        //addWeighted(catEar_left, 1, CatRoi_left, 0, 0, CatRoi_left);
        //addWeighted(catEar_right, 0.5, CatRoi_right, 0.5, 0, CatRoi_right);

        for(int i = 0; i < 3; ++i)
        {
            Roi_lefts[i] = Roi_lefts[i].mul(255.0 - source_left[3], 1/255.0);
            Roi_lefts[i] += source_left[i].mul(source_left[3], 1/255.0);
            Roi_rights[i] = Roi_rights[i].mul(255.0 - source_right[3], 1/255.0);
            Roi_rights[i] += source_right[i].mul(source_right[3], 1/255.0);
        }
        merge(Roi_lefts, CatRoi_left);
        merge(Roi_rights, CatRoi_right);
        faceROI = src(faces[0]);
        //存储找到的眼睛矩形。
        std::vector<Rect> eyes;
        eye.detectMultiScale(faceROI, eyes, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(30,30));
        if(eyes.size() == 2)
        {
            int position1_x = eyes[0].x + faces[0].x;
            int position1_y = eyes[0].y + faces[0].y + faces[0].height * 0.5;
            int position2_x = eyes[1].x + faces[0].x;
            int position2_y = eyes[1].y + faces[0].y + faces[0].height * 0.5;

            Point p1_left(position1_x, position1_y);
            Point p2_left(static_cast<unsigned int>(position1_x - faces[0].width * 0.4),
                          static_cast<unsigned int>(position1_y + faces[0].height * 0.2));
            Point p3_left(position1_x, static_cast<unsigned int>(position1_y - faces[0].height * 0.1));
            Point p4_left(static_cast<unsigned int>(position1_x - faces[0].width * 0.4),
                          static_cast<unsigned int>(position1_y - faces[0].height * 0.1 - faces[0].height * 0.2));

            Point p1_right(position2_x, position2_y);
            Point p2_right(static_cast<unsigned int>(position2_x + faces[0].width * 0.4),
                          static_cast<unsigned int>(position2_y + faces[0].height * 0.2));
            Point p3_right(position2_x, static_cast<unsigned int>(position2_y - faces[0].height * 0.1));
            Point p4_right(static_cast<unsigned int>(position2_x + faces[0].width * 0.4),
                          static_cast<unsigned int>(position2_y - faces[0].height * 0.1 - faces[0].height * 0.2));
            cv::line(frame, p1_left, p2_left, Scalar(0,0,0), 2);
            cv::line(frame, p3_left, p4_left, Scalar(0,0,0), 2);
            cv::line(frame, p1_right, p2_right, Scalar(0,0,0), 2);
            cv::line(frame, p3_right, p4_right, Scalar(0,0,0), 2);

        }
        //medianBlur(frame, frame, CV_SHAPE_CROSS);
    }

    imshow("face", frame);
    //imshow("catear", catEar_left);
}

void FaceClassifier::DrawGrass(Mat &src)
{
    cvtColor(src, src, COLOR_BGR2BGRA);
    Mat img_cao = imread("../image/yanwenzi/cao.png", CV_LOAD_IMAGE_UNCHANGED);
    Mat hotpot_left = imread("../image/yanwenzi/left_hotpot.png", CV_LOAD_IMAGE_UNCHANGED);
    Mat hotpot_right = imread("../image/yanwenzi/right_hotpot.png", CV_LOAD_IMAGE_UNCHANGED);
    Mat frame, faceROI;
    frame = src.clone();
    std::vector<Rect> faces;
    cvtColor(src, src, COLOR_BGRA2GRAY);
    equalizeHist(src, src);
    face.detectMultiScale(src, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30,30));

    if(!faces.empty()) {
        std::vector<Mat> vec_hotpot_left, vec_hotpot_right, vec_cao;

        Size sizeofface = faces[0].size();
        sizeofface.width /= 2;
        sizeofface.height /= 2;
        resize(img_cao, img_cao, sizeofface);
        sizeofface.width /= 2;
        sizeofface.height /= 2;
        resize(hotpot_left, hotpot_left, sizeofface);
        resize(hotpot_right, hotpot_right, sizeofface);
        //rectangle(frame, faces[0], Scalar(0, 0, 225), 1);
        split(img_cao, vec_cao);
        split(hotpot_left, vec_hotpot_left);
        split(hotpot_right, vec_hotpot_right);

        cv::Rect position_cao = faces[0];
        position_cao.x += position_cao.width / 4;
        position_cao.y -= position_cao.height / 2.5;
        if(position_cao.y < 0)
            position_cao.y = 0;
        position_cao.width /= 2;
        position_cao.height /= 2;
        Mat cao_ROI = frame(position_cao);
        std::vector<Mat> cao_ROIS;
        split(cao_ROI, cao_ROIS);
        for(int i = 0; i < 3; ++i)
        {
            cao_ROIS[i] = cao_ROIS[i].mul(255.0 - vec_cao[3], 1/255.0);
            cao_ROIS[i] += vec_cao[i].mul(vec_cao[3], 1/255.0);
        }
        merge(cao_ROIS, cao_ROI);

        faceROI = src(faces[0]);
        //存储找到的眼睛矩形。
        std::vector<Rect> eyes;
        eye.detectMultiScale(faceROI, eyes, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(30,30));
        if(eyes.size() == 2)
        {
            int position1_x = eyes[0].x + faces[0].x - faces[0].width * 0.15;
            int position1_y = eyes[0].y + faces[0].y + faces[0].height * 0.3;
            int position2_x = eyes[1].x + faces[0].x + faces[0].width * 0.15;
            int position2_y = eyes[1].y + faces[0].y + faces[0].height * 0.3;

            cv::Rect position_hotpot_left(position1_x, position1_y, faces[0].width / 4, faces[0].height / 4);
            cv::Rect position_hotpot_right(position2_x, position2_y, faces[0].width / 4, faces[0].height / 4);
            Mat hotpot_left_ROI = frame(position_hotpot_left);
            Mat hotpot_right_ROI = frame(position_hotpot_right);
            std::vector<Mat> hotpot_left_ROIS, hotpot_right_ROIS;
            split(hotpot_left_ROI, hotpot_left_ROIS);
            split(hotpot_right_ROI, hotpot_right_ROIS);
            for(int i = 0; i < 3; ++i)
            {
                hotpot_left_ROIS[i] = hotpot_left_ROIS[i].mul(255.0 - vec_hotpot_left[3], 1/255.0);
                hotpot_left_ROIS[i] += vec_hotpot_left[i].mul(vec_hotpot_left[3], 1/255.0);
                hotpot_right_ROIS[i] = hotpot_right_ROIS[i].mul(255.0 - vec_hotpot_right[3], 1/255.0);
                hotpot_right_ROIS[i] += vec_hotpot_right[i].mul(vec_hotpot_right[3], 1/255.0);
            }
            merge(hotpot_left_ROIS, hotpot_left_ROI);
            merge(hotpot_right_ROIS, hotpot_right_ROI);
        }
        //medianBlur(frame, frame, CV_SHAPE_CROSS);
    }

    imshow("face", frame);
    //imshow("catear", catEar_left);
}