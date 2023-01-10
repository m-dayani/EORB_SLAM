//
// Created by root on 3/9/21.
//

#include "MyCalibrator.h"

#include <utility>

namespace EORB_SLAM {

    MyCalibrator::MyCalibrator(const cv::Mat& K, const cv::Mat& distCoefs, const cv::Scalar &imageSize,
            const cv::Mat&  R, const cv::Mat& P, const bool isFishEye) :

            mCamType(PINHOLE), mImWidth(imageSize[0]), mImHeight(imageSize[1]),
            mImSize(imageSize[0], imageSize[1]),
            mK(K.clone()), mDistCoefs(distCoefs.clone()), mR(R.clone()), mP(P.clone())
    {
        if (isFishEye) {
            mCamType = FISHEYE;
            if (P.empty()) {
                // New CamMatrix is actually Projection matrix P
                cv::fisheye::estimateNewCameraMatrixForUndistortRectify(mK, mDistCoefs, mImSize, mR, mP);
            }
        }
        else if (P.empty()) {
            mP = K.clone();
        }
        this->generateUndistMaps();
    }

    bool MyCalibrator::isInImage(const float x, const float y) const {

        return (x >= 0 && x < float(mImWidth)) && (y >= 0 && y < float(mImHeight));
    }

    bool MyCalibrator::isInImage(const float x, const float y, const int imWidth, const int imHeight) {

        return (x >= 0 && x < float(imWidth)) && (y >= 0 && y < float(imHeight));
    }

    bool MyCalibrator::isInImage(const float x, const float y, const cv::Scalar &imageSize) {

        return (x >= 0 && x < imageSize[0]) && (y >= 0 && y < imageSize[1]);
    }

    bool MyCalibrator::isDistorted(const cv::Mat &distCoefs) {

        return !distCoefs.empty() && distCoefs.rows * distCoefs.cols >= 4 &&
               fabs(distCoefs.at<float>(0)) > 1e-9;
    }

    void MyCalibrator::generateUndistMaps() {

        if (this->isPinhole()) {
            DLOG(INFO) << "Generating Pinhole distortion maps...\n";
            this->generateUndistMapsPinhole();
        }
        else if (this->isFishEye()) {
            DLOG(INFO) << "Generating FishEye distortion maps...\n";
            this->generateUndistMapsFishEye();
        }
    }

    void MyCalibrator::generateUndistMapsPinhole() {

        //cv::initUndistortRectifyMap(mK, mDistCoefs, mR, mP,
        //        mImSize, CV_32FC1, mUndistMapX, mUndistMapY);

        mUndistMapX = cv::Mat(mImHeight, mImWidth, CV_32FC1);
        mUndistMapY = cv::Mat(mImHeight, mImWidth, CV_32FC1);

        for (int x = 0; x < mImWidth; x++) {
            for (int y = 0; y < mImHeight; y++) {

                cv::Point2f srcPt(x,y);
                this->undistPointPinhole(srcPt, srcPt);

                mUndistMapX.at<float>(y, x) = srcPt.x;
                mUndistMapY.at<float>(y, x) = srcPt.y;
            }
        }
    }

    void MyCalibrator::generateUndistMapsFishEye() {

        //cv::fisheye::initUndistortRectifyMap(mK, mDistCoefs, mR, mP,
        //        mImSize, CV_32FC1, mUndistMapX, mUndistMapY);

        mUndistMapX = cv::Mat(mImHeight, mImWidth, CV_32FC1);
        mUndistMapY = cv::Mat(mImHeight, mImWidth, CV_32FC1);

        for (int x = 0; x < mImWidth; x++) {
            for (int y = 0; y < mImHeight; y++) {

                cv::Point2f srcPt(x,y);
                this->undistPointFishEye(srcPt, srcPt);

                mUndistMapX.at<float>(y, x) = srcPt.x;
                mUndistMapY.at<float>(y, x) = srcPt.y;
            }
        }
    }

    void MyCalibrator::undistPoint(const cv::Point2f &srcPt, cv::Point2f &dstPt) {

        if (this->isPinhole()) {
            this->undistPointPinhole(srcPt, dstPt);
        }
        else if (this->isFishEye()) {
            this->undistPointFishEye(srcPt, dstPt);
        }
    }

    void MyCalibrator::undistPointPinhole(const cv::Point2f &srcPt, cv::Point2f &dstPt) {

        undistPointPinhole(srcPt, dstPt, mK, mDistCoefs, mR, mP);
    }

    void MyCalibrator::undistPointPinhole(const cv::Point2f &srcPt, cv::Point2f &dstPt, const cv::Mat &K,
                                          const cv::Mat &distCoefs, const cv::Mat& R, const cv::Mat& P) {

        if(!isDistorted(distCoefs)) {
            DLOG_EVERY_N(WARNING, 1000) << "MyCalibrator::undistPointPinhole: Point is not distorted -> nothing to do!\n";
            dstPt = srcPt;
            return;
        }

        cv::Mat srcMat(1, 1, CV_32FC2, cv::Scalar(srcPt.x, srcPt.y));
        cv::undistortPoints(srcMat, srcMat, K, distCoefs, R, P);
        srcMat.reshape(1);

        dstPt.x = srcMat.at<float>(0,0);
        dstPt.y = srcMat.at<float>(0,1);
    }

    void MyCalibrator::undistPointFishEye(const cv::Point2f &srcPt, cv::Point2f &dstPt) {

        undistPointFishEye(srcPt, dstPt, mK, mDistCoefs, mR, mP);
    }

    void MyCalibrator::undistPointFishEye(const cv::Point2f &srcPt, cv::Point2f &dstPt, const cv::Mat &K,
                                          const cv::Mat &distCoefs, const cv::Mat& R, const cv::Mat& P) {

        if(!isDistorted(distCoefs)) {
            DLOG_EVERY_N(WARNING, 1000) << "MyCalibrator::undistPointFishEye: Point is not distorted -> nothing to do!\n";
            dstPt = srcPt;
            return;
        }

        cv::Mat srcMat(1, 1, CV_32FC2, cv::Scalar(srcPt.x, srcPt.y));
        cv::fisheye::undistortPoints(srcMat, srcMat, K, distCoefs, R, P);
        srcMat.reshape(1);

        dstPt.x = srcMat.at<float>(0,0);
        dstPt.y = srcMat.at<float>(0,1);
    }

    void MyCalibrator::undistPointMaps(const cv::Point2f &srcPt, cv::Point2f &dstPt) {

        undistPointMaps(srcPt, dstPt, mUndistMapX, mUndistMapY);
    }

    // Attention!! cv undistMaps are like image: size = (height, width)!
    void MyCalibrator::undistPointMaps(const cv::Point2f &srcPt, cv::Point2f &dstPt,
                                   const cv::Mat &mapX, const cv::Mat &mapY) {

        int rowsX = mapX.rows;
        int colsX = mapX.cols;
        int rowsY = mapY.rows;
        int colsY = mapY.cols;

        int x = static_cast<int>(srcPt.x);
        int y = static_cast<int>(srcPt.y);

        assert(rowsX == rowsY && colsX == colsY && x >= 0 && x < colsX && y >= 0 && y < rowsX);

        dstPt.x = mapX.at<float>(y, x);
        dstPt.y = mapY.at<float>(y, x);
    }

    void MyCalibrator::undistKeyPoints(const std::vector<cv::KeyPoint> &vDistKPts, std::vector<cv::KeyPoint> &vUndistKPts) {

        if (this->isPinhole()) {
            this->undistKeyPointsPinhole(vDistKPts, vUndistKPts);
        }
        else if (this->isFishEye()) {
            this->undistKeyPointsFishEye(vDistKPts, vUndistKPts);
            //this->undistKeyPointsPinhole(vDistKPts, vUndistKPts);
            //vUndistKPts = vDistKPts;
        }
    }

    void MyCalibrator::undistKeyPointsPinhole(const std::vector<cv::KeyPoint> &vDistKPts, std::vector<cv::KeyPoint> &vUndistKPts) {

        undistKeyPointsPinhole(vDistKPts, vUndistKPts, mK, mDistCoefs, mR, mP);
    }

    void MyCalibrator::undistKeyPointsPinhole(const std::vector<cv::KeyPoint> &vDistKPts,
            std::vector<cv::KeyPoint> &vUndistKPts, const cv::Mat &K, const cv::Mat &distCoefs,
            const cv::Mat& R, const cv::Mat& P) {

        if (vDistKPts.empty()) {
            LOG(WARNING) << "MyCalibrator::undistKeyPointsPinhole: Empty key point vector -> nothing to do!\n";
            return;
        }
        if(!isDistorted(distCoefs)) {
            DLOG(WARNING) << "MyCalibrator::undistKeyPointsPinhole: Key points are not distorted -> nothing to do!\n";
            vUndistKPts = vDistKPts;
            return;
        }

        int nPts = vDistKPts.size();
        // Fill matrix with points
        cv::Mat mat(nPts,2, CV_32F);

        for(int i=0; i<nPts; i++)
        {
            mat.at<float>(i,0)=vDistKPts[i].pt.x;
            mat.at<float>(i,1)=vDistKPts[i].pt.y;
        }

        // Undistort points
        mat=mat.reshape(2);
        cv::undistortPoints(mat, mat, K, distCoefs, R, P);
        mat=mat.reshape(1);


        // Fill undistorted keypoint vector
        vUndistKPts.resize(nPts);
        for(int i=0; i<nPts; i++)
        {
            cv::KeyPoint kp = vDistKPts[i];
            kp.pt.x=mat.at<float>(i,0);
            kp.pt.y=mat.at<float>(i,1);
            vUndistKPts[i]=kp;
        }
    }

    void MyCalibrator::undistKeyPointsFishEye(const std::vector<cv::KeyPoint> &vDistKPts,
            std::vector<cv::KeyPoint> &vUndistKPts) {

        undistKeyPointsFishEye(vDistKPts, vUndistKPts, mK, mDistCoefs, mR, mP);
    }

    void MyCalibrator::undistKeyPointsFishEye(const std::vector<cv::KeyPoint> &vDistKPts,
            std::vector<cv::KeyPoint> &vUndistKPts, const cv::Mat &K, const cv::Mat &distCoefs,
            const cv::Mat& R, const cv::Mat& P) {

        if (vDistKPts.empty()) {
            LOG(WARNING) << "MyCalibrator::undistKeyPointsFishEye: Empty key point vector -> nothing to do!\n";
            return;
        }
        if(!isDistorted(distCoefs)) {
            DLOG(WARNING) << "MyCalibrator::undistKeyPointsFishEye: Key points are not distorted -> nothing to do!\n";
            vUndistKPts = vDistKPts;
            return;
        }

        int nPts = vDistKPts.size();
        // Fill matrix with points
        cv::Mat mat(nPts,2, CV_32F);

        for(int i=0; i<nPts; i++)
        {
            mat.at<float>(i,0)=vDistKPts[i].pt.x;
            mat.at<float>(i,1)=vDistKPts[i].pt.y;
        }

        // Undistort points
        mat=mat.reshape(2);
        cv::fisheye::undistortPoints(mat, mat, K, distCoefs, R, P);
        mat=mat.reshape(1);

        // Fill undistorted keypoint vector
        vUndistKPts.resize(nPts);
        for(int i=0; i<nPts; i++)
        {
            cv::KeyPoint kp = vDistKPts[i];
            kp.pt.x=mat.at<float>(i,0);
            kp.pt.y=mat.at<float>(i,1);
            vUndistKPts[i]=kp;
        }
    }

    void MyCalibrator::undistImageMaps(const cv::Mat &srcImage, cv::Mat &dstImage) {

        undistImageMaps(srcImage, mUndistMapX, mUndistMapY, dstImage);
    }

    void MyCalibrator::undistImageMaps(const cv::Mat &srcImage, const cv::Mat &mapX, const cv::Mat &mapY, cv::Mat &dstImage) {

        cv::remap(srcImage,dstImage,mapX,mapY,cv::INTER_LINEAR);
    }

    /*void MyCalibrator::undistKeyPointsMaps(const std::vector<cv::KeyPoint> &vDistKPts, std::vector<cv::KeyPoint> &vUndistKPts) {

        undistKeyPointsMaps(vDistKPts, vUndistKPts, mUndistMapX, mUndistMapY);
    }

    void MyCalibrator::undistKeyPointsMaps(const std::vector<cv::KeyPoint> &vDistKPts, std::vector<cv::KeyPoint> &vUndistKPts,
                                           const cv::Mat &mapX, const cv::Mat &mapY) {

        int nPts = vDistKPts.size();
        vUndistKPts.resize(nPts);

        for (size_t i = 0; i < nPts; i++) {

            cv::KeyPoint dstKPt = vDistKPts[i];

            cv::Point2f srcPt = dstKPt.pt;
            undistPointMaps(srcPt, srcPt, mapX, mapY);
            dstKPt.pt.x = srcPt.x;
            dstKPt.pt.y = srcPt.y;

            vUndistKPts[i] = dstKPt;
        }
    }*/

}

