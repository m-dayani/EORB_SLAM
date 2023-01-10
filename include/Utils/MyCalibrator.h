/**
 * Note: Event and Image points in a dataset can be distorted or not independently.
 *      Distortion is realized based on separate configuration settings.
 */

#ifndef ORB_SLAM3_MYCALIBRATOR_H
#define ORB_SLAM3_MYCALIBRATOR_H

#include <cmath>
#include <vector>
#include <memory>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <glog/logging.h>


namespace EORB_SLAM {

    class MyCalibrator {
    public:
        enum CameraType {
            PINHOLE,
            FISHEYE
        };

        MyCalibrator(const cv::Mat&  K, const cv::Mat&  distCoefs, const cv::Scalar& imageSize,
                     const cv::Mat&  R = cv::Mat(), const cv::Mat& P = cv::Mat(), bool isFishEye = false);

        static bool isInImage(float x, float y, const cv::Scalar& imageSize);
        static bool isInImage(float x, float y, int imWidth, int imHeight);
        bool isInImage(float x, float y) const;

        static bool isDistorted(const cv::Mat& distCoefs);

        bool isFishEye() const { return mCamType == FISHEYE; }
        bool isPinhole() const { return mCamType == PINHOLE; }


        void generateUndistMaps();
        void generateUndistMapsPinhole();
        void generateUndistMapsFishEye();


        void undistKeyPoints(const std::vector<cv::KeyPoint>& vDistKPts, std::vector<cv::KeyPoint>& vUndistKPts);

        void undistKeyPointsPinhole(const std::vector<cv::KeyPoint>& vDistKPts, std::vector<cv::KeyPoint>& vUndistKPts);
        static void undistKeyPointsPinhole(const std::vector<cv::KeyPoint>& vDistKPts,
                std::vector<cv::KeyPoint>& vUndistKPts, const cv::Mat& K, const cv::Mat& distCoefs,
                const cv::Mat& R = cv::Mat(), const cv::Mat& P = cv::Mat());

        void undistKeyPointsFishEye(const std::vector<cv::KeyPoint>& vDistKPts, std::vector<cv::KeyPoint>& vUndistKPts);
        static void undistKeyPointsFishEye(const std::vector<cv::KeyPoint>& vDistKPts,
                std::vector<cv::KeyPoint>& vUndistKPts, const cv::Mat& K, const cv::Mat& distCoefs,
                const cv::Mat& R = cv::Mat(), const cv::Mat& P = cv::Mat());


        void undistPoint(const cv::Point2f& srcPt, cv::Point2f& dstPt);

        void undistPointPinhole(const cv::Point2f& srcPt, cv::Point2f& dstPt);
        static void undistPointPinhole(const cv::Point2f& srcPt, cv::Point2f& dstPt, const cv::Mat& K,
                const cv::Mat& distCoefs, const cv::Mat& R = cv::Mat(), const cv::Mat& P = cv::Mat());

        void undistPointFishEye(const cv::Point2f& srcPt, cv::Point2f& dstPt);
        static void undistPointFishEye(const cv::Point2f& srcPt, cv::Point2f& dstPt, const cv::Mat& K,
                const cv::Mat& distCoefs, const cv::Mat& R, const cv::Mat& P = cv::Mat());


        // This is not so easy because it requires interpolation
        //void undistKeyPointsMaps(const std::vector<cv::KeyPoint>& vDistKPts, std::vector<cv::KeyPoint>& vUndistKPts);
        //static void undistKeyPointsMaps(const std::vector<cv::KeyPoint>& vDistKPts,
        //        std::vector<cv::KeyPoint>& vUndistKPts, const cv::Mat& mapX, const cv::Mat& mapY);


        // Only use these for integer points!
        void undistPointMaps(const cv::Point2f& srcPt, cv::Point2f& dstPt);
        static void undistPointMaps(const cv::Point2f& srcPt, cv::Point2f& dstPt, const cv::Mat& mapX, const cv::Mat& mapY);

        void undistImageMaps(const cv::Mat& srcImage, cv::Mat& dstImage);
        static void undistImageMaps(const cv::Mat& srcImage, const cv::Mat& mapX, const cv::Mat& mapY, cv::Mat& dstImage);

    protected:

    private:
        CameraType mCamType;
        int mImWidth, mImHeight;
        cv::Size mImSize;
        cv::Mat mK;
        cv::Mat mDistCoefs;
        cv::Mat mR; // Rectification Matrix
        cv::Mat mP; // Projection Matrix
        cv::Mat mUndistMapX, mUndistMapY;
        cv::Mat mNewCamMatrix;
    };

    typedef std::shared_ptr<MyCalibrator> MyCalibPtr;
    typedef std::pair<MyCalibPtr, MyCalibPtr> PairCalibPtr;

} // EORB_SLAM

#endif //ORB_SLAM3_MYCALIBRATOR_H
