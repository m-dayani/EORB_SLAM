//
// Created by root on 11/25/20.
//

#ifndef ORB_SLAM3_KLT_TRACKER_H
#define ORB_SLAM3_KLT_TRACKER_H

#include <cmath>

#include <iostream>
#include <vector>
#include <memory>
#include <cassert>
//#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
//#include <opencv2/highgui/highgui.hpp>

#include "EventData.h"


namespace EORB_SLAM
{

    class ELK_Tracker {
    public:
        explicit ELK_Tracker(const std::shared_ptr<EvParams>&  params);

        void setRefImage(const cv::Mat& image, const std::vector<cv::KeyPoint>& refPts);

        // Input image's type must be 8U(C1/C3)
        void trackCurrImage(const cv::Mat& currImage,
                            std::vector<cv::Point2f> &kpts, std::vector<uchar> &status, std::vector<float> &err);

        void trackCurrImage(const cv::Mat& currImage, const std::vector<cv::KeyPoint> &initPts,
                            std::vector<cv::Point2f> &kpts, std::vector<uchar> &status, std::vector<float> &err);

        unsigned refineTrackedPts(const std::vector<cv::Point2f>& currPts, const std::vector<uchar>& status,
                const std::vector<float>& err, std::vector<cv::KeyPoint>& p1, std::vector<int>& vMatches12,
                std::vector<int>& vCntMatches, std::vector<float>& vPxDisp);

        unsigned refineFirstOctaveLevel(std::vector<cv::KeyPoint>& trackedKPts, std::vector<int>& vMatches12,
                                        unsigned& nMatches, std::vector<int>& vCntMatches);
        static unsigned refineFirstOctaveLevel(const std::vector<cv::KeyPoint>& refKPts, const std::vector<cv::KeyPoint>& trackedKPts,
                                               std::vector<int>& vMatches12, unsigned& nMatches);

        //static std::vector<int> dummyCntMatches;
        //static std::vector<float> dummyPxDisps;

        //unsigned trackAndMatchCurrImage(const cv::Mat& currImage);
        unsigned trackAndMatchCurrImage(const cv::Mat& currImage, std::vector<cv::KeyPoint>& trackedKPts,
                                        std::vector<int>& vMatches12);
        unsigned trackAndMatchCurrImage(const cv::Mat& currImage, std::vector<cv::KeyPoint>& trackedKPts,
                std::vector<int>& vMatches12, std::vector<int>& vCntMatches, std::vector<float>& vPxDisp);
        unsigned trackAndMatchCurrImageInit(const cv::Mat& currImage, std::vector<cv::KeyPoint>& trackedKPts,
                std::vector<int>& vMatches12, std::vector<int>& vCntMatches, std::vector<float>& vPxDisp);

        std::vector<cv::KeyPoint> getRefPoints() {
            return this->mRefKPoints;
        }

        void getRefImageAndPoints(cv::Mat& im0, std::vector<cv::KeyPoint>& pts0) {
            im0 = mRefFrame.clone();
            pts0 = mRefKPoints;
        }

        std::vector<cv::KeyPoint> getLastTrackedPts() {
            return this->mLastTrackedKPts;
        }

        void setLastTrackedPts(const std::vector<cv::KeyPoint>& currTrackedPts);

    protected:

    private:
        cv::TermCriteria mLKCriteria;
        //const EvParamsPtr mpEvParams;

        cv::Mat mRefFrame;
        cv::Mat mCurrFrame;

        std::vector<cv::KeyPoint> mRefKPoints;
        std::vector<cv::Point2f> mRefPoints;
        std::vector<cv::KeyPoint> mLastTrackedKPts;
        std::vector<cv::Point2f> mLastTrackedPts;

        const int mPatchSz;
        const int mMaxLevel;
    };


}// namespace EORB_SLAM

#endif //ORB_SLAM3_KLT_TRACKER_H
