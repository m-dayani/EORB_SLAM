/**
 * Note: Event frames alwayes are generated with undistorted image and key points
 */

#ifndef ORB_SLAM3_EVENTFRAME_H
#define ORB_SLAM3_EVENTFRAME_H

#include <vector>
#include <cmath>

#include <opencv2/core.hpp>

#include "EventData.h"

#include "Frame.h"


namespace EORB_SLAM {

    class FeatureTrack;
    typedef std::shared_ptr<FeatureTrack> FeatureTrackPtr;

    class EvFrame : public ORB_SLAM3::Frame {
    public:
        EvFrame() : Frame(), mnTrackedPts(0), mpRefFrame(), mpPrevFrame(), mMedPxDisp(0.f)
        {}

        // Constructor for simple FAST detection
        EvFrame(unsigned long kfId, const cv::Mat& evImage, const double &timeStamp,
                const cv::Ptr<cv::FastFeatureDetector>& fastDet,
                int nMaxPts, ORB_SLAM3::GeometricCamera* pCamera, const cv::Mat &distCoef,
                const std::shared_ptr<EvFrame>& pPrevF = nullptr,
                const ORB_SLAM3::IMU::Calib &ImuCalib = ORB_SLAM3::IMU::Calib());

        // Constructor for ORB detection (no descriptors)
        EvFrame(unsigned long kfId, const cv::Mat& evImage, const double &timeStamp, ORB_SLAM3::ORBextractor* extractor,
                ORB_SLAM3::GeometricCamera* pCamera, const cv::Mat &distCoef,
                bool extDesc = false, const std::shared_ptr<EvFrame>& pPrevF = nullptr,
                const ORB_SLAM3::IMU::Calib &ImuCalib = ORB_SLAM3::IMU::Calib());

        // Detect FAST in the first octave and ORB for the other levels
        EvFrame(unsigned long kfId, const cv::Mat& evImage, const double &timeStamp,
                const cv::Ptr<cv::FastFeatureDetector>& fastDet, ORB_SLAM3::ORBextractor* extractorORB,
                int nMaxPts, ORB_SLAM3::GeometricCamera* pCamera, const cv::Mat &distCoef,
                const std::shared_ptr<EvFrame>& pPrevF = nullptr,
                const ORB_SLAM3::IMU::Calib &ImuCalib = ORB_SLAM3::IMU::Calib());

        // Constructor with known image and tracked key points
        EvFrame(unsigned long kfId, const cv::Mat& evImage, const double &timeStamp,
                const std::vector<cv::KeyPoint>& kpts, ORB_SLAM3::GeometricCamera* pCamera, const cv::Mat &distCoef,
                const std::shared_ptr<EvFrame>& pPrevF = nullptr,
                const ORB_SLAM3::IMU::Calib &ImuCalib = ORB_SLAM3::IMU::Calib());

        EvFrame(ulong id, const PoseImagePtr& pImage, const std::vector<cv::KeyPoint>& kpts,
                const std::vector<FeatureTrackPtr>& vFtTracks, ORB_SLAM3::GeometricCamera* pCamera, const cv::Mat &distCoef,
                const std::shared_ptr<EvFrame>& pPrevF = nullptr, const ORB_SLAM3::IMU::Calib &ImuCalib = ORB_SLAM3::IMU::Calib());

        // Copy constructor
        EvFrame(const EvFrame& evFrame);

        //~EvFrame();

        void setPrevFrame(const std::shared_ptr<EvFrame>& pEvFrame);
        std::shared_ptr<EvFrame> getPrevFrame();// { return mpPrevFrame; }

        void setRefFrame(const std::shared_ptr<EvFrame>& pEvFrame);
        std::shared_ptr<EvFrame> getRefFrame();// { return mpRefFrame; }

        // Extract ORB on the image. 0 for left image and 1 for right image.
        //void ExtractORB(int flag, const cv::Mat &im, int x0, int x1) override;

        //void UndistortTrackedPts();

        unsigned connectNewAndTrackedPts(const float& threshDist);

        float computeSceneMedianDepth(int q);
        float computeSceneMedianDepth(int q, const std::vector<cv::Point3f>& pts3d);
        static float computeSceneMedianDepth(const cv::Mat& Tcw, const std::vector<ORB_SLAM3::MapPoint*>& vpMPs, int q);

        //void setConnection(const std::vector<int>& matches12, const cv::Mat& Tcw);

        std::vector<int>& getTrackedMatches() { return mvMchTrackedPts; }

        void setMedianPxDisp(const float pxDisp) { mMedPxDisp = pxDisp; }
        float getMedianPxDisp() const { return mMedPxDisp; }


        //void resetAllDistKPtsMono(const std::vector<cv::KeyPoint>& vKpts, bool distorted = false);
        void setDistTrackedPts(const std::vector<cv::KeyPoint>& trackedPts);
        std::vector<cv::KeyPoint>& getDistTrackedPts() { return mvTrackedPts; }
        std::vector<cv::KeyPoint>& getUndistTrackedPts() { return mvTrackedPtsUn; }
        int numTrackedPts() const { return mnTrackedPts; }

        void keepRefImage(const cv::Mat& refIm) { this->imgLeft = refIm.clone(); }

        void setFirstAcc(const cv::Mat& firstAcc) { mFirstAcc = firstAcc.clone(); }
        void setFirstAcc(const ORB_SLAM3::IMU::Point& imuMea);
        cv::Mat getFirstAcc() { return mFirstAcc.clone(); }

        FeatureTrackPtr getFeatureTrack(const size_t& idx);
        std::vector<FeatureTrackPtr>& getAllFeatureTracks() { return mvpFeatureTracks; }
        //void setAllFeatureTracks(const std::vector<FeatureTrackPtr>& vpFtTracks);

    protected:

        EvFrame(unsigned long kfId, const cv::Mat& evImage, const double &timeStamp,
                ORB_SLAM3::GeometricCamera* pCamera, const cv::Mat &distCoef,
                const std::shared_ptr<EvFrame>& pPrevF = nullptr,
                const ORB_SLAM3::IMU::Calib &ImuCalib = ORB_SLAM3::IMU::Calib());

    private:
        int mnTrackedPts;
        std::vector<cv::KeyPoint> mvTrackedPts;
        std::vector<cv::KeyPoint> mvTrackedPtsUn;
        std::vector<FeatureTrackPtr> mvpFeatureTracks;

        std::vector<int> mvMchTrackedPts;

        std::weak_ptr<EvFrame> mpRefFrame;
        std::weak_ptr<EvFrame> mpPrevFrame;

        // Median pixel displacement of the frame
        float mMedPxDisp;

        // IMU related vars
        cv::Mat mFirstAcc;

    public:
        PoseImagePtr mpPoseImage;
    };

    typedef std::shared_ptr<EvFrame> EvFramePtr;
    typedef std::unique_ptr<EvFrame> EvFrameUPtr;

} // EORB_SLAM

#endif //ORB_SLAM3_EVENTFRAME_H
