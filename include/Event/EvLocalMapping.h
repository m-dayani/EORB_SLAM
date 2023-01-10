//
// Created by root on 8/6/21.
//

#ifndef ORB_SLAM3_EVLOCALMAPPING_H
#define ORB_SLAM3_EVLOCALMAPPING_H

#include "KeyFrame.h"
#include "Atlas.h"

#include "EventFrame.h"
#include "IMU_Manager.h"


namespace EORB_SLAM {

    class EvAsynchTracker;

    class EvLocalMapping {
    public:

        EvLocalMapping(EvAsynchTracker* pEvTracker, std::shared_ptr<ORB_SLAM3::Atlas> pAtlas, SensorConfigPtr pSensor,
                       const CamParamsPtr& pCamParams, std::shared_ptr<IMU_Manager> pImuManager = nullptr);

        ~EvLocalMapping();

        bool isReadyForNewKeyFrame();
        void insertNewKeyFrame(ORB_SLAM3::KeyFrame* pCurrKF);

        void localMapping(ORB_SLAM3::KeyFrame* pKFcur);

        void localBA();

        bool isProcessing() { return mbProcFlag == true; }
        void stop() { mbStop = true; }

        void setImuManager(const std::shared_ptr<IMU_Manager> &pImuManager) { mpImuManager = pImuManager; }

        void Run();

        void abortLBA() { mbAbortBA = true; }

        void resetAll();

        void updateInertialStateTracker();

        static void processNewKeyFrame(std::shared_ptr<ORB_SLAM3::Atlas>& mpAtlas, ORB_SLAM3::KeyFrame* mpCurrentKeyFrame,
                                       std::list<ORB_SLAM3::MapPoint*>& mlpRecentAddedMapPoints);

        void keyFrameCulling();

        static cv::Mat SkewSymmetricMatrix(const cv::Mat &v);
        static cv::Mat ComputeF12(ORB_SLAM3::KeyFrame *&pKF1, ORB_SLAM3::KeyFrame *&pKF2);

        static bool triangulateAndCheckMP(const cv::KeyPoint& kp1, const cv::KeyPoint& kp2,
                ORB_SLAM3::KeyFrame* pKF1, ORB_SLAM3::KeyFrame* pKF2, const cv::Mat& Tcw1, const cv::Mat& Rcw1,
                const cv::Mat& Rwc1, const cv::Mat& tcw1, const cv::Mat& Ow1, const cv::Mat& Tcw2, const cv::Mat& Rcw2,
                const cv::Mat& Rwc2, const cv::Mat& tcw2, const cv::Mat& Ow2, float fThCosParr, float ratioFactor, cv::Mat& x3D,
                bool checkKPtLevelInfo = false);
        static void getCameraPoseInfo(ORB_SLAM3::KeyFrame* pKF, cv::Mat& Tcw1, cv::Mat& Rcw1, cv::Mat& Rwc1, cv::Mat& tcw1, cv::Mat& Ow1);

        static unsigned addNewMapPoints(const std::shared_ptr<ORB_SLAM3::Atlas>& mpAtlas, ORB_SLAM3::KeyFrame* pKFini,
                ORB_SLAM3::KeyFrame* pKFcur, const std::vector<int>& vMatches12, const std::vector<cv::Point3f>& vP3D);

        static int createNewMapPoints(const std::shared_ptr<ORB_SLAM3::Atlas>& mpAtlas, ORB_SLAM3::KeyFrame* pKFini,
                                      ORB_SLAM3::KeyFrame* pKFcur, std::vector<ORB_SLAM3::MapPoint*>& vpMPs, std::vector<bool>& vbOutliers,
                                      const std::vector<ORB_SLAM3::MapPoint*>& vpIniMPs, const std::vector<int>& vMatches12, float medDepth = -1.f);

        static int mapPointCullingSTD(std::shared_ptr<ORB_SLAM3::Atlas>& pEvAtlas);

        static void mapPointCullingCovis(std::shared_ptr<ORB_SLAM3::Atlas>& mpAtlas, ORB_SLAM3::KeyFrame* pKFcur,
                const EvFramePtr& pFrCur, std::list<ORB_SLAM3::MapPoint*>& mlpRecentAddedMPts, bool bMono = true);

        void createNewMapPoints(std::shared_ptr<ORB_SLAM3::Atlas>& mpAtlas, ORB_SLAM3::KeyFrame* pKFcur,
                                std::list<ORB_SLAM3::MapPoint*>& mlpRecentAddedMPts);

        void createNewMapPoints(std::shared_ptr<ORB_SLAM3::Atlas>& mpAtlas, std::vector<FeatureTrackPtr>& vpFtTracks,
                                ulong minFrId, ulong maxFrId, std::list<ORB_SLAM3::MapPoint*>& mlpRecentAddedMPts);

        static void scalePoseAndMap(std::vector<ORB_SLAM3::KeyFrame*>& vpKFs, std::vector<ORB_SLAM3::MapPoint*>& vpAllMapPoints,
                                    const float& medianDepth, bool isInertial = false);

        static void scalePoseAndMap(std::vector<EvFramePtr>& vpFrs, std::vector<cv::Point3f>& vPts3D,
                                    const float& medianDepth, bool isInertial = false);

        void enableLocalMapping(const bool state) { mbLocalMappingEnabled = state; }

        void setOptimizeFlag(bool flag) { mbOptimize = flag; }

    protected:

    private:

        SharedQueue<ORB_SLAM3::KeyFrame*> mqpKeyFrames;
        ORB_SLAM3::KeyFrame* mpCurrKF;

        SharedFlag mbStop;
        SharedFlag mbProcFlag;
        SharedFlag mbOptimize;

        bool mbLocalMappingEnabled;

        bool mbAbortBA;

        SharedFlag mbImuInitialized;

        EvAsynchTracker* mpTracker;

        std::shared_ptr<ORB_SLAM3::Atlas> mpEvAtlas;

        SensorConfigPtr mpSensor;

        std::shared_ptr<IMU_Manager> mpImuManager;

        double mFirstTs, mTinit;

        const bool mbMonocular, mbInertial;
        cv::Mat mK;
        bool mbFarPoints;
        float mThFarPoints;
    };

} // EORB_SLAM


#endif //ORB_SLAM3_EVLOCALMAPPING_H
