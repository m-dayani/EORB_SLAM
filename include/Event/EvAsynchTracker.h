//
// Created by root on 1/9/21.
//

#ifndef ORB_SLAM3_EVASYNCHTRACKER_H
#define ORB_SLAM3_EVASYNCHTRACKER_H

#include <vector>
#include <chrono>
#include <thread>
#include <memory>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#ifndef CERES_FOUND
#define CERES_FOUND true
#endif
#include <opencv2/sfm.hpp>

#include "EventConversion.h"
#include "EventFrame.h"
#include "EvBaseTracker.h"

#include "KeyFrame.h"
#include "KeyFrameDatabase.h"
#include "Atlas.h"
#include "Visualization.h"


namespace EORB_SLAM {

    class EvTrackManager;
    class EvLocalMapping;

#ifndef DEF_MATCHES_WIN_NAME
#define DEF_MATCHES_WIN_NAME "Matches"
#endif
#define DEF_MAX_IM_BUFF_CAP 10
#define DEF_TH_MED_PX_DISP 15.f
#define DEF_INIT_MIN_TRACKED_PTS3D 50
#define DEF_MIN_TRACKED_PTS3D 30
#define DEF_MIN_TRACKED_PTS3D_TLM 15
#define DEF_MIN_KFS_IN_MAP 3

    class EvAsynchTracker : public EvBaseTracker {
    public:

        enum TrackState {
            IDLE,
            INIT_TRACK,
            INIT_MAP,
            TRACKING,
            //CH_REF_KF,
            //OPTIM,
            STOP
        };

        enum TrackMode {
            ODOMETRY,
            TLM,
            TLM_CH_REF
        };

        EvAsynchTracker(shared_ptr<EvParams> evParams, CamParamsPtr camParams, const MixedFtsParamsPtr& pFtParams,
                        std::shared_ptr<ORB_SLAM3::Atlas> pEvAtlas, ORB_SLAM3::ORBVocabulary* pVocab,
                        SensorConfigPtr  sConf, ORB_SLAM3::Viewer* pViewer);
        ~EvAsynchTracker() override;

        void setTrackManager(EvTrackManager* pTM) { mpTrackManager = pTM; }

        /* ---------------------------------------------------------------------------------------------------------- */

        bool isTrackerReady() override;
        bool isTrackerSaturated() override;

        bool isInputGood() override;
        virtual bool isIdle() { return mStat == IDLE; }
        virtual bool isInitTracking() { return mStat == INIT_TRACK; }
        virtual bool isTracking() { return mStat == TRACKING; }
        bool isMapInitialized() { return mbMapInitialized == true; }
        bool isImuInitialized() { return mbImuInitialized == true; }

        bool isStopped() override { return this->sepThread && this->mStat == STOP; }
        void stop() override;
        void resetAll() override;

        /* ---------------------------------------------------------------------------------------------------------- */

        void makeFrame(const PoseImagePtr& pImage, unsigned fId, EvFramePtr& frame) override;

        /* ---------------------------------------------------------------------------------------------------------- */

        // Sample L2 Tracking
        void Track() override;

        /* ---------------------------------------------------------------------------------------------------------- */

        int refineTrackedMPs(EvFramePtr& currFrame);

        /* ---------------------------------------------------------------------------------------------------------- */

        void getLastDPoseAndDepth(double& t0, double& dts, cv::Mat& Tcr, float& medDepth);
        PoseDepthPtr getLastPoseDepthInfo() const { return mpPoseInfo; }

        void getEventConstraints(std::vector<ORB_SLAM3::KeyFrame*>& vpEvKFs);

        /* ---------------------------------------------------------------------------------------------------------- */

        int getMatchesInliers() const { return mnMatchesInliers; }

        void setImuInitialized(const bool flag) { mbImuInitialized = flag; }

        void setImuManagerAndChannel(const std::shared_ptr<IMU_Manager>& pImuManager) override;

        virtual void updateFrameIMU(float s, const ORB_SLAM3::IMU::Bias &b, ORB_SLAM3::KeyFrame* pCurrentKeyFrame);

        /* ---------------------------------------------------------------------------------------------------------- */

        virtual void updateFrameDrawer();

        virtual void saveAllFramePoses(const std::string& fileName, double tsc, int ts_prec);
        virtual void saveAllPoses(const std::string& fileName, double tsc, int ts_prec);
        virtual void saveAtlas(const std::string& fileName);

    protected:

        //void setNumLoop(unsigned nLoop) { this->mNumLoop = nLoop; }
        //void setEvWinSize(unsigned szEvWin) { this->mEvWinSize = szEvWin; }

        void reset();
        void resetActiveMap();
        void resetInitMapState();
        void resetKeyFrameTracker(const PoseImagePtr& pImage);

        /* ---------------------------------------------------------------------------------------------------------- */

        static bool resolveReconstStat(bool reconstRes, const float& medPxDisp, const ORB_SLAM3::ReconstInfo& rInfo,
                                       const std::vector<std::vector<cv::Point3f>>& pts3d, std::vector<int>& vMatches12);

        bool needNewKeyFrame(const cv::Mat& image);

        //bool isCurrPoseGood(const double& ts, int& state);

        void prepReInitDisconnGraph();
        void backupLastRefInfo();
        void restoreLastRefInfo(EvFramePtr& lastRefFrame, ORB_SLAM3::KeyFrame* pLastRefKF, ORB_SLAM3::KeyFrame* pLastKF);

        void resetToInitTracking(SharedState<TrackState>& currState, const SharedState<TrackMode>& currTM);
        void changeStateToInitMap(SharedState<TrackState>& currState);
        void changeStateToTLM(SharedState<TrackState>& currState, const PoseImagePtr& pImage);

        /* ---------------------------------------------------------------------------------------------------------- */

        void makeFrame(const PoseImagePtr& pImage, unsigned int fId, unique_ptr<ELK_Tracker> &pLKTracker,
                       EvFramePtr &frame) override;
        void makeFrame(const PoseImagePtr& pImage, unsigned fId,
                       const std::vector<cv::KeyPoint>& p1, EvFramePtr& frame) override;
        //bool resolveNewIniFrame(EvFrame& newIniFrame);
        unsigned trackAndFrame(const PoseImagePtr& pImage, unsigned fId, std::unique_ptr<ELK_Tracker>& pLKTracker,
                               EvFramePtr& evFrame) override;

        /* ---------------------------------------------------------------------------------------------------------- */

        //void insertPosesAndMap(ORB_SLAM3::KeyFrame* pKFini, ORB_SLAM3::KeyFrame* pKFcur,
        //                       const std::vector<ORB_SLAM3::MapPoint*>& vpMapPoints);
        void insertRefKF(ORB_SLAM3::KeyFrame* pKFref);
        ORB_SLAM3::KeyFrame * insertRefKF(EvFramePtr& refFrame);
        virtual void insertCurrKF(ORB_SLAM3::KeyFrame* pKFcur, ORB_SLAM3::KeyFrame* pKFref);
        virtual ORB_SLAM3::KeyFrame * insertCurrKF(EvFramePtr& currFrame, ORB_SLAM3::KeyFrame* pKFref);

        void updateRefMapPoints(const std::vector<ORB_SLAM3::MapPoint*>& vpCurrMPs);

        //cv::Mat resolveIniPose();
        //void updateLastPose(const EvFramePtr& currFrame);
        //void updateLastPose(double ts, const cv::Mat& newPose);
        //void getLastPose(double& ts, cv::Mat& Tcw);
        //void grabL1PoseChange(const cv::Mat& DTcw);
        //static void getCurrPoseSmooth(const cv::Mat& DTcw, cv::Mat& currPose);
        void predictNextPose(EvFramePtr& pCurrFrame);

        void refineInertialPoseMapIni(const cv::Mat& Tcw0, std::vector<cv::Point3f>& pts3d);

        static void mergeSimilarFrames(EvFramePtr& pFr);

        ORB_SLAM3::IMU::Preintegrated* getPreintegratedFromLastKF(const EvFramePtr& pFr);

        /* ---------------------------------------------------------------------------------------------------------- */

        int init(const cv::Mat& im, const double& ts, const cv::Mat& initPose);

        int initTracking(const cv::Mat& image, double ts, int& stat);

        bool initMap(bool normalize = true);

        //static int assignMatchedMPs(EvFramePtr& iniFrame, EvFramePtr& currFrame);
        static int assignMatchedMPs(ORB_SLAM3::KeyFrame* pRefKF, EvFramePtr& currFrame);

        int estimateCurrPoseTLM(EvFramePtr& iniFrame, EvFramePtr& currFrame);
        void trackLocalMap(const cv::Mat& image, double ts, int& nTrackedPts,
                           const int& nMinTrakedPts = DEF_MIN_TRACKED_PTS3D_TLM);

        //int refreshNewInitMap(const cv::Mat& image, const double& ts);
        //int refreshNewTracking(const cv::Mat& image, const double& ts);
        //void initiateRefChange(const cv::Mat& image, const EvFrame& tempIniFrame);
        //int completeRefChange(const cv::Mat& currImg, const EvFrame& currFrame);
        //void initiateNewInitMap();
        void refreshNewTrack(const PoseImagePtr& pImage);

        void addNewKeyFrame(const cv::Mat& currImage);

        void refreshLastTrackedKPts(ORB_SLAM3::KeyFrame* pKFcur);

        //void localMapping(ORB_SLAM3::KeyFrame* pKFcur);

        //void localBA();
        //void globalBA();

    protected:

        // For multi-threaded operation,
        // we need thread-safe state vars & input/output buffers

        // States
        SharedState<TrackState> mStat;// mTempStat;
        SharedState<TrackMode> mTrackMode;
        SensorConfigPtr mpSensor;

        EvFramePtr mpIniFrame, mpIniFrameTemp;
        EvFramePtr mpLastFrame;
        EvFramePtr mpCurrFrame;

        SharedFlag mbTrackerInitialized;
        SharedFlag mbMapInitialized;
        SharedFlag mbImuInitialized;
        unsigned char mBadInitCnt;

        static unsigned long mFrameIdx;
        static unsigned long mKeyFrameIdx;

        ORB_SLAM3::KeyFrame* mpReferenceKF;
        ORB_SLAM3::KeyFrame* mpRefKfTemp;
        ORB_SLAM3::KeyFrame* mpLastKF;

        PoseDepthPtr mpPoseInfo;

        // Inputs

        // Outputs
        //std::vector<EvFrame> mvEvFrames;
        std::unique_ptr<ORB_SLAM3::KeyFrameDatabase> mpKeyFrameDB;
        std::vector<EvFramePtr> mvpRefFrames;
        // These are from ORB-SLAM Tracking
        std::mutex mMtxVpRefKFs;
        std::vector<ORB_SLAM3::KeyFrame*> mvpRefKeyFrames;
        std::vector<ORB_SLAM3::KeyFrame*> mvpLocalKeyFrames;
        std::vector<ORB_SLAM3::MapPoint*> mvpLocalMapPoints;
        //cv::Mat mVelocity;

        // Utils
        std::shared_ptr<ORB_SLAM3::Atlas> mpEvAtlas;

        //cv::Ptr<cv::FastFeatureDetector> mpFastDetector;
        //std::shared_ptr<ORB_SLAM3::ORBextractor> mpORBDetector;
        std::shared_ptr<ORB_SLAM3::ORBextractor> mpORBLevelDetector;

        std::unique_ptr<ELK_Tracker> mpLKTracker;//, mpLKTrackerTemp;
        std::unique_ptr<ELK_Tracker> mpLKTrackerKF;

        //ORB_SLAM3::GeometricCamera* mpCamera;

        //std::shared_ptr<SimpleImageDisplay> mpImageDisplay;
        ORB_SLAM3::Viewer* mpViewer;
        //thread* mptImageDisplay;

        //std::unique_ptr<std::thread> mpThreadLocalBA;

        // Params
        const float mMaxDistRefreshPts;

        // Top Track Manager
        EvTrackManager* mpTrackManager;

        // Local Event Mapper
        std::unique_ptr<EvLocalMapping> mpLocalMapper;
        std::thread* mptLocalMapping;

        int mnMatchesInliers;

        bool mbScaledTracking;

        //std::mutex mMtxUpdateState;

    public:
        static unsigned long mMapIdx;

        std::mutex mMtxUpdateState;

    };

} // EORB_SLAM

#endif //ORB_SLAM3_EVASYNCHTRACKER_H
