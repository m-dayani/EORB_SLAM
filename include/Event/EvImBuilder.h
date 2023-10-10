//
// Created by root on 12/24/20.
//

#ifndef ORB_SLAM3_EVIMBUILDER_H
#define ORB_SLAM3_EVIMBUILDER_H

#include <ctime>
#include <cstdlib>
#include <vector>
#include <map>
#include <algorithm>
#include <chrono>
#include <thread>
#include <memory>
#include <sstream>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#ifndef CERES_FOUND
#define CERES_FOUND true
#endif
#include <opencv2/sfm.hpp>

#include "EventConversion.h"
#include "EventFrame.h"
#include "EvBaseTracker.h"
#include "EvAsynchTracker.h"
//#include "EvOptimizer.h"

#include "Converter.h"
#include "KeyFrame.h"


namespace EORB_SLAM {

#define DEF_L1_SIM3_OPTIM_TH2 10.0
#define DEF_L1_SIM3_OPTIM_NITR 10
#define DEF_L1_MAX_TRACK_LOST 3
#define DEF_NKFS_TO_OPTIM 2
#define DEF_PATCH_SIZE_STD 30
#define DEF_L1_TH_NEAR_POSE 1e-2
#define DEF_CHI2_RATIO 0.1 //10 percent

    class EvTrackManager;

    class EvImBuilder : public EvBaseTracker {
    public:

        enum TrackState {
            IDLE,
            INIT,
            TRACKING,
            //OPTIM,
            JAMMED_L2,
            STOP
        };

        EvImBuilder(shared_ptr<EvParams> evParams, CamParamsPtr camParams, MixedFtsParamsPtr pFtParams, SensorConfigPtr  sConfig);
        ~EvImBuilder() override;

        void setLinkedL2Tracker(std::shared_ptr<EvAsynchTracker>& l2Tracker);
        void setTrackManager(EvTrackManager* pEvTrackManager) { mpTrackManager = pEvTrackManager; }

        bool isTrackerReady() override;
        bool isTrackerSaturated() override;

        bool needTrackTinyFrame() const;
        bool needDispatchMCF() const;

        // If we only track with events, we can use
        // more resources to build and track MC-Images with complicated
        // optimizations in parallel.
        // If other sensor(s) available, we can use simpler MC-Image generation
        // mothods to save resources for other tracker modules.
        // This is decided based on L1OpMode option
        void Track() override;

        bool isInputGood() override;
        bool isStopped() override { return this->sepThread && this->mStat == STOP; }
        void stop() override { this->mStat.update(STOP); }

        void resetAll() override;

        //void makeFrame(const cv::Mat& im, double ts, unsigned fId, EvFramePtr& frame) override;

        unsigned getL1ChunkSize();

        // Doesn't use explicit L2 Pose anymore (done internally)
        void getSynchMCI(const std::vector<EventData>& evs, PoseImagePtr& pMCI, bool useL2 = false, bool optimize = false);

    protected:
        //void setNumLoop(unsigned nLoop) { this->mNumLoop = nLoop; }
        //void setEvWinSize(unsigned szEvWin) { this->mEvWinSize = szEvWin; }
        ulong resolveWinOverlap();
        bool resolveEvWinSize(float medPxDisp, double imTs);
        void updateL1ChunkSize(unsigned int newSz);
        unsigned calcNewL1ChunkSize(float medPxDisp);

        bool isMcImageGood(const cv::Mat& mcImage);

        bool isEvGenRateGood(const double& evGenRate);
        int checkEvGenRate(const double& eventRate, const EvImBuilder::TrackState& state);

        void reset();
        void resetOptimVars();

        void makeFrame(const PoseImagePtr& pImage, unsigned fId,
                       const std::vector<cv::KeyPoint>& p1, EvFramePtr& frame) override;
        void makeFrame(const PoseImagePtr& pImage, unsigned int fId, unique_ptr<ELK_Tracker> &pLKTracker,
                       EvFramePtr &frame) override;
        unsigned int trackAndFrame(const PoseImagePtr& pImage, unsigned int fId, unique_ptr<ELK_Tracker> &pLKTracker,
                                   EvFramePtr &evFrame) override;

        void updateState(const std::vector<EventData>& vEvData, const EvFramePtr& pEvFrame);

        void updateLastPose(float imStdDP, float imStdBA, float imStdEH, bool inertial);

        int init(const cv::Mat& im, const double& ts);

        static bool resolveReconstStat(bool iniRes, const ORB_SLAM3::ReconstInfo& rInfo,
                                const std::vector<std::vector<cv::Point3f>>& pts3d, std::vector<int> vMatches12);

        int step(const cv::Mat& image, double ts, float &medPxDisp, unsigned &nMatches, bool refineRANSAC = true);

        /* ---------------------------------------------------------------------------------------------------------- */

        static void initPosesSpeed(std::vector<EvFramePtr>& vpEvFrames, const cv::Mat& speed);
        static void initPosesInertial(std::vector<EvFramePtr>& vpEvFrames);

        static void initPreRefInertial(EvFramePtr& pEvFrame);
        cv::Mat resolveIniDPoses(std::vector<EvFramePtr>& vpEvFrames, double t0, double dt, cv::Mat& Tcr, float& medDepth, bool inertial);
        int resolveIniPosesBA(std::vector<EvFramePtr>& vpEvFrames, std::vector<int>& vValidPts3D, bool inertial);

        int resolveLastDPose(double t0, double dt, cv::Mat& Tcr, float& medDepth, bool optimize, bool inertial);
        void getDPoseMCI(const std::vector<EventData>& evs, PoseImagePtr& pMCI, float& imSTD, bool optimize = false, bool inertial = false);

        int resolveLastPoseMap(cv::Mat& Tcr, float& medDepth, bool optimize, bool inertial);
        void getBAMCI(const std::vector<EventData>& evs, PoseImagePtr& pMCI, float& imSTD, bool optimize = true, bool inertial = false);

        void getEvHist(const std::vector<EventData>& evs, PoseImagePtr& pMCI, float& imSTD, unsigned long prefSize = 0);

        int resolveLastAtt2Params(cv::Mat& paramsSE2, bool optimize);
        void getAff2DMCI(const std::vector<EventData>& evs, PoseImagePtr& pMCI, float& imSTD, bool optimize = true);

        void generateMCImage(const std::vector<EventData>& evs, PoseImagePtr& pMCI, bool useL2 = false,
                             bool optimize = true, bool inertial = false);

        bool dispatchL2Image(const PoseImagePtr& pPoseImage);

        /* ---------------------------------------------------------------------------------------------------------- */

        static void refineTrackedPts(const std::vector<bool>& reconstStat, std::vector<int>& vMatches12,
                                    std::vector<int>& vCntMatches, unsigned& nMatches);

        // Utils

        //unsigned long getPointTracks(std::vector<cv::Mat>& ptTracks);

    private:
        // For multi-threaded operation,
        // we need thread-safe state vars & input/output buffers

        // States
        SharedState<TrackState> mStat, mLastStat;
        int mCntLowEvGenRate;
        SensorConfigPtr mpSensorConfig;

        EvFramePtr mpIniFrame;
        EvFramePtr mpCurrFrame;
        EvFramePtr mpLastFrame, mpLastFrameBA;

        //static unsigned long mFrameIdx;

        // Inputs
        SharedVector<EventData> mvSharedL2Evs;

        // Outputs
        std::mutex mMtxEvFrames;
        std::vector<EvFramePtr> mvpEvFrames;

        // For BA Reconstruction
        std::mutex mMtxStateBA;
        int mLastIdxBA; // Reconst. frame index, -1 : Bad Reconst.
        // This method also utilizes mvEvFrames & mvPts3D

        // For Sim3 Reconstruction
        PoseDepthPtr mpLastPoseDepth;
        PoseDepthPtr mpLastPoseDepthBA;

        InitInfoImuPtr mpImuInfo;
        InitInfoImuPtr mpImuInfoBA;

        double mIniChi2, mLastChi2;
        double mIniChi2BA, mLastChi2BA;

        // For Affine 2D Reconstruction
        std::mutex mMtxStateParams2D;
        cv::Mat mLastParams2D;

        // MC-Image Temp. holders
        PoseImagePtr mpPoseImageTemp;

        // Linked Trackers
        std::shared_ptr<EvAsynchTracker> mpL2Tracker;

        // Master Track Manager
        EvTrackManager* mpTrackManager;

        // Utils
        //cv::Ptr<cv::FastFeatureDetector> mpFastDetector;
        //std::shared_ptr<ORB_SLAM3::ORBextractor> mpORBDetector;
        std::unique_ptr<ELK_Tracker> mpLKTracker;
        //ORB_SLAM3::GeometricCamera* mpCamera;

        // Params
        const bool mbContTracking;

        std::mutex mMtxL1EvWinSize;
        unsigned long mL1EvWinSize;
        const unsigned long mInitL1EvWinSize;

        ulong mnWinOverlap;
        const bool mbFixedWinSize;
        const unsigned mL1NumLoop;

        const float mL1MaxPxDisp;
        const bool mTrackTinyFrames;
        //const bool isImageSensor;
    };

} //EORB_SLAM

#endif //ORB_SLAM3_EVIMBUILDER_H
