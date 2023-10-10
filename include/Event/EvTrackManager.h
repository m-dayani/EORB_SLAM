//
// Created by root on 11/25/20.
//

#ifndef ORB_SLAM3_EVTRACKMANAGER_H
#define ORB_SLAM3_EVTRACKMANAGER_H

#include <string>
#include <vector>
#include <queue>
#include <mutex>
#include <memory>
#include <thread>
#include <chrono>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#ifndef CERES_FOUND
#define CERES_FOUND true
#endif

//#include <opencv2/opencv.hpp>
//#include <opencv2/core/core.hpp>
#include <opencv2/core.hpp>
#include <opencv2/sfm.hpp>

#include "EventData.h"
#include "MyDataTypes.h"
#include "KLT_Tracker.h"
#include "EventConversion.h"
#include "Visualization.h"
#include "EvImBuilder.h"
#include "EvSynchTrackerU.h"
#include "EvAsynchTrackerU.h"
#include "IMU_Manager.h"

//#include "Frame.h"
//#include "KeyFrame.h"
//#include "MapPoint.h"
//#include "G2oTypes.h"
//#include "MLPnPsolver.h"
#include "Atlas.h"
#include "GeometricCamera.h"
#include "ImuTypes.h"
#include "MapDrawer.h"

//#include <Eigen/Dense>



namespace ORB_SLAM3 {
    class System;
}

namespace EORB_SLAM
{

#ifndef DEF_MATCHES_WIN_NAME
#define DEF_MATCHES_WIN_NAME "Matched Pairs"
#endif

    class EvTrackManager {
    public:
        enum TrackState {
            IDLE,
            TRACKING,
            MERGE_MAPS,
            JAMMED,
            STOP
        };

        //explicit EvTrackManager(const std::string& fSettings);
        EvTrackManager(const EvParamsPtr& pEvParams, const CamParamsPtr& camParams, const MixedFtsParamsPtr& pORBParams,
                       const SensorConfigPtr& sConfig, ORB_SLAM3::ORBVocabulary* pVocab, ORB_SLAM3::Viewer* pViewer);
        EvTrackManager(const EvParamsPtr& pEvParams, const CamParamsPtr& camParams, const MixedFtsParamsPtr& pORBParams,
                       const SensorConfigPtr& sConfig, ORB_SLAM3::ORBVocabulary* pVocab, ORB_SLAM3::Viewer* pViewer,
                       const IMUParamsPtr& pIMUParams);
        //EvTrackManager(const EvParamsPtr& pEvParams, ORB_SLAM3::GeometricCamera* mCamera, bool sepThread);
        ~EvTrackManager();

        /* ---------------------------------------------------------------------------------------------------------- */

        void setSeparateThread(bool flg);
        void setOrbSlamSystem(ORB_SLAM3::System* pSystem) { this->mpSystem = pSystem; }
        void setMapDrawer(ORB_SLAM3::MapDrawer* pMapDrawer) { this->mpMapDrawer = pMapDrawer; }
        //bool isSeparateThread() { return (bool) sepThread; }

        bool isStopped();
        void stop();
        void reset();
        void shutdown();

        bool isReady();
        bool allInputsEmpty();

        /* ---------------------------------------------------------------------------------------------------------- */

        void fillBuffer(const std::vector<EventData>& evs);
        void emptyBuffer();

        void injectEventsBegin(const std::vector<EventData>& evs);

        void fillEvImBuffer(const std::vector<EventData>& evs);
        void consumeEvImBuffBegin(unsigned long chunkSize, std::vector<EventData>& evs);
        void getEvImBuffTillImTs(double imTs, unsigned long chunkSize, std::vector<EventData>& evs, bool eraseOldEvs = true);

        /* ---------------------------------------------------------------------------------------------------------- */

        bool getRecommendedEvWinSize(unsigned long& winSize);
        //void getEventsForOptimization(std::vector<EORB_SLAM::EventData>& vEvData, bool isInitialized);

        static float calcSceneMedianDepthForEvImOpt(ORB_SLAM3::Frame* pFrame);
        static float calcSceneMedianDepthForEvImOpt(const ORB_SLAM3::Frame* pFrame);

        bool reconstSynchEvMCI(PoseImagePtr& pPoseImage, bool dispIm = true);

        static void getEmptyPoseImage(PoseImagePtr& pImage, const double& ts);

        /* ---------------------------------------------------------------------------------------------------------- */

        void eventSynchPreProcessing(ORB_SLAM3::FramePtr& pFrame);
        void eventSynchPostProcessing();

        void setInitEvFrameSynch(ORB_SLAM3::FramePtr& pFrame);
        int trackAndOptEvFrameSynch(ORB_SLAM3::FramePtr& pFrame);
        void trackEvFrameSynch(ORB_SLAM3::FramePtr& pFrame);

        void eventImageInitOptimization(ORB_SLAM3::Map* pMap, int nIterations = 20);
        void eventImageInitOptimization(ORB_SLAM3::Map* pMap, ORB_SLAM3::KeyFrame* pKFini,
                ORB_SLAM3::KeyFrame* pKFcur, int nIterations = 20);
                
        void trackEvKeyFrameSynch(ORB_SLAM3::KeyFrame* pKFcur);

        bool resolveEventMapInit(ORB_SLAM3::FramePtr& pFrIni, ORB_SLAM3::KeyFrame* pKFini,
                                 ORB_SLAM3::FramePtr& pFrCurr, ORB_SLAM3::KeyFrame* pKFcur, int nIteration);

        bool evImReconst2ViewsSynch(const ORB_SLAM3::FramePtr& pFrame1,
                ORB_SLAM3::FramePtr& pFrame2, const std::vector<int> &vMatches12,
                cv::Mat &R21, cv::Mat &t21, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated);

        int evLocalMappingSynch(ORB_SLAM3::KeyFrame* pEvKFcur);

        void applyScaleAndRotationEvSynch(const cv::Mat &R, float scale, bool bScaleVel, const cv::Mat& t);
        void updateEvFrameImuSynch(float s, const ORB_SLAM3::IMU::Bias &b, ORB_SLAM3::KeyFrame *pOrbKFcur);

        /* ---------------------------------------------------------------------------------------------------------- */

        void Track();

        /* ---------------------------------------------------------------------------------------------------------- */

        unsigned getL1ChunkSize() { return (mpL1EvImBuilder) ? mpL1EvImBuilder->getL1ChunkSize() : DEF_L1_CHUNK_SIZE; }
        void getEventConstraints(std::vector<ORB_SLAM3::KeyFrame*>& vpEvKFs);

        //std::shared_ptr<SimpleImageDisplay> getImageDisplay() { return mpImDisplay; }

        AtlasPtr getEventsAtlas() { return mpEvAtlas; }

        void grabImuData(const ORB_SLAM3::IMU::Point& imuMea) {
            if (mpImuManager) {
                mpImuManager->grabImuData(imuMea);
            }
        }

        void setCurrentCamPose(const cv::Mat& Tcw);

        /* ---------------------------------------------------------------------------------------------------------- */

        static void fuseEventTracks(const std::vector<ORB_SLAM3::KeyFrame*>& vpEvKFs,
                                    std::vector<ORB_SLAM3::KeyFrame*>& vpEvKFsFused, float thTsSelf = 1e-3);

        void saveTrajectory(const std::string& fileName, double tsc=1.0, int ts_prec=9);
        void saveFrameTrajectory(const std::string& fileName, double tsc=1.0, int ts_prec=9);
        void savePoseFileHeader(const std::string& fileName);

        void printAverageTrackingTimes();

    protected:
        bool isInputGood();

        ulong consumeEventsBegin(ulong chunkSize, std::vector<EventData>& evs);
        ulong consumeEventsBegin(ulong chunkSize, ulong overlap, std::vector<EventData>& evs);
        ulong retrieveEventsBegin(ulong chunkSize, std::vector<EventData>& evs);

    private:
        // States
        SharedFlag sepThread;
        SensorConfigPtr mpSensor;
        SharedState<TrackState> mStat;
        //SharedFlag mbSynchEvImGood;

        PoseImagePtr mpLastSynchEvIm;

        // Buffers
        std::mutex mMtxEvBuff;
        std::list<EventData> mEventBuffer;
        // This is for synch. event-image operation
        std::mutex mMtxEvImQ;
        std::list<EventData> mEvImBuffer;

        // Trackers
        std::unique_ptr<EvImBuilder> mpL1EvImBuilder;
        std::unique_ptr<std::thread> mpL1Thread;
        std::shared_ptr<EvAsynchTracker> mpL2AsynchTracker;
        std::shared_ptr<EvSynchTrackerU> mpL2SynchTracker;
        std::shared_ptr<EvAsynchTrackerU> mpL2AsynchTrackerU;
        std::unique_ptr<std::thread> mpL2Thread;

        //Utils
        //ORB_SLAM3::GeometricCamera* mpCamera;
        //ORB_SLAM3::ORBVocabulary* mpORBVocab;

        std::shared_ptr<ORB_SLAM3::Atlas> mpEvAtlas;

        // Params
        CamParamsPtr mpCamParams;
        //MixedFtsParamsPtr mpORBParams;
        EvParamsPtr mpEvParams;

        ulong mL1WinSize;
        //ulong mL1WinOverlap;
        const float mL2EvImSTD;
        const unsigned mL1NumLoop;
        const bool mbFixedWinSz;
        const bool mbContTracking;

        cv::Size mImSize;

        const std::string mMchWindow;

        // Pointer to System (for global KF insersion)
        ORB_SLAM3::System* mpSystem;
        ORB_SLAM3::Viewer* mpViewer;
        ORB_SLAM3::MapDrawer* mpMapDrawer;

        std::shared_ptr<IMU_Manager> mpImuManager;

        //std::vector<float> mvL1TrackingTimes;
        MySmartTimer mL1TrackingTimer;
    };

}// namespace EORB_SLAM

#endif //ORB_SLAM3_EVTRACKMANAGER_H
