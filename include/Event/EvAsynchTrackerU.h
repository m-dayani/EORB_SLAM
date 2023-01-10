//
// Created by root on 11/15/21.
//

#ifndef ORB_SLAM3_EVASYNCHTRACKERU_H
#define ORB_SLAM3_EVASYNCHTRACKERU_H

#include "EvAsynchTracker.h"

#include "FeatureTrack.h"


namespace EORB_SLAM {

#define DEF_GRID_PATCH_SIZE 32
#define DEF_TH_KF_TEMP_CONST 0.4

    class EvAsynchTrackerU : virtual public EvAsynchTracker {
    public:
        EvAsynchTrackerU(EvParamsPtr evParams, CamParamsPtr camParams, const MixedFtsParamsPtr& pFtParams,
                         std::shared_ptr<ORB_SLAM3::Atlas> pEvAtlas, ORB_SLAM3::ORBVocabulary* pVocab,
                         SensorConfigPtr  sConf, ORB_SLAM3::Viewer* pViewer);

        EvFramePtr getFrameById(ulong frameId);

        void resetAll() override;
        virtual void resetTracker();
        void resetState(bool wipeMem = true);
        //void resetActiveMap();

        bool isTracking() override;

        void updateFrameIMU(float s, const ORB_SLAM3::IMU::Bias &b, ORB_SLAM3::KeyFrame* pCurrentKeyFrame) override;

        void saveAllFramePoses(const std::string& fileName, double tsc, int ts_prec) override;
        void saveAllPoses(const std::string& fileName, double tsc, int ts_prec) override;

        //float getAverageTrackingTime() override;

    protected:

        bool isDisconnGraph();
        bool isTrackerInitialized();
        bool enoughFeatureTracks() const;

        /* ---------------------------------------------------------------------------------------------------------- */

        unsigned getTrackedPoints(ulong frame, std::vector<cv::KeyPoint>& vKPts, std::vector<ORB_SLAM3::MapPoint*>& vpMPts,
                                  std::vector<FeatureTrackPtr>& vFtTracks);
        static void updateTrackedPoints(ulong frame, const std::vector<cv::KeyPoint>& vKPts, const std::vector<int>& vMatches12,
                                        std::vector<FeatureTrackPtr>& vFtTracks);
        static void assignNewMPtsToTracks(ulong frId2, const std::vector<ORB_SLAM3::MapPoint*>& vpMPts2,
                const vector<int>& vMatches12, std::vector<FeatureTrackPtr>& vpFtTracks2, bool resetOldMPts = false);

        void matchCurrentFrame(EvFramePtr& pCurFrame);

        void updateTrackStats();

        void preintegrateCurrFrame(EvFramePtr& pCurrFrame);

        void createCurrFrame(const PoseImagePtr& pImage, ulong fId, EvFramePtr& frame);

        void changeReference(ulong newRefId);
        void removeOldFrames(ulong newRefId);

        void findTracksBounds(ulong& minFrId, ulong& maxFrId);

        bool needNewKeyFrame(float medPxDisp);

        void addNewKeyFrame(EvFramePtr& pCurrFrame);

        void triangulateNewMapPoints(const EvFramePtr& pRefFrame, const EvFramePtr& pCurrFrame);
        void triangulateNewMapPoints();

        //void insertCurrKF(ORB_SLAM3::KeyFrame* pKFcur, ORB_SLAM3::KeyFrame* pKFref) override;
        //ORB_SLAM3::KeyFrame * insertCurrKF(EvFramePtr& currFrame, ORB_SLAM3::KeyFrame* pKFref) override;

        bool estimateCurrMapIniPose(cv::Mat &Rb0w, cv::Mat &tb0w);
        void fuseInertialMaps();

        /* ---------------------------------------------------------------------------------------------------------- */

        static float calcMedPxDisplacement(const std::vector<cv::KeyPoint>& vp0, const std::vector<cv::KeyPoint>& vp1,
                                           std::vector<int>& vMatches12, std::vector<float>& vMedPxDisp);

        void updateFrameDrawer() override;

        /* ---------------------------------------------------------------------------------------------------------- */

        void trackLastFeatures(const PoseImagePtr& pImage);

        void estimateCurrentPose(const PoseImagePtr& pImage);

        void detectAndFuseNewFeatures(const PoseImagePtr& pImage);

        void detectAndFuseNewMapPoints();

        void checkTrackedMapPoints();

        void reconstIniMap(const PoseImagePtr& pImage);

        virtual void localMapping();

        /* ---------------------------------------------------------------------------------------------------------- */

        void Track() override;

    protected:
        //PoseImagePtr mpLastImage;

        ulong mnFirstFrameId;

        ulong mFtTrackIdx;
        std::map<ulong, FeatureTrackPtr> mmFeatureTracks;

        // or maybe some vars for tracking validity of tracks
        unsigned mnFeatureTracks; // num. initial valid tracks
        unsigned mnCurrFeatureTracks; // num. curr valid tracks
        unsigned mnLastTrackedKPts; // num. last tracked key points
        unsigned mnLastTrackedMPts; // num. last tracked map points

        bool mbSetLastPose;
        // Also uses mbTrackerInitialized, mbMapInitialized, mbImuInitialized extensively

        std::vector<int> mvLastMatches;
        std::vector<float> mvLastFtPxDisp;

        std::map<ulong, EvFramePtr> mmpWinFrames;

        std::list<ORB_SLAM3::MapPoint*> mlpRecentAddedMapPoints;

        FrameInfo mFrameInfo;
    };

} // EORB_SLAM


#endif //ORB_SLAM3_EVASYNCHTRACKERU_H
