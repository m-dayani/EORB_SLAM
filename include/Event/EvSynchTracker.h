//
// Created by root on 8/10/21.
//

#ifndef ORB_SLAM3_EVSYNCHTRACKER_H
#define ORB_SLAM3_EVSYNCHTRACKER_H

#include "EvAsynchTracker.h"


namespace EORB_SLAM {

    class EvSynchTracker : virtual public EvAsynchTracker {
    public:

        EvSynchTracker(shared_ptr<EvParams> evParams, CamParamsPtr camParams, const MixedFtsParamsPtr& pFtParams,
                       std::shared_ptr<ORB_SLAM3::Atlas> pEvAtlas, ORB_SLAM3::ORBVocabulary* pVocab,
                       SensorConfigPtr  sConf, ORB_SLAM3::Viewer* pViewer);

        ~EvSynchTracker() override;

        void TrackSynch();

        virtual int setInitEvFrameSynch(const PoseImagePtr& pImage, ORB_SLAM3::FramePtr& pFrame);
        virtual int trackEvFrameSynch(const PoseImagePtr& pImage);

        virtual int trackAndOptEvFrameSynch(const PoseImagePtr& pImage, ORB_SLAM3::FramePtr& pFrame);

        virtual int trackTinyFrameSynch(const PoseImagePtr& pImage);

        virtual int trackEvKeyFrameSynch(const PoseImagePtr& pImage, ORB_SLAM3::KeyFrame* pKFcur);

        /*virtual bool setRefEvKeyFrameSynch(ORB_SLAM3::KeyFrame* pKFini);
        virtual bool getKFsForInitBASynch(ORB_SLAM3::KeyFrame*& pKFref, ORB_SLAM3::KeyFrame*& pKFcur);
        virtual int evImInitOptimizationSynch(const cv::Mat& currMCI, ORB_SLAM3::Map* pMap, ORB_SLAM3::KeyFrame* pKFini,
                                              ORB_SLAM3::KeyFrame* pKFcur, int nIter);*/

        virtual bool resolveEventMapInit(ORB_SLAM3::FramePtr& pFrIni, ORB_SLAM3::KeyFrame* pKFini,
                                         ORB_SLAM3::FramePtr& pFrCurr, ORB_SLAM3::KeyFrame* pKFcur, int nIteration) = 0;

        static void optimizeSBASynch(ORB_SLAM3::KeyFrame* pKFref, EvFrame& refFrame);

        virtual int evImReconst2ViewsSynch(const PoseImagePtr& pImage,
                const ORB_SLAM3::FramePtr& pFrame1, ORB_SLAM3::FramePtr& pFrame2, const std::vector<int> &vMatches12,
                cv::Mat &R21, cv::Mat &t21, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated);

    protected:

        /* ---------------------------------------------------------------------------------------------------------- */

        bool isTrackerInitializedSynch();
        bool isMapInitializedSynch();
        static bool checkConsistentTs(EvFrame& evFrame, ORB_SLAM3::KeyFrame* pKF, double thTs = 1e-6);

        void insertRefKeyFrameSynch(ORB_SLAM3::KeyFrame* pKFref);

        bool initMapSynch(EvFrame& refFrame, EvFrame& curFrame, ORB_SLAM3::KeyFrame* refKF,
                          ORB_SLAM3::KeyFrame* curKF, float medDepth, bool cleanMap); // = true);

        void assignKPtsLevelDesc(const EvFrame& refFrame, EvFrame& currFrame, const cv::Mat& currIm);

        static void mergeKeyPoints(const std::vector<cv::KeyPoint>& vkpt1, const std::vector<cv::KeyPoint>& vkpt2, std::vector<cv::KeyPoint>& vkpt);
        static uint mergeMatches12(const std::vector<int>& vmch1, const std::vector<int>& vmch2, const int maxIdx12, std::vector<int>& vmch);

    protected:
        //, mRefFrameSynch;
        //, mCurrFrameSynch;
        SharedFlag mbSynchRefMatched;
        int mnLastTrackedMPsSynch;

        std::mutex mMtxSynchTracker;
        //std::unique_ptr<ELK_Tracker> mpLKTrackerSynch;

    };

} // EORB_SLAM


#endif //ORB_SLAM3_EVSYNCHTRACKER_H
