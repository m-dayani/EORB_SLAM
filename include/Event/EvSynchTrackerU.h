//
// Created by root on 1/9/22.
//

#ifndef ORB_SLAM3_EVSYNCHTRACKERU_H
#define ORB_SLAM3_EVSYNCHTRACKERU_H

#include "EvSynchTracker.h"
#include "EvAsynchTrackerU.h"



namespace EORB_SLAM {

    class EvSynchTrackerU : public EvSynchTracker, public EvAsynchTrackerU {
    public:
        EvSynchTrackerU(const EvParamsPtr& evParams, const CamParamsPtr& camParams, const MixedFtsParamsPtr& pFtParams,
                        const std::shared_ptr<ORB_SLAM3::Atlas>& pEvAtlas, ORB_SLAM3::ORBVocabulary* pVocab,
                        const SensorConfigPtr&  sConf, ORB_SLAM3::Viewer* pViewer);

        void preProcessing(const PoseImagePtr& pImage);
        void postProcessing(const PoseImagePtr& pImage);

        int setInitEvFrameSynch(ORB_SLAM3::FramePtr& pFrame);
        int trackAndOptEvFrameSynch(ORB_SLAM3::FramePtr& pFrame);
        int trackEvFrameSynch();
        int trackTinyFrameSynch(const PoseImagePtr& pImage) override;

        //bool setRefEvKeyFrameSynch(ORB_SLAM3::KeyFrame* pKFini) override;
        int trackEvKeyFrameSynch(ORB_SLAM3::KeyFrame* pKFcur);

        bool resolveEventMapInit(ORB_SLAM3::FramePtr& pFrIni, ORB_SLAM3::KeyFrame* pKFini,
                                 ORB_SLAM3::FramePtr& pFrCurr, ORB_SLAM3::KeyFrame* pKFcur, int nIteration) override;

        int evImReconst2ViewsSynch(const ORB_SLAM3::FramePtr& pFrame1, ORB_SLAM3::FramePtr& pFrame2,
                                   const std::vector<int> &vMatches12, cv::Mat &R21, cv::Mat &t21,
                                   std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated);

        int evLocalMappingSynch(ORB_SLAM3::KeyFrame* pEvKFcur);

    protected:

        void preInitMapSynch(ORB_SLAM3::KeyFrame* pKFiniORB, ORB_SLAM3::KeyFrame* pKFcurORB);

        bool postInitMapSynch();

        void localMappingSynch(EvFramePtr& pRefFrame, ORB_SLAM3::KeyFrame* pKFini, EvFramePtr& pCurrFrame, ORB_SLAM3::KeyFrame* pKFcur);

        void resetTracker() override;
        
        //void preProcessing(const PoseImagePtr& pImage);
        //void postProcessing(const PoseImagePtr& pImage);

    private:


    };
} // EORB_SLAM


#endif //ORB_SLAM3_EVSYNCHTRACKERU_H
