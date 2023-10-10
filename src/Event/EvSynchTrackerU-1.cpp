//
// Created by root on 1/9/22.
//

#include "EvSynchTrackerU.h"

#include <utility>
#include "EvAsynchTrackerU.h"
#include "EvLocalMapping.h"
#include "EvTrackManager.h"


namespace EORB_SLAM {


    EvSynchTrackerU::EvSynchTrackerU(const EvParamsPtr& evParams, const CamParamsPtr& camParams, const MixedFtsParamsPtr &pFtParams,
                                     const std::shared_ptr<ORB_SLAM3::Atlas>& pEvAtlas, ORB_SLAM3::ORBVocabulary *pVocab,
                                     const SensorConfigPtr& sConf, ORB_SLAM3::Viewer *pViewer) :

                                     EvAsynchTracker(evParams, camParams, pFtParams, pEvAtlas, pVocab, sConf, pViewer),
                                     EvSynchTracker(evParams, camParams, pFtParams, pEvAtlas, pVocab, sConf, pViewer),
                                     EvAsynchTrackerU(evParams, camParams, pFtParams, pEvAtlas, pVocab, sConf, pViewer)
    {
        //if (mpLocalMapper)
        //    mpLocalMapper->enableLocalMapping(false);

        mTrackingTimer.setName("L2 Continuous Synch Tracker");
    }

    void EvSynchTrackerU::preProcessing(const PoseImagePtr& pPoseImage) {

        mbSetLastPose = false;

        // Track last features
        this->trackLastFeatures(pPoseImage);

        if (pPoseImage->mReconstStat != 0) {

            // Let the IMU init. be completed
            if (mpSensor->isInertial() && mpImuManager) {
                while (mpImuManager->isInitializing()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }

            std::unique_lock<mutex> lock1(mMtxUpdateState);

            // If not enough tracked map points, create and inject some
            // This also resets the map if non-inertial mode
            this->checkTrackedMapPoints();

            // Is there enough feature tracks?? -> Init. if not
            // these are lock(mMtxUpdateState) safe
            if (!this->enoughFeatureTracks()) {
                this->detectAndFuseNewFeatures(pPoseImage);
            }

            // First reconstruct current frame
            this->createCurrFrame(pPoseImage, mFrameIdx++, mpCurrFrame);

            // Add IMU measurements
            this->preintegrateCurrFrame(mpCurrFrame);
        }
    }

    void EvSynchTrackerU::postProcessing(const PoseImagePtr& pPoseImage) {

        if (pPoseImage->mReconstStat != 0) {

            // Check if tracker is initialized
            if (!this->isTrackerInitialized() && !mmFeatureTracks.empty()) {
                mbTrackerInitialized.set(true);
            }

            // Visualizing frames
            this->updateFrameDrawer();
        }

        if (mbSetLastPose && mpCurrFrame) {

            mpPoseInfo->updateLastPose(mpCurrFrame->getPrevFrame(), mpCurrFrame);

            mpTrackManager->setCurrentCamPose(mpCurrFrame->mTcw);

            mFrameInfo.pushState(mpCurrFrame.get(), false);
        }

        this->popImage();
    }

    int EvSynchTrackerU::setInitEvFrameSynch(const PoseImagePtr &pImage, ORB_SLAM3::Frame* pFrame) {

        this->preProcessing(pImage);

        mpIniFrame = mpCurrFrame;
        pFrame->mpEvFrame = mpIniFrame.get();

        this->postProcessing(pImage);

        return mpIniFrame->numAllKPts();
    }

    int EvSynchTrackerU::trackAndOptEvFrameSynch(const cv::Mat &mci, ORB_SLAM3::Frame *pFrame, EvFrame &currEvFrame) {
        return EvSynchTracker::trackAndOptEvFrameSynch(mci, pFrame, currEvFrame);
    }

    int EvSynchTrackerU::trackEvFrameSynch(const cv::Mat &mci, double ts) {
        return EvSynchTracker::trackEvFrameSynch(mci, ts);
    }

    int EvSynchTrackerU::trackTinyFrameSynch(const cv::Mat &tinyIm, double ts) {
        return EvSynchTracker::trackTinyFrameSynch(tinyIm, ts);
    }

    bool EvSynchTrackerU::setRefEvKeyFrameSynch(ORB_SLAM3::KeyFrame *pKFini) {
        return EvSynchTracker::setRefEvKeyFrameSynch(pKFini);
    }

    int EvSynchTrackerU::trackEvKeyFrameSynch(const cv::Mat &mci, ORB_SLAM3::KeyFrame *pKFcur) {
        return EvSynchTracker::trackEvKeyFrameSynch(mci, pKFcur);
    }

} // EORB_SLAM

