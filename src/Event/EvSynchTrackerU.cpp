//
// Created by root on 1/9/22.
//

#include "EvSynchTrackerU.h"

#include <utility>
#include "EvAsynchTrackerU.h"
#include "EvLocalMapping.h"
#include "EvTrackManager.h"
#include "System.h"

#include "EvOptimizer.h"

using namespace ORB_SLAM3;

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

    /* -------------------------------------------------------------------------------------------------------------- */

    void EvSynchTrackerU::resetTracker() {

        //this->prepReInitDisconnGraph();

        //mbTrackerInitialized = false; // -> these 3 also done in parent class
        this->resetInitMapState();
        mbImuInitialized = false;
        mbSynchRefMatched = false;

        //mmFeatureTracks.clear();
        //mmpWinFrames.clear();
        mlpRecentAddedMapPoints.clear();

        //mnFeatureTracks = mmFeatureTracks.size();
        //mnCurrFeatureTracks = mnFeatureTracks;
        //mnLastTrackedMPts = 0;
        //mnLastTrackedKPts = 0;

        mvLastMatches.clear();
        mvLastFtPxDisp.clear();
    }

    void EvSynchTrackerU::preProcessing(const PoseImagePtr& pPoseImage) {

        mbSetLastPose = false;

        // Track last features
        this->trackLastFeatures(pPoseImage);

        if (pPoseImage->mReconstStat != 0) {

            // Let the IMU init. be completed
//            if (mpSensor->isInertial() && mpImuManager) {
//                while (mpImuManager->isInitializing()) {
//                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
//                }
//            }

            //std::unique_lock<mutex> lock1(mMtxUpdateState);

            // If not enough tracked map points, create and inject some
            // This also resets the map if non-inertial mode
            // In Synch. case tracker reset is not independent and controlled by ORB-SLAM
            //this->checkTrackedMapPoints();

            // Is there enough feature tracks?? -> Init. if not
            // these are lock(mMtxUpdateState) safe
            if (!this->enoughFeatureTracks()) {
                VLOG(30) << "EvSynchTrackerU::preProcessing: Not enough ftTracks: "
                            << mnCurrFeatureTracks << ", nInitFtTracks: " << mnFeatureTracks << endl;
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
                VLOG(20) << "EvSynchTrackerU::postProcessing: Set tracker initialized for FrameId: " << mpCurrFrame->mnId << endl;
                mbTrackerInitialized.set(true);
            }

            // Visualizing frames
            this->updateFrameDrawer();
        }

        if (mbSetLastPose && mpCurrFrame) {

            VLOG_EVERY_N(50, 100) << "EvSynchTrackerU::postProcessing: Set last pose for FrameId: " << mpCurrFrame->mnId << endl;

            mpPoseInfo->updateLastPose(mpCurrFrame->getPrevFrame(), mpCurrFrame);

            //mpTrackManager->setCurrentCamPose(mpCurrFrame->mTcw);

            //mFrameInfo.pushState(mpCurrFrame.get(), false);
        }

        //this->popImage();
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    int EvSynchTrackerU::setInitEvFrameSynch(ORB_SLAM3::FramePtr& pFrame) {

        // When we come here, 3 diff. states are possible: No frames, currFrame OK, reInit.
        // Two outcomes: Setting init. frame is possible or not

        // Always reset tracker here
        this->resetTracker();

        if (mpCurrFrame && mnFeatureTracks > DEF_TH_MIN_KPTS) {

            // change reference frame
            mpIniFrame = mpCurrFrame;
            pFrame->mpEvFrame = mpIniFrame;
            mpIniFrame->mpEvFrame = pFrame;

            // set matched ref.
            mbSynchRefMatched = true;

            VLOG_EVERY_N(50, 100) << "setInitEvFrameSynch: New ref. frame set: EvFrameId: "
                << mpIniFrame->mnId << ", OrbFrameId: " << pFrame->mnId << ", nPts: " << mpIniFrame->numAllKPts() << endl;

            return mpIniFrame->numAllKPts();
        }
        else {
            VLOG_EVERY_N(50, 100) << "setInitEvFrameSynch: Unable to set new event ref. nFtTracks: " << mnFeatureTracks << endl;
            mbSynchRefMatched = false;
        }
        return 0;
    }

    int EvSynchTrackerU::trackEvFrameSynch() {

        // nothing to do!

    }

    int EvSynchTrackerU::trackTinyFrameSynch(const PoseImagePtr& pImage) {

        this->preProcessing(pImage);

        // nothing to do!
        VLOG_EVERY_N(50, 100) << "trackTinyFrameSynch: Called with image: " << pImage->printStr();

        this->postProcessing(pImage);
    }

    int EvSynchTrackerU::evImReconst2ViewsSynch(const ORB_SLAM3::FramePtr& pFrame1,
            ORB_SLAM3::FramePtr& pFrame2, const std::vector<int> &vMatches12, cv::Mat &R21, cv::Mat &t21,
            std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated) {

        // Possible states when using this methods: matched ref. or not, and the availability of current state
        // Outcomes: bad states (-1), bad reconst. (0), tightly-coupled init. (1)
        int stat = -1;

        // currFrame must be a least the second frame, so the tracker must be init.
        if (!this->isTrackerInitialized() || !mpCurrFrame || !mbSynchRefMatched) {
            VLOG_EVERY_N(3, 10) << "EvAsynchTracker::evImReconst2ViewsSynch: Bad init. conditions: isTrackerInit? "
                << isTrackerInitialized() << ", refMatched? " << (mbSynchRefMatched == true)
                << ", currFr: " << mpCurrFrame << endl;
            return stat;
        }

        // if tightly-coupled init. not possible, return
        ORB_SLAM3::FramePtr pRefFr1 = pFrame1->mpEvFrame.lock();
        EvFramePtr pEvFrame1 = dynamic_pointer_cast<EvFrame>(pRefFr1);

        if (!pEvFrame1) {
            VLOG(4) << "EvAsynchTracker::evImReconst2ViewsSynch: Abort because of inconsistent frames\n";
            mbSynchRefMatched = false;
            return stat;
        }

        stat = 0;
        pFrame2->mpEvFrame = mpCurrFrame;
        //mpIniFrame = pEvFrame1;

        // match
        // assuming not mapInit. state, this shall do the right thing
        this->matchCurrentFrame(mpCurrFrame);

        vector<int> vMatches12ev = mpCurrFrame->getMatches();

        // merge
        vector<cv::KeyPoint> vAllKpts1, vAllKpts2;
        vector<int> vAllMatches12;
        mergeKeyPoints(pFrame1->getAllUndistKPtsMono(), mpIniFrame->getAllUndistKPtsMono(), vAllKpts1);
        mergeKeyPoints(pFrame2->getAllUndistKPtsMono(), mpCurrFrame->getAllUndistKPtsMono(), vAllKpts2);
        mergeMatches12(pFrame2->getMatches(), vMatches12ev, pFrame2->numAllMPsMono(), vAllMatches12);

        // reconst.
        bool res = mpCamera->ReconstructWithTwoViews(vAllKpts1, vAllKpts2, vAllMatches12,
                                                     R21, t21, vP3D, vbTriangulated);

        if (!res) {
            VLOG(30) << "EvAsynchTracker::evImReconst2ViewsSynch: " << pFrame2->mTimeStamp
                          << ", Cannot reconst. because of original bad response\n";
            return stat;
        }

        stat = 1;

        // recover data
        // Recover current pose
        cv::Mat currTcw = ORB_SLAM3::Converter::toCvSE3(R21, t21);
        mpCurrFrame->SetPose(currTcw);
        mpIniFrame->SetPose(cv::Mat::eye(4,4,CV_32FC1));

        // Recover current 3d points
        size_t nOrbPts1 = pFrame1->numAllKPts();
        mvPts3D.resize(mpIniFrame->numAllKPts());
        for (size_t i = 0; i < mvPts3D.size(); i++) {
            mvPts3D[i] = vP3D[i+nOrbPts1];
        }
        vP3D.resize(nOrbPts1);

        // Recover reconstruction matches
        for (size_t i = 0; i < vMatches12ev.size(); i++) {
            if (vMatches12ev[i] >= 0 && !vbTriangulated[i+nOrbPts1]) {
                vMatches12ev[i] = -1;
            }
        }
        mpCurrFrame->setMatches(vMatches12ev);
        vbTriangulated.resize(nOrbPts1);

        VLOG(30) << "EvAsynchTracker::evImReconst2ViewsSynch: " << pFrame2->mTimeStamp
                   << ", Successful reconstruction: evRefTs: " << mpIniFrame->mTimeStamp
                   << ", orbRefTs: " << pFrame1->mTimeStamp << endl;

        return stat;
    }

    bool EvSynchTrackerU::resolveEventMapInit(ORB_SLAM3::FramePtr &pFrIni, ORB_SLAM3::KeyFrame *pKFini,
                                              ORB_SLAM3::FramePtr &pFrCurr, ORB_SLAM3::KeyFrame *pKFcur, const int nIteration) {

        // Possible input states: there is a tightly-coupled init. or not
        // Outcomes: init. map and tc-gba, set event ref. with last orb info, return (event tracking not ready)

        // If there's a tightly-coupled map init. both frames must provide ref. event frame links
        FramePtr pFrame1 = pFrIni->mpEvFrame.lock();
        FramePtr pFrame2 = pFrCurr->mpEvFrame.lock();

        const bool matchedFrames = pFrame1 && pFrame2 && mpIniFrame && mpCurrFrame &&
                                   pFrame1->mnId == mpIniFrame->mnId && pFrame2->mnId == mpCurrFrame->mnId;
        bool mapInitialized = false;

        if (matchedFrames) {

            VLOG(30) << "EvSynchTrackerU::resolveEventMapInit: Matched refs. available\n";

            const bool tcInit = !mpIniFrame->mTcw.empty() && !mpCurrFrame->mTcw.empty() && !mvPts3D.empty();

            if (tcInit) {
                VLOG(30) << "EvSynchTrackerU::resolveEventMapInit: Tightly-coupled reconst. available\n";

                // Init. Event Map
                this->preInitMapSynch(pKFini, pKFcur);

                EvOptimizer::GlobalBundleAdjustemnt(pKFcur->GetMap(), nIteration);

                // Post Init. Map Processing
                mapInitialized = this->postInitMapSynch();

                if (mapInitialized) {
                    VLOG(30) << "EvSynchTrackerU::resolveEventMapInit: Successfully initialized event map\n";
                    return true;
                }
            }
        }
        if (!mapInitialized && (mpCurrFrame && pFrame2 && pFrame2->mnId == mpCurrFrame->mnId)) {

            // Reset frames and continue
            mpCurrFrame->SetPose(pKFcur->GetPose());
            mpIniFrame = mpCurrFrame;

            // Insert a ref. key frame
            // This sets both reference KF and last KF (same KF)
            KeyFrame* pRefEvKF = this->insertRefKF(mpIniFrame);
            pRefEvKF->mpSynchOrbKF = pKFcur;
            pKFcur->mpSynchEvKF = pRefEvKF;

            mbSynchRefMatched = true;

            VLOG_EVERY_N(3, 100) << "EvSynchTrackerU::resolveEventMapInit: Reset ref. frame and key frame\n";
        }
        return false;
    }

    void EvSynchTrackerU::preInitMapSynch(ORB_SLAM3::KeyFrame* pKFiniORB, ORB_SLAM3::KeyFrame* pKFcurORB) {

        vector<int> vIniMatches = mpCurrFrame->getMatches();

        if (mpIniFrame->mTcw.empty()) {
            mpIniFrame->SetPose(pKFiniORB->GetPose());
        }
        if (mpCurrFrame->mTcw.empty()) {
            mpCurrFrame->SetPose(pKFcurORB->GetPose());
        }

        // Create and insert KeyFrames
        KeyFrame* pKFini = this->insertRefKF(mpIniFrame);
        KeyFrame* pKFcur = this->insertCurrKF(mpCurrFrame, pKFini);

        // Add new map points
        uint nPts = EvLocalMapping::addNewMapPoints(mpEvAtlas, pKFini, pKFcur, vIniMatches, mvPts3D);

        //mpIniFrame->setAllMapPointsMono(pKFini->GetMapPointMatches());
        //mpCurrFrame->setAllMapPointsMono(pKFcur->GetMapPointMatches());

        // Update Connections
        pKFini->UpdateConnections();
        pKFcur->UpdateConnections();

        VLOG(50) << "EvSynchTrackerU::preInitMapSynch: New Map created with " +
                      to_string(mpEvAtlas->MapPointsInMap()) + " points\n";
        VLOG(50) << pKFcur->printPointDistribution();

        // Set up references & links
        pKFiniORB->mpSynchEvKF = pKFini;
        pKFcurORB->mpSynchEvKF = pKFcur;

        pKFini->mpSynchOrbKF = pKFiniORB;
        pKFcur->mpSynchOrbKF = pKFcurORB;

        // reference KF always points to the last KF
        //mpReferenceKF = mpLastKF;
    }

    bool EvSynchTrackerU::postInitMapSynch() {

        vector<int> vIniMatches = mpCurrFrame->getMatches();
        KeyFrame* pLastKF = mpLastKF;
        IMU::Preintegrated* pPreIntFromLastKF = getPreintegratedFromLastKF(mpIniFrame);

        KeyFrame* pKFini = mpReferenceKF;
        KeyFrame* pKFcur = mpLastKF;

        assert(pKFini && pKFcur && pKFini != pKFcur);

        float medianDepth = pKFini->ComputeSceneMedianDepth(2);

        if(medianDepth<0 || pKFcur->TrackedMapPoints(1) < DEF_INIT_MIN_TRACKED_PTS3D) {

            LOG(WARNING) << "EvSynchTrackerU::postInitMapSynch: Wrong initialization, reseting...\n";
            return false;
        }

        // Scale
        vector<KeyFrame*> vpKFs = {pKFini, pKFcur};
        set<MapPoint*> spCurrMPs = pKFini->GetMapPoints();
        vector<MapPoint*> vpAllMapPoints(spCurrMPs.begin(), spCurrMPs.end());

        EvLocalMapping::scalePoseAndMap(vpKFs, vpAllMapPoints, medianDepth, mpSensor->isInertial());

        if (mpSensor->isInertial()) {

            pKFcur->mPrevKF = pKFini;
            pKFini->mNextKF = pKFcur;

            if (!pLastKF) {
                pKFini->mpImuPreintegrated = (IMU::Preintegrated *) (nullptr);
            }
            else {
                // Establish inertial connections with last map
                pKFini->mpImuPreintegrated = pPreIntFromLastKF;
                pKFini->mPrevKF = pLastKF;
                pLastKF->mNextKF = pKFini;
            }
            pKFcur->mpImuPreintegrated = mpImuManager->getImuPreintegratedFromLastKF(mnImuID);

            mpImuManager->refreshPreintFromLastKF(mnImuID);
        }

        mpCurrFrame->SetPose(pKFcur->GetPose());
        // We don't have the pose of the real prev. frames
        mpCurrFrame->setPrevFrame(mpIniFrame);
        //mpIniFrame->setPrevFrame(nullptr);
        mpIniFrame->setRefFrame(nullptr);

        this->updateRefMapPoints(vpAllMapPoints);

        //saveActivePose();
        //saveActiveMap();

        vector<MapPoint *> vMPts = pKFcur->GetMapPointMatches();
        this->assignNewMPtsToTracks(mpCurrFrame->mnId, vMPts, vIniMatches, mpCurrFrame->getAllFeatureTracks());

        for (auto pMpt : vMPts) {
            if (pMpt)
                mlpRecentAddedMapPoints.push_back(pMpt);
        }

        mnFirstFrameId = pKFini->mnFrameId;
        //mpReferenceKF = mpLastKF;
        mnLastTrackedMPts = mlpRecentAddedMapPoints.size();
        mbSetLastPose = true;

        mbMapInitialized.set(true);

        VLOG(50) << "EvSynchTrackerU::postInitMapSynch: New Init. with iniKF: " << pKFini->mnId << ", curKF: "
                << pKFcur->mnId << ", medDepth: " << pKFcur->ComputeSceneMedianDepth(2)
                << ", nMPts: " << mnLastTrackedMPts << endl;

        return true;
    }


    int EvSynchTrackerU::trackAndOptEvFrameSynch(ORB_SLAM3::FramePtr& pFrame) {

        // Possible input states: map init., other (ref. matched, tracker init., no frames yet, ...)
        int stat = -1;

        // Assign map points and required info. to the current event frame
        this->matchCurrentFrame(mpCurrFrame);

        // if there is a map, track it and optimize
        if (this->isMapInitialized()) {

            // Setup links
            pFrame->mpEvFrame = mpCurrFrame;

            mpCurrFrame->SetPose(pFrame->mTcw);
            mbSetLastPose = true;

            stat = 1;
        }
        return stat;
    }

    // TODO: Remember to update last pose info
    int EvSynchTrackerU::trackEvKeyFrameSynch(ORB_SLAM3::KeyFrame *pKFcur) {

        // Possible input states: map init., ref. matched (init. map now),
        //      tracker init. or first evFrame (match refs here), no evFrames
        // Outcomes: localMapping with the 3rd KF, init. map with ORB poses, match refs, nothing (return)

        int stat = -1;
        if (!mpCurrFrame) {
            return stat;
        }

        // if tracker not init., set ref. info. and return
        if (this->isMapInitialized() || mbSynchRefMatched) {

            // Resolve ref. event info
            KeyFrame* pRefEvKF = mpReferenceKF;
            if (this->isMapInitialized()) {
                pRefEvKF = mpLastKF;
            }
            EvFramePtr pRefEvFr = mmpWinFrames[pRefEvKF->mnFrameId];

            // This is not needed, instead call trackAndOpt regularly, doing it doesn't heart anyway
            this->matchCurrentFrame(mpCurrFrame);

            // do the local mapping to create new KF and Map points
            // map can be init. here (with local mapping)

            // first insert current KeyFrame
            mpCurrFrame->SetPose(pKFcur->GetPose());
            KeyFrame* pEvKFcur = this->insertCurrKF(mpCurrFrame, pRefEvKF);

            // set the links
            pEvKFcur->mpSynchOrbKF = pKFcur;
            pKFcur->mpSynchEvKF = pEvKFcur;

            if (!pEvKFcur->mPrevKF || pEvKFcur->mPrevKF != pRefEvKF) {
                LOG(ERROR) << "trackEvKeyFrameSynch: Bad initialization key frames\n";
            }

            stat = this->evLocalMappingSynch(pEvKFcur);
            if (stat > 0)
                return stat;
        }

        if (stat <= 0 && mnFeatureTracks > DEF_TH_MIN_KPTS) {

            // if we can set a new reference key frame, do it
            mpIniFrame = mpCurrFrame;

            mpIniFrame->SetPose(pKFcur->GetPose());

            KeyFrame* pKFiniEv = this->insertRefKF(mpIniFrame);
            // pKFiniEv must be equal to mpReferenceKF
            //assert(pKFiniEv == mpReferenceKF);
            //mpIniFrame->mpReferenceKF = mpReferenceKF;

            pKFiniEv->mpSynchOrbKF = pKFcur;
            pKFcur->mpSynchEvKF = pKFiniEv;

            mbSynchRefMatched = true;

            stat = 0;
        }

        return stat;
    }

    int EvSynchTrackerU::evLocalMappingSynch(ORB_SLAM3::KeyFrame* pEvKFcur) {

        int stat = -1;
        if (!pEvKFcur) {
            return stat;
        }

        // Resolve KeyFrames and Frames
        KeyFrame *pEvKFref = pEvKFcur->mPrevKF, *pKFcurORB = pEvKFcur->mpSynchOrbKF;
        if (!pEvKFref || !pKFcurORB) {
            return stat;
        }

        if (mmpWinFrames.find(pEvKFref->mnFrameId) == mmpWinFrames.end() ||
            mmpWinFrames.find(pEvKFcur->mnFrameId) == mmpWinFrames.end()) {
            return stat;
        }

        EvFramePtr pRefEvFr = mmpWinFrames[pEvKFref->mnFrameId];
        EvFramePtr pCurEvFr = mmpWinFrames[pEvKFcur->mnFrameId];

        // Choose the right local mapping based on the state
        if (this->isMapInitialized()) {

            this->localMappingSynch(pRefEvFr, pEvKFref, pCurEvFr, pEvKFcur);

            stat = 1;
        }
        else if (mbSynchRefMatched) {

            stat = 0;

            float medDepth = pKFcurORB->ComputeSceneMedianDepth(2);
            bool res = initMapSynch(*pRefEvFr, *pCurEvFr, pEvKFref, pEvKFcur, medDepth, false);

            if (res) {
                assignNewMPtsToTracks(pCurEvFr->mnId, pEvKFcur->GetMapPointMatches(),
                                      pCurEvFr->getMatches(), pCurEvFr->getAllFeatureTracks());
                mbMapInitialized = true;
                stat = 1;
            }
        }

        return stat;
    }

    void EvSynchTrackerU::localMappingSynch(EvFramePtr& pRefFrame, ORB_SLAM3::KeyFrame* pKFini,
                                            EvFramePtr& pCurrFrame, ORB_SLAM3::KeyFrame* pKFcur) {

        // if map not initialized or just init. abort
        if (!this->isMapInitialized() || !pCurrFrame || !pKFcur) {
            return;
        }

        // refresh the connections
        EvLocalMapping::processNewKeyFrame(mpEvAtlas, pKFcur, mlpRecentAddedMapPoints);

        // map point culling? -> partially done in detect & merge new tracks
        EvLocalMapping::mapPointCullingCovis(mpEvAtlas, pKFcur, pCurrFrame, mlpRecentAddedMapPoints, mpSensor->isMonocular());

        // triangulate new map points
        //this->triangulateNewMapPoints(pRefFrame, pCurrFrame);

        vector<int> vMatches12 = pCurrFrame->getMatches();

        assert(pKFini && pKFcur && pKFini->mnFrameId == pRefFrame->mnId && pKFcur->mnFrameId == pCurrFrame->mnId);

        mpLocalMapper->createNewMapPoints(mpEvAtlas, pKFcur, mlpRecentAddedMapPoints);

        // update feature track map points
        assignNewMPtsToTracks(pCurrFrame->mnId, pKFcur->GetMapPointMatches(), vMatches12, pCurrFrame->getAllFeatureTracks());

        // No need to use event local mapper here
        // TODO: Impelement tightly-coupled local mapping
    }

} // EORB_SLAM

