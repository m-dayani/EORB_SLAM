//
// Created by root on 8/10/21.
//

#include "EvSynchTracker.h"

#include <utility>
#include "MyOptimizer.h"
#include "EvLocalMapping.h"

using namespace cv;
using namespace ORB_SLAM3;

namespace EORB_SLAM {

    EvSynchTracker::EvSynchTracker(shared_ptr<EvParams> evParams, CamParamsPtr camParams, const MixedFtsParamsPtr& pFtParams,
            std::shared_ptr<ORB_SLAM3::Atlas> pEvAtlas, ORB_SLAM3::ORBVocabulary *pVocab,
            SensorConfigPtr sConf, ORB_SLAM3::Viewer* pViewer)
                    : EvAsynchTracker(std::move(evParams), std::move(camParams), pFtParams, std::move(pEvAtlas),
                            pVocab, std::move(sConf), pViewer), mnLastTrackedMPsSynch(0)
    {}

    EvSynchTracker::~EvSynchTracker() = default;

    void EvSynchTracker::TrackSynch() {

        while (!this->isStopped()) {

            if (isInputGood()) {

                if (mStat == IDLE) {
                    mStat.update(INIT_TRACK);
                }

                PoseImagePtr pPoseImage = this->frontImage();

                if (pPoseImage->mImage.empty()) {
                    LOG(ERROR) << "* L2 Asynch. Tracker, " << pPoseImage->ts << ": Empty image\n";
                    this->popImage();
                    continue;
                }

                if (pPoseImage->mReconstStat == 0) {
                    this->trackTinyFrameSynch(pPoseImage);
                }

                this->popImage();
            }
            if (mStat == IDLE) {

                VLOG_EVERY_N(3, 1000) << "L2: Idle state...\n";
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }

            if (!this->sepThread) { break; }
        }
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    // Big Note: mpLastFrame is KF tracker's ref. frame (must have descriptors)

    bool EvSynchTracker::isTrackerInitializedSynch() {

        return mbTrackerInitialized == true;
    }

    bool EvSynchTracker::isMapInitializedSynch() {

        return mbMapInitialized == true;
    }

    bool EvSynchTracker::checkConsistentTs(EvFrame &evFrame, ORB_SLAM3::KeyFrame *pKF, const double thTs) {

        return evFrame.mpReferenceKF && pKF && abs(evFrame.mTimeStamp-pKF->mTimeStamp) < thTs &&
               abs(evFrame.mTimeStamp-evFrame.mpReferenceKF->mTimeStamp) < thTs;
    }

    int EvSynchTracker::trackTinyFrameSynch(const PoseImagePtr& pImage) {

        vector<cv::KeyPoint> trackedPts, trackedPtsKF;
        vector<int> vMatches12, vMatches12KF;

        double ts = pImage->ts;
        cv::Mat tinyIm = pImage->mImage;

        if (!isTrackerInitializedSynch()) {
            DLOG_EVERY_N(WARNING, 1000) << "EvAsynchTracker::trackTinyFrameSynch: "
                                        << ts << ", Synch tracker not init., abort\n";
            return 0;
        }
        std::unique_lock<mutex> lock1(mMtxSynchTracker);
        mpLKTrackerKF->trackAndMatchCurrImage(tinyIm, trackedPtsKF, vMatches12KF);
        unsigned nTrackedPts = mpLKTracker->trackAndMatchCurrImage(tinyIm, trackedPts, vMatches12);

        DLOG_EVERY_N(INFO, 1000) << "EvAsynchTracker::trackTinyFrameSynch: " << ts
                                 << ", Tracked tiny frame with " << nTrackedPts << " tracked points\n";

        return static_cast<int>(nTrackedPts);
    }

    int EvSynchTracker::setInitEvFrameSynch(const PoseImagePtr& pImage, ORB_SLAM3::FramePtr& pFrame) {

        this->resetInitMapState();
        mbTrackerInitialized = false;
        mbSynchRefMatched = false;
        mnLastTrackedMPsSynch = 0;

        this->prepReInitDisconnGraph();

        std::unique_lock<mutex> lock1(mMtxSynchTracker);
        int nKPts = this->init(pImage->mImage, pImage->ts, pImage->mTcw);
        //mpIniFrameTemp = EvFrame(mpIniFrame);

        if (nKPts > DEF_TH_MIN_KPTS) {
            DLOG(INFO) << "EvAsynchTracker::setInitEvFrameSynch: " << pImage->ts
                       << ": Init. with " << nKPts << " KPts\n";
            mbTrackerInitialized = true;
            return nKPts;
        }
        else {
            DLOG(WARNING) << "EvAsynchTracker::setInitEvFrameSynch: " << pImage->ts
                          << ": Not enough key points, " << nKPts << endl;
            return 0;
        }
    }

    int EvSynchTracker::trackEvFrameSynch(const PoseImagePtr& pImage) {

        double ts = pImage->ts;

        if (!isTrackerInitializedSynch()) {

            DLOG(WARNING) << "EvAsynchTracker::trackEvFrameSynch: " << ts
                          << ", Tracker not init., attempting to init. tracker\n";

            FramePtr pDummyFr = nullptr;
            return setInitEvFrameSynch(pImage, pDummyFr);
        }
        else {
            std::unique_lock<mutex> lock1(mMtxSynchTracker);
            EvFramePtr dummyFrame;
            trackAndFrame(pImage, -1, mpLKTrackerKF, dummyFrame);
            // Current frame is tracked with respect to reference frame
            unsigned nTrackedPts = trackAndFrame(pImage, mFrameIdx++, mpLKTracker, mpCurrFrame);

            DLOG(INFO) << "EvAsynchTracker::trackEvFrameSynch: " << ts
                       << ", Tracked " << nTrackedPts << " KPts successfully\n";

            return static_cast<int>(nTrackedPts);
        }
    }

    int EvSynchTracker::trackAndOptEvFrameSynch(const PoseImagePtr& pImage, ORB_SLAM3::FramePtr& pFrame) {

        // Check both tracker and local map are initialized
        if (!isTrackerInitializedSynch() || !isMapInitializedSynch()) {

            DLOG_EVERY_N(WARNING, 100) << "EvAsynchTracker::trackAndOptEvFrameSynch: " << pFrame->mTimeStamp
                                       << ", Tracker or Map are not initialized, abort\n";
            mnLastTrackedMPsSynch = 0;
            return 0;
        }

        assert(checkConsistentTs(*mpIniFrame, mpReferenceKF) && checkConsistentTs(*mpLastFrame, mpLastKF));

        const double imTs = pFrame->mTimeStamp;
        // Track reference frame (im0, p0) using
        // last tracked pts and current image -> find matches
        unsigned nMatches = 0;

        {
            std::unique_lock<mutex> lock1(mMtxSynchTracker);
            nMatches = this->trackAndFrame(pImage, mFrameIdx++, mpLKTracker, mpCurrFrame);
            // Update KF tracker
            EvFramePtr dummyFrame;
            this->trackAndFrame(pImage, -1, mpLKTrackerKF, dummyFrame);
        }

        int nTrackedPts = static_cast<int>(nMatches);
        // Not enough matches (ORB-SLAM min th is 15)
        if (nTrackedPts < DEF_MIN_TRACKED_PTS3D_TLM) {//DEF_MIN_TRACKED_PTS3D) {
            DLOG(WARNING) << "EvAsynchTracker::trackAndOptEvFrameSynch: " << pFrame->mTimeStamp
                          << ", Not enough tracked points: " << nTrackedPts << endl;

            mnLastTrackedMPsSynch = 0;
            return 0;
        }

        mpCurrFrame->SetPose(pFrame->mTcw);

        // TODO: Usage of init. frame after init. is wrong because only KFs are optimized!
        // We extensively use kpt levels in this mode, so assign based on descriptors
        this->assignKPtsLevelDesc(*mpIniFrame, *mpCurrFrame, pImage->mImage);

        // Use reference map points to track current frame
        int nMPs = assignMatchedMPs(mpReferenceKF, mpCurrFrame);

        EvFrame currEvFrame = EvFrame(*mpCurrFrame);

        DLOG(INFO) << "EvAsynchTracker::trackAndOptEvFrameSynch: " << pFrame->mTimeStamp
                   << ", EvFrame tracked successfully with: " << nTrackedPts
                   << " tracked points, " << nMPs << " map point matches\n";
        // Optimize TODO: Do these here

        // Discard outliers
        //return refineTrackedMPs(currFrame);
        mnLastTrackedMPsSynch = nMPs;
        return nMPs;
    }

    void EvSynchTracker::insertRefKeyFrameSynch(ORB_SLAM3::KeyFrame *pKFref) {

        mbSynchRefMatched = true;
        mpIniFrame->SetPose(pKFref->GetPose());
        mpReferenceKF = insertRefKF(mpIniFrame);
        mpReferenceKF->mpSynchOrbKF = pKFref;
        mpIniFrame->mpReferenceKF = pKFref;
    }

    int EvSynchTracker::trackEvKeyFrameSynch(const PoseImagePtr& pImage, ORB_SLAM3::KeyFrame *pKFcur) {

        cv::Mat mci = pImage->mImage;
        FramePtr pDummyFr = nullptr;

        // If we didn't init. tracker with a valid ORB KeyFrame, init. tracker
        if (!isTrackerInitializedSynch() || mbSynchRefMatched == false) {

            DLOG(INFO) << "EvAsynchTracker::trackEvKeyFrameSynch: " << pKFcur->mTimeStamp
                       << " No init. available or init. missmatch, reInit.\n";

            this->setInitEvFrameSynch(pImage, pDummyFr);
            // Insert new key frame in graph
            this->insertRefKeyFrameSynch(pKFcur);
            return 1;
        }
            // This is the second key frame and we need to init. map
        else if (!isMapInitializedSynch()) {

            DLOG(INFO) << "EvAsynchTracker::trackEvKeyFrameSynch: " << pKFcur->mTimeStamp
                       << " Map not initialized, attempting to init. map.\n";
            // Prepare reference & current frames and kfs
            // Assure LK tracker init. before calling this method!
            unsigned nMatched = 0;
            {
                unique_lock<mutex> lock(mMtxSynchTracker);
                nMatched = this->trackAndFrame(pImage, mFrameIdx++, mpLKTracker, mpLastFrame);
            }

            mpLastFrame->SetPose(pKFcur->GetPose());
            mpLastKF = insertCurrKF(mpLastFrame, mpReferenceKF);
            mpLastKF->mpSynchOrbKF = pKFcur;
            mpLastFrame->mpReferenceKF = pKFcur;

            this->assignKPtsLevelDesc(*mpIniFrame, *mpLastFrame, mci);

            // Since we cannot compute medDepth from Ev ref KF, we use ORB ref. KF
            const float medDepth = mpIniFrame->mpReferenceKF->ComputeSceneMedianDepth(2);

            // Init. map using curr and ref. ORB KFs
            bool resMap = this->initMapSynch(*mpIniFrame, *mpLastFrame, mpReferenceKF, mpLastKF, medDepth, true);

            if (!resMap) {
                // At very least set KFs bad flags, maybe event erase them from atlas & map & delete
                mpLastKF->SetBadFlag();
                mpReferenceKF->SetBadFlag();

                PoseImagePtr pImage = make_shared<PoseImage>(pKFcur->mTimeStamp, mci, pKFcur->GetPose(), "");
                this->setInitEvFrameSynch(pImage, pDummyFr);
                // Insert new key frame in graph
                this->insertRefKeyFrameSynch(pKFcur);
                return 1;
            }
            mbMapInitialized = true;
            // Also update KF tracker
            {
                unique_lock<mutex> lock(mMtxSynchTracker);
                this->makeFrame(pImage, mFrameIdx++, mpIniFrameTemp);
                mpLKTrackerKF->setRefImage(mci, mpIniFrameTemp->getAllUndistKPtsMono());
            }
            return 2;
        }
            // We had an init. for both tracker and map (this is the >= 3rd KF)
        else {
            DLOG(INFO) << "EvAsynchTracker::trackEvKeyFrameSynch: " << pKFcur->mTimeStamp
                       << " Tracker and map initialized, refreshing tracker state...\n";

            cv::Mat refIm;
            vector<cv::KeyPoint> refKPts;
            {
                unique_lock<mutex> lock(mMtxSynchTracker);
                mpLKTrackerKF->getRefImageAndPoints(refIm, refKPts);

                // Use temp vars so if map is not initialized, we don't loss current map!
                // Always initialize mpIniFrameTemp with KF tracker
                assert(mpIniFrameTemp->numAllKPts() > 0);
                //this->makeFrame(refIm, mpLastFrame.mTimeStamp, mFrameIdx++, refKPts, mpIniFrameTemp);
                this->trackAndFrame(pImage, mFrameIdx++, mpLKTrackerKF, mpCurrFrame);
            }

            // Since KF insertion is invasive, save important KFs
            mpRefKfTemp = mpReferenceKF;
            //KeyFrame* pEvKFlast = mpLastKF;

            mpIniFrameTemp->SetPose(mpLastFrame->mTcw);
            mpReferenceKF = insertRefKF(mpIniFrameTemp);
            mpReferenceKF->mpSynchOrbKF = mpLastFrame->mpReferenceKF;
            mpIniFrameTemp->mpReferenceKF = mpLastFrame->mpReferenceKF;

            mpCurrFrame->SetPose(pKFcur->GetPose());
            mpLastKF = insertCurrKF(mpCurrFrame, mpReferenceKF);
            mpLastKF->mpSynchOrbKF = pKFcur;
            mpCurrFrame->mpReferenceKF = pKFcur;

            this->assignKPtsLevelDesc(*mpIniFrameTemp, *mpCurrFrame, mci);

            float medDepth = mpLastFrame->mpReferenceKF->ComputeSceneMedianDepth(2);

            // Init. map using curr and ref. ORB KFs
            bool resMap = this->initMapSynch(*mpIniFrameTemp, *mpCurrFrame, mpReferenceKF, mpLastKF, medDepth, true);

            if (!resMap && mnLastTrackedMPsSynch < DEF_MIN_TRACKED_PTS3D_TLM) {

                DLOG(WARNING) << "EvAsynchTracker::trackEvKeyFrameSynch: " << pKFcur->mTimeStamp
                              << ", Failed to reInit map, reInit tracker.\nnLastTrackedMPs: " << mnLastTrackedMPsSynch
                              << ", iniTs: " << mpIniFrameTemp->mTimeStamp << ", currTs: " << mpCurrFrame->mTimeStamp << endl;

                // Clean up Temp KFs
                mpLastKF->SetBadFlag();
                mpReferenceKF->SetBadFlag();

                PoseImagePtr pImage = make_shared<PoseImage>(pKFcur->mTimeStamp, mci, pKFcur->GetPose(), "");
                this->setInitEvFrameSynch(pImage, pDummyFr);
                // Insert new key frame in graph
                this->insertRefKeyFrameSynch(pKFcur);
                return 1;
            }
            else if (resMap) {

                DLOG(WARNING) << "EvAsynchTracker::trackEvKeyFrameSynch: " << pKFcur->mTimeStamp
                              << ", Map reInit. successfully, finalize update.\nnLastTrackedMPs: " << mnLastTrackedMPsSynch
                              << ", iniTs: " << mpIniFrameTemp->mTimeStamp << ", currTs: " << mpCurrFrame->mTimeStamp << endl;

                // Map Response is good, finalize temp vars
                mpIniFrame = make_shared<EvFrame>(*mpIniFrameTemp);
                mpLastFrame = make_shared<EvFrame>(*mpCurrFrame);
                //mpReferenceKF = mpRefKfTemp;
                //mpLastKF = pEvKFcur;

                // Update trackers' info.
                unique_lock<mutex> lock(mMtxSynchTracker);
                mpLKTracker->setRefImage(refIm, refKPts);
                mpLKTracker->setLastTrackedPts(mpLKTrackerKF->getLastTrackedPts());
            }
            else if (mnLastTrackedMPsSynch >= DEF_MIN_TRACKED_PTS3D_TLM) {

                DLOG(WARNING) << "EvAsynchTracker::trackEvKeyFrameSynch: " << pKFcur->mTimeStamp
                              << ", Map failure but last map is good, trying to find more MPs with currFrame\n"
                              << "nLastTrackedMPs: " << mnLastTrackedMPsSynch << ", iniTs: "
                              << mpIniFrame->mTimeStamp << ", currTs: " << pKFcur->mTimeStamp << endl;

                // Clean up Temp KFs & Restore
                mpLastKF->SetBadFlag();
                mpReferenceKF->SetBadFlag();

                mpReferenceKF = mpRefKfTemp;
                mpIniFrame->mpReferenceKF = mpReferenceKF;
                mpIniFrame->SetPose(mpReferenceKF->GetPose());

                // Refine old map points before triangulating new ones
                optimizeSBASynch(mpReferenceKF, *mpIniFrame);

                // Update main tracker
                this->trackAndFrame(pImage, mFrameIdx++, mpLKTracker, mpLastFrame);

                // See if we can add more map points
                mpLastFrame->SetPose(pKFcur->GetPose());
                mpLastKF = this->insertCurrKF(mpLastFrame, mpReferenceKF);
                mpLastKF->mpSynchOrbKF = pKFcur;
                mpLastFrame->mpReferenceKF = pKFcur;

                this->assignKPtsLevelDesc(*mpIniFrame, *mpLastFrame, mci);

                resMap = this->initMapSynch(*mpIniFrame, *mpLastFrame, mpReferenceKF, mpLastKF,
                                            mpReferenceKF->ComputeSceneMedianDepth(2), false);

                // Map response is not important here because we have a map
            }
            mbSynchRefMatched = true;
            mbMapInitialized = true;
            {
                // Update KF tracker
                unique_lock<mutex> lock(mMtxSynchTracker);
                this->makeFrame(pImage, mFrameIdx++, mpIniFrameTemp);
                mpLKTrackerKF->setRefImage(mci, mpIniFrameTemp->getAllUndistKPtsMono());
            }

            return 3;
        }
    }

    bool EvSynchTracker::initMapSynch(EvFrame& refFrame, EvFrame& curFrame,
            ORB_SLAM3::KeyFrame* refKF, ORB_SLAM3::KeyFrame* curKF, const float medDepth, const bool cleanMap) {

        vector<MapPoint*> vpIniMPs = curKF->GetMapPointMatches();
        vector<MapPoint*> vpMPs = curFrame.getAllMapPointsMono();
        vector<bool> vbOutliers = curFrame.getAllOutliersMono();

        int nMPs = EvLocalMapping::createNewMapPoints(mpEvAtlas, refKF, curKF,
                                            vpMPs, vbOutliers, vpIniMPs, curFrame.getMatches(), medDepth);

        if (nMPs < DEF_MIN_TRACKED_PTS3D_TLM) {
            DLOG(WARNING) << "EvAsynchTracker::initMapSynch: Not enough map points: " << nMPs
                          << ", currTs: " << refKF->mTimeStamp << endl;
            // Maybe do some cleaning (map points)
            if (cleanMap) {
                for (auto &pMP : vpMPs) {
                    if (pMP) {
                        pMP->SetBadFlag();
                        refKF->EraseMapPointMatch(pMP);
                        curKF->EraseMapPointMatch(pMP);
                        mpEvAtlas->GetCurrentMap()->EraseMapPoint(pMP);
                        delete pMP;
                        pMP = static_cast<MapPoint *>(nullptr);
                    }
                }
            }
            //mbMapInitialized = false;
            return false;
        }

        // Update Connections
        refKF->UpdateConnections();
        curKF->UpdateConnections();

        //refFrame.setAllMapPointsMono(vpMPs);
        //refFrame.setAllMPOutliers(vbOutliers);
        curFrame.setAllMapPointsMono(vpMPs);
        curFrame.setAllMPOutliers(vbOutliers);

        DLOG(INFO) << "EvAsynchTracker::initMapSynch: Successful reconst. at " << refKF->mTimeStamp
                   << ", nMPs: " << nMPs << endl;
        //mbMapInitialized = true;
        return true;
    }

    /*bool EvSynchTracker::setRefEvKeyFrameSynch(ORB_SLAM3::KeyFrame *pKFini) {

        if (!isTrackerInitializedSynch()) {

            DLOG(WARNING) << "EvAsynchTracker::setRefEvKeyFrameSynch: " << pKFini->mTimeStamp
                          << " Tracker is not initialized, abort\n";
            mbSynchRefMatched = false;
            return false;
        }

        if (abs(mpIniFrame->mTimeStamp - pKFini->mTimeStamp) < 1e-6) {

            DLOG(INFO) << "EvAsynchTracker::setRefEvKeyFrameSynch: " << pKFini->mTimeStamp
                       << " Tracker is initialized and ts match at: " << mpIniFrame->mTimeStamp
                       << ", inserting ref. ORB KF\n";
            this->insertRefKeyFrameSynch(pKFini);
            return true;
        }
        DLOG(WARNING) << "EvAsynchTracker::setRefEvKeyFrameSynch: " << pKFini->mTimeStamp
                      << " Tracker is initialized but reference ts missmatch at: "
                      << mpIniFrame->mTimeStamp << ", abort\n";
        mbSynchRefMatched = false;
        return false;
    }

    bool EvSynchTracker::getKFsForInitBASynch(ORB_SLAM3::KeyFrame*& pKFref, ORB_SLAM3::KeyFrame*& pKFcur) {

        if (mbSynchRefMatched == true && isMapInitializedSynch()) {

            DLOG(INFO) << "EvAsynchTracker::getKFsForInitBASynch: ORB ref: " << mpIniFrame->mpReferenceKF->mTimeStamp
                       << ", ORB cur: " << mpLastFrame->mpReferenceKF->mTimeStamp << ", Ev ref: "
                       << mpReferenceKF->mTimeStamp << ", Ev cur: " << mpLastKF->mTimeStamp << endl;
            pKFref = mpReferenceKF;
            pKFcur = mpLastKF;
            return true;
        }
        else {
            DLOG_EVERY_N(WARNING, 100) << "EvAsynchTracker::getKFsForInitBASynch: No Ev reconst. available, abort\n";
            pKFref = nullptr;
            pKFcur = nullptr;
            return false;
        }
    }

    int EvSynchTracker::evImInitOptimizationSynch(const cv::Mat &currMCI, ORB_SLAM3::Map *pMap,
            ORB_SLAM3::KeyFrame *pKFini, ORB_SLAM3::KeyFrame *pKFcur, int nIter) {

        int stat = -1;
        PoseImagePtr pImage = make_shared<PoseImage>(pKFcur->mTimeStamp, currMCI, pKFcur->GetPose(), "");
        if (!isTrackerInitializedSynch()) {

            DLOG(WARNING) << "EvAsynchTracker::evImInitOptimizationSynch: " << pKFcur->mTimeStamp
                          << ", Tracker not init., attempting to init. tracker\n";

            FramePtr pDummyFr = nullptr;
            setInitEvFrameSynch(pImage, pDummyFr);
            return stat;
        }
        // Assert ref. & current Ts consistency
        const bool consistentTs = abs(mpIniFrame->mTimeStamp - pKFini->mTimeStamp) < 1e-6 &&
                                  abs(mpCurrFrame->mTimeStamp - pKFcur->mTimeStamp) < 1e-6;

        if (!consistentTs) {
            trackEvKeyFrameSynch(pImage, pKFcur);
            return stat;
        }

        mbSynchRefMatched = true;

        // If all good, init. event KFs and map
        mpIniFrame->SetPose(pKFini->GetPose());
        mpReferenceKF = insertRefKF(mpIniFrame);
        mpIniFrame->mpReferenceKF = pKFini;
        mpReferenceKF->mpSynchOrbKF = pKFini;

        mpCurrFrame->SetPose(pKFcur->GetPose());
        mpLastKF = insertCurrKF(mpCurrFrame, mpReferenceKF);
        mpCurrFrame->mpReferenceKF = pKFcur;
        mpLastKF->mpSynchOrbKF = pKFcur;

        EvLocalMapping::addNewMapPoints(mpEvAtlas, mpReferenceKF, mpLastKF, mpCurrFrame->getMatches(), mvPts3D);

        mpIniFrame->setAllMapPointsMono(mpReferenceKF->GetMapPointMatches());
        mpCurrFrame->setAllMapPointsMono(mpLastKF->GetMapPointMatches());

        mpLastFrame = make_shared<EvFrame>(*mpCurrFrame);

        // Joint Ev-Im optimization
        vector<KeyFrame*> vpEvKFs{mpReferenceKF, mpLastKF};

        MyOptimizer::GlobalBundleAdjustment(pMap, vpEvKFs, nIter);

        // Normalize & reScale event map
        set<MapPoint*> spEvMPs = mpReferenceKF->GetMapPoints();
        vector<MapPoint*> vpEvMPs(spEvMPs.begin(), spEvMPs.end());

        const float medDepth = mpReferenceKF->ComputeSceneMedianDepth(2);

        EvLocalMapping::scalePoseAndMap(vpEvKFs, vpEvMPs, medDepth);

        mbMapInitialized = true;

        {
            // Set KeyFrame Tracker
            unique_lock <mutex> lock(mMtxSynchTracker);
            this->makeFrame(pImage, mFrameIdx++, mpIniFrameTemp);
            mpLKTrackerKF->setRefImage(currMCI, mpIniFrameTemp->getAllUndistKPtsMono());
        }

        return 1;
    }*/

    void EvSynchTracker::optimizeSBASynch(ORB_SLAM3::KeyFrame* pKFref, EvFrame& refFrame) {

        vector<KeyFrame*> vpKFs;
        vpKFs.reserve(pKFref->GetChilds().size()+1);

        // Update all Event Poses before SBA
        // Assure KFs consistency
        assert(pKFref->mpSynchOrbKF && abs(pKFref->mpSynchOrbKF->mTimeStamp-pKFref->mTimeStamp) < 1e-6);
        pKFref->SetPose(pKFref->mpSynchOrbKF->GetPose());
        vpKFs.push_back(pKFref);

        for (KeyFrame* pKFi : pKFref->GetChilds()) {

            if (pKFi && !pKFi->isBad()) {
                assert(pKFi->mpSynchOrbKF && abs(pKFi->mpSynchOrbKF->mTimeStamp-pKFi->mTimeStamp) < 1e-6);
                pKFi->SetPose(pKFi->mpSynchOrbKF->GetPose());
                vpKFs.push_back(pKFi);
            }
        }

        // Structure only BA
        MyOptimizer::StructureBA(vpKFs, 5);

        // Refine Reference Frame map points
        assert(checkConsistentTs(refFrame, pKFref));
        for (int i = 0; i < refFrame.numAllKPts(); i++) {

            MapPoint* pMP = pKFref->GetMapPoint(i);
            if (!pMP || pMP->isBad()) {
                refFrame.setMapPoint(i, static_cast<MapPoint*>(nullptr));
                refFrame.setMPOutlier(i, false);
            }
        }

    }

    void EvSynchTracker::mergeKeyPoints(const std::vector<cv::KeyPoint> &vkpt1, const std::vector<cv::KeyPoint> &vkpt2,
                                        std::vector<cv::KeyPoint> &vkpt) {

        size_t nKpts1 = vkpt1.size(), nKpts2 = vkpt2.size();

        size_t numAllKpts = nKpts1 + nKpts2;
        vkpt.resize(numAllKpts);

        for (size_t i = 0; i < numAllKpts; i++) {
            if (i < nKpts1) {
                vkpt[i] = vkpt1[i];
            }
            else {
                vkpt[i] = vkpt2[i - nKpts1];
            }
        }
    }

    uint EvSynchTracker::mergeMatches12(const std::vector<int> &vmch1, const std::vector<int> &vmch2, const int maxIdx12,
                                        std::vector<int> &vmch) {

        size_t nmch1 = vmch1.size(), nmch2 = vmch2.size();

        size_t nAllMch = nmch1+nmch2;
        vmch.resize(nAllMch);
        uint cnt = 0;

        for (size_t i = 0; i < nAllMch; i++) {
            if (i < nmch1) {
                int currIdx = vmch1[i];
                vmch[i] = currIdx;
                if (currIdx >= 0) {
                    cnt++;
                }
            }
            else {
                int currIdx = vmch2[i - nmch1];
                if (currIdx >= 0) {
                    vmch[i] = currIdx+maxIdx12;
                    cnt++;
                }
                else {
                    vmch[i] = currIdx;
                }
            }
        }
        return cnt;
    }

    int EvSynchTracker::evImReconst2ViewsSynch(const PoseImagePtr& pImage,
            const ORB_SLAM3::FramePtr& pFrame1, ORB_SLAM3::FramePtr& pFrame2, const std::vector<int> &vMatches12,
            cv::Mat &R21, cv::Mat &t21, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated) {

        int stat = -1;
        cv::Mat currMCI = pImage->mImage;
        GeometricCamera* pCamera = pFrame1->mpCamera; // TODO: Careful! The calibration of Evs & Ims might vary

        if (!isTrackerInitializedSynch()) {

            DLOG(WARNING) << "EvAsynchTracker::evImReconst2ViewsSynch: " << pFrame2->mTimeStamp
                          << ", Tracker not init., attempting to init. tracker\n";

            FramePtr pDummyFr = nullptr;
            setInitEvFrameSynch(pImage, pDummyFr);
            return stat;
        }
        else {
            // Update trackers with current frame
            unsigned nTrackedPts = 0;
            {
                std::unique_lock<mutex> lock1(mMtxSynchTracker);
                // Keep key frame tracker tunned
                EvFramePtr dummyFrame;
                trackAndFrame(pImage, -1, mpLKTrackerKF, dummyFrame);

                // Current frame is tracked with respect to reference frame
                vector<KeyPoint> p1;
                vector<int> vMatches12ev;
                vector<int> vCntMatches; //= mvMatchesCnt;
                vector<float> vPxDisp;

                if (mFtDtMode == 0) { // Simple FAST
                    nTrackedPts = mpLKTracker->trackAndMatchCurrImage(currMCI, p1, vMatches12ev, vCntMatches, vPxDisp);
                }
                else {
                    // ORB -> Track only first level key points!
                    nTrackedPts = mpLKTracker->trackAndMatchCurrImageInit(currMCI, p1, vMatches12ev, vCntMatches, vPxDisp);
                }
                this->makeFrame(pImage, mFrameIdx++, p1, mpCurrFrame);
                mpCurrFrame->setMatches(vMatches12ev);
            }

            // Check ini. frame Ts consistency
            if (abs(pFrame1->mTimeStamp - mpIniFrame->mTimeStamp) >= 1e-6) {

                DLOG(WARNING) << "EvAsynchTracker::evImReconst2ViewsSynch: " << pFrame2->mTimeStamp
                              << ", Cannot reconst. because of ref. Ts miss-match, evRefTs: " << mpIniFrame->mTimeStamp
                              << ", fr1Ts: " << pFrame1->mTimeStamp << endl;
                mbSynchRefMatched = false;
                return stat;
            }

            if (nTrackedPts < DEF_TH_MIN_MATCHES) { // Not enough matches

                DLOG(INFO) << "EvAsynchTracker::evImReconst2ViewsSynch: " << pFrame2->mTimeStamp
                           << ", Not enough matches: " << nTrackedPts << endl;
                return stat;
            }

            // If all is good, try to initialize:
            // Assign level info. to tracked KPts based on best matching descriptors
            this->assignKPtsLevelDesc(*mpIniFrame, *mpCurrFrame, currMCI);

            stat = 0;
            // Merge key points and matches
            vector<cv::KeyPoint> vKeys1 = pFrame1->getAllUndistKPtsMono();
            vector<cv::KeyPoint> vKeys2 = pFrame2->getAllUndistKPtsMono();

            // Normally all key points are undistorted, but careful about KPts distortion miss-match
            vector<cv::KeyPoint> vKeysUn1 = pCamera->UndistortKeyPoints(vKeys1);
            vector<cv::KeyPoint> vKeysUn2 = pCamera->UndistortKeyPoints(vKeys2);

            size_t nOrbPts1 = vKeys1.size(), nOrbPts2 = vKeys2.size();
            size_t numAllKeys1 = nOrbPts1 + mpIniFrame->numAllKPts();
            size_t numAllKeys2 = nOrbPts2 + mpCurrFrame->numAllKPts();

            vector<cv::KeyPoint> vAllKeysUn1(numAllKeys1), vAllKeysUn2(numAllKeys2);
            vector<int> vAllMatches12(numAllKeys1, -1);
            vector<int> vMatches12ev = mpCurrFrame->getMatches();

            for (size_t i = 0; i < numAllKeys1; i++) {
                if (i < nOrbPts1) {
                    vAllKeysUn1[i] = vKeysUn1[i];
                    vAllMatches12[i] = vMatches12[i];
                }
                else {
                    size_t currIdx = i - nOrbPts1;
                    vAllKeysUn1[i] = mpIniFrame->getUndistKPtMono(currIdx);
                    if (vMatches12ev[currIdx] >= 0) {
                        vAllMatches12[i] = vMatches12ev[currIdx] + nOrbPts2;
                    }
                }
            }
            for (size_t i = 0; i < numAllKeys2; i++) {
                if (i < nOrbPts2) {
                    vAllKeysUn2[i] = vKeysUn2[i];
                }
                else {
                    vAllKeysUn2[i] = mpCurrFrame->getUndistKPtMono(static_cast<int>(i - nOrbPts2));
                }
            }

            bool res = mpCamera->ReconstructWithTwoViews(vAllKeysUn1, vAllKeysUn2, vAllMatches12,
                                                         R21, t21, vP3D, vbTriangulated);

            if (!res) {
                DLOG(WARNING) << "EvAsynchTracker::evImReconst2ViewsSynch: " << pFrame2->mTimeStamp
                              << ", Cannot reconst. because of original bad response\n";
                return stat;
            }

            // If reconst. successful, fill important stuff for later
            stat = 1;

            // Recover current pose
            cv::Mat currTcw = ORB_SLAM3::Converter::toCvSE3(R21, t21);
            mpCurrFrame->SetPose(currTcw);

            // Recover current 3d points
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

            DLOG(INFO) << "EvAsynchTracker::evImReconst2ViewsSynch: " << pFrame2->mTimeStamp
                       << ", Successful reconstruction: evRefTs: " << mpIniFrame->mTimeStamp
                       << ", orbRefTs: " << pFrame1->mTimeStamp << endl;

            return stat;
        }
    }

    void EvSynchTracker::assignKPtsLevelDesc(const EvFrame& refFrame, EvFrame& currFrame, const cv::Mat& currIm) {

        vector<cv::KeyPoint> trackedKPts = currFrame.getAllDistKPtsMono();
        mpORBLevelDetector->AssignKPtLevelByBestDesc(refFrame.getAllORBDescMono(), currIm, trackedKPts);
        currFrame.setDistTrackedPts(trackedKPts);
    }

} // EORB_SLAM