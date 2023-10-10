//
// Created by root on 11/15/21.
//

#include "EvAsynchTrackerU.h"

#include <utility>

#include "Optimizer.h"
#include "MyOptimizer.h"
#include "EvLocalMapping.h"


using namespace std;
using namespace cv;
using namespace ORB_SLAM3;


namespace EORB_SLAM {

#define DEF_TH_NUM_FTT 0.8
#define DEF_MIN_TRACKED_PTS3D_LOW 10
#define DEF_TH_AREA_RATIO 0.4f

    // TODO: Check KF & MPts connections and ...

    EvAsynchTrackerU::EvAsynchTrackerU(EvParamsPtr evParams, CamParamsPtr camParams, const MixedFtsParamsPtr& pFtParams,
            std::shared_ptr<ORB_SLAM3::Atlas> pEvAtlas, ORB_SLAM3::ORBVocabulary *pVocab, SensorConfigPtr sConf,
            ORB_SLAM3::Viewer* pViewer) :
                EvAsynchTracker(std::move(evParams), std::move(camParams), pFtParams, std::move(pEvAtlas),
                                pVocab, std::move(sConf), pViewer),
                mnFirstFrameId(0), mFtTrackIdx(0), mnFeatureTracks(0), mnCurrFeatureTracks(0), mnLastTrackedKPts(0),
                mnLastTrackedMPts(0), mbSetLastPose(false)
    {
        if (mpLocalMapper)
            mpLocalMapper->enableLocalMapping(false);

        mTrackingTimer.setName("L2 Continuous Tracker");
        mTrackingWD.setName("L2 Continuous Tracker");
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    void printBadTracks(const std::map<ulong, FeatureTrackPtr>& mpFeatureTracks) {

        cout << "Tracks Size: " << mpFeatureTracks.size() << ", track stats: (id, lost, valid):\n";
        for (const auto& ppTrack : mpFeatureTracks) {

            FeatureTrackPtr pTrack = ppTrack.second;
            if (pTrack && (pTrack->isLost() || !pTrack->isValid()))
                cout << "(" << ppTrack.first << ", " << ppTrack.second->isLost() << ", " << ppTrack.second->isValid() << "), ";
        }
        cout << endl;
    }

    void printMismatches(const EvFramePtr& pRefFrame, const EvFramePtr& pCurFrame) {

        vector<FeatureTrackPtr> vpRefTracks = pRefFrame->getAllFeatureTracks();
        vector<FeatureTrackPtr> vpCurTracks = pCurFrame->getAllFeatureTracks();
        vector<int> vMatches12 = pCurFrame->getMatches();

        cout << "RefTracks size: " << vpRefTracks.size() << ", CurTracks size: "
             << vpCurTracks.size() << ", vMatches size: " << vMatches12.size() << endl;
        if (vpRefTracks.size() != vMatches12.size()) {
            cout << "Ref. tracks and matches size mismatch, abort...\n";
            return;
        }
        uint nMisMatches = 0;
        for (size_t i = 0; i < vMatches12.size(); i++) {
            const int j = vMatches12[i];
            if (j >= 0 && j < vpCurTracks.size()) {
                FeatureTrackPtr pRefTrack = vpRefTracks[i];
                FeatureTrackPtr pCurTrack = vpCurTracks[j];
                if (pRefTrack && pCurTrack && pRefTrack->getId() != pCurTrack->getId()) {
                    cout << "(" << pRefTrack->getId() << ", " << pRefTrack->getId() << "), ";
                    nMisMatches++;
                }
            }
        }
        cout << endl << nMisMatches << " mismatches discovered\n";
    }

    void EvAsynchTrackerU::saveAllFramePoses(const std::string &fileName, const double tsc, const int ts_prec) {

        while (this->isInputGood())
            this_thread::sleep_for(std::chrono::milliseconds(3));

        if (mpSensor->isInertial() && mpImuManager)
            mpImuManager->abortInit(true);

        if (isDisconnGraph()) {
            // for disconnected graph save all poses (disconnected)
            for (auto& pMap : mpEvAtlas->GetAllMaps()) {

                vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
                sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

                mFrameInfo.setIsLostRange(vpKFs[0]->mnFrameId, vpKFs.back()->mnFrameId, false, true);

                Visualization::saveAllFramePoses(vpKFs, mFrameInfo, fileName, nullptr,
                                                 mpSensor->isInertial(), tsc, ts_prec);

                // append disconnection signs
                ofstream poseFile;
                poseFile.open(fileName, std::ios_base::app);
                if (poseFile.is_open()) {
                    poseFile << "# DISCONNECTED POSE GRAPH" << endl;
                }
                poseFile.close();
            }
        }
        else {
            Visualization::saveAllFramePoses(mpEvAtlas.get(), mFrameInfo, fileName, false,
                                             mpSensor->isInertial(), tsc, ts_prec);
        }
    }

    void EvAsynchTrackerU::saveAllPoses(const std::string &fileName, const double tsc, const int ts_prec) {

        while (this->isInputGood())
            this_thread::sleep_for(std::chrono::milliseconds(3));

        if (mpSensor->isInertial() && mpImuManager)
            mpImuManager->abortInit(true);

        if (isDisconnGraph()) {
            // for disconnected graph save all poses (disconnected)
            for (auto &pMap : mpEvAtlas->GetAllMaps()) {

                vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
                sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

                Visualization::saveAllKeyFramePoses(vpKFs, fileName, mpSensor->isInertial(), tsc, ts_prec);

                // append disconnection signs
                ofstream poseFile;
                poseFile.open(fileName, std::ios_base::app);
                if (poseFile.is_open()) {
                    poseFile << "# DISCONNECTED POSE GRAPH" << endl;
                }
                poseFile.close();
            }
        }
        else {
            Visualization::saveAllKeyFramePoses(mpEvAtlas.get(), fileName, false, mpSensor->isInertial(), tsc, ts_prec);
        }
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    bool EvAsynchTrackerU::isDisconnGraph() {

        return !mpSensor->isInertial() || mTrackMode == TLM_CH_REF;
    }

    bool EvAsynchTrackerU::isTrackerInitialized() {

        return mbTrackerInitialized == true;
    }

    bool EvAsynchTrackerU::enoughFeatureTracks() const {

        const bool cNumFeatures = mnFeatureTracks > 0 && mnCurrFeatureTracks > DEF_TH_NUM_FTT * mnFeatureTracks;

        vector<FeatureTrackPtr> vpTracks;
        FeatureTrack::mapToVectorTracks(mmFeatureTracks, vpTracks);
        float areaFeatures = FeatureTrack::calcAreaFeatureTracks(mFrameIdx, vpTracks);
        float areaRatio = areaFeatures / static_cast<float>(imHeight * imWidth);

        VLOG_EVERY_N(1, 1000) << "EvAsynchTrackerU::enoughFeatureTracks: areaFeatures: "
                                            << areaFeatures << ", areaRatio: " << areaRatio << endl;

        const bool cAreaFeatures = areaRatio > DEF_TH_AREA_RATIO;

        return cNumFeatures && cAreaFeatures;
    }

    bool EvAsynchTrackerU::isTracking() {
        return this->isTrackerInitialized() && this->isMapInitialized();
    }

    void EvAsynchTrackerU::resetAll() {

        EvAsynchTracker::resetAll();

        this->resetState();
        mFrameInfo.reset();

        mnFirstFrameId = 0;

        mFtTrackIdx = 0;
    }

    void EvAsynchTrackerU::resetState(const bool wipeMem) {

        mbTrackerInitialized = false; // -> these 3 also done in parent class
        this->resetInitMapState();
        mbImuInitialized = false;

        mmFeatureTracks.clear();
        mmpWinFrames.clear();
        mlpRecentAddedMapPoints.clear();

        mnFeatureTracks = mmFeatureTracks.size();
        mnCurrFeatureTracks = mnFeatureTracks;
        mnLastTrackedMPts = 0;
        mnLastTrackedKPts = 0;

        // Keep something to merge graph pieces in inertial case
        if (wipeMem) {
            mpIniFrame = nullptr; // these 3 also done in first parent
            mpLastFrame = nullptr;
            mpCurrFrame = nullptr;

            mpReferenceKF = nullptr;
            mpLastKF = nullptr;
        }

        mvLastMatches.clear();
        mvLastFtPxDisp.clear();
    }

    void EvAsynchTrackerU::resetTracker() {

        Verbose::PrintMess("Active map Reseting", Verbose::VERBOSITY_NORMAL);
        if(mpViewer)
        {
            mpViewer->RequestStop();
            while(!mpViewer->isStopped())
                std::this_thread::sleep_for(std::chrono::milliseconds(3));
        }

        //Map* pMap = mpEvAtlas->GetCurrentMap();

        // there is no point in resetting local mapper

        // reset IMU manager
        if (mpSensor->isInertial() && mpImuManager) {

            // never do this! it causes a deadlock!
            // TODO: Better state management??
//            while (mpImuManager->isInitializing())
//                std::this_thread::sleep_for(std::chrono::milliseconds(3));

            mpImuManager->reset();
            mpImuManager->refreshPreintFromLastKF(mnImuID);
        }

        // reset active map
//        if (!mpSensor->isInertial()) {
//            while (mpLocalMapper->isProcessing())
//                std::this_thread::sleep_for(std::chrono::microseconds(500));
//        }
        this->prepReInitDisconnGraph();

        // reset frames
        ulong iniFrId = mnFirstFrameId;
        ulong lastFrId = mmpWinFrames.rbegin()->first;
        mFrameInfo.setIsLostRange(iniFrId, lastFrId, true);

        ulong num_lost = lastFrId-iniFrId+1;
        LOG(INFO) << num_lost << " Frames set to lost" << endl;

//        mnInitialFrameId = mpCurrentFrame->mnId;
//        mnLastRelocFrameId = mpCurrentFrame->mnId;

        this->resetState(false);

        if(mpViewer)
            mpViewer->Release();

        Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    EvFramePtr EvAsynchTrackerU::getFrameById(const ulong frameId) {

        if (mmpWinFrames.find(frameId) != mmpWinFrames.end()) {
            return mmpWinFrames.find(frameId)->second;
        }
        return nullptr;
    }

    // for tracking
    unsigned EvAsynchTrackerU::getTrackedPoints(const ulong frame, std::vector<cv::KeyPoint> &vKPts,
            std::vector<ORB_SLAM3::MapPoint*>& vpMPts, std::vector<FeatureTrackPtr>& vFtTracks) {

        const size_t nAllTracks = mmFeatureTracks.size();
        unsigned nTracks = 0;

        vKPts.clear();
        vKPts.reserve(nAllTracks);
        vFtTracks.clear();
        vFtTracks.reserve(nAllTracks);
        vpMPts.clear();
        vpMPts.reserve(nAllTracks);

        for (const auto& ppTrack : mmFeatureTracks) {

            FeatureTrackPtr pTrack = ppTrack.second;
            cv::KeyPoint kpt;
            MapPoint* pMpt = nullptr;

            if (pTrack && pTrack->isValid() && pTrack->getFeatureAndMapPoint(frame, kpt, pMpt)) {

                vKPts.push_back(kpt);
                vFtTracks.push_back(pTrack);
                vpMPts.push_back(pMpt);
                nTracks++;
            }
        }

        return nTracks;
    }

    void EvAsynchTrackerU::updateTrackedPoints(const ulong frame, const std::vector<cv::KeyPoint> &vKPts,
                                               const std::vector<int>& vMatches12, std::vector<FeatureTrackPtr>& vFtTracks) {

        const size_t nTracks = vFtTracks.size();
        assert(vKPts.size() == nTracks && nTracks == vMatches12.size());

        for (size_t i = 0; i < nTracks; i++) {

            FeatureTrackPtr pTrack = vFtTracks[i];

            if (pTrack) {
                // based on frame idx, we decide to add a new feature or update the last one
                pTrack->updateTrackedFeature(frame, vKPts[i], vMatches12[i]);
            }
        }
    }

    void EvAsynchTrackerU::updateTrackStats() {

        mnCurrFeatureTracks = 0;
        for (const auto& ppTrack : mmFeatureTracks) {

            FeatureTrackPtr pTrack = ppTrack.second;
            if (pTrack && pTrack->isValid()) {
                mnCurrFeatureTracks++;
            }
        }
    }

    void EvAsynchTrackerU::assignNewMPtsToTracks(const ulong frId2, const std::vector<ORB_SLAM3::MapPoint*>& vpMPts2,
            const vector<int>& vMatches12, std::vector<FeatureTrackPtr>& vpFtTracks2, const bool resetOldMPts) {

        const size_t nTracks = vpFtTracks2.size();
        assert(vpMPts2.size() == nTracks);

        for (const int& idx2 : vMatches12) {

            if (idx2 >= 0 && idx2 < nTracks) {

                MapPoint* pMpt2 = vpMPts2[idx2];
                FeatureTrackPtr pTrack = vpFtTracks2[idx2];

                if (pMpt2 && pTrack && pTrack->isValid()) {

                    cv::KeyPoint kpt;
                    MapPoint* pMpt = nullptr;
                    pTrack->getFeatureAndMapPoint(frId2, kpt, pMpt);

                    if (resetOldMPts || (!resetOldMPts && (!pMpt || pMpt->isBad())))
                        pTrack->updateMapPoint(frId2, pMpt2);
                }
            }
        }
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    void EvAsynchTrackerU::preintegrateCurrFrame(EvFramePtr &pCurrFrame) {

        if (mpSensor->isInertial() && mpImuManager) {

            pCurrFrame->mImuCalib = *(mpImuManager->getImuCalib());
            pCurrFrame->mImuBias = mpImuManager->getLastImuBias();
            mpImuManager->preintegrateIMU(mnImuID, pCurrFrame, pCurrFrame->mImuBias);

            // To prevent ugly behavior, wait until IMU is initialized
            //while (mpImuManager->isInitializing()) {
            //    std::this_thread::sleep_for(std::chrono::microseconds (500));
            //}
        }
    }

    void EvAsynchTrackerU::createCurrFrame(const PoseImagePtr &pImage, ulong fId, EvFramePtr &frame) {

        // retrieve all features with map points
        vector<cv::KeyPoint> vAllKPts;
        vector<MapPoint*> vAllMPts;
        vector<FeatureTrackPtr> vpFtTracks;

        this->getTrackedPoints(fId, vAllKPts, vAllMPts, vpFtTracks);

        // create current frame
        frame = make_shared<EvFrame>(fId, pImage, vAllKPts, vpFtTracks, mpCamera, mDistCoefs);
        //frame->setAllMapPointsMono(vAllMPts); // we match these later

        // set Prior Pose from L1 Tracker
        if (!pImage->mTcw.empty())
            frame->setPosePrior(pImage->mTcw);

        // Update State & Connections
        if (!this->isTrackerInitialized()) {
            mpIniFrame = frame;
        }
        if (frame != mpIniFrame && mpIniFrame) {
            frame->setRefFrame(mpIniFrame);
        }
        if (mpLastFrame) {
            frame->setPrevFrame(mpLastFrame);
        }
        if (mpLastKF) {
            frame->mpLastKeyFrame = mpLastKF;
            frame->mpReferenceKF = mpLastKF;
        }
        mpLastFrame = frame;

        // insert frame in local sliding window
        mmpWinFrames.insert(make_pair(fId, frame));

        // keep the image
        frame->imgLeft = pImage->mImage.clone();
    }

    void EvAsynchTrackerU::matchCurrentFrame(EvFramePtr &pCurFrame) {

        if (!this->isTrackerInitialized()) {
            return;
        }

        if (this->isMapInitialized()) {
            EvFramePtr pRefFrame = mmpWinFrames[mpLastKF->mnFrameId];
            FeatureTrack::matchFeatureTracks(pRefFrame, pCurFrame);
        }
        else {
            FeatureTrack::matchFeatureTracks(mpIniFrame, pCurFrame);
        }
    }

    void EvAsynchTrackerU::changeReference(const ulong newRefId) {

        // find the first frame after iniFrame
        if (mmpWinFrames.find(newRefId) == mmpWinFrames.end()) {
            return;
        }

        // commit the change
        mpIniFrame = mmpWinFrames[newRefId];
    }

    void EvAsynchTrackerU::removeOldFrames(ulong newRefId) {

        auto newRefItr = mmpWinFrames.find(newRefId);

        if (mmpWinFrames.empty() || newRefItr == mmpWinFrames.end() || newRefItr == mmpWinFrames.begin()) {
            return;
        }

        auto frIter = mmpWinFrames.begin();
        while(frIter != newRefItr) {

            frIter->second->imgLeft = cv::Mat();
            frIter->second->setRefFrame(nullptr);

            frIter = mmpWinFrames.erase(frIter);
        }
        // TODO: marginalize last iniFrame info -> care memory!
    }

    void EvAsynchTrackerU::findTracksBounds(ulong& minFrId, ulong& maxFrId) {

        ulong curMin = -1, curMax = 0;
        minFrId = curMin;
        maxFrId = curMax;

        for (const auto& ppTrack : mmFeatureTracks) {

            FeatureTrackPtr pTrack = ppTrack.second;

            if (pTrack && pTrack->isValid()) {

                curMin = pTrack->getFrameIdMin();
                curMax = pTrack->getFrameIdMax();

                if (curMin < minFrId)
                    minFrId = curMin;
                if (curMax > maxFrId)
                    maxFrId = curMax;
            }
        }
    }

    bool EvAsynchTrackerU::needNewKeyFrame(const float medPxDisp) {

        // enough median pixel displacement of tracked features
        const bool bLargeMedFtDisp = medPxDisp > DEF_TH_MED_PX_DISP;

        // enough valid feature tracks?
        const bool bLowFtTracks = !this->enoughFeatureTracks();

        // number of frames passed from last key frame
        const bool bEnoughNumFrames = mpCurrFrame->mnId > mpLastKF->mnFrameId + static_cast<ulong>(mpCamParams->mMaxFrames);

        // temporal constraint for inertial mode
        bool bTemporalConst = false;
        if (mpSensor->isInertial() && mpImuManager) {
            bTemporalConst = mpCurrFrame->mTimeStamp - mpLastKF->mTimeStamp > DEF_TH_KF_TEMP_CONST;
        }

        // idle state of local mapper???
        // what about num. tracked map points or features? -> only in inertial mode
        const bool bLowTrackedMPts = mpSensor->isInertial() && mnLastTrackedMPts < DEF_MIN_TRACKED_PTS3D;

        return (bLargeMedFtDisp || bLowFtTracks || bEnoughNumFrames || bTemporalConst || bLowTrackedMPts);
    }

    void EvAsynchTrackerU::addNewKeyFrame(EvFramePtr& pCurrFrame) {

        // Create and insert new KF
        KeyFrame* pKFcur = this->insertCurrKF(pCurrFrame, mpLastKF);
        mpReferenceKF = mpLastKF;

        if(mpSensor->isInertial() && mpEvAtlas->isImuInitialized()) {
            pKFcur->bImu = true;
        }

        pKFcur->SetNewBias(pCurrFrame->mImuBias);

        // Reset preintegration from last KF (Create new object)
        if (mpSensor->isInertial() && mpImuManager) {

            // Need to assign KF preInt. to the last KF? -> No, takes from ref. frame
            mpImuManager->refreshPreintFromLastKF(mnImuID);
        }
    }

    void EvAsynchTrackerU::triangulateNewMapPoints(const EvFramePtr &pRefFrame, const EvFramePtr &pCurrFrame) {

        KeyFrame* pRefKF = mpLastKF->mPrevKF;
        KeyFrame* pCurKF = mpLastKF;
        vector<int> vMatches12 = pCurrFrame->getMatches();

        assert(pRefKF && pCurKF && pRefKF->mnFrameId == pRefFrame->mnId && pCurKF->mnFrameId == pCurrFrame->mnId);

        mpLocalMapper->createNewMapPoints(mpEvAtlas, pCurKF, mlpRecentAddedMapPoints);

        // update feature track map points
        this->assignNewMPtsToTracks(pCurrFrame->mnId, pCurKF->GetMapPointMatches(), vMatches12, pCurrFrame->getAllFeatureTracks());
    }

    void EvAsynchTrackerU::triangulateNewMapPoints() {

        // retrieve feature tracks
        vector<FeatureTrackPtr> vpTracks;
        FeatureTrack::mapToVectorTracks(mmFeatureTracks, vpTracks);

        // retrieve frame bounds
        ulong minFrId = -1, maxFrId = 0, frame_depth = 10;
        this->findTracksBounds(minFrId, maxFrId);

        // resolve frame depth
        if (minFrId > maxFrId || maxFrId - minFrId <= 2) {
            return;
        }
        frame_depth = min(maxFrId-minFrId, 10ul);
        minFrId = maxFrId - frame_depth;

        // call the appropriate reconstruction method
        mpLocalMapper->createNewMapPoints(mpEvAtlas, vpTracks, minFrId, maxFrId, mlpRecentAddedMapPoints);
    }

    void EvAsynchTrackerU::updateFrameIMU(float s, const ORB_SLAM3::IMU::Bias &b, ORB_SLAM3::KeyFrame *pCurrentKeyFrame) {

        {
            //std::unique_lock<mutex> lock1(mMtxUpdateState);

            Map *pMap = pCurrentKeyFrame->GetMap();
            //unsigned int index = mnFirstFrameId;

            list<cv::Mat> mlRelativeFramePoses;
            list<KeyFrame *> mlpReferences;
            list<double> mlFrameTimes;
            list<bool> mlbLost;
            mFrameInfo.getAllState(mlRelativeFramePoses, mlpReferences, mlFrameTimes, mlbLost);

            list<ORB_SLAM3::KeyFrame *>::iterator lRit = mlpReferences.begin();
            list<bool>::iterator lbL = mlbLost.begin();
            for (list<cv::Mat>::iterator lit = mlRelativeFramePoses.begin(), lend = mlRelativeFramePoses.end();
                 lit != lend; lit++, lRit++, lbL++) {
                if (*lbL)
                    continue;

                KeyFrame *pKF = *lRit;

                while (pKF->isBad()) {
                    pKF = pKF->GetParent();
                }

                if (pKF->GetMap() == pMap) {
                    (*lit).rowRange(0, 3).col(3) = (*lit).rowRange(0, 3).col(3) * s;
                }
            }

            mFrameInfo.setAllState(mlRelativeFramePoses, mlpReferences, mlFrameTimes, mlbLost);
        }

        EvAsynchTracker::updateFrameIMU(s, b, pCurrentKeyFrame);
    }

    bool EvAsynchTrackerU::estimateCurrMapIniPose(cv::Mat &Rb0w, cv::Mat &tb0w) {

        // retrieve the set of frames connecting the curr initial KF and prev last KF
        KeyFrame* pCurrIniKF = mpEvAtlas->GetCurrentMap()->GetOriginKF();
        if (!pCurrIniKF)
            return false;
        KeyFrame* pPrevLastKF = pCurrIniKF->mPrevKF;
        if (!pPrevLastKF || pPrevLastKF->GetMap() == pCurrIniKF->GetMap())
            return false;

        // Both maps must be inertially initialized
        if (!pCurrIniKF->GetMap()->isImuInitialized() || !pPrevLastKF->GetMap()->isImuInitialized()) {

            LOG(INFO) << "estimateCurrMapIniPose: Abort because at least one map is not inertially init.\n";
            return false;
        }

        EvFramePtr pCurrIniFr = nullptr;
        if (mmpWinFrames.find(pCurrIniKF->mnFrameId) != mmpWinFrames.end()) {
            pCurrIniFr = mmpWinFrames.find(pCurrIniKF->mnFrameId)->second;
        }
        if (!pCurrIniFr)
            return false;
        vector<EvFramePtr> vpConnFrames;
        vpConnFrames.reserve(10);
        vpConnFrames.push_back(pCurrIniFr);
        EvFramePtr pPrevLastFr = pCurrIniFr->getPrevFrame();

        while(pPrevLastFr && pPrevLastFr->mnId != pPrevLastKF->mnFrameId) {

            vpConnFrames.push_back(pPrevLastFr);
            pPrevLastFr = pPrevLastFr->getPrevFrame();
        }
        vpConnFrames.push_back(pPrevLastFr);

        if (!pPrevLastFr || pPrevLastFr->mnId != pPrevLastKF->mnFrameId) {

            if (pPrevLastFr)
                LOG(INFO) << "estimateCurrMapIniPose: Abort because last frame : keyframe mismatch: "
                          << pPrevLastFr->mnId << ", " << pPrevLastKF->mnFrameId << endl;
            return false;
        }

        pPrevLastFr->SetImuPoseVelocity(pPrevLastKF->GetImuRotation(), pPrevLastKF->GetImuPosition(), pPrevLastKF->GetVelocity());

        for (auto rIter = vpConnFrames.rbegin(); rIter != vpConnFrames.rend(); rIter++) {

            if (rIter == vpConnFrames.rbegin())
                continue;

            IMU_Manager::predictStateIMU((*rIter)->getPrevFrame().get(), (*rIter).get());
        }

        // this sends Rwb0 so reverse it
        Rb0w = vpConnFrames[0]->GetImuRotation().t();
        tb0w = -Rb0w * vpConnFrames[0]->GetImuPosition();

        // extract & return just the heading
        vector<float> euAngles = Converter::toEuler(Rb0w);
        LOG(INFO) << "estimateCurrMapIniPose: Euler angles: "
                  << euAngles[0] << ", " << euAngles[1] << ", " << euAngles[2] << endl;

        float psi = euAngles[2];
        Rb0w = cv::Mat::eye(3,3, CV_32F);
        Rb0w.at<float>(0,0) = cos(psi);
        Rb0w.at<float>(0,1) = -sin(psi);
        Rb0w.at<float>(1,0) = sin(psi);
        Rb0w.at<float>(1,1) = cos(psi);

        return true;
    }

    void EvAsynchTrackerU::fuseInertialMaps() {

        // estimate initial pose from inertial integrations
        cv::Mat Rb0w = cv::Mat::eye(3,3, CV_32F);
        cv::Mat tb0w = cv::Mat::zeros(3,1, CV_32F);
        bool res = estimateCurrMapIniPose(Rb0w, tb0w);

        if (!res) {
            LOG(INFO) << "EvAsynchTrackerU::fuseInertialMaps: Can't retrieve merge transformation\n";
            return;
        }

        // transform active map
        {
            unique_lock<mutex> lock(mpEvAtlas->GetCurrentMap()->mMutexMapUpdate);
            mpEvAtlas->GetCurrentMap()->ApplyScaledRotation(Rb0w, 1.f, false, tb0w);
        }

        // local IBA optimization to enhance the result

        bool b_doneLBA = false, bAbortBA = false;
        int num_FixedKF_BA = 0;
        int num_OptKF_BA = 0;
        int num_MPs_BA = 0;
        int num_edges_BA = 0;
        KeyFrame* pCurrKF = mpEvAtlas->GetCurrentMap()->GetOriginKF();
        for (size_t i = 0; i < 5; i++) {
            KeyFrame* pNextKF = pCurrKF->mNextKF;
            if (!pNextKF) {
                break;
            }
            else {
                pCurrKF = pNextKF;
                pCurrKF->mnBALocalForKF = 0;
            }
        }

        MyOptimizer::LocalInertialBA(pCurrKF, &bAbortBA, pCurrKF->GetMap(), num_FixedKF_BA,num_OptKF_BA,
                                     num_MPs_BA,num_edges_BA, 10, 10, false,
                                     !pCurrKF->GetMap()->GetIniertialBA2(), true);
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    void EvAsynchTrackerU::trackLastFeatures(const PoseImagePtr &pImage) { // changes tracks

        // return if feature tracks are not initialized
        if (!this->isTrackerInitialized()) {
            return;
        }

        // retrieve last features
        vector<KeyPoint> vAllKPts;
        vector<MapPoint*> vMapPoints;
        vector<FeatureTrackPtr> vFtTracks;
        ulong lastFrameId = mpLastFrame->mnId;
        unsigned nTracks = this->getTrackedPoints(lastFrameId, vAllKPts, vMapPoints, vFtTracks);

        if (nTracks <= 0 || vAllKPts.empty()) {
            return;
        }

        // init. LK Tracker
        // TODO: With this we reset valuable last ft. tracks info every time!
        mpLKTracker->setRefImage(mpLastFrame->imgLeft, vAllKPts);

        // track
        mvLastMatches.clear();
        mnLastTrackedKPts = mpLKTracker->trackAndMatchCurrImage(pImage->mImage, vAllKPts, mvLastMatches);

        // process tracked points
        ulong currFrameIdx = (pImage->mReconstStat == 0) ? lastFrameId : mFrameIdx;
        this->updateTrackedPoints(currFrameIdx, vAllKPts, mvLastMatches, vFtTracks);

        // update track stats
        this->updateTrackStats();
    }

    void EvAsynchTrackerU::estimateCurrentPose(const PoseImagePtr &pImage) {

        // check if there is actually a map to track
        if (!this->isMapInitialized() || !mpCurrFrame) {
            return;
        }

        // predict & init. current pose
        this->predictNextPose(mpCurrFrame);

        // optimize
        Optimizer::PoseOptimization(mpCurrFrame.get());

        // post processing
        mnLastTrackedMPts = this->refineTrackedMPs(mpCurrFrame);

        // if not enough inliers, abort the optimization in favor of inertial pose estimation
//        if (mnLastTrackedMPts < DEF_MIN_TRACKED_PTS3D_TLM) {
//            this->predictNextPose(mpCurrFrame);
//            mpLocalMapper->setOptimizeFlag(false);
//        }
//        else {
//            mpLocalMapper->setOptimizeFlag(true);
//        }
        mbSetLastPose = true;
    }

    void EvAsynchTrackerU::detectAndFuseNewFeatures(const PoseImagePtr& pImage) {

        // detect new feature points
        ulong currFrameId = mFrameIdx; // mpCurrFrame->mnId;
        this->makeFrame(pImage, currFrameId, mpIniFrameTemp);

        // retrieve points
        vector<cv::KeyPoint> vkpts0, vkpts1, vNewKpts;
        vector<FeatureTrackPtr> vpFtTracks;
        vector<MapPoint*> vpMpts;

        vkpts1 = mpIniFrameTemp->getAllUndistKPtsMono();

        if (this->isTrackerInitialized()) {

            this->getTrackedPoints(currFrameId, vkpts0, vpMpts, vpFtTracks);

            // select new key points uniformly distributed across image
            cv::Size imSize(imWidth, imHeight);
            const int patchSize = DEF_GRID_PATCH_SIZE;
            FeatureTrack::selectNewKPtsUniform(vkpts0, vkpts1, imSize, patchSize, mnEvPts, vNewKpts, true);
        }
        else {
            // much better results for EvMVSEC dataset
            vNewKpts = vkpts1;
        }

        // create & merge new tracks
        for (const auto & kpt : vNewKpts) {

            FeatureTrackPtr pNewTrack = make_shared<FeatureTrack>(mFtTrackIdx, currFrameId, kpt);
            mmFeatureTracks.insert(make_pair(mFtTrackIdx, pNewTrack));
            mFtTrackIdx++;
        }

        // update stats
        mnFeatureTracks = mmFeatureTracks.size();
        mnCurrFeatureTracks = mnFeatureTracks;

        if (!this->isTrackerInitialized()) {
            return;
        }

        // remove bad tracks
        for (auto trackItr = mmFeatureTracks.begin(), endTrack = mmFeatureTracks.end(); trackItr != endTrack;) {

            FeatureTrackPtr pTrack = trackItr->second;

            if (pTrack && !pTrack->isValid())
                pTrack->removeTrack();

            if (!pTrack || !pTrack->isValid()) {
                trackItr = mmFeatureTracks.erase(trackItr);
            }
            else {
                trackItr++;
            }
        }

        // remove old frames
        ulong minFrId = -1, maxFrId = 0;
        this->findTracksBounds(minFrId, maxFrId);
        this->removeOldFrames(minFrId);
    }

    void EvAsynchTrackerU::detectAndFuseNewMapPoints() {

        if (!this->isTrackerInitialized() || !this->isMapInitialized() || mnLastTrackedMPts > DEF_MIN_TRACKED_PTS3D_LOW) {
            return;
        }

        //this->triangulateNewMapPoints();
    }

    void EvAsynchTrackerU::checkTrackedMapPoints() {

        if (!this->isTrackerInitialized() || !this->isMapInitialized() || mnLastTrackedMPts > DEF_MIN_TRACKED_PTS3D_LOW) {
            return;
        }

        // in event-only case, if num tracked mpts is really low,
        // we can't help it -> reset to initialization
        if (mnLastTrackedMPts <= DEF_MIN_TRACKED_PTS3D_LOW && this->isDisconnGraph()) {

            LOG(INFO) << "Tracking lost because disconnected graph and tracked mps: " << mnLastTrackedMPts << endl;

            if (mpSensor->isInertial() && mpEvAtlas->GetAllMaps().size() > 1) {

                //LOG(INFO) << "Inertial case with Tracking Mode 2: Fuse inertial maps\n";
                //this->fuseInertialMaps();
            }

            this->resetTracker();
        }
    }

    void EvAsynchTrackerU::reconstIniMap(const PoseImagePtr& pImage) {

        if (!this->isTrackerInitialized() || this->isMapInitialized() || !mpIniFrame || !mpCurrFrame) {
            return;
        }

        // retrieve frame info.
        vector<KeyPoint> p0 = mpIniFrame->getAllUndistKPtsMono();
        vector<KeyPoint> p1 = mpCurrFrame->getAllUndistKPtsMono();
        vector<int> vMatches12 = mpCurrFrame->getMatches();
        int nMatches = mpCurrFrame->getNumMatches();
        size_t nLenMch = vMatches12.size();

        bool abortInit = false;
        if (p0.size() < DEF_TH_MIN_KPTS || nMatches < DEF_TH_MIN_MATCHES) {
            abortInit = true;
        }

        if (!abortInit) {
            // scene reconstruction
            vector<cv::Mat> Rl1, tl1;
            vector<vector<cv::Point3f>> pts3d;
            vector<bool> vbReconstStat(nLenMch, false);
            vector<vector<bool>> vbTriangStat;
            ORB_SLAM3::ReconstInfo reconstInfo;

            bool reconstRes = mpCamera->ReconstructWithTwoViews(p0, p1, vMatches12, Rl1, tl1, pts3d,
                                                                vbTriangStat, vbReconstStat, reconstInfo);

            float medPxDisp = calcMedPxDisplacement(p0, p1, vMatches12, mvLastFtPxDisp);
            reconstRes = resolveReconstStat(reconstRes, medPxDisp, reconstInfo, pts3d, vMatches12);

            // reconst. successful? -> init. map if yes
            if (reconstRes) {
                // Refine matches and reject outliers
                for (size_t i = 0; i < nLenMch; i++) {

                    if (vMatches12[i] >= 0 && !vbTriangStat[0][i]) {

                        vMatches12[i] = -1;
                        nMatches--;
                    }
                }
                mpCurrFrame->setMatches(vMatches12);

                cv::Mat currTcw = ORB_SLAM3::Converter::toCvSE3(Rl1[0], tl1[0]);
                mvPts3D = pts3d[0];
                mpIniFrame->SetPose(cv::Mat::eye(4, 4, CV_32F));
                mpCurrFrame->SetPose(currTcw);

                this->initMap();

                vector<MapPoint *> vMPts = mpLastKF->GetMapPointMatches();
                this->assignNewMPtsToTracks(mpCurrFrame->mnId, vMPts, vMatches12, mpCurrFrame->getAllFeatureTracks());

                for (auto pMpt : vMPts) {
                    if (pMpt)
                        mlpRecentAddedMapPoints.push_back(pMpt);
                }

                mnFirstFrameId = mpLastKF->mPrevKF->mnFrameId;
                mpReferenceKF = mpLastKF;
                mnLastTrackedMPts = mlpRecentAddedMapPoints.size();
                mbSetLastPose = true;
            }
                // if no, change ref. frame if low tracked points!
            else if (reconstInfo.mnBestGood < DEF_TH_MIN_MATCHES) {

                abortInit = true;
            }
        }

        if (abortInit) {
            //this->changeReference(mpIniFrame->mnId+1);
            mpIniFrame = mpCurrFrame;
        }
    }

    void EvAsynchTrackerU::localMapping() {

        // if map not initialized or just init. abort
        if (!this->isMapInitialized() || !mpCurrFrame || !mpLastKF || mpCurrFrame->mnId == mpLastKF->mnFrameId) {
            return;
        }

        // retrieve current & ref. info.
        EvFramePtr pRefFrame = mmpWinFrames[mpLastKF->mnFrameId];
        vector<cv::KeyPoint> vkpts0 = pRefFrame->getAllUndistKPtsMono();
        vector<cv::KeyPoint> vkpts1 = mpCurrFrame->getAllUndistKPtsMono();
        vector<int> vMatches12 = mpCurrFrame->getMatches();

        const float medFtDisp = calcMedPxDisplacement(vkpts0, vkpts1, vMatches12, mvLastFtPxDisp);

        if (this->needNewKeyFrame(medFtDisp)) {

            // create new key frame if required
            this->addNewKeyFrame(mpCurrFrame);

            // refresh the connections
            EvLocalMapping::processNewKeyFrame(mpEvAtlas, mpLastKF, mlpRecentAddedMapPoints);

            // map point culling? -> partially done in detect & merge new tracks
            EvLocalMapping::mapPointCullingCovis(mpEvAtlas, mpLastKF, mpCurrFrame, mlpRecentAddedMapPoints, mpSensor->isMonocular());

            // triangulate new map points
            this->triangulateNewMapPoints(pRefFrame, mpCurrFrame);

            // send KF to local mapping, only process one key frame at a time
            const bool isInertial = mpSensor->isInertial() && mpImuManager;
            mTrackingWD.reset();
            while (!mpLocalMapper->isReadyForNewKeyFrame() && (!isInertial || !mpImuManager->isInitializing())) { // beware of deadlocks

                //LOG_EVERY_N(WARNING, 1000) << "L2 TrackerU Local Mapping: Waiting for local mapper: "
                //                              << mpCurrFrame->mTimeStamp << endl;
                std::this_thread::sleep_for(std::chrono::microseconds(100));

#ifdef ACTIVATE_WATCH_DOG
                mTrackingWD.step();
#endif
                if (mTrackingWD.getWaitTimeSec() > 1.5f) {
                    mpLocalMapper->abortLBA();
                }
            }
            mpLocalMapper->insertNewKeyFrame(mpLastKF);
        }
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    void EvAsynchTrackerU::Track() {

        while (!this->isStopped()) {

            if (isInputGood()) {

                mTrackingTimer.tic();

                mbIsProcessing = true;

                if (mStat == IDLE) {
                    mStat.update(INIT_TRACK);
                }

                mbSetLastPose = false;
                bool abortIter = false;

                PoseImagePtr pPoseImage = this->frontImage();
                double imTs = pPoseImage->ts;

                // Check image order
                if (mpCurrFrame && imTs < mpCurrFrame->mTimeStamp) {
                    LOG(WARNING) << "EvAsynchTracker::Track: Lower new image ts: "
                               << imTs << " <= " << mpCurrFrame->mTimeStamp << endl;
                    abortIter = true;
                }

                // Check image is valid
                if (pPoseImage->mImage.empty()) {
                    LOG(WARNING) << "* L2 Asynch. Tracker, " << imTs << ": Empty image\n";
                    abortIter = true;
                }

                if (abortIter) {
                    this->popImage();
                    continue;
                }

                // Track last features
                this->trackLastFeatures(pPoseImage);

                if (pPoseImage->mReconstStat != 0) {

                    // Let the IMU init. be completed
                    if (mpSensor->isInertial() && mpImuManager) {
                        while (mpImuManager->isInitializing()) {

                            LOG_EVERY_N(WARNING, 1000) << "L2 TrackerU: Waiting for IMU manager to init.: " << imTs << endl;
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

                    // Match curr. feature tracks against the reference
                    this->matchCurrentFrame(mpCurrFrame);

                    // Track local map to estimate current pose
                    this->estimateCurrentPose(pPoseImage);

                    // KF Generation and MP Triangulation (this must be done here not local mapping)
                    this->localMapping();

                    // Initialize local map if not reconst. before
                    this->reconstIniMap(pPoseImage);

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

                mbIsProcessing = false;

                mTrackingTimer.toc();
                mTrackingTimer.push();
            }
            if (mStat == IDLE) {

                VLOG_EVERY_N(3, 1000) << "L2: Idle state...\n";
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }

            if (!this->sepThread) { break; }
        }
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    float EvAsynchTrackerU::calcMedPxDisplacement(const std::vector<cv::KeyPoint> &vp0, const std::vector<cv::KeyPoint> &vp1,
                                                 std::vector<int> &vMatches12, std::vector<float> &vMedPxDisp) {

        const size_t p0Size = vp0.size();
        assert(p0Size == vMatches12.size());

        vMedPxDisp.clear();
        vMedPxDisp.reserve(p0Size);

        for (size_t i = 0; i < p0Size; i++) {

            int currMatch = vMatches12[i];

            if (currMatch >= 0) {

                cv::KeyPoint p0 = vp0[i];
                cv::KeyPoint p1 = vp1[currMatch];

                float medDisp = sqrtf(powf(p0.pt.x-p1.pt.x, 2)+powf(p0.pt.y-p1.pt.y, 2));
                vMedPxDisp.push_back(medDisp);
            }
        }

        float medPxDisp = 0.f;
        if (!vMedPxDisp.empty()) {
            sort(vMedPxDisp.begin(), vMedPxDisp.end());
            medPxDisp = vMedPxDisp[vMedPxDisp.size() / 2];
        }
        return medPxDisp;
    }

    void EvAsynchTrackerU::updateFrameDrawer() {

        if (!mpViewer || !mpFrameDrawer) {
            return;
        }

        if (this->isMapInitialized()) {
            mpFrameDrawer->updateLastTrackerState(mFrDrawerId, EORB_SLAM::FrameDrawerState::TRACKING);
            mpFrameDrawer->pushNewFrame(mFrDrawerId, mpCurrFrame);
        }
        else if (this->isTrackerInitialized() && mpCurrFrame) {

            mpFrameDrawer->updateLastTrackerState(mFrDrawerId, EORB_SLAM::FrameDrawerState::INIT_MAP);
            mpFrameDrawer->updateIniFrame(mFrDrawerId, mpIniFrame);
            mpFrameDrawer->pushNewFrame(mFrDrawerId, mpCurrFrame);
        }
        else if (this->isTrackerInitialized()){

            mpFrameDrawer->updateLastTrackerState(mFrDrawerId, EORB_SLAM::FrameDrawerState::INIT_TRACK);
            mpFrameDrawer->pushNewFrame(mFrDrawerId, mpCurrFrame);
        }
        else {
            mpFrameDrawer->updateLastTrackerState(mFrDrawerId, EORB_SLAM::FrameDrawerState::IDLE);
        }
    }

    /*float EvAsynchTrackerU::getAverageTrackingTime() {

        if (mvTrackingTimes.empty())
            return 0.0;

        size_t nTotTTrack = mvTrackingTimes.size();
        return static_cast<float>(1.0 * std::accumulate(mvTrackingTimes.begin(), mvTrackingTimes.end(), 0.0) / nTotTTrack);
    }*/

} // EORB_SLAM
