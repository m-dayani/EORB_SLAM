//
// Created by root on 1/9/21.
//
/**
 * 1. Seems like using a single Camera* for all tracking threads
 *    introduces ugly exceptions!
 * 2. TODO: We can use L1 pose to choose between init. motion models.
 * 3. Most methods for reinitializing weak track are almost the same,
 *    Major difference is how they treat newly added KFs and MPs.
 * 4. Two most importance reinitialization are:
 *    a) Renew reference points by both tracking new images with old LK tracker (to find currPose)
 *       and trying to init. map using newly detected points and estimated poses.
 *    b) Find new poses like above but completely reInit new map based on
 *       epipolar geometry.
 */

#include <utility>

#include "EvAsynchTracker.h"
//#include "EvTrackManager.h"

#include "G2oTypes.h"
#include "Optimizer.h"
#include "MyOptimizer.h"
#include "EvLocalMapping.h"

using namespace std;
using namespace cv;
using namespace ORB_SLAM3;

namespace EORB_SLAM {

    unsigned long EvAsynchTracker::mFrameIdx = 0, EvAsynchTracker::mKeyFrameIdx = 0, EvAsynchTracker::mMapIdx = 0;

    EvAsynchTracker::EvAsynchTracker(shared_ptr<EvParams> evParams, CamParamsPtr camParams, const MixedFtsParamsPtr& pFtParams,
            std::shared_ptr<ORB_SLAM3::Atlas> pEvAtlas, ORB_SLAM3::ORBVocabulary* pVocab, SensorConfigPtr  sConf,
            ORB_SLAM3::Viewer* pViewer) :

            EvBaseTracker(std::move(evParams), std::move(camParams)), mStat(), //mTempStat(INIT_TRACK),
            mTrackMode(static_cast<TrackMode>(mpEvParams->l2TrackMode % 3)), mpSensor(std::move(sConf)),
            mpIniFrame(nullptr), mpIniFrameTemp(nullptr), mpLastFrame(nullptr), mpCurrFrame(nullptr),
            mbMapInitialized(false), mbImuInitialized(false), mBadInitCnt(0),
            mpReferenceKF(nullptr), mpRefKfTemp(nullptr), mpLastKF(nullptr), mpEvAtlas(std::move(pEvAtlas)),
            mpViewer(pViewer), mMaxDistRefreshPts(mpEvParams->kltMaxDistRefreshPts),
            mpTrackManager(nullptr), mnMatchesInliers(0), mbScaledTracking(false)
    {
        mpPoseInfo = make_shared<PoseDepthInfo>();

        mImSigma = mpEvParams->l2ImSigma;

        mnEvPts *= 2;
        DLOG(INFO) << "L2 Num. Features is set to " << mnEvPts << endl;

        if (mpSensor->isImage()) {
            mFtDtMode = 1;
            mbScaledTracking = true;
        }

        cv::Size imSize(mpEvParams->imWidth, mpEvParams->imHeight);
        ORB_SLAM3::ORBxParams parORB(mnEvPts, mpEvParams->l2ScaleFactor, mpEvParams->l2NLevels, mThFAST,
                                     0, pFtParams->imMargin, imSize);
        this->initFeatureExtractors(mFtDtMode, parORB);
        if (mFtDtMode == 1) {
            mpORBLevelDetector = make_shared<ORB_SLAM3::ORBextractor>(parORB);
        }

        mpLKTracker = unique_ptr<ELK_Tracker>(new ELK_Tracker(mpEvParams.get()));
        mpLKTrackerKF = unique_ptr<ELK_Tracker>(new ELK_Tracker(mpEvParams.get()));
        //mpLKTrackerTemp = unique_ptr<ELK_Tracker>(new ELK_Tracker(mpEvParams.get()));

        mpKeyFrameDB = unique_ptr<ORB_SLAM3::KeyFrameDatabase>(new ORB_SLAM3::KeyFrameDatabase(*pVocab));

        // Reconstruct a fine tuned camera
        ORB_SLAM3::Params2VR params2VR;
        params2VR.mThMaxGoodR_H = 0.85f;
        params2VR.mThMaxGoodR_F = 0.8f;
        auto* mpTVR = new ORB_SLAM3::TwoViewReconstruction(mK, params2VR);
        // Use a calibrated camera model if events are rectified:
        MyParameters::createCameraObject(*(mpCamParams.get()), mpTVR, mpCamera,
                MyParameters::dummyCamera, mpEvParams->isRectified);

        mpLocalMapper = std::unique_ptr<EvLocalMapping>(new EvLocalMapping(this, mpEvAtlas,
                mpSensor, mpCamParams.get(), mpImuManager));
        mptLocalMapping = new thread(&EvLocalMapping::Run, mpLocalMapper.get());

        if (mpViewer)
            this->setFrameDrawerChannel(mpViewer->getFrameDrawer());
    }

    EvAsynchTracker::~EvAsynchTracker() {

        // All key frames must be deleted in system when destroyed
        mpReferenceKF = nullptr;
        mpLastKF = nullptr;
        for (auto& pkf : mvpLocalKeyFrames) {
            delete pkf;
            pkf = nullptr;
        }
        for (auto& pMP : mvpLocalMapPoints) {
            delete pMP;
            pMP = nullptr;
        }
        mpTrackManager = nullptr;
//        if (mpCamParams->isPinhole()) {
//            delete mpCamera;
//        }
        delete mpCamera;
        mpCamera = nullptr;
        mpLocalMapper->stop();
        mptLocalMapping->join();
        delete mptLocalMapping;
        mptLocalMapping = nullptr;
        mpViewer = nullptr;
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    bool EvAsynchTracker::isTrackerReady() {

        return !this->isTrackerSaturated();
    }

    bool EvAsynchTracker::isTrackerSaturated() {

        return this->isImageBuffSaturated(DEF_MAX_IM_BUFF_CAP);
    }

    bool EvAsynchTracker::isInputGood() {

        return this->isImageBuffGood();
    }

    void EvAsynchTracker::stop() {

        //this->mpThreadLocalBA->join();
        this->mStat.update(STOP);
    }

    void EvAsynchTracker::resetAll() {

        DLOG(INFO) << "EvAsynchTracker::resetAll >> Stats:\n"
                   << "\tInput: -Num. EvImages: " << mqImageBuff.size() << ", last ts: ?\n"
                   //<< ((!mqImageBuff.isEmpty()) ? to_string(std::prev(mqImageBuff.end())->first.second) : "??") << endl
                   << "\tOutput: -Num. ref. key frames: " << mvpRefKeyFrames.size() << endl
                   << "\t\t-Num. maps in atlas: " << mpEvAtlas->GetAllMaps().size() << endl
                   << "\t\t-Num. all key frames: " << mpEvAtlas->GetAllMaps().size() << endl
                   << "\t\t-Num. map points: " << mpEvAtlas->GetAllMapPoints().size() << endl;

        // Empty input image queue
        EvBaseTracker::resetAll();

        mStat.update(IDLE);

        mpIniFrame = nullptr;
        mpIniFrameTemp = mpIniFrame;
        mpCurrFrame = nullptr;
        mpLastFrame = nullptr;

        mbTrackerInitialized = false;
        mbMapInitialized = false;
        mbImuInitialized = false;
        mBadInitCnt = 0;

        mFrameIdx = 0;
        mKeyFrameIdx = 0;

        //mpPoseInfo->reset(); //-> not different from bellow
        mpPoseInfo = make_shared<PoseDepthInfo>();

        mpReferenceKF = nullptr;
        mpRefKfTemp = nullptr;
        mpLastKF = nullptr;

        // reset local mapper
        while (mpLocalMapper->isProcessing())
            std::this_thread::sleep_for(std::chrono::milliseconds (3));
        mpLocalMapper->resetAll();

        mMapIdx = 0;
        mpKeyFrameDB->clear();
        mpEvAtlas->clearMap();
        mpEvAtlas->clearAtlas();
        mpEvAtlas->CreateNewMap((int)mMapIdx++);

        mnMatchesInliers = 0;

        try {
            if (!mvpRefFrames.empty()) {
                mvpRefFrames.clear();
            }
            if (!mvPts3D.empty()) {
                mvPts3D.clear();
            }
            if (!mvpLocalKeyFrames.empty()) {
                mvpLocalKeyFrames.clear();
            }
            if (!mvpLocalMapPoints.empty()) {
                mvpLocalMapPoints.clear();
            }
            {
                std::unique_lock<mutex> lock1(mMtxVpRefKFs);

                for (KeyFrame* pRefKF : mvpRefKeyFrames) {

                    // delete map points
                    for (MapPoint* pMP : pRefKF->GetMapPoints()) {
                        delete pMP;
                        pMP = static_cast<MapPoint*>(nullptr);
                    }

                    // delete all children
                    for (KeyFrame* pKF : pRefKF->GetChilds()) {
                        delete pKF;
                        pKF = static_cast<KeyFrame*>(nullptr);
                    }

                    delete pRefKF;
                    pRefKF = static_cast<KeyFrame*>(nullptr);
                }
                mvpRefKeyFrames.clear();
            }
        }
        catch (std::bad_alloc& e) {
            LOG(ERROR) << "EvImBuilder::reset, " << e.what() << endl;
        }
        catch (Exception& e) {
            LOG(ERROR) << "EvImBuilder::reset, " << e.what() << endl;
        }
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    void EvAsynchTracker::reset() {

        this->resetInitMapState();

        //mFrameIdx = 0;
        //mKeyFrameIdx = 0;
        //mMapIdx = 0;

        //mpEvAtlas->clearMap();
        //mpEvAtlas->clearAtlas();
        //mpEvAtlas->CreateNewMap((int)mMapIdx++);

//        mvpRefFrames.clear();
//        for (ORB_SLAM3::MapPoint* MP : mvpLocalMapPoints) {
//            delete MP;
//            MP = nullptr;
//        }
//        mvpLocalMapPoints.clear();
//
//        for (ORB_SLAM3::KeyFrame* KF : mvpLocalKeyFrames) {
//            delete KF;
//            KF = nullptr;
//        }
//        mvpLocalKeyFrames.clear();
    }

    void EvAsynchTracker::resetInitMapState() {

        mbMapInitialized = false;
        mBadInitCnt = 0;
    }

    void EvAsynchTracker::resetActiveMap()
    {
        Map* pMap = mpEvAtlas->GetCurrentMap();
        vector<KeyFrame*> allKFs = pMap->GetAllKeyFrames();
        int nAllKFs = allKFs.size();

        // Clear local key frame holders
        if (nAllKFs > 0) {
            std::unique_lock<mutex> lock1(mMtxVpRefKFs);
            mvpRefKeyFrames.pop_back();
        }
        for (int i = 0; i < nAllKFs; i++) {
            // TODO: This is wrong!!
            mvpLocalKeyFrames.pop_back();
        }

        // Clear BoW Database
        DVLOG(1) << "Reseting Database\n";
        mpKeyFrameDB->clearMap(pMap); // Only clear the active map references
        DVLOG(1) << "Reseting Database Done\n";

        // Clear Map (this will erase MapPoints and KeyFrames)
        mpEvAtlas->clearMap();

        //mpCurrFrame = EvFrame();
        //mpLastFrame = EvFrame();
        //mpReferenceKF = static_cast<KeyFrame*>(nullptr);
        //mpLastKF = static_cast<KeyFrame*>(nullptr);
        //mvIniMatches.clear();

        this->resetInitMapState();

        DVLOG(1) << "   End reseting!\n";
    }

    void EvAsynchTracker::resetKeyFrameTracker(const PoseImagePtr& pImage) {

        this->makeFrame(pImage, mFrameIdx++, mpIniFrameTemp);
        // TODO: Also copy data like Tcw and IMU from currFrame???
        if (mpLastKF) {
            mpIniFrameTemp->SetPose(mpLastKF->GetPose());
        }
        mpLKTrackerKF->setRefImage(pImage->mImage, mpIniFrameTemp->getAllUndistKPtsMono());
    }

    bool EvAsynchTracker::resolveReconstStat(const bool reconstRes, const float& medPxDisp,
            const ORB_SLAM3::ReconstInfo &rInfo, const std::vector<std::vector<cv::Point3f>> &pts3d,
            std::vector<int>& vMatches12) {

        // If successful reconstruction, return true
        if (reconstRes) {
            DLOG(INFO) << "EvAsynchTracker::resolveReconstStat: Good Reconst. by original response.\n";
            return reconstRes;
        }
        // Else if the median pixel displacement is good,
        // we have good parallex but reconstRes is false (because of 2 similar found Transformations)
        // This mostly happen when we have homography and small images! -> why?????
        else if (medPxDisp > DEF_TH_MED_PX_DISP) {

            // In this case, both the number of matches and parallax must be good
            DVLOG(1) << "EvAsynchTracker::resolveReconstStat: ReconstInfo: " << rInfo.print();

            bool preCond = rInfo.mnBestGood > rInfo.mnSecondBest && rInfo.mBestParallax > rInfo.mSecondBestPar &&
                   rInfo.mnBestGood > DEF_TH_MIN_MATCHES && rInfo.mBestParallax > DEF_MIN_PARALLAX;

            if (!preCond) {
                DLOG(INFO) << "EvAsynchTracker::resolveReconstStat: Bad Reconst. -> pre-condition is not fulfilled\n";
                return preCond;
            }
            else {
                return isReconstDepthGood(pts3d, vMatches12);
            }
        }
        else {
            DLOG(INFO) << "EvAsynchTracker::resolveReconstStat: Bad Reconst. -> cannot choose a good solution!\n";
            return reconstRes;
        }
    }

    // Decide based on median pixel displacement
    bool EvAsynchTracker::needNewKeyFrame(const cv::Mat& image) {

        // LK Optical Flow matching
        vector<KeyPoint> p1;
        vector<int> vMatches12;
        vector<int> vCntMatches;
        vector<float> vPxDisp;

        mpLKTrackerKF->trackAndMatchCurrImage(image, p1, vMatches12, vCntMatches, vPxDisp);

        sort(vPxDisp.begin(), vPxDisp.end());
        float medPxDisp = vPxDisp[vPxDisp.size()/2];

        return medPxDisp > DEF_TH_MED_PX_DISP;
    }

    /*bool EvAsynchTracker::isCurrPoseGood(const double& ts, int& stat) {

        stat = 0;
        // Current frame must match the image (ts) and currPose is estimates
        bool goodCurrFrame = abs(ts-mpCurrFrame.mTimeStamp) < 1e-9 && !mpCurrFrame.mTcw.empty();
        if (!goodCurrFrame) {
            stat = 1;
            return false;
        }
        // Last Key frame must not be a reference key frame
        // This means that the tracked is lost
        bool isLastKfEqRefKf = mpLastKF == mpReferenceKF;
        if (isLastKfEqRefKf) {
            stat = 2;
            return false;
        }
        // Current frame must not be the same as last key frame
        // This should not happen
        bool isCurrFrameEqLastKF = abs(ts-mpLastKF->mTimeStamp) < 1e-2;
        if (isCurrFrameEqLastKF) {
            stat = 3;
            return false;
        }
        // TODO: Maybe even check for good parallax (medPxDisp)
        return true;
    }*/

    void EvAsynchTracker::prepReInitDisconnGraph() {

        vector<KeyFrame*> vpCurrKfs = mpEvAtlas->GetCurrentMap()->GetAllKeyFrames();
        if (vpCurrKfs.empty()) {

            LOG(INFO) << "* L2 Tracking: Current map is already empty, nothing to do.\n";
            return;
        }
        // If not much KFs in current map, clear map and retry
        else if (vpCurrKfs.size() < DEF_MIN_KFS_IN_MAP) {

            // Do this only when tracking local map!
            if (!(mTrackMode == ODOMETRY)) {

                LOG(WARNING) << "* L2 Tracking: Not enough key frames in current map, resetting active amp.\n";
                // Reset current map and reInit.
                this->resetActiveMap();
            }
        }
        // Else, create a new map and try to reinit.
        else {
            LOG(INFO) << "* L2 Tracking: Creating new map #" << mMapIdx << endl;
            // Create a new map
            mpEvAtlas->CreateNewMap((int)mMapIdx++);
            this->resetInitMapState();
        }
    }

    void EvAsynchTracker::backupLastRefInfo() {

        mpIniFrame->setMatches(mpCurrFrame->getMatches());
        mpIniFrame->setDistTrackedPts(mpCurrFrame->getDistTrackedPts());
        mvpRefFrames.push_back(mpIniFrame);
        // Insist keeping ref. image
        mvpRefFrames.back()->imgLeft = mpIniFrame->imgLeft.clone();
        // + mLastKF????
    }

    void EvAsynchTracker::restoreLastRefInfo(EvFramePtr& lastRefFrame, ORB_SLAM3::KeyFrame* pLastRefKF,
            ORB_SLAM3::KeyFrame* pLastKF) {

        lastRefFrame = mvpRefFrames.back();
        lastRefFrame->imgLeft = mvpRefFrames.back()->imgLeft.clone();
        pLastRefKF = lastRefFrame->mpReferenceKF;
        pLastKF = *(pLastRefKF->GetChilds().end());
    }

    void EvAsynchTracker::resetToInitTracking(SharedState<TrackState> &currState, const SharedState<TrackMode>& currTM) {

        const bool inertialMap = mpSensor->isInertial() && mpImuManager && mbImuInitialized;

        if (currState == INIT_TRACK) {

            // Reset last Frame (for proper preInt.)
            if (inertialMap && mpIniFrame->getPrevFrame()) {
                mpLastFrame = mpIniFrame->getPrevFrame();
            }
            else {
                this->prepReInitDisconnGraph();
            }
        }
        else if (currState == INIT_MAP) {

            if (currTM.getState() == ODOMETRY) {

                // Reset Init. State
                this->resetInitMapState();
                // Backup last Ref. state
                this->backupLastRefInfo();
                // Do not merge (not very useful, we only need relative constraints
                // this->mergeRefKeyFrames(image, imTs)

                //VLOG_EVERY_N(2, 1000) << "* L2 Only Tracking: " << imTs << ", Beginning next round,\n";
            }
            else {
                if (inertialMap && mpIniFrame->imuIsPreintegrated()) {

                    // Get the KF preInt. data (from IniFrame until currFrame)
                    mpCurrFrame->mpImuPreintegrated = mpImuManager->getImuPreintegratedFromLastKF(mnImuID);
                    // Merge preIntegration for more accurate relative pose
                    mpCurrFrame->mpImuPreintegrated->MergePrevious(mpIniFrame->mpImuPreintegrated);
                    // TODO: Create a new key frame???
                    mpLastFrame = mpCurrFrame;
                    mpLastFrame->setPrevFrame(mpIniFrame->getPrevFrame());
                    // Estimate the pose of curr/last frame based on preint. data from the prev. frame of IniFrame
                    mpImuManager->predictStateIMU(mpIniFrame->getPrevFrame().get(), mpLastFrame.get());
                    // Refresh KF preintegration tracker
                    mpImuManager->refreshPreintFromLastKF(mnImuID);
                }
                else {
                    // Connection is broken, prepare new reInit.
                    this->prepReInitDisconnGraph();
                }
            }

        }
        else if (currState == TRACKING) {

            if (!inertialMap) {
                this->resetInitMapState();
            }
            this->backupLastRefInfo();
            //mpIniFrameTemp = mpLastFrame;
        }
        else {
            DLOG(WARNING) << "EvAsynchTracker::resetToInitTracking: Current State Not Supported\n";
        }

        currState.update(INIT_TRACK);
    }

    void EvAsynchTracker::changeStateToInitMap(SharedState<TrackState> &currState) {

        // If inertial, stop KF preInt. from accumulating measurements
        if (mpSensor->isInertial() && mpImuManager && mbImuInitialized && mpIniFrame->getPrevFrame()) {

            // This is required because not always IniFrame contains the preInt. data (multiple resets)
            mpIniFrame->mpImuPreintegrated = mpImuManager->getImuPreintegratedFromLastKF(mnImuID);
            // Try to update the pose based on preint. data
            mpImuManager->predictStateIMU(mpIniFrame->getPrevFrame().get(), mpIniFrame.get());
            // If current ref. is similar to last frame, merge them. Don't change the place of this line!
            //mergeSimilarFrames(mpIniFrame);

            mpImuManager->refreshPreintFromLastKF(mnImuID);
        }
        currState.update(INIT_MAP);
    }

    void EvAsynchTracker::changeStateToTLM(SharedState<TrackState> &currState, const PoseImagePtr& pImage) {

        // Try stitch new map with the old one -> Don't!
        //if (!mvpRefFrames.empty()) {
        //this->mergeMaps(image, refP1, refMatches);
        //bool resMergeMaps = this->mergeRefKeyFrames(image, imTs);
        //}

        this->resetKeyFrameTracker(pImage);
        currState.update(TRACKING);

        VLOG(2) << "* L2 Feature tracking, " << pImage->ts
                           << ": Init. local map, ready for tracking\n";
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    void EvAsynchTracker::makeFrame(const PoseImagePtr& pImage, const unsigned fId, EvFramePtr &frame) {
        EvBaseTracker::makeFrame(pImage, fId, frame);
    }

    /*void EvAsynchTracker::makeFrame(const cv::Mat& im, const double ts, const unsigned fId, EvFramePtr &frame) {

        if (mFtDtMode == 1 || mFtDtMode == 2) {

            if (mpORBDetector) {
                if (mFtDtMode == 1) {
                    frame = make_shared<EvFrame>(fId, im, ts, mpORBDetector.get(), mpCamera, mDistCoefs);
                }
                else {
                    frame = make_shared<EvFrame>(fId, im, ts, mpFastDetector, mpORBDetector.get(), mnEvPts, mpCamera, mDistCoefs);
                }
            }
            else {
                LOG(ERROR) << "* L2 Init, " << ts << ": Feature detection mode set to ORB, but empty ORBDetector object.\n";
                frame = make_shared<EvFrame>();
                return;
            }
        }
        else {
            frame = make_shared<EvFrame>(fId, im, ts, mpFastDetector, mnEvPts, mpCamera, mDistCoefs);
        }
    }*/

    void EvAsynchTracker::makeFrame(const PoseImagePtr& pImage, const unsigned int fId,
                                const std::vector<cv::KeyPoint> &p1, EvFramePtr &frame) {

        frame = make_shared<EvFrame>(fId, pImage->mImage, pImage->ts, p1, mpCamera, mDistCoefs);
    }

    void EvAsynchTracker::makeFrame(const PoseImagePtr& pImage, const unsigned int fId,
                                    unique_ptr<ELK_Tracker> &pLKTracker, EvFramePtr &frame) {

        cv::Mat im = pImage->mImage.clone();
        double ts = pImage->ts;

        if (mStat == INIT_TRACK) {

            // Construct initial event frame
            this->makeFrame(pImage, fId, mpIniFrame);
            // Because we are using KLT Tracker, all reference frames must keep ref. image
            mpIniFrame->keepRefImage(im);
            frame = mpIniFrame;
        }
        else if (mStat == INIT_MAP) {

            // LK Optical Flow matching
            vector<KeyPoint> p1;
            vector<int> vMatches12;
            vector<int> vCntMatches; //= mvMatchesCnt;
            vector<float> vPxDisp;
            uint nMatches = 0;

            if (mFtDtMode == 0) { // Simple FAST
                nMatches = pLKTracker->trackAndMatchCurrImage(im, p1, vMatches12, vCntMatches, vPxDisp);
            }
            else {
                // ORB -> Track only first level key points!
                nMatches = pLKTracker->trackAndMatchCurrImageInit(im, p1, vMatches12, vCntMatches, vPxDisp);
            }

            sort(vPxDisp.begin(), vPxDisp.end());
            float medPxDisp = vPxDisp[vPxDisp.size()/2];

            this->makeFrame(pImage, fId, p1, frame);
            frame->setNumMatches(nMatches);
            frame->setMedianPxDisp(medPxDisp);
            frame->setMatches(vMatches12);
            //mvMatchesCnt = vCntMatches;
        }
        else if (mStat == TRACKING) {

            // Track reference frame (im0, p0) using
            // last tracked pts and current image -> find matches
            unsigned nMatches = this->trackAndFrame(pImage, fId, pLKTracker, frame);
            frame->setNumMatches(nMatches);
        }
        else {
            frame = make_shared<EvFrame>();
        }

        // Update State & Connections
        if (frame != mpIniFrame && mpIniFrame) {
            frame->setRefFrame(mpIniFrame);
        }
        if (mpLastFrame) {
            frame->setPrevFrame(mpLastFrame);
        }
        if (mpLastKF) {
            frame->mpLastKeyFrame = mpLastKF;
        }
        mpLastFrame = frame;
    }

    unsigned EvAsynchTracker::trackAndFrame(const PoseImagePtr& pImage, const unsigned int fId,
                                        std::unique_ptr<ELK_Tracker>& pLKTracker, EvFramePtr &evFrame) {

        // LK Optical Flow matching
        vector<KeyPoint> p1;
        vector<int> vMatches12;
        unsigned nMatches = pLKTracker->trackAndMatchCurrImage(pImage->mImage, p1, vMatches12);

        // Establish and update connections with last key frames and map points
        this->makeFrame(pImage, fId, p1, evFrame);
        evFrame->setMatches(vMatches12);

        return nMatches;
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    void EvAsynchTracker::insertRefKF(ORB_SLAM3::KeyFrame *pKFref) {

        assert(pKFref);

        // Add to Atlas
        mpEvAtlas->AddKeyFrame(pKFref);

        mpReferenceKF = pKFref;
        mpLastKF = pKFref;

        mpCurrFrame->mpReferenceKF = pKFref;
        //mpLastFrame.mpReferenceKF = pKFref;
        mpIniFrame->mpReferenceKF = pKFref;

        {
            std::unique_lock<mutex> lock1(mMtxVpRefKFs);
            mvpRefKeyFrames.push_back(pKFref);
        }
        mvpLocalKeyFrames.push_back(pKFref);
        mpEvAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFref); //????

        //mpTrackManager->insertGlobalKeyFrame(pKFini);
    }

    ORB_SLAM3::KeyFrame * EvAsynchTracker::insertRefKF(EvFramePtr &refFrame) {

        // Use Global KF indexing or like this locally????
        auto* pKFref = new KeyFrame(*refFrame, (int)mKeyFrameIdx++,
                                    mpEvAtlas->GetCurrentMap(), mpKeyFrameDB.get());
        this->insertRefKF(pKFref);
        return pKFref;
    }

    void EvAsynchTracker::insertCurrKF(ORB_SLAM3::KeyFrame *pKFcur, ORB_SLAM3::KeyFrame *pKFref) {

        assert(pKFcur && pKFref);

        // Add to Atlas
        mpEvAtlas->AddKeyFrame(pKFcur);

        // Establish connections
        //pKFcur->mPrevKF = pKFref;
        //pKFcur->ChangeParent(pKFref);
        //pKFref->AddChild(pKFcur); // this is done inside change parent

        if (mpLastKF) {
            pKFcur->mPrevKF = mpLastKF;
            pKFcur->mPrevKF->mNextKF = pKFcur;
        }

        mpLastKF = pKFcur;

        mvpLocalKeyFrames.push_back(pKFcur);

        //mpTrackManager->insertGlobalKeyFrame(pKFcur);
    }

    ORB_SLAM3::KeyFrame * EvAsynchTracker::insertCurrKF(EvFramePtr &currFrame, ORB_SLAM3::KeyFrame *pKFref) {

        auto* pKFcur = new KeyFrame(*currFrame, (int)mKeyFrameIdx++,
                                    mpEvAtlas->GetCurrentMap(), mpKeyFrameDB.get());
        this->insertCurrKF(pKFcur, pKFref);
        return pKFcur;
    }

    void EvAsynchTracker::updateRefMapPoints(const vector<MapPoint*>& vpCurrMPs) {

        mvpLocalMapPoints = vpCurrMPs;
        mpEvAtlas->SetReferenceMapPoints(vpCurrMPs); //????
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    void EvAsynchTracker::getLastDPoseAndDepth(double& t0, double &dts, cv::Mat &Tcr, float &medDepth) {

        mpPoseInfo->getDPose(t0, dts, Tcr, medDepth);
    }

    /*void EvAsynchTracker::getCurrPoseSmooth(const cv::Mat &DTcw, cv::Mat& currPose) {

        // TODO: This is based on python eval script -> make sure it's correct
        if (currPose.empty() || DTcw.empty()) {
            return;
        }
        // Stitch pose and dpose so that base lines are comparable
        cv::Mat dTcw = DTcw;
        Converter::scaleSE3(dTcw, currPose);
        currPose = Converter::getCurrTcw(currPose, dTcw);
    }*/

    /*cv::Mat EvAsynchTracker::resolveIniPose() {

        cv::Mat iniTcw = cv::Mat::eye(4, 4, CV_32FC1);

        if (mpSensor->isInertial() && mpImuManager) {

            if (isImuInitialized() && mpCurrFrame->getPrevFrame() && mpCurrFrame->imuIsPreintegrated()) {

                mpImuManager->predictStateIMU(mpCurrFrame->getPrevFrame().get(), mpCurrFrame.get());

                cv::Mat Twb = cv::Mat::eye(4, 4, CV_32FC1);
                mpCurrFrame->mPredRwb.copyTo(Twb.rowRange(0,3).colRange(0,3));
                mpCurrFrame->mPredtwb.copyTo(Twb.rowRange(0,3).col(3));

                cv::Mat Twc = Twb * mpCurrFrame->mImuCalib.Tbc;
                iniTcw = Twc.inv(DECOMP_SVD);
            }
        }
        else {
            // TODO: Maybe we can estimate iniPose using PosePrior
        }

        return iniTcw.clone();
    }*/

    void EvAsynchTracker::predictNextPose(EvFramePtr& pCurrFrame) {

        if (!pCurrFrame) {
            return;
        }

        if (mpSensor->isInertial() && mpImuManager && mbImuInitialized) {

            if (mpLastKF) {
                mpImuManager->predictStateIMU(mnImuID, mpLastKF, pCurrFrame.get());
            }
            else if (pCurrFrame->getPrevFrame()) {
                mpImuManager->predictStateIMU(pCurrFrame->getPrevFrame().get(), pCurrFrame.get());
            }
        }
        else if (pCurrFrame->getPrevFrame()) {

            pCurrFrame->SetPose(pCurrFrame->getPrevFrame()->mTcw);
        }
    }

    // Set the poses of iniFrame and currFrame correctly and remap pts3d if required
    void EvAsynchTracker::refineInertialPoseMapIni(const cv::Mat& Tcw0, std::vector<cv::Point3f>& pts3d) {

        if (mpSensor->isInertial() && mpImuManager && mbImuInitialized && mpIniFrame->getPrevFrame()) {
            // if it's inertial, we can estimate init. pose using inertial measurements

            float sco = static_cast<float>(mpImuManager->getInitInfoIMU()->getLastScale()); //mpLastKF->ComputeSceneMedianDepth(2);
            mpIniFrame->setMatches(mpCurrFrame->getMatches());
            float scc = mpIniFrame->computeSceneMedianDepth(2, pts3d);
            float scRatio = sco/scc;

            // Set First Pose if it's empty (Tw0w1 = Tc0w0^-1 * Tc0w1)
            if (!mpIniFrame->getPosePrior().empty())
                // Reset the IniFrame's identity pose
                mpIniFrame->SetPose(mpIniFrame->getPosePrior());
            // The previous frame of current frame could be much further!
            //mpImuManager->predictStateIMU(mpIniFrame->getPrevFrame().get(), mpIniFrame.get());

            // Set Curr Pose, Careful about the scale ambiguity
            cv::Mat currTcw = Tcw0.clone();
            // Scale
            currTcw.rowRange(0,3).col(3) *= scRatio;
            // Tc1w1 = Tc1w0 * Tw0w1
            mpCurrFrame->SetPose(currTcw * mpIniFrame->mTcw);

            // Remap Pts3D
            cv::Mat Rwc0 = mpIniFrame->GetRotationInverse();
            cv::Mat twc0 = -Rwc0 * mpIniFrame->mTcw.rowRange(0,3).col(3);
            for (auto & currPt : pts3d) {

                cv::Mat Pw0 = cv::Mat(currPt);
                cv::Mat Pw1 = scRatio * Rwc0 * Pw0 + twc0;
                currPt = cv::Point3f(Pw1.at<float>(0),Pw1.at<float>(1),Pw1.at<float>(2));
            }
        }
    }

    void EvAsynchTracker::mergeSimilarFrames(EvFramePtr &pFr) {

        EvFramePtr pPreFr = pFr->getPrevFrame();

        while (pPreFr) {

            if (abs(pPreFr->mTimeStamp - pFr->mTimeStamp) < 1e-6 ||
                (pFr->mpImuPreintegrated && pFr->mpImuPreintegrated->dT < 1e-6)) {

                // Merge preintegration and pose info.
                if (pPreFr->mpImuPreintegrated) { // Inertial
                    pFr->mpImuPreintegrated = pPreFr->mpImuPreintegrated;
                    pFr->SetImuPoseVelocity(pPreFr->GetImuRotation(), pPreFr->GetImuPosition(), pPreFr->mVw);
                }
                else if (!pPreFr->mTcw.empty()) { // Non-inertial
                    pFr->SetPose(pPreFr->mTcw);
                }

                pPreFr = pPreFr->getPrevFrame();
            }
            else {
                break;
            }
        }
    }

    IMU::Preintegrated* EvAsynchTracker::getPreintegratedFromLastKF(const EvFramePtr &pFr) {

        IMU::Preintegrated* pCurrPrInt = pFr->mpImuPreintegrated;
        EvFramePtr pPreFr = pFr->getPrevFrame();

        if (!mbImuInitialized) {
            return pCurrPrInt;
        }

        while (pPreFr && pPreFr->mpImuPreintegrated && abs(pPreFr->mTimeStamp - mpLastKF->mTimeStamp) > 1e-6) {

            pCurrPrInt->MergePrevious(pPreFr->mpImuPreintegrated);

            pPreFr = pPreFr->getPrevFrame();
        }

        return pCurrPrInt;
    }

    void EvAsynchTracker::getEventConstraints(std::vector<ORB_SLAM3::KeyFrame *> &vpEvKFs) {

        std::unique_lock<mutex> lock1(mMtxVpRefKFs);
        vpEvKFs = mvpRefKeyFrames;
    }

    void EvAsynchTracker::setImuManagerAndChannel(const std::shared_ptr<IMU_Manager> &pImuManager) {

        mnImuID = pImuManager->setNewIntegrationChannel();
        mpImuManager = pImuManager;
        mpLocalMapper->setImuManager(pImuManager);
    }

    void EvAsynchTracker::updateFrameIMU(const float s, const IMU::Bias &b, KeyFrame* pCurrentKeyFrame)
    {
        /*Map * pMap = pCurrentKeyFrame->GetMap();
        //unsigned int index = mnFirstFrameId;
        list<ORB_SLAM3::KeyFrame*>::iterator lRit = mlpReferences.begin();
        list<bool>::iterator lbL = mlbLost.begin();
        for(list<cv::Mat>::iterator lit=mlRelativeFramePoses.begin(),lend=mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lbL++)
        {
            if(*lbL)
                continue;

            KeyFrame* pKF = *lRit;

            while(pKF->isBad())
            {
                pKF = pKF->GetParent();
            }

            if(pKF->GetMap() == pMap)
            {
                (*lit).rowRange(0,3).col(3)=(*lit).rowRange(0,3).col(3)*s;
            }
        }*/

        //mLastBias = b;

        //mpLastKF = pCurrentKeyFrame;
        //mpLastKF->SetNewBias(b);

        //std::unique_lock<mutex> lock1(mMtxUpdateState);

        mpLastFrame->SetNewBias(b);
        if (mpLastFrame->mnId != mpCurrFrame->mnId) {
            mpCurrFrame->SetNewBias(b);
        }

        // Correct the pose of last key frame (Scale and Rotation)
        cv::Mat Rwg = mpImuManager->getLastRwg();
        cv::Mat Ryw = Rwg.clone();
        cv::Mat Tyw = cv::Mat::eye(4, 4, CV_32FC1);
        Rwg.copyTo(Tyw.rowRange(0, 3).colRange(0, 3));

        if (mpLastKF->mnId != pCurrentKeyFrame->mnId) {

            cv::Mat Twc = mpLastKF->GetPoseInverse();
            cv::Mat Tyc = Tyw * Twc;

            cv::Mat Tcy = cv::Mat::eye(4, 4, CV_32FC1);
            Tcy.rowRange(0,3).colRange(0,3) = Tyc.rowRange(0,3).colRange(0,3).t();
            Tcy.rowRange(0,3).col(3) = -Tcy.rowRange(0,3).colRange(0,3)*Tyc.rowRange(0,3).col(3);
            mpLastKF->SetPose(Tcy);

            if (mpLastKF->mnFrameId == mpLastFrame->mnId) {
                mpLastFrame->SetPose(Tcy);
            }
            if (mpLastKF->mnFrameId == mpCurrFrame->mnId) {
                mpCurrFrame->SetPose(Tcy);
            }

            cv::Mat Vw = mpLastKF->GetVelocity();
            mpLastKF->SetVelocity(Ryw*Vw*s);
        }

        cv::Mat Gz = (cv::Mat_<float>(3,1) << 0, 0, -IMU::GRAVITY_VALUE);

        cv::Mat twb1;
        cv::Mat Rwb1;
        cv::Mat Vwb1;
        float t12;

        //while(!mpCurrFrame->imuIsPreintegrated())// && !mpImuManager->isInitializing())
        //{
        //    std::this_thread::sleep_for(std::chrono::microseconds(500));
        //}


        if(mpLastFrame->mnId == mpLastFrame->mpLastKeyFrame->mnFrameId)
        {
            mpLastFrame->SetImuPoseVelocity(mpLastFrame->mpLastKeyFrame->GetImuRotation(),
                                            mpLastFrame->mpLastKeyFrame->GetImuPosition(),
                                            mpLastFrame->mpLastKeyFrame->GetVelocity());
        }
        else
        {
            twb1 = mpLastFrame->mpLastKeyFrame->GetImuPosition();
            Rwb1 = mpLastFrame->mpLastKeyFrame->GetImuRotation();
            Vwb1 = mpLastFrame->mpLastKeyFrame->GetVelocity();
            t12 = mpLastFrame->mpImuPreintegrated->dT;

            mpLastFrame->SetImuPoseVelocity(Rwb1 * mpLastFrame->mpImuPreintegrated->GetUpdatedDeltaRotation(),
                                            twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1 * mpLastFrame->mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                            Vwb1 + Gz*t12 + Rwb1 * mpLastFrame->mpImuPreintegrated->GetUpdatedDeltaVelocity());
        }

        if (mpCurrFrame->mpImuPreintegrated)
        {
            twb1 = mpCurrFrame->mpLastKeyFrame->GetImuPosition();
            Rwb1 = mpCurrFrame->mpLastKeyFrame->GetImuRotation();
            Vwb1 = mpCurrFrame->mpLastKeyFrame->GetVelocity();
            t12 = mpCurrFrame->mpImuPreintegrated->dT;

            mpCurrFrame->SetImuPoseVelocity(Rwb1 * mpCurrFrame->mpImuPreintegrated->GetUpdatedDeltaRotation(),
                                               twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1 * mpCurrFrame->mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                               Vwb1 + Gz*t12 + Rwb1 * mpCurrFrame->mpImuPreintegrated->GetUpdatedDeltaVelocity());
        }

        //mnFirstImuFrameId = mpCurrFrame->mnId;
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    int EvAsynchTracker::init(const cv::Mat &im, const double& ts, const cv::Mat& iniPose) {

        // Set Initial Pose
        if (!iniPose.empty())
            mpIniFrame->SetPose(iniPose);

        vector<KeyPoint> p0 = mpIniFrame->getAllUndistKPtsMono();
        int nKPts = p0.size();

        if (nKPts <= 0) {
            return nKPts;
        }

        mvPts3D.resize(nKPts);

        // Init LK Trackers (internally inits last tracked pts)
        mpLKTracker->setRefImage(im, p0);
        mpLKTrackerKF->setRefImage(im, p0);

#ifdef SAVE_DEBUG_IMAGES_L2
        cv::Mat outIm;
        cv::drawKeypoints(im, p0, outIm);
        Visualization::saveImage(mRootImPath+"/l2_init_"+to_string(mIdxImSave++)+'_'+to_string(ts)+".png", outIm);
#endif

        DVLOG(1) << "Curr asynch tracker index: " << mCurrIdx << endl;
        DVLOG(1) << "Init completed with " << p0.size() << " key points.\n";

        return nKPts;
    }

    int EvAsynchTracker::initTracking(const cv::Mat& image, double ts, int& stat) {

        stat = 0;
        unsigned nMatches = mpCurrFrame->getNumMatches();

        // LK Optical Flow matching
        vector<KeyPoint> p1 = mpCurrFrame->getUndistTrackedPts();
        vector<int> vMatches12 = mpCurrFrame->getMatches();
        float medPxDisp = mpCurrFrame->getMedianPxDisp();

        if (nMatches < DEF_TH_MIN_MATCHES) { // Not enough matches
            stat = 1;
            return static_cast<int>(nMatches);
        }

        // Try to find relative pose between frames
        vector<cv::Mat> Rl1, tl1;
        vector<vector<cv::Point3f>> pts3d;
        vector<KeyPoint> p0 = mpLKTracker->getRefPoints();
        size_t nLenMch = vMatches12.size();
        vector<bool> vbReconstStat(nLenMch, false);
        vector<vector<bool>> vbTriangStat;
        ORB_SLAM3::ReconstInfo reconstInfo;

        bool reconstRes = mpCamera->ReconstructWithTwoViews(p0, p1, vMatches12,Rl1, tl1, pts3d,
                vbTriangStat, vbReconstStat, reconstInfo);

        reconstRes = resolveReconstStat(reconstRes, medPxDisp, reconstInfo, pts3d, vMatches12);

        if (reconstInfo.mnBestGood < DEF_TH_MIN_MATCHES) { // Not enough inliers
            stat = 2;
            return reconstInfo.mnBestGood;
        }
        if (!reconstRes) { // Bad reconstruction
            stat = 3;
            return reconstInfo.mnBestGood;
        }

        // Refine matches and reject outliers
        for(size_t i = 0; i < nLenMch; i++) {

            if(vMatches12[i]>=0 && !vbTriangStat[0][i]) {

                vMatches12[i]=-1;
                nMatches--;
            }
        }

        mpCurrFrame->setMatches(vMatches12);

        // Store the reconst. info somewhere safe!
        cv::Mat currTcw = ORB_SLAM3::Converter::toCvSE3(Rl1[0], tl1[0]);
        mvPts3D = pts3d[0];

        // IniFrame pose in some cases is calculated with difficulty, don't throw it away!
        if (!mpIniFrame->mTcw.empty()) {
            mpIniFrame->setPosePrior(mpIniFrame->mTcw);
        }
        mpIniFrame->SetPose(cv::Mat::eye(4, 4, CV_32FC1));
        mpCurrFrame->SetPose(currTcw);
        //this->refineInertialPoseMapIni(currTcw, mvPts3D);

#ifdef SAVE_DEBUG_IMAGES_L2
        cv::Mat mchIm;
        cv::Mat im0;
        mpLKTracker->getRefImageAndPoints(im0, p0);
        Visualization::myDrawMatches(im0, p0, image, p1, vMatches12, mchIm);
        Visualization::saveImage(mRootImPath+"/l2_matches_"+to_string(mIdxImSave++)+'_'+to_string(ts)+".png", mchIm);
#endif

        DVLOG(1) << "Curr Idx: " << mCurrIdx << endl;
        DVLOG(1) << "Median pixel displacement: " << medPxDisp << endl;
        DVLOG(1) << "L2 Tracking: Num. Matched Inliers: " << nMatches << endl;

        return reconstInfo.mnBestGood;
    }

    // This only have to be called for the first initialization
    bool EvAsynchTracker::initMap(const bool normalize) {

        // First, Optimize
        // Only work with newly created data (in case of odometry & ...)s
        vector<int> vIniMatches = mpCurrFrame->getMatches();
        vector<EvFramePtr> vpEvFrames = {mpIniFrame, mpCurrFrame};

        MyOptimizer::OptimInitV(vpEvFrames, mvPts3D, vIniMatches, 20);

        mpIniFrame->setMatches(vIniMatches);
        float medianDepth = mpIniFrame->computeSceneMedianDepth(2, mvPts3D);

        KeyFrame* pLastKF = mpLastKF;
        IMU::Preintegrated* pPreIntFromLastKF = getPreintegratedFromLastKF(mpIniFrame);

        if (normalize) {
            // V-SLAM: Normalize map scale
            EvLocalMapping::scalePoseAndMap(vpEvFrames, mvPts3D, medianDepth);
        }
        else {
            //VI-SLAM: Merge maps using inertial measurements
            refineInertialPoseMapIni(mpCurrFrame->mTcw, mvPts3D);
        }

        // Create and insert KeyFrames
        KeyFrame* pKFini = this->insertRefKF(mpIniFrame);
        KeyFrame* pKFcur = this->insertCurrKF(mpCurrFrame, pKFini);

        // Add new map points
        EvLocalMapping::addNewMapPoints(mpEvAtlas, pKFini, pKFcur, vIniMatches, mvPts3D);

        mpIniFrame->setAllMapPointsMono(pKFini->GetMapPointMatches());
        mpCurrFrame->setAllMapPointsMono(pKFcur->GetMapPointMatches());

        // Update Connections
        pKFini->UpdateConnections();
        pKFcur->UpdateConnections();

        DLOG(INFO) << "EvAsynchTracker::initMap: New Map created with " +
                      to_string(mpEvAtlas->MapPointsInMap()) + " points\n";
        DLOG(INFO) << pKFcur->printPointDistribution();

        set<MapPoint*> spCurrMPs = pKFini->GetMapPoints();
        vector<MapPoint*> vpCurrMapPoints(spCurrMPs.begin(), spCurrMPs.end());

        if(medianDepth<0 || pKFcur->TrackedMapPoints(1) < DEF_INIT_MIN_TRACKED_PTS3D) {

            Verbose::PrintMess("Wrong initialization, reseting...", Verbose::VERBOSITY_NORMAL);
            return false;
        }

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

        this->updateRefMapPoints(vpCurrMapPoints);

        //saveActivePose();
        //saveActiveMap();

        mbMapInitialized.set(true);
        return true;
    }

    int EvAsynchTracker::assignMatchedMPs(ORB_SLAM3::KeyFrame* pRefKF, EvFramePtr& currFrame) {

        vector<int> vMatches12 = currFrame->getMatches();
        vector<MapPoint*> vpMapPointMatches;
        {
            //unique_lock<mutex> lock(pRefKF->GetMap()->mMutexMapUpdate); //-> There's no point, map points will change!
            vpMapPointMatches = pRefKF->GetMapPointMatches(); //iniFrame->getAllMapPointsMono();
        }
        assert(vpMapPointMatches.size() == vMatches12.size());

        int nMPs = 0;
        for (size_t i = 0; i < vMatches12.size(); i++) {

            if (vMatches12[i] < 0) {
                vpMapPointMatches[i] = static_cast<MapPoint *>(nullptr);
                //mpCurrFrame.mvbOutlier[i] = true;
            }
            else if (vpMapPointMatches[i]) {
                nMPs++;
            }
        }

        currFrame->setAllMapPointsMono(vpMapPointMatches);
        return nMPs;
    }

    int EvAsynchTracker::refineTrackedMPs(EvFramePtr &currFrame) {

        // Discard outliers
        int nmatchesMap = 0;
        for(int i =0; i < currFrame->numAllKPts(); i++)
        {
            MapPoint* pMP = currFrame->getMapPoint(i);

            if(pMP)
            {
                if(currFrame->getMPOutlier(i))
                {


                    currFrame->setMapPoint(i, static_cast<MapPoint*>(nullptr));
                    currFrame->setMPOutlier(i,false);

                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = currFrame->mnId;

                    //nMatches--;
                }
                else if(currFrame->getMapPoint(i)->Observations() > 0)
                    nmatchesMap++;
            }
        }

        mnMatchesInliers = nmatchesMap;
        return nmatchesMap;
    }

    // TODO: Usage of init. frame after init. is wrong because only KFs are optimized!
    int EvAsynchTracker::estimateCurrPoseTLM(EvFramePtr& iniFrame, EvFramePtr& currFrame) {

        // Use reference map points to track current frame
        assignMatchedMPs(iniFrame->mpReferenceKF, currFrame);

        // Find current pose with PnP using matches and active map
        // Current ORB-SLAM finds pose by optimization
        Optimizer::PoseOptimization(currFrame.get());

        // Discard outliers
        return this->refineTrackedMPs(currFrame);
    }

    void EvAsynchTracker::trackLocalMap(const cv::Mat& image, double ts,
                                        int& nTrackedPts, const int& nMinTrakedPts) {

        uint nMatches = mpCurrFrame->getNumMatches();

        nTrackedPts = static_cast<int>(nMatches);
        // Not enough matches (ORB-SLAM min th is 15)
        if (nTrackedPts < nMinTrakedPts) {//DEF_MIN_TRACKED_PTS3D) {
            return;
        }

        this->predictNextPose(mpCurrFrame);

        int nmatchesMap = this->estimateCurrPoseTLM(mpIniFrame, mpCurrFrame);

        DLOG(INFO) << "* L2 Tracking, Calculated dTcc0:\n"
                   << Converter::toString(Converter::getTc1c0(mpLastFrame->mTcw, mpCurrFrame->mTcw));

        //this->updateLastPose(mpCurrFrame);
        nTrackedPts = nmatchesMap;
    }

    // TODO: Add most of the work to local mapping class
    // This is not very simple, we need to do all we did in initMap
    void EvAsynchTracker::addNewKeyFrame(const cv::Mat& currImage) {

        // Create and insert new KF
        KeyFrame* pKFcur = this->insertCurrKF(mpCurrFrame, mpReferenceKF);

        if(mpSensor->isInertial() && mpEvAtlas->isImuInitialized()) {
            pKFcur->bImu = true;
        }

        pKFcur->SetNewBias(mpCurrFrame->mImuBias);
        //mpReferenceKF = pKF;
        //mpCurrentFrame->mpReferenceKF = pKF;

        // Reset preintegration from last KF (Create new object)
        if (mpSensor->isInertial() && mpImuManager) {
            mpImuManager->refreshPreintFromLastKF(mnImuID);
            // mpPreintFromLast... = new IMU::Preintegrated(pKF->GetImuBias(),pKF->mImuCalib);
        }

        // Only process one key frame at a time
        while(mpLocalMapper->isProcessing() && !mpImuManager->isInitializing()) { // beware of deadlocks
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
        mpLocalMapper->insertNewKeyFrame(pKFcur);

        // Update Connections
        //mpReferenceKF->UpdateConnections();
        //pKFcur->UpdateConnections();

        //this->updateLastPose(mpCurrFrame->mTimeStamp, pKFcur->GetPose());

        // If new KF is inserted, reset KF Tracker
        PoseImagePtr pImage = make_shared<PoseImage>(mpCurrFrame->mTimeStamp, currImage, cv::Mat(), "");
        this->resetKeyFrameTracker(pImage);

        // Refresh local reference LK tracker with map points and poses
        //this->refreshLastTrackedKPts(pKFcur);

        //saveActivePose();
        //saveActiveMap();
    }

    void EvAsynchTracker::refreshLastTrackedKPts(KeyFrame* pKFcur) {

        vector<MapPoint*> allMPs = mpIniFrame->getAllMapPointsMono();
        //cv::Mat currPose = pKFcur->GetPose();
        //cv::Mat Pcur = mK * currPose.rowRange(0,3).colRange(0,4);
        cv::Mat Rcur = pKFcur->GetRotation();
        cv::Mat tcur = pKFcur->GetTranslation();

        vector<cv::KeyPoint> currTrackedKPts = mpLKTracker->getLastTrackedPts();

        // Project each map point to current KF to find new KPts
        for (size_t i = 0; i < allMPs.size(); i++) {

            MapPoint* pMP = allMPs[i];
            if (pMP) {// && !pMP->isBad() -> TODO

                cv::Mat newPt = Rcur * pMP->GetWorldPos() + tcur;
                newPt = newPt.rowRange(0,3).col(0) / newPt.at<float>(2, 0);
                newPt = mK * newPt;

                currTrackedKPts[i].pt.x = newPt.at<float>(0,0);
                currTrackedKPts[i].pt.y = newPt.at<float>(1,0);
            }
        }

        mpLKTracker->setLastTrackedPts(currTrackedKPts);
    }

    void EvAsynchTracker::refreshNewTrack(const PoseImagePtr& pImage) {

        // Assumption: we've init. and tracked last KF before (LKTrackerKF)
        if (this->needNewKeyFrame(pImage->mImage)) {

            // Track new image again (to produce new curr. frame)
            EvFramePtr pCurrFrameTemp;
            this->trackAndFrame(pImage, mFrameIdx++, mpLKTrackerKF, pCurrFrameTemp);
            pCurrFrameTemp->SetPose(mpCurrFrame->mTcw);
            // TODO: Maybe also copy IMU Data??

            // First copy ini/cur frames and all required connections and data
            KeyFrame* pLastRefKF = mpReferenceKF;
            KeyFrame* pLastKF = mpLastKF;
            float medDepth = mpLastKF->ComputeSceneMedianDepth(2);
            //mpIniFrameTemp->SetPose(pLastKF->GetPose()); -> do this in resetKFTracker

            // Refresh ref. key frame
            // Remember these methods will change ref. and last KF states
            KeyFrame* pKFini = this->insertRefKF(mpIniFrameTemp);
            KeyFrame* pKFcur = this->insertCurrKF(pCurrFrameTemp, pKFini);
            pKFini->mpOrigKF = mpLastKF;

            // Triangulate New Map Points
            vector<MapPoint*> vpIniMPts = mpIniFrameTemp->getAllMapPointsMono();
            vector<bool> vbIniOutliers = mpIniFrameTemp->getAllOutliersMono();


            int nNewMPts = EvLocalMapping::createNewMapPoints(mpEvAtlas, pKFini, pKFcur, vpIniMPts, vbIniOutliers,
                                                              vpIniMPts, pCurrFrameTemp->getMatches(), medDepth);

            // If good number of map points, ini. is complete
            if (nNewMPts > DEF_INIT_MIN_TRACKED_PTS3D) {

                mpIniFrame = mpIniFrameTemp;

                mpLKTracker->setRefImage(mpIniFrameTemp->imgLeft, mpIniFrameTemp->getAllUndistKPtsMono());
                mpLKTracker->setLastTrackedPts(mpLKTrackerKF->getLastTrackedPts());

                this->resetKeyFrameTracker(pImage);
            }
            // else, do the house-keeping and exit
            else {
                pKFcur->SetBadFlag();
                pKFini->SetBadFlag();

                mpReferenceKF = pLastRefKF;
                mpLastKF = pLastKF;
            }
            //mpIniFrameTemp = nullptr;
        }
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    void EvAsynchTracker::Track() {

        while (!this->isStopped()) {

            if (isInputGood()) {

                mbIsProcessing = true;

                if (mStat == IDLE) {
                    mStat.update(INIT_TRACK);
                }

                PoseImagePtr pPoseImage = this->frontImage();
                double imTs = pPoseImage->ts;
                cv::Mat image = pPoseImage->mImage.clone();

                if (mpCurrFrame && imTs < mpCurrFrame->mTimeStamp) {
                    LOG(FATAL) << "EvAsynchTracker::Track: Lower new image ts: "
                               << imTs << " <= " << mpCurrFrame->mTimeStamp << endl;
                }

                if (image.empty()) {
                    LOG(ERROR) << "* L2 Asynch. Tracker, " << imTs << ": Empty image\n";
                    this->popImage();
                    continue;
                }

//                if (mpImageDisplay) {
//
//                    ImageTsPtr pImage = make_shared<ImageTs>(imTs, image, pPoseImage->uri);
//                    mpImageDisplay->addImage(pImage);
//                    DLOG_EVERY_N(INFO, 10) << "EvAsynchTracker::Track: Added image at: " << imTs << endl;
//                }

                // Do not process tiny frames
                if (pPoseImage->mReconstStat == 0) {

                    if (mStat == INIT_MAP || mStat == TRACKING) {
                        EvFramePtr dummyFrame;
                        this->trackAndFrame(pPoseImage, -1, mpLKTrackerKF, dummyFrame);
                        this->trackAndFrame(pPoseImage, -1, mpLKTracker, dummyFrame);
                    }
                    this->popImage();
                    continue;
                }

                std::unique_lock<mutex> lock1(mMtxUpdateState);

                this->makeFrame(pPoseImage, mFrameIdx++, mpLKTracker, mpCurrFrame);

                // Set Prior Pose from L1 Tracker
                mpCurrFrame->setPosePrior(pPoseImage->mTcw);

                // IMU Preintegration
                if (mpSensor->isInertial() && mpImuManager) {

                    mpCurrFrame->mImuCalib = *(mpImuManager->getImuCalib());
                    mpCurrFrame->mImuBias = mpImuManager->getLastImuBias();
                    mpImuManager->preintegrateIMU(mnImuID, mpCurrFrame, mpCurrFrame->mImuBias);

                    // To prevent ugly behavior, wait until IMU is initialized
                    //while (mpImuManager->isInitializing()) {
                    //    std::this_thread::sleep_for(std::chrono::microseconds (500));
                    //}
                }

                bool setLastPose = false;

                // Tcw0 from L1 is not currently used in init. stages
                if (mStat == INIT_TRACK) { // All in one init. state

                    // Provide a reliable pose before entering this state
                    // Retaining first pose is not that simple (all later pose and depths must match)
                    cv::Mat iniPose;// = this->resolveIniPose();

                    int nKpts = this->init(image, imTs, iniPose);

                    // Success -> There is enough features, complete init.
                    if (nKpts > DEF_TH_MIN_KPTS) {

                        changeStateToInitMap(mStat);
                    }
                    // Failure -> the pose graph will be disconnected
                    else {
                        LOG(WARNING) << "* L2 Init., " << imTs << ": Not enough key points, " << nKpts << endl;

                        resetToInitTracking(mStat, mTrackMode);
                    }
                }
                else if (mStat == INIT_MAP) {

                    int stat;
                    int nMatches = this->initTracking(image, imTs, stat);

                    if (stat == 0) { // Successful reconstruction

                        // Try to Init. Map
                        bool mapInitialized = mbMapInitialized == true;
                        const bool inertialMap = mpSensor->isInertial() && mbImuInitialized;
                        if (!mbMapInitialized || inertialMap) {
                            mapInitialized = this->initMap(!inertialMap);
                            setLastPose = mapInitialized;
                        }

                        if (mTrackMode == ODOMETRY || !mapInitialized) {
                            resetToInitTracking(mStat, mTrackMode);
                            continue; // Keep the current image
                        }
                        else {
                            changeStateToTLM(mStat, pPoseImage);
                        }
                    }
                    // Mostly happens when we fit wrong model (Fundamental instead of Homography)
                    else if (stat == 1 || stat == 2) { // Failure -> Not enough tracked inliers

                        // Use current image to reInit.
                        LOG(WARNING) << "* L2 Feature tracking, " << imTs << ": Not enough matches, " << nMatches << endl;
                        mBadInitCnt++;
                        if (mBadInitCnt >= 0) { // Give it chance to recover!

                            this->resetToInitTracking(mStat, mTrackMode);
                            continue; // currImage info is not popped on purpose
                        }

                    }
                    else { // Bad reconstruction

                        mBadInitCnt = 0;
                        VLOG_EVERY_N(2, 100) << "* L2 Feature tracking, " << imTs <<
                              ": Bad reconstruction with " << nMatches << " matches\n";
                    }
                }
                else if (mStat == TRACKING) {

                    // We do use L1 pose change here
                    //this->grabL1PoseChange(tcw0);
                    DLOG(INFO) << "* L2 Tracking, Suggested dTcc0:\n" << Converter::toString(pPoseImage->mTcw);

                    int nTrackedMPts = 0;
                    this->trackLocalMap(image, imTs, nTrackedMPts);

                    // If tracking is failed
                    if (nTrackedMPts < DEF_MIN_TRACKED_PTS3D) {

                        // Prepare for reInit. stage
                        this->addNewKeyFrame(image);

                        LOG(WARNING) << "* L2 Tracking, " << imTs << ": Track lost, reInit.\n";
                        resetToInitTracking(mStat, mTrackMode);
                        continue; // Keep current image
                    }
                    // If the condition is favorable for initiating a reference change
                    else if (mTrackMode == TLM_CH_REF && nTrackedMPts < DEF_INIT_MIN_TRACKED_PTS3D) {

                        LOG(INFO) << "* L2 Tracking, " << imTs << ": Checking ref. change condition with "
                             << nTrackedMPts << " tracked map points.\n";

                        // Go to ch_ref state imediately (without parallax check)
                        // If we have an estimate of the new pose
                        //this->initiateRefChange(image, *mpCurrFrame);
                        this->refreshNewTrack(pPoseImage);

                        LOG(INFO) << "* L2 Tracking, " << imTs << ": Initiating ref. change.\n";
                        //mStat.update(CH_REF_KF);
                    }
                    // Else if we can have a new regular key frame (enough parallax)
                    // TODO: Check imReconstStat so we don't add KFs from tiny images
                    else if (this->needNewKeyFrame(image)) {

                        // create new key frame
                        this->addNewKeyFrame(image);

                        DLOG(INFO) << "* L2 Tracking, " << imTs << ": Add new key frame because of large baseline.\n";
                    }

                    setLastPose = true;

                    // Local BA -> After each KF insertion and/or RefKF change
                    // Global BA -> 4-DOF pose alignment (system wide)
                }

                if (setLastPose) {
                    mpPoseInfo->updateLastPose(mpCurrFrame->getPrevFrame(), mpCurrFrame);
                    mpTrackManager->setCurrentCamPose(mpCurrFrame->mTcw);
                }

                // Visualizing frames
                mpCurrFrame->imgLeft = image.clone();
                this->updateFrameDrawer();

                this->popImage();

                mbIsProcessing = false;
            }
            if (mStat == IDLE) {

                VLOG_EVERY_N(3, 1000) << "L2: Idle state...\n";
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }

            if (!this->sepThread) { break; }
        }
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    void EvAsynchTracker::updateFrameDrawer() {

        if (!mpViewer || !mpFrameDrawer) {
            return;
        }

        switch (mStat.getState()) {

            case INIT_TRACK:
                mpFrameDrawer->updateLastTrackerState(mFrDrawerId, EORB_SLAM::FrameDrawerState::INIT_TRACK);
                mpFrameDrawer->pushNewFrame(mFrDrawerId, mpCurrFrame);
                break;
            case INIT_MAP:
                mpFrameDrawer->updateLastTrackerState(mFrDrawerId, EORB_SLAM::FrameDrawerState::INIT_MAP);
                mpFrameDrawer->updateIniFrame(mFrDrawerId, mpIniFrame);
                mpFrameDrawer->pushNewFrame(mFrDrawerId, mpCurrFrame);
                break;
            case TRACKING:
                mpFrameDrawer->updateLastTrackerState(mFrDrawerId, EORB_SLAM::FrameDrawerState::TRACKING);
                mpFrameDrawer->pushNewFrame(mFrDrawerId, mpCurrFrame);
                break;
            case IDLE:
            default:
                mpFrameDrawer->updateLastTrackerState(mFrDrawerId, EORB_SLAM::FrameDrawerState::IDLE);
                break;
        }
    }
    // Important variables: ini & curr frames, curr image, tracking matches, curr map points and outliers

    void EvAsynchTracker::saveAllFramePoses(const std::string &fileName, double tsc, int ts_prec) {

        // This method can't do this because we don't implement a FrameInfo mechanism here
    }

    void EvAsynchTracker::saveAllPoses(const std::string &fileName, double tsc, int ts_prec) {

        //Visualization::saveAllPoseLV(mvpLocalKeyFrames, fileName);
        Visualization::saveAllPoseRefs(mvpRefKeyFrames, fileName, tsc, ts_prec);
    }

    void EvAsynchTracker::saveAtlas(const std::string &fileName) {

        Visualization::saveAtlas(mpEvAtlas, fileName);
    }

} // EORB_SLAM

/* Draft
                            // Rescale current pose to match other poses
                            float medDepth = mpReferenceKF->ComputeSceneMedianDepth(2);
                            cv::Mat currPose = mpCurrFrame.mTcw.clone();
                            cv::Mat currTrans = currPose.rowRange(0, 3).col(3);
                            cv::Mat newTrans = currTrans / cv::norm(currTrans) * medDepth;
                            newTrans.copyTo(currPose.rowRange(0, 3).col(3));
                            mpCurrFrame.SetPose(mpIniFrame.mTcw * currPose);

                            // add new key frame
                            this->addNewKeyFrame();

                            else {
                            cout << "* L2 Feature tracking, " << imTs
                                 << ": initialized before, trying to recover.\n";
                            mBadInitCnt++;
                            // Give it change to recover or else we have to go to init.
                            if (mBadInitCnt >= 3) {
                                cout << "* L2 Feature tracking, " << imTs << ": could not recover, ReInit.\n";
                                mpEvAtlas->CreateNewMap((int) mMapIdx++);
                                mStat.update(HARD_INIT);
                                continue;
                            }
                        }
*/
// This will set the init. state to go to TRACKING state
// Use the last key frame and current pose to change the state
/*int EvAsynchTracker::refreshNewTracking(const cv::Mat &image, const double& ts) {

    // Check the last KF and current frame are not the same
    int stat = 0;
    if (!this->isCurrPoseGood(ts, stat)) {
        cerr << "EvAsynchTracker::refreshNewTracking, Close KF & F TimeStamps\n";
        return 0;
    }

    mvpRefFrames.push_back(mpIniFrame);

    // Initialize init. state: Image, New Features, Tracked Features, ...
    cv::Mat im0;
    vector<cv::KeyPoint> p0;
    mpLKTrackerKF->getRefImageAndPoints(im0, p0);

    // What is the right Id???
    mpIniFrame = EvFrame(mpCurrFrame.mnId-1, im0, mpLastKF->mTimeStamp,
                        mpFastDetector, mnEvPts, mpCamera, mDistCoefs);
    mpIniFrame.SetPose(mpLastKF->GetPose());
    mpIniFrame.keepRefImage(im0);
    mpIniFrame.setDistTrackedPts(mpLastKF->mvKeys);

    vector<cv::KeyPoint> p0ini = mpIniFrame.getAllUndistKPtsMono();
    //mLastTrackedPtsKF = mLastTrackedPts;

    mpLKTracker->setRefImage(im0, p0ini);

    // Refresh current frame with new KPts
    // LK Optical Flow matching
    vector<KeyPoint> p1;
    vector<int> vMatches12;
    unsigned nKLTMatches = mpLKTracker->trackAndMatchCurrImage(image, p1, vMatches12);

    cv::Mat currPose = mpCurrFrame.mTcw.clone();
    mpCurrFrame = EvFrame(mpCurrFrame.mnId, image, ts, p1, mpCamera, mDistCoefs, !this->isEvsRectified, &mpIniFrame);
    mpCurrFrame.setMatches(vMatches12);
    mpCurrFrame.SetPose(currPose);

    vector<MapPoint*> mIniMapPoints = mpLastKF->GetMapPointMatches();
    mpLastKF->SetBadFlag();

    // Refresh last key frame based on new InitFrame
    auto* pKFini = new KeyFrame(mpIniFrame, (int)mpLastKF->mnId,
                                mpEvAtlas->GetCurrentMap(), mpKeyFrameDB.get());
    // Create new KF for current state
    auto* pKFcur = new KeyFrame(mpCurrFrame, (int)mKeyFrameIdx++,
                                    mpEvAtlas->GetCurrentMap(), mpKeyFrameDB.get());

    // Keep old connections
    unsigned nMatches = mpIniFrame.connectNewAndTrackedPts(mMaxDistRefreshPts);

    // Update map points based on newly found matches
    vector<int> vNewMatches = mpIniFrame.getTrackedMatches();
    // Since ref. frames don't have matches, this might be a good idea
    mpIniFrame.setMatches(vNewMatches);

    // TODO: Is this right???
    vector<MapPoint*> vMPs = mpIniFrame.mvpMapPoints;
    vector<bool> vOutliers = mpIniFrame.mvbOutlier;

    this->reAssignMapPoints(pKFini, pKFcur, vMPs, vOutliers, mIniMapPoints, vNewMatches);

    mpIniFrame.mvpMapPoints = vMPs;
    mpIniFrame.mvbOutlier = vOutliers;

    vector<cv::KeyPoint> mvIniKPts = mpIniFrame.getAllUndistKPtsMono();
    vector<cv::KeyPoint> mvKPts = p1; //mpCurrFrame.getAllUndistKPtsMono();

    vector<MapPoint*> vIniMPs = mpIniFrame.mvpMapPoints;
    vector<bool> vIniOutliers = mpIniFrame.mvbOutlier;

    this->addNewMapPoints(mpEvAtlas, pKFini, pKFcur, vIniMPs, vIniOutliers, vIniMPs, vMatches12);

    mpIniFrame.mvpMapPoints = vIniMPs;
    mpIniFrame.mvbOutlier = vIniOutliers;
    mpCurrFrame.mvpMapPoints = mpIniFrame.mvpMapPoints;
    mpCurrFrame.mvbOutlier = mpIniFrame.mvbOutlier;

    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    // Replace refreshed last KF with the old one
    //mpLastKF->SetBadFlag();

    this->insertPosesAndMap(pKFini, pKFcur, mpEvAtlas->GetAllMapPoints());

    // We can (event only tracking) refine
    // last Ref. kf to last kf with local BA
//        if (mpThreadLocalBA) {
//            mpThreadLocalBA->join();
//        }
//        mpThreadLocalBA = unique_ptr<thread>(new thread(&EvAsynchTracker::localBA, this));
    this->localBA();

    // Update state to init. map (in main tracking thread)
    return static_cast<int>(nMatches);
}

// When tracked points are lower than a thresh
// This sets the init. state to go to INIT_MAP state
int EvAsynchTracker::refreshNewInitMap(const cv::Mat& image, const double& ts) {

    // Initialize init. state: Image, New Features, Tracked Features, ...
    mvpRefFrames.push_back(mpIniFrame);

    mpIniFrame = EvFrame(mpCurrFrame.mnId, image, ts, mpFastDetector, mnEvPts, mpCamera, mDistCoefs);
    mpIniFrame.SetPose(mpCurrFrame.mTcw);
    mpIniFrame.keepRefImage(image);
    mpIniFrame.setDistTrackedPts(mpCurrFrame.getDistTrackedPts());
    // Set Median depth for later???

    vector<cv::KeyPoint> p0 = mpIniFrame.getAllUndistKPtsMono();
    //mLastTrackedPtsKF = mLastTrackedPts;

    mpLKTracker->setRefImage(image, p0);

    // We still want to keep the connection between new pts and last tracked pts
    // so we might merge maps latter
    unsigned nMatches = mpIniFrame.connectNewAndTrackedPts(mMaxDistRefreshPts);

    // Update map points based on newly found matches
    vector<int> vNewMatches = mpIniFrame.getTrackedMatches();
    // Since ref. frames don't have matches, this might be a good idea
    mpIniFrame.setMatches(vNewMatches);

    // Add a new key frame based on new InitFrame
    // Create new KF
    auto* pKFcur = new KeyFrame(mpIniFrame, (int)mKeyFrameIdx++,
                                mpEvAtlas->GetCurrentMap(), mpKeyFrameDB.get());

    mpEvAtlas->AddKeyFrame(pKFcur);

    vector<MapPoint*> mIniMapPoints = mvpRefFrames.back().mvpMapPoints;
    vector<MapPoint*> vpMPs = mpIniFrame.mvpMapPoints;
    vector<bool> vbOutliers = mpIniFrame.mvbOutlier;

    this->reAssignMapPoints(pKFcur, vpMPs, vbOutliers, mIniMapPoints, vNewMatches);

    mpIniFrame.mvpMapPoints = vpMPs;
    mpIniFrame.mvbOutlier = vbOutliers;

    // Update Connections
    pKFcur->UpdateConnections();

    // currently this is not considered here!
    // These are important for me, the rest is a copy of ORB-SLAM!
    mpReferenceKF = pKFcur;
    mpLastKF = pKFcur;

    mpIniFrame.mpReferenceKF = mpReferenceKF;

    mpCurrFrame.mpReferenceKF = mpReferenceKF;

    //mpLastFrame = mpCurrFrame;

    mvpLocalKeyFrames.push_back(pKFcur);

    // Insert KFs in global pose chain (System)
    // Only add init KF if it is the first init.
    mpTrackManager->insertGlobalKeyFrame(pKFcur);

    mpEvAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(mpReferenceKF);

    // We can (event only tracking) refine
    // last Ref. kf to last kf with local BA
//        if (mpThreadLocalBA) {
//            mpThreadLocalBA->join();
//        }
//        mpThreadLocalBA = unique_ptr<thread>(new thread(&EvAsynchTracker::localBA, this));
    this->localBA();

    // Create a new map because ref. points are changed
    // and we don't have descriptors to find the matches
    //mpEvAtlas->CreateNewMap((int)mMapIdx++);

    // Update state to init. map (in main tracking thread)
    return static_cast<int>(nMatches);
}*/
