/**
 * Refine frame links (prev., ref., ...) and set preintegration for IMU case -> done
 * Reconstruction with median scene depth (or 2d trans.) is
 * good for low to medium range depth, but fails for large depth variations (Fisheye camera).
 * consider multi-depth for these -> poor performance!
 */

#include "EvImBuilder.h"

#include <utility>
#include <cmath>

//#include "Optimizer.h"
#include "MyOptimizer.h"
#include "EvOptimizer.h"
#include "EvLocalMapping.h"
#include "EvTrackManager.h"


using namespace std;
using namespace cv;

namespace EORB_SLAM {

    /* -------------------------------------------------------------------------------------------------------------- */

    EvImBuilder::EvImBuilder(shared_ptr<EvParams> evParams, CamParamsPtr camParams, MixedFtsParamsPtr pFtParams, SensorConfigPtr  sConfig) :

            EvBaseTracker(std::move(evParams), std::move(camParams)), mStat(), mLastStat(),
            mCntLowEvGenRate(0), mpSensorConfig(std::move(sConfig)), mpLastFrame(nullptr), mLastIdxBA(-1),
            mIniChi2(-1.0), mLastChi2(-1.0), mIniChi2BA(-1.0), mLastChi2BA(-1.0), mpTrackManager(nullptr),
            mbContTracking(mpEvParams->continTracking), mL1EvWinSize(mpEvParams->l1ChunkSize),
            mInitL1EvWinSize(mpEvParams->l1ChunkSize), mbFixedWinSize(mpEvParams->l1FixedWinSz),
            mL1NumLoop(mpEvParams->l1NumLoop), mL1MaxPxDisp(mpEvParams->maxPixelDisp),
            mTrackTinyFrames(mpEvParams->trackTinyFrames) {

        this->reset();
        this->resetOptimVars();

        mnWinOverlap = static_cast<ulong>(mpEvParams->l1WinOverlap * static_cast<double>(mL1NumLoop * mL1EvWinSize));

        // Use a calibrated camera model if events are rectified:
        MyParameters::createCameraObject(*(mpCamParams.get()), mpCamera,
                MyParameters::dummyCamera, mpEvParams->isRectified);

        DLOG(INFO) << "L1 Num. Features is set to " << mnEvPts << endl;

        // For simple image building, simple FAST detector will suffice.
        mFtDtMode = 0;
        cv::Size imSize(mpEvParams->imWidth, mpEvParams->imHeight);
        ORB_SLAM3::ORBxParams parORB(mnEvPts, mpEvParams->l1ScaleFactor, mpEvParams->l1NLevels, mThFAST,
                                     0, pFtParams->imMargin, imSize);
        this->initFeatureExtractors(mFtDtMode, parORB);

        mpLKTracker = unique_ptr<ELK_Tracker>(new ELK_Tracker(mpEvParams.get()));

        mpLastPoseDepth = make_shared<PoseDepthInfo>();
        mpLastPoseDepthBA = make_shared<PoseDepthInfo>();

        mTrackingTimer.setName("Event Image Builder");

        mTrackingWD.setName("Event Image Builder");
        if (mpEvParams->trackTinyFrames) {
            mTrackingWD.setWaitTimeSec(6.f);
        }

//        if (mpSensorConfig->isInertial() && mpImuManager) {
//
//            mpImuInfo = make_shared<InitInfoIMU>();
//            mpImuInfoBA = make_shared<InitInfoIMU>();
//        }
    }

    EvImBuilder::~EvImBuilder() {

        delete mpCamera;
        mpCamera = nullptr;
        mpTrackManager = nullptr;
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    void EvImBuilder::resetAll() {

        DLOG(INFO) << "EvImBuilder::resetAll >> Stats:\n"
                   << "\tInput: -Num. Chunks: " << mqEvChunks.size() << ", last ts of first chunck: "
                   << ((!mqEvChunks.isEmpty() && !mqEvChunks.front().empty()) ? to_string(mqEvChunks.front().back().ts) : "??") << endl
                   << "\t\t-Size of shared L2 ev. buff.: " << mvSharedL2Evs.size() << endl
                   << "\tOutput: -Num. processed frames: " << mvpEvFrames.size() << endl;

        // Empty input queues
        EvBaseTracker::resetAll();

        mStat.update(IDLE);
        mLastStat.update(IDLE);

        mpIniFrame = nullptr;
        mpCurrFrame = nullptr;
        mpLastFrame = nullptr;
        mpLastFrameBA = nullptr;

        this->updateL1ChunkSize(mInitL1EvWinSize);

        this->reset();

        this->resetOptimVars();
        //mLastDepthBA = 1.0;
        //mLastPoseBA = cv::Mat::eye(4,4, CV_32F);

        mIniChi2 = -1;
        mLastChi2 = mIniChi2;
        mIniChi2BA = -1;
        mLastChi2BA = mIniChi2BA;

        mpImuInfo = nullptr;
        mpImuInfoBA = nullptr;

        mpLastPoseDepth = make_shared<PoseDepthInfo>();
        mpLastPoseDepthBA = make_shared<PoseDepthInfo>();

        mpPoseImageTemp = nullptr;
    }

    void EvImBuilder::reset() {

        mCurrIdx = 0; // is reset in base tracker
        mCntLowEvGenRate = 0;

        try {
            if (!mvSharedL2Evs.isEmpty()) {
                mvSharedL2Evs.clear();
                std::unique_lock<mutex> lock1(mMtxL1EvWinSize);
                mvSharedL2Evs.reserve(mL1NumLoop * mL1EvWinSize);
            }
            {
                std::unique_lock<mutex> lock(mMtxEvFrames);
                if (!mvpEvFrames.empty()) {
                    mvpEvFrames.clear();
                    mvpEvFrames.reserve(mL1NumLoop);
                }
            }
            if (!mvMatchesCnt.empty()) {
                mvMatchesCnt.clear(); //-> in base tracker
            }
            {
                std::unique_lock<mutex> lock(mMtxPts3D);
                if (!mvPts3D.empty()) {
                    mvPts3D.clear(); //-> in base tracker
                }
            }
        }
        catch (std::bad_alloc& e) {
            LOG(ERROR) << "EvImBuilder::reset, " << e.what() << endl;
        }
        catch (Exception& e) {
            LOG(ERROR) << "EvImBuilder::reset, " << e.what() << endl;
        }
        {
            std::unique_lock<mutex> lock1(mMtxStateBA);
            mLastIdxBA = -1;
        }
        if (mpSensorConfig->isInertial() && mpImuManager) {
            mpImuManager->refreshPreintFromLastKF(mnImuID); //-> done in track manager
        }
    }

    void EvImBuilder::resetOptimVars() {

        {
            std::unique_lock<mutex> lock(mMtxStateParams2D);
            mLastParams2D = cv::Mat::zeros(3,1, CV_32F);
        }

        //mpImuInfo = make_shared<InitInfoIMU>();
        //mpImuInfoBA = make_shared<InitInfoIMU>();

        //mpLastPoseDepth = make_shared<PoseDepthInfo>();
        //mpLastPoseDepthBA = make_shared<PoseDepthInfo>();
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    void EvImBuilder::setLinkedL2Tracker(std::shared_ptr<EvAsynchTracker> &l2Tracker) {

        mpL2Tracker = l2Tracker;
    }

    void EvImBuilder::updateL1ChunkSize(unsigned int newSz) {

        //LOG_EVERY_N(INFO, 100) << "EvImBuilder::updateL1ChunkSize: New chunk size: " << newSz << endl;

        std::unique_lock<mutex> lock1(mMtxL1EvWinSize);
        mL1EvWinSize = newSz;
        mpEvParams->l1ChunkSize = newSz;
    }

    unsigned EvImBuilder::calcNewL1ChunkSize(const float medPxDisp) {

        std::unique_lock<mutex> lock1(mMtxL1EvWinSize);
        return static_cast<unsigned>(floorf(((float) (mCurrIdx + 1) / medPxDisp) * ((float) mL1EvWinSize)));
    }

    unsigned EvImBuilder::getL1ChunkSize() {

        std::unique_lock<mutex> lock1(mMtxL1EvWinSize);
        return mL1EvWinSize;
    }

    bool EvImBuilder::resolveEvWinSize(const float medPxDisp, const double imTs) {

        // Decide based on pixel displacement to start a new Optimization and recycle
        if (!mbFixedWinSize && medPxDisp > mL1MaxPxDisp) {

            // Update Parameter: l1ChunkSize
            unsigned newL1ChunkSize = calcNewL1ChunkSize(medPxDisp);
            updateL1ChunkSize(newL1ChunkSize);
            DLOG(INFO) << "* L1 Tracking, " << imTs <<
                       ": Event window size updated with: " << newL1ChunkSize << endl;

            return true;
        }
        // If window size is set to fixed, decide based on the number of windows
        else if (mbFixedWinSize && mCurrIdx >= mL1NumLoop) {

            std::unique_lock<mutex> lock1(mMtxL1EvWinSize);
            DLOG(INFO) << "* L1 Tracking, " << imTs << ": CurrIdx: " << mCurrIdx <<
                       ", Window size is fixed, " << mL1EvWinSize << endl;

            return true;
        }
        return false;
    }

    ulong EvImBuilder::resolveWinOverlap() {

        if (mbFixedWinSize)
            return mnWinOverlap;
        return mvSharedL2Evs.size() * mpEvParams->l1WinOverlap;
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    // Decide based on Tracker state and the capacity of input/output
    bool EvImBuilder::isTrackerReady() {

        return mStat == IDLE || !this->isTrackerSaturated();
    }

    // Need better condition??
    bool EvImBuilder::isTrackerSaturated() {

        return mqEvChunks > 2 * mL1NumLoop;
    }

    bool EvImBuilder::isInputGood() {

        return mqEvChunks > 0;
    }

    bool EvImBuilder::isMcImageGood(const cv::Mat& mcImage) {

        // Check if image is good in terms of number of detected kpts for next level
        EvFramePtr dummyFrame;
        PoseImagePtr pImage = make_shared<PoseImage>(0.0, mcImage, cv::Mat(), "");
        mpL2Tracker->makeFrame(pImage, -1, dummyFrame);
        return dummyFrame->numAllKPts() > DEF_TH_MIN_KPTS;
    }

    bool EvImBuilder::isEvGenRateGood(const double &evGenRate) {
        return evGenRate > mpEvParams->minEvGenRate;
    }

    bool EvImBuilder::needTrackTinyFrame() const {

        // Checking L2 tracker state is not necessary, imReconstStat flag is used now
        return mTrackTinyFrames;// && !mpL2Tracker->isIdle() && !mpL2Tracker->isInitTracking();
    }

    bool EvImBuilder::needDispatchMCF() const {

        if (mpSensorConfig->isImage()) {
            // Do not dispatch mcImages when sensor config is event-image
            DLOG_EVERY_N(INFO, 10000) << "* L1 Tracking, Asynch. MCI tracking "
                                      << "is not supported in Ev-Im sensor config.\n";
            return false;
        }
        return true;
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    int EvImBuilder::checkEvGenRate(const double &eventRate, const EvImBuilder::TrackState &state) {

        if (isEvGenRateGood(eventRate)) {
            mCntLowEvGenRate = 0;
            return 0; // OK
        }

        // Don't check event generation rate in continuous tracking mode
        if (mbContTracking) {
            // in variable window case, there is room for skipping bad frames if not initialized
            if (!mbFixedWinSize && state == INIT)
                return -1;
            return 0;
        }
        
        if (state == INIT) {

            DLOG(WARNING) << "* L1 Init: Low event generation rate, " << eventRate << endl;
            return -1; // Reject & reset to init.
        }
        else if (state == TRACKING) {

            DLOG(WARNING) << "* L1 Tracking: Low event gen. rate, " << eventRate << endl;
            mCntLowEvGenRate++;
            // If evGenRate is low for few consecutive frames, tracking is lost and abort
            if (mCntLowEvGenRate > DEF_L1_MAX_TRACK_LOST) {
                DLOG(WARNING) << "* L1 Tracking: Low event gen. rate for "
                              << mCntLowEvGenRate << " frames, abort tracking..." << endl;
                return -1;
            }
            return 1; // Keep events & state & continue
        }
    }

    bool EvImBuilder::resolveReconstStat(const bool iniRes, const ORB_SLAM3::ReconstInfo& rInfo,
                                         const std::vector<std::vector<cv::Point3f>>& pts3d, std::vector<int> vMatches12) {

        // If successful reconstruction, return true
        if (iniRes) {
            DLOG(INFO) << "EvImBuilder::resolveReconstStat: Good Reconst. by original response.\n";
            return iniRes;
        }
        // If not successful, consider less rigorous condition (no parallax)
        else {
            // In this case, both the number of matches and parallax must be good
            DVLOG(1) << "EvImBuilder::resolveReconstStat: ReconstInfo: " << rInfo.print();

            bool preCond = rInfo.mnBestGood > rInfo.mnSecondBest && rInfo.mBestParallax > rInfo.mSecondBestPar &&
                   rInfo.mnBestGood > DEF_TH_MIN_MATCHES;

            if (!preCond) {
                DLOG(INFO) << "EvImBuilder::resolveReconstStat: Bad Reconst. -> pre-condition is not fulfilled\n";
                return preCond;
            }
            else {
                return isReconstDepthGood(pts3d, vMatches12);
            }
        }
    }

    /*unsigned long  EvImBuilder::getPointTracks(std::vector<cv::Mat>& ptTracks) {

        int nFrames = 0;
        {
            std::unique_lock<mutex> lock(mMtxEvFrames);
            nFrames = mvEvFrames.size();
        }
        if (nFrames < 3) {
            cerr << nFrames << " frames is not enough for multiple view reconstruction.\n";
            return 0;
        }
        size_t nKpts = mvEvFrames[0].numAllKPts();
        int nTrackedPts = 0;
        EvFrame refFrame = mvEvFrames[0];

        for (size_t fr = 0; fr < nFrames; fr++) {

            cv::Mat_<double> frame(2, (int)nKpts);

            for (unsigned tr = 0; tr < nKpts; tr++) {
                // If curr point is tracked in all frames, add it to track
                if (mvMatchesCnt[tr] < nFrames) {
                    continue;
                }

                if (fr == 0) {
                    frame(0, (int) tr) = mvEvFrames[fr].getUndistKPtMono(tr).pt.x;
                    frame(1, (int) tr) = mvEvFrames[fr].getUndistKPtMono(tr).pt.y;
                }
                else {
                    frame(0, (int) tr) = mvEvFrames[fr].getUndistTrackedPts()[tr].pt.x;
                    frame(1, (int) tr) = mvEvFrames[fr].getUndistTrackedPts()[tr].pt.y;
                }

                nTrackedPts++;
            }
            ptTracks.push_back(frame);
        }
        return nTrackedPts;
    }*/

    void EvImBuilder::refineTrackedPts(const std::vector<bool>& vbReconstStat,
            vector<int>& vMatches12, vector<int>& vCntMatches, unsigned& nMatches) {

        size_t nMch = vMatches12.size();
        assert(nMch == vCntMatches.size() && nMch >= vbReconstStat.size());

        // Refine matches and reject outliers
        // Triangulation status
        if (nMch == vbReconstStat.size()) {

            for (size_t i = 0; i < nMch; i++) {

                if (vMatches12[i] >= 0 && !vbReconstStat[i]) {

                    vMatches12[i] = -1;
                    vCntMatches[i]--;
                    nMatches--;
                }
            }
        }
        // Transformation (Homo. or Fund.) status
        else if (nMatches == vbReconstStat.size()) {

            for (size_t i = 0, statCnt = 0; i < nMch; i++) {

                if (vMatches12[i] >= 0) {

                    if (!vbReconstStat[statCnt]) {

                        vMatches12[i] = -1;
                        vCntMatches[i]--;
                        nMatches--;
                    }
                    statCnt++;
                }
            }
        }
        else {
            LOG(WARNING) << "** EvImBuilder::refineTrackedPts, vbReconstStat size is not correct!\n";
        }
    }

    void EvImBuilder::updateState(const std::vector<EventData>& vEvData, const EvFramePtr& pEvFrame) {

        mCurrIdx++;
        mvSharedL2Evs.fillBuffer(vEvData);
        {
            std::unique_lock<mutex> lock(mMtxEvFrames);
            mvpEvFrames.push_back(pEvFrame);
        }
    }

    void EvImBuilder::updateLastPose(const float imStdDP, const float imStdBA, const float imStdEH, const bool inertial) {

        if (mIniChi2 < 0) {
            mIniChi2 = mLastChi2;
        }
        if (mIniChi2BA < 0) {
            mIniChi2BA = mLastChi2BA;
        }

        if (inertial) {
            if (mIniChi2 >= 0 && mLastChi2 >= 0) {

                // Reset inertial info. if total error grows
                if (mLastChi2 * DEF_CHI2_RATIO > mIniChi2) {

                    // Only last dPose's medDepth is required for inertial DP method
                    mpLastPoseDepth->updateLastPose(0, cv::Mat::eye(4, 4, CV_32FC1),
                                                    1, cv::Mat::eye(4, 4, CV_32FC1), 1.f);
                    mpImuInfo = make_shared<InitInfoIMU>();
                    mIniChi2 = -1;
                } else {
                    if (mLastChi2 < mIniChi2)
                        mLastChi2 = mIniChi2;

                    if (mpLastFrame)
                        mpLastFrame->SetNewBias(mpImuInfo->getLastImuBias());
                }
            }

            if (mIniChi2BA >= 0 && mLastChi2BA >= 0)
            {
                if (mLastChi2BA * DEF_CHI2_RATIO > mIniChi2BA) {

                    mpLastPoseDepthBA->updateLastPose(0, cv::Mat::eye(4, 4, CV_32FC1),
                                                      1, cv::Mat::eye(4, 4, CV_32FC1), 1.f);
                    cv::Mat baRwg = mpImuInfoBA->getLastRwg();
                    mpImuInfoBA = make_shared<InitInfoIMU>();
                    mpImuInfoBA->setLastRwg(baRwg);

                    mIniChi2BA = -1;
                } else if (mLastChi2BA < mIniChi2BA) {
                    mLastChi2BA = mIniChi2BA;
                }
            }

            // Only update mpImuManager if L2 is not initialized!
            if (!mpL2Tracker->isImuInitialized()) {
                // Update Important Parameters
                //InitInfoImuPtr pImuInfo = mpImuManager->getInitInfoIMU();
                //pImuInfo->setLastImuBias(mpImuInfo->getLastImuBias());
                //pImuInfo->setLastScale(mpImuInfoBA->getLastScale());
                //pImuInfo->setLastRwg(mpLastFrame->GetRotationInverse());
                //mpImuManager->updateState(pImuInfo); // Don't update this!
            }
        }

        // update last dPose based on L2 or BA
        if (!mpLastPoseDepth->isInitialized()) {
            const PoseDepthPtr pPoseInfo = mpL2Tracker->getLastPoseDepthInfo();
            if (mpLastPoseDepthBA->isInitialized()) {
                mpLastPoseDepth = make_shared<PoseDepthInfo>(mpLastPoseDepthBA);
            }
            else if (pPoseInfo->isInitialized()) {
                mpLastPoseDepth = make_shared<PoseDepthInfo>(pPoseInfo);
            }
        }
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    void EvImBuilder::makeFrame(const PoseImagePtr& pImage, unsigned int fId,
                                const std::vector<cv::KeyPoint> &p1, EvFramePtr &frame) {

        frame = make_shared<EvFrame>(mCurrIdx, pImage->mImage, pImage->ts, p1, mpCamera, mDistCoefs);
    }

    void EvImBuilder::makeFrame(const PoseImagePtr& pImage, unsigned int fId, unique_ptr<ELK_Tracker> &pLKTracker,
                                EvFramePtr &frame) {

        if (mStat == INIT) {

            // Construct initial event frame
            // Ref. frame doesn't have any previous siblings, except if it's inertial
            EvBaseTracker::makeFrame(pImage, fId, mpIniFrame);

            EvFramePtr pPreFrame = nullptr;
            if (mpLastFrame) {
                //pPreFrame = mpLastFrame; //make_shared<EvFrame>(*(mpCurrFrame));
            }
            mpIniFrame->setPrevFrame(pPreFrame);

            mpIniFrame->SetPose(cv::Mat::eye(4, 4, CV_32F));

            frame = mpIniFrame;
        }
        else if (!pLKTracker->getRefPoints().empty()){

            // TODO: Extract vMchCnt from LK Tracker and put it somewhere else

            // Level 1 Tracking
            // LK Optical Flow matching
            vector<KeyPoint> p1;
            vector<int> vMatches12;
            vector<int> vCntMatches = mvMatchesCnt;
            vector<float> vPxDisp;

            uint nMatches = pLKTracker->trackAndMatchCurrImage(pImage->mImage, p1, vMatches12, vCntMatches, vPxDisp);

            sort(vPxDisp.begin(), vPxDisp.end());
            float medPxDisp = vPxDisp[vPxDisp.size()/2];

            this->makeFrame(pImage, fId, p1, frame);
            frame->setRefFrame(mpIniFrame);//make_shared<EvFrame>(*mpIniFrame));
            frame->setPrevFrame(mvpEvFrames.back());//make_shared<EvFrame>(*(mvpEvFrames.back())));
            frame->setNumMatches(nMatches);
            frame->setMedianPxDisp(medPxDisp);
            frame->setMatches(vMatches12);
            mvMatchesCnt = vCntMatches;
        }
    }

    unsigned int
    EvImBuilder::trackAndFrame(const PoseImagePtr& pImage, unsigned int fId, unique_ptr<ELK_Tracker> &pLKTracker,
                               EvFramePtr &evFrame) {
        return 0;
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    int EvImBuilder::init(const cv::Mat& im, const double& ts) {

        vector<KeyPoint> p0 = mpIniFrame->getAllUndistKPtsMono();
        int nkpts = p0.size();

        mvMatchesCnt.resize(nkpts, 1);
        {
            std::unique_lock<mutex> lock(mMtxPts3D);
            mvPts3D.resize(nkpts);
        }

        // Init LK Tracker
        mpLKTracker->setRefImage(im, p0);

#ifdef SAVE_DEBUG_IMAGES
//        cv::Mat outIm;
//        cv::drawKeypoints(im, p0, outIm);
//        EvAsynchTracker::saveImage("l1_init_"+to_string(mIdxImSave++)+'_'+to_string(ts)+".png", outIm);
#endif

        DVLOG(1) << "EvImBuilder::step: Curr asynch tracker index: " << mCurrIdx << endl;
        DVLOG(1) << "EvImBuilder::step: Init completed with " << nkpts << " key points.\n";

        return nkpts;
    }

    int EvImBuilder::step(const cv::Mat& image, double ts, float &medPxDisp,
                          unsigned& nMatches, const bool refineRANSAC) {

        vector<KeyPoint> p1 = mpCurrFrame->getUndistTrackedPts();
        vector<int> vMatches12 = mpCurrFrame->getMatches();
        nMatches = mpCurrFrame->getNumMatches();
        medPxDisp = mpCurrFrame->getMedianPxDisp();

        // Disabled refine RANSAC or not enough matches
        if (!refineRANSAC || nMatches < DEF_TH_MIN_MATCHES) {
            return 1;
        }

        // Further refine matched points with RANSAC to reject outliers
        vector<KeyPoint> p0 = mpLKTracker->getRefPoints();
        size_t nLenMch = vMatches12.size();
        vector<bool> vbReconstStat(nLenMch, false);
        vector<vector<bool>> vbTriangStat;
        vector<cv::Mat> vR, vt;
        vector<vector<cv::Point3f>> vvP3D;
        ORB_SLAM3::ReconstInfo reconstInfo;

        bool reconstRes = mpCamera->ReconstructWithTwoViews(p0, p1, vMatches12,
                vR, vt, vvP3D, vbTriangStat, vbReconstStat, reconstInfo);

        reconstRes = this->resolveReconstStat(reconstRes, reconstInfo, vvP3D, vMatches12);

        // Unable to reconstruct
        if (!reconstRes) {
            this->refineTrackedPts(vbReconstStat, vMatches12, mvMatchesCnt, nMatches);
            mpCurrFrame->setMatches(vMatches12);
            //mvMatchesCnt = vCntMatches;
            return 2;
        }

        this->refineTrackedPts(vbTriangStat[0], vMatches12, mvMatchesCnt, nMatches);

        // Store the reconst. info somewhere safe!
        mpCurrFrame->setMatches(vMatches12);
        //mvMatchesCnt = vCntMatches;

        cv::Mat currTcw = cv::Mat::eye(4,4, CV_32F); //mLastDPose.clone();
        vR[0].copyTo(currTcw.rowRange(0,3).colRange(0,3));
        vt[0].copyTo(currTcw.rowRange(0,3).col(3));

        mpCurrFrame->SetPose(currTcw);
        {
            std::unique_lock<mutex> lock(mMtxPts3D);
            mvPts3D = vvP3D[0];
        }
        {
            std::unique_lock<mutex> lock1(mMtxStateBA);
            mLastIdxBA = static_cast<int>(mCurrIdx);
        }

#ifdef SAVE_DEBUG_IMAGES
        cv::Mat mchIm;
        cv::Mat im0;
        mpLKTracker->getRefImageAndPoints(im0, p0);
        Visualization::myDrawMatches(im0, p0, image, p1, vMatches12, mchIm);
        Visualization::saveImage(mRootImPath+"/l1_matches_"+to_string(mIdxImSave++)+'_'+to_string(ts)+".png", mchIm);
#endif

        DVLOG(1) << "EvImBuilder::step: Curr Idx: " << mCurrIdx << endl;
        DVLOG(1) << "EvImBuilder::step: Median pixel displacement: " << medPxDisp << endl;
        DVLOG(1) << "EvImBuilder::step: Num. Matched Inliers: " << nMatches << endl;

        return 0;
    }

    bool EvImBuilder::dispatchL2Image(const PoseImagePtr& pPoseImage) {

        if (mpL2Tracker->isTrackerReady()) {

            mpL2Tracker->fillImage(pPoseImage);
            return true;
        }
        else {
            mpPoseImageTemp = pPoseImage;
            return false;
        }
    }

    /*void EvImBuilder::refine(cv::Mat &R, cv::Mat &t, float &medDepth, int &stat) {

        // Multiple-view optimization -> Get MC-Image
        // Either BA or Contrast maximization
        // Multiple-view Rconstruction
        stat = 0;
        vector<cv::Mat> kpts, slimKpts;
        unsigned nKpts = this->getPointTracks(kpts);
        if (kpts.empty() || nKpts/kpts.size() < DEF_TH_MIN_MATCHES) {
            cerr << "Empty point track, continue...\n";
            stat = 1;
            return;
        }
        slimKpts.push_back(kpts[0]);
        slimKpts.push_back(kpts[kpts.size()-2]);
        slimKpts.push_back(kpts[kpts.size()-1]);
        cv::Mat K = mpCamera->toK();
        vector<cv::Mat> Rs, ts, pts3d;
        cv::sfm::reconstructM(kpts, Rs, ts, K, pts3d, true, false);

        if (Rs.empty() || ts.empty()) {
            cerr << "L1 Optim: Cannot find any suitable transformation.\n";
            stat = 2;
            return;
        }

        // BA Pose optimization
        //vector<vector<KeyPoint>> vKpts = mpKfInfo->getKeyPoints();
        //ORB_SLAM3::Optimizer::performPoseOnlyBA(vKpts, R, t, pts3d, mpCamera);

        Rs.back().copyTo(R.rowRange(0,3).colRange(0,3));
        ts.back().copyTo(t.rowRange(0,3).col(0));

        vector<float> depths;
        depths.resize(pts3d.size());
        for (size_t i = 0; i < pts3d.size(); i++) {

            cv::Vec3f pt3d;
            pts3d[i].copyTo(pt3d);
            depths[i] = sqrtf(powf(pt3d(0),2) + powf(pt3d(1),2) + pow(pt3d(2),2));
        }

        sort(depths.begin(), depths.end());
        medDepth = depths[depths.size()/2];

        // Compensate motion using last optimized pose
        cv::Mat Tcw = cv::Mat::eye(4,4, CV_32F);
        R.copyTo(Tcw.rowRange(0,3).colRange(0,3));
        t.copyTo(Tcw.rowRange(0,3).col(3));
        //mpKfInfo->addPose(Tcw.clone());

        cout << "Reconstructed with " << nKpts << " key points" << endl;
        cout << "Last Tcw: \n";
        for (int i = 0; i < Tcw.rows; i++) {
            for (int j = 0; j < Tcw.cols; j++) {
                cout << Tcw.at<float>(i,j) << ", ";
            }
            cout << endl;
        }
        cout << "Median depth: " << medDepth << endl;
    }*/

    /* -------------------------------------------------------------------------------------------------------------- */

    void EvImBuilder::initPosesSpeed(std::vector<EvFramePtr> &vpEvFrames, const cv::Mat &speed) {

        if (speed.empty() || vpEvFrames.empty()) {
            return;
        }

        // Init. first pose if not init. before
        EvFramePtr pIniFrame = vpEvFrames[0];

        cv::Mat Tcw = pIniFrame->mTcw.clone();
        double t0 = pIniFrame->mTimeStamp;

        // Perform the conversion and assignment;
        for (int i = 1; i < vpEvFrames.size(); i++) {

            EvFramePtr pFri = vpEvFrames[i];

            double dt = pFri->mTimeStamp-t0;

            Tcw = ORB_SLAM3::Converter::se3_to_SE3(speed, dt) * Tcw;

            pFri->SetPose(Tcw);

            t0 = pFri->mTimeStamp;
        }
    }

    void EvImBuilder::initPosesInertial(std::vector<EvFramePtr> &vpEvFrames) {

        // IMU must be initialized before and curr ref. frame has a prev. frame
        if (vpEvFrames.empty()) {
            return;
        }

        for (EvFramePtr& pFri : vpEvFrames) {

            if (!pFri->getPrevFrame())
                continue;

            IMU_Manager::predictStateIMU(pFri->getPrevFrame().get(), pFri.get());
        }
    }

    void EvImBuilder::initPreRefInertial(EvFramePtr& pEvFrame) {

        if (!pEvFrame->getPrevFrame()) {

            // For inertial optimization, the orientation of the first ref. must be correct
            cv::Mat twb0 = cv::Mat::zeros(3, 1, CV_32FC1);
            cv::Mat Vwb0 = cv::Mat::zeros(3, 1, CV_32FC1);
            cv::Mat Rwb0 = cv::Mat::eye(3,3,CV_32FC1);

            if (!pEvFrame->getFirstAcc().empty()) {

                InitInfoImuPtr pInfoImu = make_shared<InitInfoIMU>();
                pInfoImu->initRwg(pEvFrame->getFirstAcc());
                // Rwg == Rbw so Rwb = Rbw.t()
                Rwb0 = pInfoImu->getIniRwg().t();
            }

            pEvFrame->SetImuPoseVelocity(Rwb0, twb0, Vwb0);
        }
    }

    // The following 2 methods copy frames so other methods are not affected!
    // This only considers mpLastPoseDepth so update this if dpose from other methods is required
    cv::Mat EvImBuilder::resolveIniDPoses(std::vector<EvFramePtr>& vpEvFrames, const double t0, const double dt,
                                          cv::Mat& Tcr, float& medDepth, const bool inertial) {

        if (inertial) {
            {
                std::unique_lock<mutex> lock1(mMtxEvFrames);
                copyConnectedEvFrames(mvpEvFrames, vpEvFrames);
            }
            // Don't change any frame!
            initPreRefInertial(vpEvFrames[0]);
            initPosesInertial(vpEvFrames);
            if (mpLastPoseDepth->isInitialized()) {
                medDepth = static_cast<float>(mpLastPoseDepth->getLastDepth());
            }
            else {
                medDepth = 1;
            }
            return cv::Mat();
        }
        else {
            // From L1 (Last DPose or L1 BA)
            double ts0, lastDTs;
            mpLastPoseDepth->getDPose(ts0, lastDTs, Tcr, medDepth);

            // The TimeStamp must not be far from currTs,aAlso we must have a pose and medDepth
            if (!mpLastPoseDepth->isInitialized() || abs(ts0 + lastDTs - t0) > DEF_L1_TH_NEAR_POSE ||
                Tcr.empty() || lastDTs < 1e-6) {

                DLOG(WARNING) << "EvImBuilder::getDPoseMCI, could not reconst. using L2 info: DTimeStamps = "
                              << abs(ts0 + lastDTs - t0) << endl;
                return cv::Mat();
            }

            // Turn DPose to speed and calculate new DPose
            cv::Mat speed = ORB_SLAM3::Converter::SE3_to_se3(Tcr, lastDTs);

            {
                std::unique_lock<mutex> lock1(mMtxEvFrames);
                copyConnectedEvFrames({mpIniFrame, mvpEvFrames.back()}, vpEvFrames);
            }
            initPosesSpeed(vpEvFrames, speed);
            return speed.clone();
        }
    }

    int EvImBuilder::resolveIniPosesBA(std::vector<EvFramePtr> &vpEvFrames, vector<int>& vValidPts3D, bool inertial) {

        // Retrieve Info to see if we had a successful reconstruction
        int nFrames = 0;
        {
            std::unique_lock<mutex> lock(mMtxEvFrames);
            nFrames = mvpEvFrames.size();
        }
        int lastIdxBA = 0;
        {
            std::unique_lock<mutex> lock(mMtxStateBA);
            lastIdxBA = mLastIdxBA;
        }

        // If we never had a reconstructions, abort
        if (nFrames < 2 || ((lastIdxBA < 0 || lastIdxBA >= nFrames))) {

            DLOG(WARNING) << "EvImBuilder::getBAMCI, No successful reconstruction available yet.\n";
            return -1;
        }

        EvFramePtr pCurrFrame;
        {
            std::unique_lock<mutex> lock(mMtxEvFrames);
            pCurrFrame = mvpEvFrames[lastIdxBA];
        }

        vValidPts3D = pCurrFrame->getMatches();

        cv::Mat speed = ORB_SLAM3::Converter::SE3_to_se3(pCurrFrame->mTcw, pCurrFrame->mTimeStamp-mpIniFrame->mTimeStamp);

        if (inertial) {
            {
                std::unique_lock<mutex> lock1(mMtxEvFrames);
                copyConnectedEvFrames(mvpEvFrames, vpEvFrames);
            }
            initPosesSpeed(vpEvFrames, speed);
        }
        else {
            // It's not a good idea to use last frame and init. pose with const. speed model!
            copyConnectedEvFrames({mpIniFrame, pCurrFrame}, vpEvFrames);
        }
        return 1;
    }

    // Can't send prePoseDepth in this version!
    int EvImBuilder::resolveLastDPose(const double t0, const double dt, cv::Mat& Tcr, float& medDepth,
                                      const bool optimize, const bool inertial) {

        // Resolve DPose
        vector<EvFramePtr> vpEvFrames;
        cv::Mat speed = resolveIniDPoses(vpEvFrames, t0, dt, Tcr, medDepth, inertial);

        if ((!inertial && speed.empty()) || vpEvFrames.empty()) {
            return -1;
        }

        // Optimize if required
        if (optimize) {

            vector<cv::Point3f> vdumyPts3d;
            vector<int> vdumyValid;
            double depth = medDepth;
            const int nIter = 10;

            if (inertial) {

                if (!mpImuInfo) {
                    mpImuInfo = make_shared<InitInfoIMU>();
                    mpImuInfo->updateState(mpImuManager->getInitInfoIMU());
                }
                if (!std::isfinite(depth)) {
                    LOG(WARNING) << "EvImBuilder::resolveLastDPose: Bad median depth detected: " << depth << endl;
                    depth = 1.0;
                }

                mLastChi2 = MyOptimizer::OptimInitVI(vpEvFrames, vdumyPts3d, vdumyValid, depth, mpImuInfo,
                                         nIter, true, false, true, false);
            }
            else {
                mLastChi2 = MyOptimizer::optimizeSim3L1(vpEvFrames, depth, 5.99f, nIter);
            }

            medDepth = static_cast<float>(depth);
        }
        else if (!inertial) {

            Tcr = ORB_SLAM3::Converter::se3_to_SE3(speed, dt);
            return 1;
        }

        // Copy last frame for later processing
        mpLastFrame = vpEvFrames.back(); // Need to copy??
        mpLastPoseDepth->updateLastPose(vpEvFrames[0].get(), vpEvFrames.back().get(), medDepth);
        Tcr = mpLastPoseDepth->getDPose();

        return 1;
    }

    // WARNING: For this to work, l2Tcr must be the relative transform spanning the EvWin
    void EvImBuilder::getDPoseMCI(const vector<EventData>& evs, PoseImagePtr& pMCI, float &imSTD,
                                  const bool optimize, const bool inertial) {

        float medDepth = 1.f;
        cv::Mat Tcw;
        int res = resolveLastDPose(evs[0].ts, evs.back().ts - evs[0].ts, Tcw, medDepth, optimize, inertial);
        if (res < 0)
            return;

        // Reconstruct image and get image variance (std)
        // For comparable image std, do not normalize image
        cv::Mat mcImage = EvImConverter::ev2mci_gg_f(evs, mpCamera, Tcw, medDepth, imWidth, imHeight,
                                                     mImSigma, false, false);

        // Normalize and compute image std
        imSTD = EvImConverter::measureImageFocus(mcImage);
        double ts = evs.back().ts;

        cv::normalize(mcImage, mcImage, 255, 0,  cv::NORM_MINMAX, CV_8UC1);

        pMCI = make_shared<PoseImage>(ts, mcImage, Tcw, 1, "DP");
    }

    // Checks evs buffer state and nFrames outside here!
    int EvImBuilder::resolveLastPoseMap(cv::Mat& Tcw, float& medDepth, const bool optimize, const bool inertial) {

        vector<EvFramePtr> vpEvFrames;
        vector<int> vValidPts3D;
        int res = resolveIniPosesBA(vpEvFrames, vValidPts3D, inertial);
        if (res < 0)
            return res;

        vector<cv::Point3f> vPts3D;
        {
            std::unique_lock<mutex> lock(mMtxPts3D);
            vPts3D = mvPts3D;
        }

        mpLastFrameBA = vpEvFrames.back();

        if (optimize) {
            // Optimize and refine reconstructed data using BA
            double depth = 1.0;
            const int nIter = 10;

            if (inertial) {

                if (!mpImuInfoBA) {
                    mpImuInfoBA = make_shared<InitInfoIMU>();
                    mpImuInfoBA->updateState(mpImuManager->getInitInfoIMU());
                    // Due to unstability issues in pts3d depth, transpose the Rwg dir.
                    mpImuInfoBA->setLastRwg(mpImuInfoBA->getLastRwg().t());
                }

                medDepth = mpLastFrameBA->computeSceneMedianDepth(2, vPts3D);
                EvLocalMapping::scalePoseAndMap(vpEvFrames, vPts3D, medDepth);

                mLastChi2BA = MyOptimizer::OptimInitVI(vpEvFrames, vPts3D, vValidPts3D, depth, mpImuInfoBA,
                                         nIter, true, false, true, true);
            }
            else {
                mLastChi2BA = MyOptimizer::OptimInitV(vpEvFrames, vPts3D, vValidPts3D, nIter, true, false);
            }
        }

        // Calculate median depth
        medDepth = mpLastFrameBA->computeSceneMedianDepth(2, vPts3D);
        mpLastPoseDepthBA->updateLastPose(mpLastFrameBA->getRefFrame().get(), mpLastFrameBA.get(), medDepth);
        Tcw = mpLastPoseDepthBA->getDPose();

        return 1;
    }

    // We might hold and use pose and medDepth for
    // next reconstruction (even if reconst. fails, we can triangulate 3d points)
    void EvImBuilder::getBAMCI(const vector<EventData>& evs, PoseImagePtr& pMCI, float &imSTD,
                               const bool optimize, const bool inertial) {

        float medDepth = 1.f;
        cv::Mat Tcw;
        int res = resolveLastPoseMap(Tcw, medDepth, optimize, inertial);
        if (res < 0)
            return;

        // Reconstruct image and get image variance (std)
        cv::Mat mcImage = EvImConverter::ev2mci_gg_f(evs, mpCamera, Tcw, medDepth,
                                                     imWidth, imHeight, mImSigma, false, false);
        // Multi-depth reconst is not stable at all!!!!!
//        MyDepthMap mDmap;
//        mDmap.populate(mpIniFrame.getAllUndistKPtsMono(), mvPts3D, currFrame.getMatches());
//        mcImage = EvImConverter::ev2mci_gg_f(l2Evs, mpCamera, Tcw, mDmap,
//                                             imWidth, imHeight, mImSigma, false, false);

        // Normalize and compute image std
        imSTD = EvImConverter::measureImageFocus(mcImage);
        double ts = evs.back().ts;

        cv::normalize(mcImage, mcImage, 255, 0,  cv::NORM_MINMAX, CV_8UC1);

        pMCI = make_shared<PoseImage>(ts, mcImage, Tcw, 1, "BA");
    }

    void EvImBuilder::getEvHist(const vector<EventData>& evs, PoseImagePtr& pMCI, float &imSTD, const unsigned long prefSize) {

        auto evsEnd = evs.end();
        auto evsBegin = evs.begin();
        if (prefSize > 0 && prefSize <= evs.size()) {
            evsBegin = evsEnd - prefSize;
        }
        vector<EventData> vEvWin(evsBegin, evsEnd);

        // Reconstruct image and get image variance (std)
        cv::Mat image = EvImConverter::ev2im_gauss(vEvWin, imWidth, imHeight, mImSigma, false, false);

        // Normalize and compute image std
        imSTD = EvImConverter::measureImageFocus(image);
        double ts = vEvWin.back().ts;

        cv::normalize(image, image, 255, 0,  cv::NORM_MINMAX, CV_8UC1);

        pMCI = make_shared<PoseImage>(ts, image, cv::Mat::eye(4,4,CV_32FC1), 1, "EH");
    }

    int EvImBuilder::resolveLastAtt2Params(cv::Mat &paramsSE2, const bool optimize) {

        // Optimize tracked points to find a 2D Affine transformation (rotation and translation)
        {
            std::unique_lock<mutex> lock(mMtxStateParams2D);
            if (!mLastParams2D.empty()) {
                paramsSE2 = mLastParams2D.clone();
            }
        }

        if (optimize) {
            int nEvFrames = 0;
            EvFramePtr lastFrame;
            {
                std::unique_lock<mutex> lock(mMtxEvFrames);
                nEvFrames = mvpEvFrames.size();
                if (nEvFrames > 0) {
                    lastFrame = mvpEvFrames.back();
                }
            }
            if (nEvFrames < 2 || paramsSE2.empty()) {
                DLOG(WARNING) << "EvImBuilder::getAff2DMCI: Empty ev win. or not enough tracked frames, abort\n";
                return -1;
            }

            vector<EvFrame> vEvFrames2Opt = {*mpIniFrame, *lastFrame};
            MyOptimizer::optimize2D(vEvFrames2Opt, paramsSE2, 20);

            {
                std::unique_lock<mutex> lock(mMtxStateParams2D);
                mLastParams2D = paramsSE2.clone();
            }
            return 1;
        }
        else {
            if (paramsSE2.empty()) {
                DLOG_EVERY_N(WARNING, 100) << "EvImBuilder::resolveLastAtt2Params: Last 2D opt. params not available, abort\n";
                return -1;
            }
            return 2;
        }
    }

    void EvImBuilder::getAff2DMCI(const vector<EventData>& evs, PoseImagePtr& pMCI, float &imSTD, const bool optimize) {

        cv::Mat paramsSE2 = cv::Mat::zeros(3, 1, CV_32FC1);
        // Resolve Last 2D Parameters
        int res = resolveLastAtt2Params(paramsSE2, optimize);
        if (res < 0)
            return;

        // Reconstruct image and get image variance (std)
        cv::Mat mcImage = EvImConverter::ev2mci_gg_f(evs, mpCamera, paramsSE2,
                imWidth, imHeight, mImSigma, false, false);

        // Normalize and compute image std
        imSTD = EvImConverter::measureImageFocus(mcImage);
        double ts = evs.back().ts;

        cv::normalize(mcImage, mcImage, 255, 0,  cv::NORM_MINMAX, CV_8UC1);

        pMCI = make_shared<PoseImage>(ts, mcImage, cv::Mat::eye(4,4,CV_32FC1), 1, "Opt");
    }

    // TODO: Check for empty image (when no method can reconst. one)
    void EvImBuilder::generateMCImage(const vector<EventData>& evs, PoseImagePtr& pMCI, const bool useL2,
                                      const bool optimize, const bool inertial) {

        if (evs.empty()) {
            DLOG(WARNING) << "EvImBuilder::generateMCImage: Empty ev buffer, abort\n";
            return;
        }
        {
            std::unique_lock<mutex> lock1(mMtxEvFrames);
            int nFrames = mvpEvFrames.size();
        }

        auto t0_prog = std::chrono::high_resolution_clock::now();

        // We can use the latest median depth and pose from L2 Tracker (if available)
        PoseImagePtr mciL2;
        float imSTDL2 = -1.f;
        thread* thMciL2;
        if (useL2) {
            thMciL2 = new thread(&EvImBuilder::getDPoseMCI, this, ref(evs), ref(mciL2), ref(imSTDL2), optimize, inertial);
//            this->getDPoseMCI(evs, mciL2, imSTDL2, optimize, inertial);
        }

        auto tp_l2 = std::chrono::high_resolution_clock::now();

        // We might use the reconstruction info (if available) & BA
        PoseImagePtr mciBA;
        float imSTDBA = -1.f;
        auto* thMciBA = new thread(&EvImBuilder::getBAMCI, this, ref(evs), ref(mciBA), ref(imSTDBA), optimize, inertial);
//        this->getBAMCI(evs, mciBA, imSTDBA, optimize, inertial);

        auto tp_ba = std::chrono::high_resolution_clock::now();

        // 2 Reconst. options are always available:
        // 1. No motion compensation (Simple Event Histogram)
        PoseImagePtr mciEH;
        float imSTDEH = -1.f;
        auto* thMciEH = new thread(&EvImBuilder::getEvHist, this, ref(evs), ref(mciEH), ref(imSTDEH), 0);
//        this->getEvHist(evs, mciEH, imSTDEH);

        auto tp_eh = std::chrono::high_resolution_clock::now();

        // Since the second method (optimization) is more involved, do it only if needed ->
        // It's advantageous to always do it.
        // 2. Fit a model with optimization
        PoseImagePtr mciOpt;
        float imSTDOpt = -1.f;
        auto* thMciOpt = new thread(&EvImBuilder::getAff2DMCI, this, ref(evs), ref(mciOpt), ref(imSTDOpt), optimize);
//        this->getAff2DMCI(evs, mciOpt, imSTDOpt, optimize);

        auto tp_opt = std::chrono::high_resolution_clock::now();

        if (useL2 && thMciL2) {
            thMciL2->join();
        }
        thMciBA->join();
        thMciEH->join();
        thMciOpt->join();

        // return the model with maximum image variance (contrast)
        MciInfo bestReconst;
        if (useL2) {
            bestReconst.insert(make_pair(imSTDL2, mciL2));
        }
        bestReconst.insert(make_pair(imSTDBA, mciBA));
        bestReconst.insert(make_pair(imSTDEH, mciEH));
        bestReconst.insert(make_pair(imSTDOpt, mciOpt));

        // Currently, imSTD is not saved and reused for later
        pMCI = bestReconst.begin()->second;

        // If the winner is EH, return the events constructed from the latest half of events
        string methodLabel = pMCI->uri;
        if (methodLabel == "EH") {
            this->getEvHist(evs, pMCI, imSTDEH, evs.size()/2);
        }

        // Update important variables
        updateLastPose(imSTDL2, imSTDBA, imSTDEH, inertial);

        if (useL2) {
            delete thMciL2;
        }
        delete thMciBA;
        delete thMciEH;
        delete thMciOpt;

        DLOG(INFO) << "EvImBuilder::generateMCImage: Best reconstruction is: " << methodLabel
                   << ", with imSTD: " << bestReconst.begin()->first << endl;

#ifdef SAVE_DEBUG_IMAGES
        auto dl2 = chrono::duration_cast<chrono::microseconds>(tp_l2 - t0_prog);
        auto dba = chrono::duration_cast<chrono::microseconds>(tp_ba - tp_l2);
        auto deh = chrono::duration_cast<chrono::microseconds>(tp_eh - tp_ba);
        auto dopt = chrono::duration_cast<chrono::microseconds>(tp_opt - tp_eh);

        DLOG(INFO) << "EvImBuilder::generateMCImage: Timing stats: tl2 = " << dl2.count()
                   << ", tba = " << dba.count() << ", teh = " << deh.count() << ", topt = " << dopt.count() << endl;
        // Visualization
        Visualization::saveReconstImages(bestReconst, mRootImPath);
#endif
    }

    void EvImBuilder::getSynchMCI(const std::vector<EventData> &evs, PoseImagePtr& pMCI, const bool useL2, const bool optimize) {

        if (evs.empty()) {
            DLOG(WARNING) << "EvImBuilder::getDPoseMCI: Empty ev buffer, abort\n";
            return;
        }

        // If ImBuilder state is good, return generateMCImage
        int nFrames = 0;
        {
            std::unique_lock<mutex> lock(mMtxEvFrames);
            nFrames = mvpEvFrames.size();
        }
        if (optimize && nFrames > 1) {

            this->generateMCImage(evs, pMCI, useL2, optimize);
        }
        else {
            // We can always do this because we don't optimize
            this->generateMCImage(evs, pMCI, useL2, false);
        }
        // Else, simply return EH from the most recent events
//        else {
//            float imSTD = -1.f;
//            this->getEvHist(evs, mcImage, imTs, imSTD, evs.size()/2);
//
//            // For Visualization purposes
//#ifdef SAVE_DEBUG_IMAGES
//            DLOG(INFO) << "EvImBuilder::getSynchMCI: Return EvHist of " << evs.size()/2
//                       << " events of all " << evs.size() << " events\n";
//
//            cv::Mat im = mcImage.clone();
//            cv::cvtColor(im, im, CV_GRAY2BGR);
//            std::ostringstream imText;
//            imText << "EH, Best, ts: " << imTs << ", STD: " << imSTD;
//            cv::putText(im, imText.str(), Point2f(10,10),cv::FONT_HERSHEY_COMPLEX_SMALL,
//                        ((float)im.cols*1.5f)/720.f, cv::Scalar(0, 180, 0), 1);
//            Visualization::saveImage(mRootImPath+"/l1_reconst_im_"+std::to_string(0)+".png", im);
//#endif
//        }
    }

    /*void EvImBuilder::refineFocusPlan2d(double& omega0, double& vx0, double& vy0, int &stat) {

        EvOptimizer::optimizeFocus_MS_RT2D(mvSharedL2Evs.getVector(),
                mpEvParams, mpCamera, omega0, vx0, vy0);
    }*/

    /* -------------------------------------------------------------------------------------------------------------- */

    // This might further broken into 2 or more threads
    void EvImBuilder::Track() {

        while (!this->isStopped()) {

            if (isInputGood()) {

                mbIsProcessing = true;

                if (mStat == IDLE) {
                    mStat.update(INIT);
                }

                if (mStat == INIT) {
                    // Reset states and data output buffers
                    this->reset();
                }

                // Retrieve events
                vector<EventData> l1Evs = mqEvChunks.front();
                if (l1Evs.empty()) {
                    DLOG(ERROR) << "* L1 Image Builder: empty event chunk\n";
                    mqEvChunks.pop();
                    mbIsProcessing = false;
                    continue;
                }

                double imTs = l1Evs.back().ts;
                double evGenRate = calcEventGenRate(l1Evs, this->imWidth, this->imHeight);
                int evRateStat = checkEvGenRate(evGenRate, mStat.getState());

                if (evRateStat != 0) {

                    if (evRateStat == -1) {
                        this->resetOptimVars();
                        mStat.update(INIT);
                    }
                    else if (evRateStat == 1) {
                        mvSharedL2Evs.fillBuffer(l1Evs);
                    }
                    mqEvChunks.pop();
                    mbIsProcessing = false;
                    continue;
                }

                // Turn events to image
                cv::Mat image = EvImConverter::ev2im_gauss(l1Evs, this->imWidth, this->imHeight, this->mImSigma);
                PoseImagePtr pImage = make_shared<PoseImage>(imTs, image, cv::Mat(), 0, "");

                this->makeFrame(pImage, mCurrIdx, mpLKTracker, mpCurrFrame);

                // IMU Preintegration
                if (mpSensorConfig->isInertial() && mpImuManager) {

                    mpCurrFrame->mImuCalib = *(mpImuManager->getImuCalib());
                    // TODO: Maybe better to get current bias from last frame??
                    mpCurrFrame->mImuBias = mpImuManager->getLastImuBias();
                    mpImuManager->preintegrateIMU(mnImuID, mpCurrFrame, mpCurrFrame->mImuBias);
                }

                bool sendTinyFrame = false, sendMCF = false;

                if (mStat == INIT) {

                    // Initialize mpIniFrame & LK Tracker
                    int nKpts = this->init(image, imTs);

                    // Also check there is enough features
                    // In continuous mode we go to next step if it's fixed win. size too
                    if (nKpts > DEF_TH_MIN_KPTS || (mbContTracking && mbFixedWinSize)) {

                        updateState(l1Evs, mpIniFrame);

                        mStat.update(TRACKING);
                        sendTinyFrame = true;
                    }
                    else {
                        // TODO: What about resetting other vars??
                        this->updateL1ChunkSize(mInitL1EvWinSize);
                        DLOG(WARNING) << "* L1 Init, " << imTs << ": Not enough key points, " << nKpts << endl;
                    }
                }
                else if (mStat == TRACKING) {

                    unsigned nMatches = 0;
                    float medPxDisp = 0.f;
                    int reconstRes = this->step(image, imTs, medPxDisp, nMatches);

                    // If there is not enough matches, reset every thing
                    // Abort and reinit. only if low frames tracked...
                    if (nMatches < DEF_TH_MIN_MATCHES && !mbContTracking) {

                        DLOG(WARNING) << "* L1 Tracking, " << imTs << ": Not enough tracked points, " << nMatches
                             << ", Reinit...\n";

                        if (mCurrIdx < 3) {
                            // TODO: Maybe better to get this update out????
                            this->updateL1ChunkSize(mInitL1EvWinSize);
                            mStat.update(INIT);
                            mqEvChunks.pop();
                            mbIsProcessing = false;
                            continue;
                        }
                    }

                    updateState(l1Evs, mpCurrFrame);

                    bool dispatchMCI = this->resolveEvWinSize(medPxDisp, imTs);

                    if (dispatchMCI || nMatches < DEF_TH_MIN_MATCHES) {

                        mStat.update(INIT);

                        sendMCF = true;
                    }
                    else {
                        sendTinyFrame = true;
                    }
                }

                mqEvChunks.pop();


                PoseImagePtr pPoseImage = make_shared<PoseImage>(imTs, image, cv::Mat(), 0, "");

                // Also track raw frame if we need raw frame tracking
                // Note: This is done because LK method works better with small pixel displacements
                if (needTrackTinyFrame() && sendTinyFrame) {

                    // For SynchTrackerU, tiny frame must be supplied manually!
                    if (mpSensorConfig->isImage()) {

                        std::shared_ptr<EvSynchTrackerU> pSynchTracker = dynamic_pointer_cast<EvSynchTrackerU>(mpL2Tracker);
                        if (pSynchTracker) {
                            pSynchTracker->trackTinyFrameSynch(pPoseImage);
                            sendTinyFrame = false;
                        }
                    }
                    // If L2 is ready, trigger L2 tracker with new reconstructed image
                    //bool res = dispatchL2Image(pPoseImage);

                    //if (!res) {
                        //mLastStat.update(TRACKING);
                        //mStat.update(JAMMED_L2);
                    //}
                }

                if (needDispatchMCF() && sendMCF) {

                    // Removed the optimization step and do things here
                    // TODO: We can even spawn a thread here

                    const bool isInertial = mpSensorConfig->isInertial() && !mpCamParams->isFisheye(); // && mpL2Tracker->isImuInitialized()

                    vector<EventData> vAccEvs = mvSharedL2Evs.getVector();
                    this->generateMCImage(vAccEvs, pPoseImage, true, true, isInertial);

                    if (isMcImageGood(pPoseImage->mImage) || mbContTracking) {

                        // If L2 is ready, trigger L2 tracker with new reconstructed image
                        //bool res = dispatchL2Image(pPoseImage);

                        //if (!res) {
                        mLastStat.update(INIT);
                            //mStat.update(JAMMED_L2);
                        //}
                    }
                    else {
                        sendMCF = false;
                        DLOG(WARNING) << "* L1 Discrete Tracking: MC-Image is not good in "
                                      << "terms of the number of detected kpts for L2\n";
                    }

                    // Refill the overlapped buffer in continuous mode
                    if (mbContTracking) {
                        ulong nWinOverlap = this->resolveWinOverlap();
                        vector<EventData> vEvsOV(vAccEvs.end()-nWinOverlap, vAccEvs.end());
                        mpTrackManager->injectEventsBegin(vEvsOV);
                    }
                }

                if ((needTrackTinyFrame() && sendTinyFrame) || (needDispatchMCF() && sendMCF)) {

                    mTrackingWD.reset();
                    while (!dispatchL2Image(pPoseImage)) {

                        //LOG_EVERY_N(WARNING, 1000) << "* L2 Tracker is Jammed at Image Timestamp: " << imTs << endl;
                        this_thread::sleep_for(std::chrono::microseconds(500));

#ifdef ACTIVATE_WATCH_DOG
                        mTrackingWD.step();
#endif
                    }
                }

                mbIsProcessing = false;
            }
            /*if (mStat == JAMMED_L2) {

                DLOG(INFO) << "L1: Waiting for L2 Tracker to recover..." << endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(3));

                if (mpL2Tracker->isTrackerReady()) {

                    mpL2Tracker->fillImage(mpPoseImageTemp);
                    mStat.update(mLastStat.getState());
                }
            }*/

            if (mStat == IDLE) {
                VLOG_EVERY_N(3, 1000) << "L1: Idle state...\n";
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            if (!this->sepThread) { break; }
        }
    }


} //EORB_SLAM

