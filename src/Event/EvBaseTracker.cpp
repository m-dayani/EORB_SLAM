//
// Created by root on 1/17/21.
//

#include "EvBaseTracker.h"

#include <utility>

namespace EORB_SLAM {

    EvBaseTracker::EvBaseTracker() :
            sepThread(), mbIsProcessing(false), mCurrIdx(0), mbContTracking(false), isEvsRectified(false), mnEvPts(100), mThFAST(1),
            mFtDtMode(0), imWidth(240), imHeight(180), mImSigma(1.f), mDistCoefs(cv::Mat::zeros(4,1, CV_32F)),
            mK(cv::Mat::eye(3,3, CV_32F)), mMchWindow(DEF_MATCHES_WIN_NAME), mIdxImSave(0),
            mRootImPath("../data/image"), mnImuID(0), mpImuManager(nullptr), mpCamera(nullptr), mFrDrawerId(-1),
            mpFrameDrawer(nullptr), mTrackingWD("BaseTracker", 3)
{}

    EvBaseTracker::EvBaseTracker(EvParamsPtr evParams, CamParamsPtr camParams) :

            sepThread(), mbIsProcessing(false), mCurrIdx(0), mpEvParams(std::move(evParams)),
            mpCamParams(std::move(camParams)), mbContTracking(mpEvParams->continTracking), isEvsRectified(false),
            mnEvPts(mpEvParams->maxNumPts), mThFAST(mpEvParams->fastTh), mFtDtMode(mpEvParams->detMode),
            imWidth(static_cast<int>(mpEvParams->imWidth)), imHeight(static_cast<int>(mpEvParams->imHeight)),
            mImSigma(mpEvParams->l1ImSigma), mDistCoefs(mpCamParams->mDistCoef), mK(mpCamParams->mK),
            mMchWindow(DEF_MATCHES_WIN_NAME), mIdxImSave(0), mRootImPath("../data/image"), mnImuID(0),
            mpImuManager(nullptr), mpCamera(nullptr), mFrDrawerId(-1), mpFrameDrawer(nullptr),
            mTrackingWD("BaseTracker", 3)
    {}

    EvBaseTracker::~EvBaseTracker() {
        mpFrameDrawer = nullptr;
        //mpCamera = nullptr;
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    void EvBaseTracker::resetAll() {

        mCurrIdx = 0;
        mIdxImSave = 0;

        // Empty Queues
        if (!mqImuMeas.isEmpty()) {
            mqImuMeas.clear();
        }
        if (!mqEvChunks.isEmpty()) {
            mqEvChunks.clear();
        }
        if (!mqImageBuff.isEmpty()) {
            mqImageBuff.clear();
        }

        mvMatchesCnt.clear();

        {
            std::unique_lock<mutex> lock(mMtxPts3D);
            mvPts3D.clear();
        }

        mTrackingTimer.reset();
    }

    void EvBaseTracker::makeFrame(const PoseImagePtr& pImage, const unsigned fId, EvFramePtr &frame) {

        if (mpORBDetector) {

            frame = make_shared<EvFrame>(fId, pImage->mImage, pImage->ts, mpORBDetector.get(), mpCamera, mDistCoefs);
        }
        else {
            LOG(ERROR) << "* L1 Init, " << pImage->ts << ": Feature detection mode set to ORB, but empty ORBDetector object.\n";
            frame = nullptr;
            return;
        }
    }

    void EvBaseTracker::copyConnectedEvFrames(const std::vector<EvFramePtr>& vpInFrames, std::vector<EvFramePtr>& vpOutFrames) {

        const size_t nFrames = vpInFrames.size();
        vpOutFrames.resize(nFrames);
        EvFramePtr pNewRef;

        for (size_t i = 0; i < nFrames; i++) {

            vpOutFrames[i] = make_shared<EvFrame>(*(vpInFrames[i]));

            if (i == 0) {
                pNewRef = vpOutFrames[i];
            }
            else {
                vpOutFrames[i]->setPrevFrame(vpOutFrames[i-1]);
                vpOutFrames[i]->setRefFrame(pNewRef);
            }
        }
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    bool compare_depth(const cv::Point3f& first, const cv::Point3f& second) {
        return first.z < second.z;
    }

    bool EvBaseTracker::isReconstDepthGood(const std::vector<std::vector<cv::Point3f>>& pts3d, std::vector<int>& vMatches12) {

        int nMatches = vMatches12.size();
        if (pts3d.size() < 2 || pts3d[0].size() != nMatches) {
            DLOG(INFO) << "EvBaseTracker::isReconstDepthGood: Reconst. depth is not good because there are no reconst.!\n";
            return false;
        }
        if (pts3d[1].size() != nMatches) {
            DLOG(INFO) << "EvBaseTracker::isReconstDepthGood: Reconst. depth is good because there is no other reconst.!\n";
            return true;
        }

        // calc. median depths
        int hSz = nMatches / 2;
        vector<cv::Point3f> pts3d0(pts3d[0].begin(), pts3d[0].end());
        vector<cv::Point3f> pts3d1(pts3d[1].begin(), pts3d[1].end());
        // sort points by depth
        std::sort(pts3d0.begin(), pts3d0.end(), compare_depth);
        std::sort(pts3d1.begin(), pts3d1.end(), compare_depth);

        float d0 = pts3d0[hSz].z;
        float d1 = pts3d1[hSz].z;

        bool res = d0 > 1.f && (d1 < 1.f || fabs(d0) < fabs(d1));

        DLOG(INFO) << "EvBaseTracker::isReconstDepthGood: d0 = " << d0 << ", d1 = " << d1 << ", res = " << res << endl;

        return res;
    }

    void EvBaseTracker::fillEvents(const vector<EventData> &evs) {

        mqEvChunks.push(evs);
    }

    bool EvBaseTracker::isImageBuffGood() {

        //std::unique_lock<mutex> lock1(mMtxImBuff);
        return !mqImageBuff.isEmpty();
    }

    bool EvBaseTracker::isImageBuffSaturated(std::size_t maxSize) {

        //unique_lock<mutex> lock1(mMtxImBuff);
        return mqImageBuff.size() >= maxSize;
    }

    void EvBaseTracker::initFeatureExtractors(int ftDtMode, const ORB_SLAM3::ORBxParams& parORB) {

        ORB_SLAM3::ORBxParams prORB(parORB);

        if (ftDtMode == 1 || ftDtMode == 2) {
            DLOG(INFO) << "L1-ImageBuilder feature detector is ORB\n";
        }
        else {
            prORB.nlevels = 1;
            prORB.scaleFactor = 1.f;
            DLOG(INFO) << "L1-ImageBuilder feature detector is FAST (ORB with 1 scale level)\n";
        }

        mpORBDetector = make_shared<ORB_SLAM3::ORBextractor>(prORB);
    }

    void EvBaseTracker::fillImage(const PoseImagePtr& pPoseImage) {

        //std::unique_lock<mutex> lock1(mMtxImBuff);
        mqImageBuff.push(pPoseImage);
    }

    PoseImagePtr EvBaseTracker::frontImage() {

        //std::unique_lock<mutex> lock1(mMtxImBuff);
        return mqImageBuff.front();
    }

    void EvBaseTracker::popImage() {

        //std::unique_lock<mutex> lock1(mMtxImBuff);
        mqImageBuff.pop();
    }

    void EvBaseTracker::fillIMU(const std::vector<ImuData> &imuMeas) {

        mqImuMeas.push(imuMeas);
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    void EvBaseTracker::setImuManagerAndChannel(const std::shared_ptr<IMU_Manager> &pImuManager) {

        mnImuID = pImuManager->setNewIntegrationChannel();
        mpImuManager = pImuManager;
    }

    void EvBaseTracker::getValidInitPts3D(const std::vector<EvFramePtr> &vpFrames, std::vector<int> &vValidPts3D) {

        if (vpFrames.empty()) {
            return;
        }

        for (int i = 0; i < vValidPts3D.size(); i++) {

            int nMatches = 0;

            for (int j = 0; j < vpFrames.size(); j++) {

                int matches12 = vpFrames[j]->getMatches(i);
                if (matches12 >= 0)
                    nMatches++;
            }

            if (nMatches > 0)
                vValidPts3D[i] = nMatches;
        }
    }

    void EvBaseTracker::setFrameDrawerChannel(MyFrameDrawer *pFrDrawer) {

        mpFrameDrawer = pFrDrawer;
        FrameDrawFilterPtr pFrDrFilter = make_shared<SimpleFrameDrawFilter>();
        mFrDrawerId = mpFrameDrawer->requestNewChannel(pFrDrFilter);
    }

} // EORB_SLAM