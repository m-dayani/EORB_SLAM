//
// Created by root on 12/24/20.
//

#include "MyDataTypes.h"

#include <utility>
#include <thread>
#include <chrono>

#include "Frame.h"
#include "KeyFrame.h"
#include "EventFrame.h"

using namespace std;


namespace EORB_SLAM {

    ORB_SLAM3::IMU::Point ImuData::toImuPoint(const double tsFactor) {
        return ORB_SLAM3::IMU::Point{accel[0], accel[1], accel[2],
                                     gyro[0], gyro[1], gyro[2], ts/tsFactor};
    }

    /* ------------------------------------------------------------------------------------------------------------- */

    string MySensorConfig::mapConfig(const SensorConfig dsConfig) {

        switch (dsConfig) {
            case MySensorConfig::MONOCULAR:
                return "mono_im";
            case MySensorConfig::IMU_MONOCULAR:
                return "mono_im_imu";
            case MySensorConfig::STEREO:
                return "stereo_im";
            case MySensorConfig::IMU_STEREO:
                return "stereo_im_imu";
            case MySensorConfig::RGBD:
                return "rgbd";
            case MySensorConfig::EVENT_ONLY:
                return "mono_ev";
            case MySensorConfig::EVENT_MONO:
                return "mono_ev_im";
            case MySensorConfig::EVENT_IMU:
                return "mono_ev_imu";
            case MySensorConfig::EVENT_IMU_MONO:
                return "mono_ev_im_imu";
            default:
                return "idle";
        }
    }

    MySensorConfig::SensorConfig MySensorConfig::mapConfig(const string &dsConfig) {

        if (dsConfig == "mono_im")
            return MySensorConfig::MONOCULAR;
        else if (dsConfig == "mono_im_imu")
            return MySensorConfig::IMU_MONOCULAR;
        else if (dsConfig == "stereo_im")
            return MySensorConfig::STEREO;
        else if (dsConfig == "stereo_im_imu")
            return MySensorConfig::IMU_STEREO;
        else if (dsConfig == "rgbd")
            return MySensorConfig::RGBD;
        else if (dsConfig == "mono_ev")
            return MySensorConfig::EVENT_ONLY;
        else if (dsConfig == "mono_ev_im")
            return MySensorConfig::EVENT_MONO;
        else if (dsConfig == "mono_ev_imu")
            return MySensorConfig::EVENT_IMU;
        else if (dsConfig == "mono_ev_im_imu")
            return MySensorConfig::EVENT_IMU_MONO;
        else
            return MySensorConfig::IDLE;
    }

    string MySensorConfig::toDsStr() const {
        return MySensorConfig::mapConfig(mSensor);
    }

    /*System::eSensor System::convertSensorConfig(EORB_SLAM::MyDataTypes::SensorConfig sConf) {

        switch (sConf) {
            case EORB_SLAM::MyDataTypes::MONOCULAR:
                return System::MONOCULAR;
            case EORB_SLAM::MyDataTypes::STEREO:
                return System::STEREO;
            case EORB_SLAM::MyDataTypes::IMU_MONOCULAR:
                return System::IMU_MONOCULAR;
            case EORB_SLAM::MyDataTypes::IMU_STEREO:
                return System::IMU_STEREO;
            case EORB_SLAM::MyDataTypes::RGBD:
                return System::RGBD;
            case EORB_SLAM::MyDataTypes::EVENT_ONLY:
                return System::EVENT_ONLY;
            case EORB_SLAM::MyDataTypes::EVENT_IMU:
                return System::EVENT_IMU;
            case EORB_SLAM::MyDataTypes::EVENT_MONO:
                return System::EVENT_MONO;
            case EORB_SLAM::MyDataTypes::EVENT_IMU_MONO:
                return System::EVENT_IMU_MONO;
            default:
                return System::IDLE;
        }
    }*/

    string MySensorConfig::toStr() const {

        switch (mSensor) {
            case MySensorConfig::MONOCULAR:
                return "Monocular";
            case MySensorConfig::STEREO:
                return "Stereo";
            case MySensorConfig::IMU_MONOCULAR:
                return "Mono-Inertial";
            case MySensorConfig::IMU_STEREO:
                return "Stereo-Inertial";
            case MySensorConfig::RGBD:
                return "RGBD";
            case MySensorConfig::EVENT_ONLY:
                return "Event";
            case MySensorConfig::EVENT_IMU:
                return "Event-Inertial";
            case MySensorConfig::EVENT_MONO:
                return "Event-Image";
            case MySensorConfig::EVENT_IMU_MONO:
                return "Event-Image-Inertial";
            default:
                return "Idle";
        }
    }

    bool MySensorConfig::isMonocular() const {
        return mSensor == MONOCULAR || mSensor == IMU_MONOCULAR || mSensor == EVENT_ONLY ||
               mSensor == EVENT_MONO || mSensor == EVENT_IMU_MONO || mSensor == EVENT_IMU;
    }

    bool MySensorConfig::isStereo() const {
        return mSensor == STEREO || mSensor == IMU_STEREO;
    }

    bool MySensorConfig::isRGBD() const {
        return mSensor == RGBD;
    }

    bool MySensorConfig::isImage() const {
        return mSensor == MONOCULAR || mSensor == IMU_MONOCULAR || mSensor == STEREO || mSensor == IMU_STEREO ||
               mSensor == RGBD || mSensor == EVENT_MONO || mSensor == EVENT_IMU_MONO;
    }

    bool MySensorConfig::isEvent() const {
        return mSensor == EVENT_ONLY || mSensor == EVENT_IMU ||
               mSensor == EVENT_MONO || mSensor == EVENT_IMU_MONO;
    }

    bool MySensorConfig::isInertial() const {
        return mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO ||
               mSensor == EVENT_IMU || mSensor == EVENT_IMU_MONO;
    }

    bool MySensorConfig::isEventOnly() const {
        return mSensor == EVENT_ONLY;
    }

    bool MySensorConfig::isImageOnly() const {
        return mSensor == MONOCULAR || mSensor == IMU_MONOCULAR || mSensor == STEREO || mSensor == IMU_STEREO ||
               mSensor == RGBD;
    }

    bool MySensorConfig::operator==(MySensorConfig::SensorConfig sConf) {
        return this->mSensor == sConf;
    }

    bool MySensorConfig::operator!=(MySensorConfig::SensorConfig sConf) {
        return this->mSensor != sConf;
    }

    /* ------------------------------------------------------------------------------------------------------------- */

    bool MyDepthMap::checkBounds(float x, float y) const {

        if (vX.empty() || vY.empty()) {
            return false;
        }
        return (x >= vX[0] && x <= vX[mnPts-1]) && (y >= vY[0] && y <= vY[mnPts-1]);
    }

    void MyDepthMap::populate(const std::vector<cv::KeyPoint> &vKPts, const std::vector<cv::Point3f> &pts3d,
                              const std::vector<int> &matches) {

        assert(vKPts.size() == matches.size() && vKPts.size() == pts3d.size());

        mnPts = 0;
        int nPts = vKPts.size();
        vX.reserve(nPts);
        vY.reserve(nPts);
        for (size_t i = 0; i < nPts; i++) {
            if (matches[i] < 0) {
                continue;
            }
            cv::KeyPoint kpt = vKPts[i];
            float x = kpt.pt.x;
            float y = kpt.pt.y;
            vX.push_back(x);
            vY.push_back(y);
            mDepthMap.insert(make_pair(make_pair(x, y), pts3d[i].z));
            mnPts++;
        }
        sort(vX.begin(), vX.end());
        sort(vY.begin(), vY.end());
    }

    float MyDepthMap::getDepth(float x, float y) const {

        auto dIter = mDepthMap.lower_bound(make_pair(x, y));
        if (dIter != mDepthMap.end()) {
            return dIter->second;
        }
        return 1.f;
    }

    float MyDepthMap::getDepthLinInterp(float x, float y) const {

        float x0, x1, y0, y1;
        if (x <= vX[0]) {
            x0 = vX[0];
            if (y <= vY[0]) {
                y0 = vY[0];
                return mDepthMap.lower_bound(make_pair(x0,y0))->second;
            }
            else if (y >= vY[mnPts-1]) {
                y1 = vY[mnPts-1];
                return mDepthMap.lower_bound(make_pair(x0,y1))->second;
            }
            else {
                auto yIter = std::lower_bound(vY.begin(), vY.end(), y);
                y0 = *(yIter-1);
                y1 = *(yIter);
                float dy0 = mDepthMap.lower_bound(make_pair(x0,y0))->second;
                float dy1 = mDepthMap.lower_bound(make_pair(x0,y1))->second;
                return (dy0+dy1)*0.5f;
            }
        }
        else if (x >= vX[mnPts-1]) {
            x1 = vX[mnPts-1];
            if (y <= vY[0]) {
                y0 = vY[0];
                return mDepthMap.lower_bound(make_pair(x1,y0))->second;
            }
            else if (y >= vY[mnPts-1]) {
                y1 = vY[mnPts-1];
                return mDepthMap.lower_bound(make_pair(x1,y1))->second;
            }
            else {
                auto yIter = std::lower_bound(vY.begin(), vY.end(), y);
                y0 = *(yIter-1);
                y1 = *(yIter);
                float dy0 = mDepthMap.lower_bound(make_pair(x1,y0))->second;
                float dy1 = mDepthMap.lower_bound(make_pair(x1,y1))->second;
                return (dy0+dy1)*0.5f;
            }
        }
        else {
            auto xIter = std::lower_bound(vX.begin(), vX.end(), x);
            x0 = *(xIter-1);
            x1 = *(xIter);
            if (y <= vY[0]) {
                y0 = vY[0];
                float d0 = mDepthMap.lower_bound(make_pair(x0,y0))->second;
                float d1 = mDepthMap.lower_bound(make_pair(x1,y0))->second;
                return (d0+d1)*0.5f;
            }
            else if (y >= vY[mnPts-1]) {
                y1 = vY[mnPts-1];
                float d0 = mDepthMap.lower_bound(make_pair(x0,y1))->second;
                float d1 = mDepthMap.lower_bound(make_pair(x1,y1))->second;
                return (d0+d1)*0.5f;
            }
            else {
                auto yIter = std::lower_bound(vY.begin(), vY.end(), y);
                y0 = *(yIter - 1);
                y1 = *(yIter);
                float d0 = mDepthMap.lower_bound(make_pair(x0, y0))->second;
                float d1 = mDepthMap.lower_bound(make_pair(x0, y1))->second;
                float d2 = mDepthMap.lower_bound(make_pair(x1, y0))->second;
                float d3 = mDepthMap.lower_bound(make_pair(x1, y1))->second;
                return ((d0 + d1) * 0.5f + (d2 + d3) * 0.5f) * 0.5f;
            }
        }
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    void EvKfInfo::addKeyPoints(const std::vector<cv::KeyPoint>& kpts) {

        mvKpts.push_back(kpts);
    }

    void EvKfInfo::addConnection(const cv::Mat& R, const cv::Mat& t, const std::vector<cv::Point3f>& pts3d,
                                 const std::vector<int>& mch, const std::vector<bool>& vbInliers, const std::vector<float>& mchErr) {

        mvRot.push_back(R);
        mvTrans.push_back(t);
        mvPts3d.push_back(pts3d);
        mvMatches.push_back(mch);
        mvMchErr.push_back(mchErr);
        mvReconstInliers.push_back(vbInliers);
    }

    void EvKfInfo::addPose(const cv::Mat &Tcw) {

        mvTcw.push_back(Tcw.clone());
    }

    cv::Mat EvKfInfo::getLastPose() {

        if (mvTcw.empty()) {
            return cv::Mat();
        }
        return mvTcw.back().clone();
    }

    unsigned EvKfInfo::getPointTracks(std::vector<cv::Mat> &kpts) {

        if (mvKpts.empty() || mvKpts[0].empty()) {
            cerr << "EvKfInfo::getPointTracks call on empty frame set.\n";
            return 0;
        }
        unsigned nFrames = mvKpts.size();
        //unsigned nKpts = mvKpts[0].size();

        kpts.clear();
        //kpts.resize(nFrames);

        unsigned nPts = 0;
        for (unsigned fr = 0; fr < nFrames; fr++) {

            //unsigned cnt = 0;
            unsigned nKpts = mvKpts[fr].size();
            /*if (nKpts <= 0 || mvMatches.empty() || mvReconstInliers.empty() ||
                mvReconstInliers[fr].size() != mvMatches[fr].size() || nKpts != mvMatches[fr].size()) {
                continue;
            }*/
            cv::Mat_<double> frame(2, (int)nKpts);
            for (unsigned tr = 0; tr < nKpts; tr++) {

                /*if (fr == 0) {
                    frame(0,(int)tr) = mvKpts[fr][tr].pt.x;
                    frame(1,(int)tr) = mvKpts[fr][tr].pt.y;
                    nPts++;
                }
                else if (mvMatches[fr-1][tr] >= 0) {// && mvReconstInliers[fr-1][tr]) {
                    frame(0,(int)tr) = mvKpts[fr][tr].pt.x;
                    frame(1,(int)tr) = mvKpts[fr][tr].pt.y;
                    nPts++;
                }
                else {
                    frame(0,(int)tr) = -1;
                    frame(1,(int)tr) = -1;
                }*/
                frame(0,(int)tr) = mvKpts[fr][tr].pt.x;
                frame(1,(int)tr) = mvKpts[fr][tr].pt.y;
                nPts++;
            }
            kpts.push_back(frame);
        }
        return nPts;
    }

    void EvKfInfo::clear() {

        mvKpts.clear();
        mvRot.clear();
        mvTrans.clear();
        mvPts3d.clear();
        mvMatches.clear();
        mvMchErr.clear();
        mvReconstInliers.clear();
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    PoseDepthInfo::PoseDepthInfo() {

        this->reset();
    }

    PoseDepthInfo::PoseDepthInfo(const EvFramePtr& pFrame0, const EvFramePtr& pFrame1) : PoseDepthInfo() {

        this->updateLastPose(pFrame0, pFrame1);
    }

    PoseDepthInfo::PoseDepthInfo(const std::shared_ptr<PoseDepthInfo> &pPoseInfo)
            : mTc1w(pPoseInfo->mTc1w.clone()), mTc0w(pPoseInfo->mTc0w.clone()), mTc1c0(pPoseInfo->mTc1c0.clone()),
              mLastTimeStamp(pPoseInfo->mLastTimeStamp), mIniTimeStamp(pPoseInfo->mIniTimeStamp), mDTs(pPoseInfo->mDTs),
              mLastDepth(pPoseInfo->mLastDepth), mbIsInitialized(pPoseInfo->mbIsInitialized)
    {}

    void PoseDepthInfo::updateLastPose(const EvFramePtr& pFrame0, const EvFramePtr& pFrame1) {

        if (!pFrame0 || !pFrame1 || pFrame0->mTcw.empty() || pFrame1->mTcw.empty()) {
            return;
        }

        std::unique_lock<std::mutex> lock1(mMtxUpdate);

        mTc0w = pFrame0->mTcw.clone();

        mTc1w = pFrame1->mTcw.clone();

        mTc1c0 = mTc1w * mTc0w.inv(cv::DECOMP_SVD);

        mIniTimeStamp = pFrame0->mTimeStamp;
        mLastTimeStamp = pFrame1->mTimeStamp;
        mDTs = mLastTimeStamp - mIniTimeStamp;

        mLastDepth = pFrame1->computeSceneMedianDepth(2);

        mbIsInitialized = true;
    }

    void PoseDepthInfo::updateLastPose(ORB_SLAM3::Frame *pFr0, ORB_SLAM3::Frame *pFr1, const float depth) {

        if (!pFr0 || !pFr1) {
            return;
        }

        std::unique_lock<std::mutex> lock1(mMtxUpdate);

        mTc0w = pFr0->mTcw.clone();
        mTc1w = pFr1->mTcw.clone();
        mTc1c0 = mTc1w * mTc0w.inv(cv::DECOMP_SVD);

        mIniTimeStamp = pFr0->mTimeStamp;
        mLastTimeStamp = pFr1->mTimeStamp;
        mDTs = mLastTimeStamp - mIniTimeStamp;

        mLastDepth = depth;

        mbIsInitialized = true;
    }

    void PoseDepthInfo::updateLastPose(ORB_SLAM3::KeyFrame *pKF0, ORB_SLAM3::KeyFrame *pKF1) {

        if (!pKF0 || !pKF1) {
            return;
        }

        std::unique_lock<std::mutex> lock1(mMtxUpdate);

        mTc0w = pKF0->GetPose().clone();
        mTc1w = pKF1->GetPose().clone();
        mTc1c0 = mTc1w * mTc0w.inv(cv::DECOMP_SVD);

        mIniTimeStamp = pKF0->mTimeStamp;
        mLastTimeStamp = pKF1->mTimeStamp;
        mDTs = mLastTimeStamp - mIniTimeStamp;

        mLastDepth = pKF1->ComputeSceneMedianDepth(2);

        mbIsInitialized = true;
    }

    void PoseDepthInfo::updateLastPose(const double ts0, const cv::Mat &Tc0w, const double ts1, const cv::Mat &Tc1w, const float depth) {

        std::unique_lock<std::mutex> lock1(mMtxUpdate);

        mTc0w = Tc0w.clone();
        mTc1w = Tc1w.clone();
        mTc1c0 = mTc1w * mTc0w.inv(cv::DECOMP_SVD);

        mIniTimeStamp = ts0;
        mLastTimeStamp = ts1;
        mDTs = mLastTimeStamp - mIniTimeStamp;

        mLastDepth = depth;

        mbIsInitialized = true;
    }

    void PoseDepthInfo::setLastDepth(const float d) {

        std::unique_lock<std::mutex> lock1(mMtxUpdate);
        mLastDepth = d;
    }

    double PoseDepthInfo::getLastDepth() {

        std::unique_lock<std::mutex> lock1(mMtxUpdate);
        return mLastDepth;
    }

    void PoseDepthInfo::getLastInfo(double &ts, cv::Mat &Tc1w, float &lastDepth) {

        std::unique_lock<std::mutex> lock1(mMtxUpdate);
        ts = mLastTimeStamp;
        Tc1w = mTc1w.clone();
        lastDepth = mLastDepth;
    }

    void PoseDepthInfo::getIniInfo(double &ts, cv::Mat &Tc0w) {

        std::unique_lock<std::mutex> lock1(mMtxUpdate);
        ts = mIniTimeStamp;
        Tc0w = mTc0w.clone();
    }

    void PoseDepthInfo::getDPose(double& ts0, double &dts, cv::Mat &Tc1c0, float &lastDepth) {

        std::unique_lock<std::mutex> lock1(mMtxUpdate);
        ts0 = mIniTimeStamp;
        dts = mDTs;
        Tc1c0 = mTc1c0.clone();
        lastDepth = mLastDepth;
    }

    cv::Mat PoseDepthInfo::getDPose() {

        std::unique_lock<std::mutex> lock1(mMtxUpdate);
        return mTc1c0.clone();
    }

    void PoseDepthInfo::reset() {

        std::unique_lock<std::mutex> lock1(mMtxUpdate);
        mTc0w = cv::Mat();
        mIniTimeStamp = 0.0;
        mTc1w = cv::Mat();
        mTc1c0 = cv::Mat();
        mLastTimeStamp = 0.0;
        mDTs = 0.0;
        mLastDepth = 1.f;
        mbIsInitialized = false;
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    bool FrameInfo::checkDataIntegrity() const {

        return mlpReferences.size() == mlRelativeFramePoses.size() && mlFrameTimes.size() == mlbLost.size() &&
               mlbLost.size() == mlpReferences.size() && mlbLost.size() == msFrameIds.size();
    }

    void FrameInfo::reset() {

        mlRelativeFramePoses.clear();
        mlpReferences.clear();
        mlFrameTimes.clear();
        mlbLost.clear();

        msFrameIds.clear();
    }

    void FrameInfo::pushState(ORB_SLAM3::Frame* pCurFr, const bool isLost) {

        cv::Mat Tcr = pCurFr->mTcw * pCurFr->mpReferenceKF->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(pCurFr->mpReferenceKF);
        mlFrameTimes.push_back(pCurFr->mTimeStamp);
        mlbLost.push_back(isLost);

        msFrameIds.push_back(pCurFr->mnId);
    }

    void FrameInfo::pushState(ulong frId, const cv::Mat &Tcr, ORB_SLAM3::KeyFrame *pRefKF, double ts, bool isLost) {

        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(pRefKF);
        mlFrameTimes.push_back(ts);
        mlbLost.push_back(isLost);

        msFrameIds.push_back(frId);
    }

    void FrameInfo::pushLastState(const bool isLost) {

        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(isLost);

        msFrameIds.push_back(msFrameIds.back());
    }

    void FrameInfo::getAllState(std::list<cv::Mat> &lRelFramePoses, std::list<ORB_SLAM3::KeyFrame *> &lpRefKFs,
                                std::list<double> &lTs, std::list<bool> &lbLost) const {

        // check data integrity
        assert(this->checkDataIntegrity());

        lRelFramePoses = mlRelativeFramePoses;
        lpRefKFs = mlpReferences;
        lTs = mlFrameTimes;
        lbLost = mlbLost;
    }

    std::list<bool>& FrameInfo::getIsLostList() {

        // check data integrity
        assert(this->checkDataIntegrity());

        return mlbLost;
    }

    cv::Mat FrameInfo::getLastRelPose() {

        if (mlRelativeFramePoses.empty())
            return cv::Mat();
        return mlRelativeFramePoses.back();
    }

    void FrameInfo::setAllState(const std::list<cv::Mat> &lRelFramePoses, const std::list<ORB_SLAM3::KeyFrame *> &lpRefKFs,
                                const std::list<double> &lTs, const std::list<bool> &lbLost) {

        mlRelativeFramePoses = lRelFramePoses;
        mlpReferences = lpRefKFs;
        mlFrameTimes = lTs;
        mlbLost = lbLost;

        // check data integrity
        assert(this->checkDataIntegrity());
    }

    void FrameInfo::setIsLostList(const std::list<bool> &lbLost) {

        if (!mlbLost.empty())
            assert(lbLost.size() == mlbLost.size());
        mlbLost = lbLost;
    }

    void FrameInfo::setIsLostRange(ulong iniFrId, ulong lastFrId, const bool isLost, const bool resetAll) {

        // check data integrity
        assert(this->checkDataIntegrity());

        auto firstIter = std::find(msFrameIds.begin(), msFrameIds.end(), iniFrId);
        auto lastIter = std::find(msFrameIds.begin(), msFrameIds.end(), lastFrId);

        if (firstIter == msFrameIds.end() || lastIter == msFrameIds.end()) {
            LOG(WARNING) << "FrameInfo::setIsLostRange: Can't set frame range: " << iniFrId << ", " << lastFrId << endl;
            return;
        }

        size_t firstIdx = std::distance(msFrameIds.begin(), firstIter);
        size_t lastIdx = std::distance(msFrameIds.begin(), lastIter);

        // set all as lost except range
        size_t i = 0;
        for (auto iter = mlbLost.begin(); iter != mlbLost.end(); iter++, i++) {

            if (resetAll) {
                *iter = (i >= firstIdx && i <= lastIdx) == isLost;
            }
            else {
                if (i >= firstIdx && i <= lastIdx)
                    *iter = true;
            }
        }
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    MySmartTimer::MySmartTimer(string name) : mName(std::move(name)), mt0(), mt1() {

        this->reserve(100000);
    }

    void MySmartTimer::reserve(const size_t& n) {
        mvDeltaTimes.reserve(n);
    }

    void MySmartTimer::tic() {
        mt0 = std::chrono::steady_clock::now();
    }

    void MySmartTimer::toc() {
        mt1 = std::chrono::steady_clock::now();
    }

    void MySmartTimer::push() {

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(mt1 - mt0).count();
        mvDeltaTimes.push_back(ttrack);
    }

    double MySmartTimer::getAverageTime() {
        if (mvDeltaTimes.empty()) {
            LOG(WARNING) << "MySmartTimer::getAverageTime: Empty dTimes vector!\n";
            return 0.0;
        }
        return static_cast<double>(1.0 * std::accumulate(mvDeltaTimes.begin(), mvDeltaTimes.end(), 0.0) / mvDeltaTimes.size());
    }

    std::string MySmartTimer::getCommentedTimeStat() {

        ostringstream oss;
        oss << fixed << setprecision(6);

        double tmin = 0.0, tmax = 0.0, tmed = 0.0, tavg = 0.0;

        if (!mvDeltaTimes.empty()) {

            sort(mvDeltaTimes.begin(), mvDeltaTimes.end());
            tmin = mvDeltaTimes[0];
            tmax = mvDeltaTimes.back();
            size_t ntot = mvDeltaTimes.size();
            tmed = mvDeltaTimes[ntot/2];
            tavg = this->getAverageTime();
        }

        oss << "# " << mName << ": min = " << tmin << ", max = " << tmax << ", med = " << tmed << ", avg = " << tavg << endl;

        return oss.str();
    }

    void MySmartTimer::reset() {

        mvDeltaTimes.clear();
    }

    MySmartWatchDog::MySmartWatchDog(string name, float waitTimeSec, ulong tStepUsec) :
            mName(std::move(name)), mWaitTimeSec(waitTimeSec), mTimeStepUsec(tStepUsec), mCount(0)
    {
        setWaitTimeSec(waitTimeSec);
    }

    void MySmartWatchDog::step() {

        float timeTillNow = getWaitTimeSec();

        LOG_EVERY_N(WARNING, 1000) << mName << ": Waiting for " << timeTillNow << "...\n";

        std::this_thread::sleep_for(std::chrono::microseconds(mTimeStepUsec));

        if (mCount > mWaitCount) {

            LOG(FATAL) << mName << ": Abort because maximum time has reached\n";
        }

        ++mCount;
    }

    void MySmartWatchDog::reset() {

        mCount = 0;
    }

    float MySmartWatchDog::getWaitTimeSec() const {

        return static_cast<float>(mCount * mTimeStepUsec) / 1e6f;
    }

    void MySmartWatchDog::setWaitTimeSec(float waitTimeSec) {

        mWaitTimeSec = waitTimeSec;
        mWaitCount = (mWaitTimeSec * 1e6) / mTimeStepUsec;
    }

} // EORB_SLAM
