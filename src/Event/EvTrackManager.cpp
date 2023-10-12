//
// Created by root on 11/25/20.
//

// TODO: Important: mpEvParams must be shared and thread safe!

#include "EvTrackManager.h"

//#include "Converter.h"
//#include "OptimizableTypes.h"
//#include "Optimizer.h"
#include "ORBextractor.h"
#include "System.h"
#include "MyOptimizer.h"
#include "EvLocalMapping.h"

using namespace cv;
using namespace std;

namespace EORB_SLAM
{
    /* -------------------------------------------------------------------------------------------------------------- */

    EvTrackManager::EvTrackManager(const EvParamsPtr& pEvParams, const CamParamsPtr& camParams,
            const MixedFtsParamsPtr& pORBParams, const SensorConfigPtr& sConfig, ORB_SLAM3::ORBVocabulary* pVocab,
            ORB_SLAM3::Viewer* pViewer):
            sepThread(false), mStat(), mpLastSynchEvIm(nullptr), mpL2AsynchTracker(nullptr),
            mpL2SynchTracker(nullptr), mpL2AsynchTrackerU(nullptr), mL1WinSize(pEvParams->l1ChunkSize),
            mL2EvImSTD(pEvParams->l2ImSigma), mL1NumLoop(pEvParams->l1NumLoop), mbFixedWinSz(pEvParams->l1FixedWinSz),
            mbContTracking(pEvParams->continTracking), mImSize(pEvParams->imWidth, pEvParams->imHeight),
            mMchWindow(DEF_MATCHES_WIN_NAME), mpSystem(nullptr), mpViewer(pViewer), mpMapDrawer(nullptr),
            mL1TrackingTimer("L1 Image Builder Tracking") {

        mpSensor = sConfig;
        mpEvParams = pEvParams;
        mpCamParams = camParams;
        //mpORBParams = pORBParams;

        mpEvAtlas = make_shared<ORB_SLAM3::Atlas>(0, EvAsynchTracker::mMapIdx++);

        mpL1EvImBuilder = unique_ptr<EvImBuilder>(new EvImBuilder(mpEvParams, mpCamParams, pORBParams, mpSensor));
        mpL1EvImBuilder->setTrackManager(this);

        // We still use L2 tracker with images but now it is synchronized with images!
        bool setL2Thread = true;
        if (mpSensor->isImage()) {
            mpL2SynchTracker = make_shared<EvSynchTrackerU>(mpEvParams, mpCamParams, pORBParams, mpEvAtlas, pVocab, mpSensor, mpViewer);
            mpL2AsynchTracker = mpL2SynchTracker;
            setL2Thread = false;
        }
        else {
            if (!mpEvParams->continTracking) {
                mpL2AsynchTracker = make_shared<EvAsynchTracker>(mpEvParams, mpCamParams, pORBParams, mpEvAtlas, pVocab, mpSensor,
                                                                 mpViewer);
            }
            else {
                mpL2AsynchTrackerU = make_shared<EvAsynchTrackerU>(mpEvParams, mpCamParams, pORBParams, mpEvAtlas, pVocab, mpSensor,
                                                                   mpViewer);
                mpL2AsynchTracker = mpL2AsynchTrackerU;
            }
        }
        if (setL2Thread) {
            mpL2AsynchTracker->setSeparateThread(true);
            mpL2Thread = unique_ptr<thread>(new thread(&EvAsynchTracker::Track, mpL2AsynchTracker.get()));
        }
        mpL2AsynchTracker->setTrackManager(this);
        mpL1EvImBuilder->setLinkedL2Tracker(mpL2AsynchTracker);

        if (mpViewer) {

            mpMapDrawer = mpViewer->getMapDrawer();

            if (!mpSensor->isImage() && mpMapDrawer) {

                mpMapDrawer->mpAtlas = mpEvAtlas.get();
                if (mpViewer->getFrameDrawer())
                    mpViewer->getFrameDrawer()->updateAtlas(mpEvAtlas.get());
            }
        }
    }

    EvTrackManager::EvTrackManager(const EvParamsPtr &pEvParams, const CamParamsPtr &camParams,
            const MixedFtsParamsPtr &pORBParams, const SensorConfigPtr &sConfig, ORB_SLAM3::ORBVocabulary *pVocab,
            ORB_SLAM3::Viewer* pViewer, const IMUParamsPtr &pIMUParams) :
            EvTrackManager(pEvParams, camParams, pORBParams, sConfig, pVocab, pViewer) {

        if (mpSensor->isInertial()) {
            mpImuManager = make_shared<IMU_Manager>(pIMUParams);//, mpCamParams->mMaxFrames);

            mpL2AsynchTracker->setImuManagerAndChannel(mpImuManager);
            mpL1EvImBuilder->setImuManagerAndChannel(mpImuManager);
        }
    }

    EvTrackManager::~EvTrackManager() {

        mpL1EvImBuilder->stop();
        mpL2AsynchTracker->stop();
        //cv::destroyWindow(mMchWindow);
        mpSystem = nullptr;
        mpMapDrawer = nullptr;
        mpViewer = nullptr;

//        if (mptImageDisplay) {
//            mpImDisplay->mbStop = true;
//            mptImageDisplay->join();
//            delete mptImageDisplay;
//        }
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    void EvTrackManager::setSeparateThread(bool flg) {

        this->sepThread.set(flg);
    }

    void EvTrackManager::shutdown() {

        // Stop Asynch. tracking threads
        while (!mpL1EvImBuilder->isStopped() && !mpL2AsynchTracker->isStopped()) {
            cout << "Trying to stop tracking threads...\n";
            mpL1EvImBuilder->stop();
            mpL2AsynchTracker->stop();
        }
        // Empty input buffers
        this->emptyBuffer();

        // Stop current thread if it's running on separate thread
        if (this->sepThread) {
            while (!this->isStopped()) {
                cout << "Trying to stop current thread...\n";
                this->stop();
            }
        }
    }

    bool EvTrackManager::isStopped() {

        return this->sepThread && this->mStat == STOP;
    }

    void EvTrackManager::stop() {

        this->mStat.update(STOP);
    }

    void EvTrackManager::reset() {

        DLOG(INFO) << "EvTrackManager::reset >> Stats:\n"
                   << "\tInput: -Size EvBuffer: " << mEventBuffer.size() << ", first ts: "
                   << ((!mEventBuffer.empty()) ? to_string(mEventBuffer.front().ts) : "??") << endl;

        // first reset manager -> prevent it from loading events to other modules
        {
            std::unique_lock<mutex> lock(mMtxEvBuff);
            mEventBuffer.clear();
        }
        {
            std::unique_lock<mutex> lock(mMtxEvImQ);
            mEvImBuffer.clear();
        }

        mStat.update(IDLE);

        // let imBuilder consume all input
        while (mpL1EvImBuilder->isInputGood())
            this_thread::sleep_for(std::chrono::microseconds(500));
        // now reset imBuilder -> stop loading l2 tracker
        mpL1EvImBuilder->resetAll();

        // let l2 tracker consume all input
        while (mpL2AsynchTracker->isInputGood())
            this_thread::sleep_for(std::chrono::microseconds(500));
        mpL2AsynchTracker->resetAll();

        if (mpSensor->isInertial() && mpImuManager) {
            while (mpImuManager->isInitializing())
                this_thread::sleep_for(std::chrono::microseconds(500));
            mpImuManager->resetAll();
        }

        mL1WinSize = mpEvParams->l1ChunkSize;

        mL1TrackingTimer.reset();

        // L2 Tracker will take care of atlas!
        //mpEvAtlas;
    }

    bool EvTrackManager::isReady() {

        if (this->sepThread) {
            // Also check if this runs in separate thread, the input buffer must be exhausted
            // before accepting new input
            return mpL1EvImBuilder->isTrackerReady() && mpL2AsynchTracker->isTrackerReady() && !isInputGood();
        }
        else {
            // If L1 is not running on separate thread and is not ready, instigate a tracking seq.
            if (!mpL1EvImBuilder->isTrackerReady() && !mpL1Thread) {
                mpL1EvImBuilder->Track();
            }
            return mpL1EvImBuilder->isTrackerReady() && mpL2AsynchTracker->isTrackerReady();
        }
    }

    // Must satisfy minimum requirements for buffer consumption
    bool EvTrackManager::isInputGood() {

        return !mEventBuffer.empty() && mEventBuffer.size() >= mpEvParams->l1ChunkSize;
    }

    bool EvTrackManager::allInputsEmpty() {

        return !this->isInputGood() && !mpL1EvImBuilder->isInputGood() && !mpL2AsynchTracker->isInputGood();
    }

    // We must have enough events and last Tcw to build mcImage
    /*bool EvTrackManager::isL2BuffGood() {

        unsigned long l2ChunkSize = mpEvParams->l1ChunkSize * mpEvParams->l1NumLoop;
        return mpL2EvBuff->size() >= l2ChunkSize && mpL2AsynchTracker->hasL1LastPose();
    }*/

    /* -------------------------------------------------------------------------------------------------------------- */

    void EvTrackManager::fillBuffer(const vector<EventData> &evs) {

        {
            std::unique_lock<mutex> lock(mMtxEvBuff);

            for (const auto & ev : evs) {
                mEventBuffer.push_back(ev);
            }
        }

        // If event-image config., also fill synch. event buffer
        if (mpSensor->isImage()) {
            this->fillEvImBuffer(evs);
        }
    }

    void EvTrackManager::emptyBuffer() {

        {
            std::unique_lock<mutex> lock(mMtxEvBuff);
            while (!mEventBuffer.empty())
                mEventBuffer.pop_front();
        }

        {
            std::unique_lock<mutex> lock(mMtxEvImQ);
            while(!mEvImBuffer.empty())
                mEvImBuffer.pop_front();
        }
    }

    void EvTrackManager::injectEventsBegin(const std::vector<EventData> &evs) {

        if (evs.empty()) {
            return;
        }

        std::unique_lock<mutex> lock(mMtxEvBuff);

        for (auto iter = evs.rbegin(), evsEnd = evs.rend(); iter != evsEnd; iter++) {

            mEventBuffer.push_front(*(iter));
        }
    }

    ulong EvTrackManager::consumeEventsBegin(const ulong chunkSize, vector<EventData> &evs) {

        ulong nEvs = min(chunkSize, mEventBuffer.size());
        evs.reserve(nEvs);

        std::unique_lock<mutex> lock(mMtxEvBuff);

        for (ulong i = 0; i < nEvs; i++) {

            evs.push_back(mEventBuffer.front());
            mEventBuffer.pop_front();
        }

        return evs.size();
    }

    ulong EvTrackManager::consumeEventsBegin(ulong chunkSize, ulong overlap, std::vector<EventData> &evs) {

        ulong nEvs = min(chunkSize, mEventBuffer.size());
        ulong n_ov = nEvs - static_cast<ulong>(overlap * ((float) nEvs / (float) chunkSize));
        evs.reserve(nEvs);
        list<EventData>::iterator lEvItr;

        std::unique_lock<mutex> lock(mMtxEvBuff);

        for (ulong i = 0; i < nEvs; i++) {

            if (i < n_ov) {
                evs.push_back(mEventBuffer.front());
                mEventBuffer.pop_front();
            }
            else {
                if (i == n_ov) {
                    lEvItr = mEventBuffer.begin();
                }
                evs.push_back(*(lEvItr++));
            }
        }

        return evs.size();
    }

    ulong EvTrackManager::retrieveEventsBegin(ulong chunkSize, std::vector<EventData> &evs) {

        ulong nEvs = min(chunkSize, mEventBuffer.size());
        evs.reserve(nEvs);

        std::unique_lock<mutex> lock(mMtxEvBuff);

        ulong i = 0;
        for (auto & ev : mEventBuffer) {

            if (i >= nEvs)
                break;

            evs.push_back(ev);

            i++;
        }

        return evs.size();
    }


    /* -------------------------------------------------------------------------------------------------------------- */

    double firstTs = 0.0;

    void EvTrackManager::Track() {

        //  Main Track loop
        while (!isStopped()) {

            // Level 1 Tracking
            // Only if L1 has capacity add new data
            if (this->isInputGood() && mpL1EvImBuilder->isTrackerReady()) {

                if (mStat == IDLE) {
                    mStat.update(TRACKING);
                }

                // Get Next Events
                vector<EventData> l1Evs;
                this->consumeEventsBegin(mpEvParams->l1ChunkSize, l1Evs);

                mpL1EvImBuilder->fillEvents(l1Evs);
                
                if (l1Evs.size() > 0) {
                    if (firstTs == 0) {
                        firstTs = l1Evs[0].ts;
                        cout << "First timestamp: " << firstTs << endl;
                    }
                    // save the map at a specific time
                    double lastTs = l1Evs.back().ts - firstTs;
                    if (lastTs > 15.0 && lastTs < 15.1) {
                        mpL2AsynchTracker->saveAtlas("/home_dir/event-map.txt");
                        saveTrajectory("/home_dir/event-kf-pose.txt");
                        saveFrameTrajectory("/home_dir/event-fr-pose.txt");
                        cout << "Saved atlas at: " << lastTs << endl;
                    }
                }
            }

            mL1TrackingTimer.tic();

            // If we don't have images, L1 tracker runs in main thread
            // so we need to call it explicitly -> Not anymore! (both share the same thread)
            mpL1EvImBuilder->Track();

            mL1TrackingTimer.toc();
            mL1TrackingTimer.push();

            if (mStat == IDLE) {
                VLOG_EVERY_N(3, 1000) << "EvTrackManager: Idle state...\n";
                this_thread::sleep_for(std::chrono::milliseconds(3));
            }

            if (!this->sepThread && !this->isInputGood()) { break; }
        }
    }

    void EvTrackManager::savePoseFileHeader(const std::string& fileName) {

        ofstream poseFile;
        poseFile.open(fileName, std::ios_base::app);
        if (poseFile.is_open()) {
            poseFile << mL1TrackingTimer.getCommentedTimeStat();
            poseFile << mpL2AsynchTracker->getTrackingTimeStat();
        }
        poseFile.close();
    }

    void EvTrackManager::saveTrajectory(const std::string& fileName, const double tsc, const int ts_prec) {

        if (mpL2AsynchTracker) {

            this->savePoseFileHeader(fileName);
            mpL2AsynchTracker->saveAllPoses(fileName, tsc, ts_prec);
        }
    }

    void EvTrackManager::saveFrameTrajectory(const std::string &fileName, double tsc, int ts_prec) {

        if (mpL2AsynchTracker) {

            this->savePoseFileHeader(fileName);
            mpL2AsynchTracker->saveAllFramePoses(fileName, tsc, ts_prec);
        }
    }

    void EvTrackManager::getEventConstraints(std::vector<ORB_SLAM3::KeyFrame *> &vpEvKFs) {

        if (mpL2AsynchTracker) {
            mpL2AsynchTracker->getEventConstraints(vpEvKFs);
        }
    }

    // Merge close relative poses based on their timestamp
    void EvTrackManager::fuseEventTracks(const std::vector<ORB_SLAM3::KeyFrame*>& vpEvKFs,
                                         std::vector<ORB_SLAM3::KeyFrame*>& vpEvKFsFused, const float thTsSelf) {

        if (vpEvKFs.empty()) {
            vpEvKFsFused = vpEvKFs;
            LOG(WARNING) << "EvTrackManager::fuseEventTracks, Call on empty event tracks, return\n";
            return;
        }

        const size_t nKFs = vpEvKFs.size();
        vpEvKFsFused.reserve(nKFs);

        size_t i = 0;

        ORB_SLAM3::KeyFrame* pCurrRefKF = vpEvKFs[i];

        vpEvKFsFused.push_back(pCurrRefKF);
        double refTs = pCurrRefKF->mTimeStamp;

        // Find the last child of current ref. kf
        ORB_SLAM3::KeyFrame* pLastChildKF = *(std::prev(pCurrRefKF->GetChilds().end()));
        double lastTs = pLastChildKF->mTimeStamp;

        // Retrieve the next ref. key frame to compare ts
        ORB_SLAM3::KeyFrame* pNextRefKF;
        while (i < nKFs-1) {
            pNextRefKF = vpEvKFs[i+1];
            double nextRefTs = pNextRefKF->mTimeStamp;

            if (abs(nextRefTs-lastTs) < thTsSelf) {

                cv::Mat lastTcw0 = pLastChildKF->GetPose();
                const float lastSc = pLastChildKF->ComputeSceneMedianDepth(2);

                g2o::SE3Quat Tcw0(ORB_SLAM3::Converter::toMatrix3d(pLastChildKF->GetRotation()),
                                  ORB_SLAM3::Converter::toVector3d(pLastChildKF->GetTranslation()));
                g2o::SE3Quat Tcw1(ORB_SLAM3::Converter::toMatrix3d(pNextRefKF->GetRotation()),
                                  ORB_SLAM3::Converter::toVector3d(pNextRefKF->GetTranslation()));
                g2o::SE3Quat Tw1w0 = Tcw1.inverse() * Tcw0;
                g2o::SE3Quat Tw0w1 = Tw1w0.inverse();

                // Even though nextRefKF is equal to lastChildKF, we have to add it
                // because all map points belongs to it!
                pNextRefKF->SetPose(lastTcw0);
                pCurrRefKF->AddChild(pNextRefKF);

                // Remap the map points
                for (ORB_SLAM3::MapPoint* pMP : pNextRefKF->GetMapPoints()) {

                    if (pMP && !pMP->isBad()) {

                        Eigen::Vector3d p3d = lastSc * ORB_SLAM3::Converter::toVector3d(pMP->GetWorldPos());
                        // Note the scale of Tw0w1 is inverse of lastSc -> is this right???
                        // -> Do it without Sim3 complications: SE3Quat
                        pMP->SetWorldPos(ORB_SLAM3::Converter::toCvMat(Tw0w1.map(p3d)));
                    }
                }

                // Update the pose of all children and add them
                for (ORB_SLAM3::KeyFrame* pNextChildKF : pNextRefKF->GetChilds()) {

                    g2o::SE3Quat Tciw1(ORB_SLAM3::Converter::toMatrix3d(pNextChildKF->GetRotation()),
                                       lastSc * ORB_SLAM3::Converter::toVector3d(pNextChildKF->GetTranslation()));
                    g2o::SE3Quat Tciw0 = Tciw1 * Tw1w0;

                    pNextChildKF->SetPose(ORB_SLAM3::Converter::toCvMat(Tciw0));

                    pCurrRefKF->AddChild(pNextChildKF);
                }

                // Update last child and last ts
                //pCurrRefKF = pNextRefKF;
                pLastChildKF = *(std::prev(pNextRefKF->GetChilds().end()));
                lastTs = pLastChildKF->mTimeStamp;
            }
            else {
                vpEvKFsFused.push_back(pNextRefKF);

                pCurrRefKF = pNextRefKF;
                pLastChildKF = *(std::prev(pNextRefKF->GetChilds().end()));
                lastTs = pLastChildKF->mTimeStamp;
            }
            i++;
        }
    }

    /*inline void EvTrackManager::grabImuData(const ORB_SLAM3::IMU::Point &imuMea) {

        if (mpImuManager) {
            mpImuManager->grabImuData(imuMea);
        }
    }*/

    void EvTrackManager::setCurrentCamPose(const cv::Mat &Tcw) {

        // When we have images, current map pose is controlled by ORB-SLAM
        if (mpMapDrawer && !mpSensor->isImage()) {
            mpMapDrawer->SetCurrentCameraPose(Tcw);
        }
    }

    void EvTrackManager::printAverageTrackingTimes() {

        ostringstream oss;
        oss << mL1TrackingTimer.getCommentedTimeStat();
        oss << mpL2AsynchTracker->getTrackingTimeStat();

        cout << oss.str();
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    void EvTrackManager::fillEvImBuffer(const vector<EventData> &evs) {

        std::unique_lock<mutex> lock(mMtxEvImQ);
        for (const EventData& ev : evs) {
            mEvImBuffer.push_back(ev);
        }
    }

    void EvTrackManager::consumeEvImBuffBegin(unsigned long chunkSize, std::vector<EventData> &evs) {

        evs.reserve(chunkSize);

        std::unique_lock<mutex> lock(mMtxEvImQ);
        for (unsigned long i = 0; i < chunkSize; i++) {
            evs.push_back(mEvImBuffer.front());
            mEvImBuffer.pop_front();
        }
    }

    // This method consumes events before chunkSize Evs before imTs
    void EvTrackManager::getEvImBuffTillImTs(const double imTs, const unsigned long chunkSize,
            std::vector<EventData>& evs, const bool eraseOldEvs) {

        evs.reserve(chunkSize);
        unsigned long cntEvsTillImTs = 0;

        std::unique_lock<mutex> lock(mMtxEvImQ);

        // Count events till image timestamp
        for (auto iter = mEvImBuffer.begin(); iter != mEvImBuffer.end() && iter->ts < imTs; iter++) {
            cntEvsTillImTs++;
        }
        // If we have more than chunkSize evs till imTs, erase them
        if (eraseOldEvs && cntEvsTillImTs > chunkSize) {

            // Do not use consume begin or else caught in dead lock!!!
            for (unsigned long i = 0; i < cntEvsTillImTs-chunkSize; i++) {
                mEvImBuffer.pop_front();
            }
        }
        // Do not consume these events here
        for (auto iter = mEvImBuffer.begin(); iter != mEvImBuffer.end() && iter->ts < imTs; iter++) {
            evs.push_back(*(iter));
            //mEvImBuffer.pop_front();
        }

        VLOG_EVERY_N(50, 100) << "getEvImBuffTillImTs: " << "nEvs before erase: " << cntEvsTillImTs
                        << ", chunk size: " << chunkSize << ", curr. buff. size: " << mEvImBuffer.size() << endl;
    }

    bool EvTrackManager::getRecommendedEvWinSize(unsigned long& winSize) {

        // We still get the recommended image size from EvImBuilder
        unsigned l1ChunkSize = this->getL1ChunkSize();
        unsigned long recommendedWinSize = l1ChunkSize * mL1NumLoop;
        unsigned long currBuffSize = 0;
        {
            std::unique_lock<mutex> lock(mMtxEvImQ);
            currBuffSize = mEvImBuffer.size();
        }
        winSize = min(recommendedWinSize, currBuffSize);

        return winSize >= l1ChunkSize;
    }

    /*void EvTrackManager::getEventsForOptimization(std::vector<EORB_SLAM::EventData> &vEvData, const bool isInitialized) {

        // We still get the recommended image size from EvImBuilder
        unsigned recommendedWinSize = this->getL1ChunkSize() * mL1NumLoop;
        unsigned long currBuffSize;
        {
            std::unique_lock<mutex> lock(mMtxEvImQ);
            currBuffSize = mEvImBuffer.size();
        }
        recommendedWinSize = min(static_cast<unsigned long>(recommendedWinSize), currBuffSize);
        if (currBuffSize > recommendedWinSize) {
            DLOG_EVERY_N(WARNING, 1000) << "Tracking::getEventsForOptimization: EvBuffSize " << currBuffSize
                                        << " is bigger than recommended win. size: " << recommendedWinSize << endl;
            // Normally we would want to get the most recent events
            if (!isInitialized) { // Map not initialized
                this->consumeEvImBuffBegin(currBuffSize-recommendedWinSize, vEvData);
                // Throw away redundant events
                vEvData.clear();
            }
        }
        this->consumeEvImBuffBegin(recommendedWinSize, vEvData);
    }*/

    float EvTrackManager::calcSceneMedianDepthForEvImOpt(ORB_SLAM3::Frame* pFrame) {

        // TODO: We can either use all map points in active map or matched map points
        return EvFrame::computeSceneMedianDepth(pFrame->mTcw, pFrame->getAllMapPointsMono(), 2);
    }

    float EvTrackManager::calcSceneMedianDepthForEvImOpt(const ORB_SLAM3::Frame* pFrame) {

        return EvFrame::computeSceneMedianDepth(pFrame->mTcw, pFrame->getAllMapPointsMono(), 2);
    }

    /*int EvTrackManager::trackAndOptEvFrameSynch(ORB_SLAM3::Frame* pFrame) {

        // Event-Image Global BA
        vector<EORB_SLAM::EventData> vEvData;
        getEventsForOptimization(vEvData, true);
        // Compute median depth
        const float medDepth = this->calcSceneMedianDepthForEvImOpt(pFrame);
        return EORB_SLAM::MyOptimizer::PoseOptimization(pFrame, vEvData, medDepth, mL2EvImSTD, mImSize);
    }*/

    bool EvTrackManager::reconstSynchEvMCI(PoseImagePtr& pPoseImage, const bool dispImage) {

        if (!pPoseImage) {
            VLOG(2) << "reconstSynchEvMCI: Empty pose image, abort...\n";
            return false;
        }

        // Wait until all tiny frames are consumed
        while(!this->allInputsEmpty() || mpL1EvImBuilder->isProcessing()) {
            this_thread::sleep_for(std::chrono::microseconds(500));
        }

        // Get recommended evWinSize;
        double imTs = pPoseImage->ts;
        ulong evWinSize = 0;
        bool winSizeRes = getRecommendedEvWinSize(evWinSize);
        if (!winSizeRes) {
            VLOG(4) << "EvTrackManager::reconstSynchEvMCI: " << imTs << ", Bad EvWinSize: " << evWinSize << endl;
            return winSizeRes;
        }

        // Get current events
        vector<EORB_SLAM::EventData> vEvData;
        getEvImBuffTillImTs(imTs, evWinSize, vEvData);

        // Reconstruct MCI using L1 EvImBuilder
        // Do not use L2 since Tcw0 is not regularly the relative trans in EvWin time span
        mpL1EvImBuilder->getSynchMCI(vEvData, pPoseImage, false, false);
        VLOG_EVERY_N(50, 100) << "EvTrackManager::reconstSynchEvMCI: " << imTs
                    << "\n\tReconst. successfully: nEvs: " << vEvData.size()
                    << "\n\tL1 WinSize: " << mpEvParams->l1ChunkSize
                    << "\n\tcurrent SynchBuffSize: " << mEvImBuffer.size()
                    << "\n\tL1 EvBuffSize: " << mEventBuffer.size()
                    << "\n\timage label: " << pPoseImage->uri << endl;

        if (pPoseImage->mReconstStat == 0)
            pPoseImage->mReconstStat = 1;

        return true;
    }

    void EvTrackManager::getEmptyPoseImage(PoseImagePtr& pImage, const double& ts) {

        pImage = make_shared<PoseImage>(ts, cv::Mat(), cv::Mat(), "Synch-Image");
    }



    void EvTrackManager::eventSynchPreProcessing(ORB_SLAM3::FramePtr& pFrame) {

        mpLastSynchEvIm = nullptr;

        if (!pFrame) {
            VLOG(2) << "eventSynchPreProcessing: Empty frame pointer, abort...\n";
            return;
        }

        PoseImagePtr pImage;
        getEmptyPoseImage(pImage, pFrame->mTimeStamp);

        // Reconstruct MCI using L1 EvImBuilder
        const bool res = this->reconstSynchEvMCI(pImage);

        if (res) {
            mpLastSynchEvIm = pImage;
            mpL2SynchTracker->preProcessing(pImage);
        }
    }

    void EvTrackManager::eventSynchPostProcessing() {

        if (mpLastSynchEvIm) {
            mpL2SynchTracker->postProcessing(mpLastSynchEvIm);
        }
    }


    void EvTrackManager::setInitEvFrameSynch(ORB_SLAM3::FramePtr& pFrame) {

        // Signal EvAsynchTracker to init. tracker
        if (mpLastSynchEvIm) {
            // Do not use supplied Tcw0, because it is normally the wrong transformation
            mpL2SynchTracker->setInitEvFrameSynch(pFrame);
        }
    }

    void EvTrackManager::trackEvFrameSynch(ORB_SLAM3::FramePtr& pFrame) {

        // Dispatch MCI to main pipline of EvAsynchTracker
        if (mpLastSynchEvIm) {
            VLOG(50) << "EvTrackManager::trackEvFrameSynch: Called with orbFrame at " << pFrame->mTimeStamp << endl;
            mpL2SynchTracker->trackEvFrameSynch();
        }
    }

    int EvTrackManager::trackAndOptEvFrameSynch(ORB_SLAM3::FramePtr& pFrame) {

        // Track MCI with LK Tracker and find the matches
        int res = -1;
        if (mpLastSynchEvIm) {
            res = mpL2SynchTracker->trackAndOptEvFrameSynch(pFrame);
        }
        // TODO: Do we need evFrame refinement?? ->
        //  In that case maybe better to do it here
        return res;
    }

    void EvTrackManager::trackEvKeyFrameSynch(ORB_SLAM3::KeyFrame *pKFcur) {

        // Dispatch current MCI
        if (mpLastSynchEvIm) {
            mpL2SynchTracker->trackEvKeyFrameSynch(pKFcur);
        }
    }

    /*void EvTrackManager::setRefEvKeyFrameSynch(ORB_SLAM3::KeyFrame *pKFcur) {

        mpL2SynchTracker->setRefEvKeyFrameSynch(pKFcur);
    }

    void EvTrackManager::eventImageInitOptimization(ORB_SLAM3::Map *pMap, const int nIterations) {

        ORB_SLAM3::KeyFrame *pEvKFref, *pEvKFcur;
        bool res = mpL2SynchTracker->getKFsForInitBASynch(pEvKFref, pEvKFcur);
        if (res) {
            vector<ORB_SLAM3::KeyFrame*> vEvKFs{pEvKFref, pEvKFcur};
            MyOptimizer::GlobalBundleAdjustment(pMap, vEvKFs, nIterations);

            // Remember to scale Event Map to unit scale
            const float evMedDepth = pEvKFref->ComputeSceneMedianDepth(2);
            vector<ORB_SLAM3::MapPoint*> vpEvMPs = pEvKFref->GetMapPointMatches();
            EvLocalMapping::scalePoseAndMap(vEvKFs, vpEvMPs, evMedDepth);
        }
        else {

            ORB_SLAM3::Optimizer::GlobalBundleAdjustemnt(pMap, nIterations);
        }
    }

    void EvTrackManager::eventImageInitOptimization(ORB_SLAM3::Map *pMap, ORB_SLAM3::KeyFrame* pKFini,
                                                    ORB_SLAM3::KeyFrame* pKFcur, const int nIterations) {

        PoseImagePtr pImage;
        getEmptyPoseImage(pImage, pKFcur->mTimeStamp);

        // Reconstruct MCI using L1 EvImBuilder
        const bool reconstRes = this->reconstSynchEvMCI(pImage);

        // Dispatch current MCI
        int stat = -1;
        if (reconstRes) {
            stat = mpL2SynchTracker->evImInitOptimizationSynch(pImage->mImage, pMap, pKFini, pKFcur, nIterations);
        }
        if (!reconstRes || stat < 0) {

            ORB_SLAM3::Optimizer::GlobalBundleAdjustemnt(pMap, nIterations);
        }
    }*/

    bool EvTrackManager::resolveEventMapInit(ORB_SLAM3::FramePtr &pFrIni, ORB_SLAM3::KeyFrame *pKFini,
                                             ORB_SLAM3::FramePtr &pFrCurr, ORB_SLAM3::KeyFrame *pKFcur, const int nIteration) {

        if (mpL2SynchTracker && mpLastSynchEvIm)
            return mpL2SynchTracker->resolveEventMapInit(pFrIni, pKFini, pFrCurr, pKFcur, nIteration);
        else
            return false;
    }

    bool EvTrackManager::evImReconst2ViewsSynch(const ORB_SLAM3::FramePtr& pFrame1,
            ORB_SLAM3::FramePtr& pFrame2, const std::vector<int> &vMatches12, cv::Mat &R21, cv::Mat &t21,
            std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated) {

        // Dispatch MCI to main pipline of EvAsynchTracker
        int rStat = -1;
        if (mpLastSynchEvIm) {
            rStat = mpL2SynchTracker->evImReconst2ViewsSynch(pFrame1, pFrame2, vMatches12,R21, t21, vP3D, vbTriangulated);
        }

        // If cannot reconstruct using events, reconstruct with image points
        if (rStat < 0) {
            return pFrame1->mpCamera->ReconstructWithTwoViews(pFrame1->getAllUndistKPtsMono(),
                    pFrame2->getAllUndistKPtsMono(), vMatches12, R21, t21, vP3D, vbTriangulated);
        }
        else {
            return rStat != 0;
        }
    }

    int EvTrackManager::evLocalMappingSynch(ORB_SLAM3::KeyFrame* pEvKFcur) {

        if (mpL2SynchTracker) {
            return mpL2SynchTracker->evLocalMappingSynch(pEvKFcur);
        }
        return -1;
    }

    void EvTrackManager::applyScaleAndRotationEvSynch(const cv::Mat &R, const float scale, const bool bScaleVel, const cv::Mat& t) {

        if (mpEvAtlas) {
            mpEvAtlas->GetCurrentMap()->ApplyScaledRotation(R, scale, bScaleVel, t);
        }
    }

    void EvTrackManager::updateEvFrameImuSynch(const float s, const ORB_SLAM3::IMU::Bias &b, ORB_SLAM3::KeyFrame *pOrbKFcur) {

        if (mpL2SynchTracker && pOrbKFcur) {
            mpL2SynchTracker->updateFrameIMU(s, b, pOrbKFcur->mpSynchEvKF);
        }
    }

}// namespace EORB_SLAM

