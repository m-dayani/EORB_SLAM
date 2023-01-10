//
// Created by root on 7/28/21.
//

#include "IMU_Manager.h"

#include "EvAsynchTracker.h"

#include "System.h"
#include "Optimizer.h"
#include "Converter.h"

#include "EvAsynchTracker.h"

using namespace std;
using namespace ORB_SLAM3;


namespace EORB_SLAM {

    IMU_Manager::IMU_Manager(const IMUParamsPtr& pIMUParams) :
            mbInitializing(false), mbAbortInit(false)  {

        //b_parse_imu = ParseIMUParamFile(fSettings);
        if(pIMUParams->missParams) {
            LOG(ERROR) << "*Error with the IMU parameters in the config file*" << std::endl;
        }
        else {
            mpImuCalib = new IMU::Calib(pIMUParams->Tbc, pIMUParams->Ng * pIMUParams->sf,
                                        pIMUParams->Na * pIMUParams->sf,
                                        pIMUParams->Ngw / pIMUParams->sf, pIMUParams->Naw / pIMUParams->sf);
            //mpImuPreintegratedFromLastKF = new IMU::Preintegrated(mLastBias, *mpImuCalib);
            //mnFramesToResetIMU = nFramesToResetIMU;
            std::unique_lock<std::mutex> lock1(mMtxInfoIMU);
            mpInitInfoIMU = make_shared<InitInfoIMU>();
        }
    }

    IMU_Manager::~IMU_Manager() {

        for (ORB_SLAM3::IMU::Preintegrated * & pPreInt : mvpImuPreintegratedFromLastKF) {

            pPreInt = nullptr;
        }
        delete mpImuCalib;
        mpImuCalib = nullptr;
    }

    uint IMU_Manager::setNewIntegrationChannel() {

        unique_lock<mutex> lock(mMutexImuQueue);
        uint Nchannels = mvlQueueImuData.size();
        mvlQueueImuData.resize(Nchannels+1);
        mvpImuPreintegratedFromLastKF.resize(Nchannels+1);
        {
            std::unique_lock<std::mutex> lock1(mMtxInfoIMU);
            mvpImuPreintegratedFromLastKF[Nchannels] = new IMU::Preintegrated(mpInitInfoIMU->getLastImuBias(),
                                                                              *mpImuCalib);
        }
        return Nchannels;
    }

    void IMU_Manager::refreshPreintFromLastKF(uint preIntId) {

        if (preIntId >= 0 && preIntId < mvpImuPreintegratedFromLastKF.size()) {

            std::unique_lock<std::mutex> lock1(mMtxInfoIMU);
            mvpImuPreintegratedFromLastKF[preIntId] = new IMU::Preintegrated(mpInitInfoIMU->getLastImuBias(), *mpImuCalib);
        }
    }

    void IMU_Manager::grabImuData(const IMU::Point &imuMeasurement)
    {
        unique_lock<mutex> lock(mMutexImuQueue);
        for (auto& lQueueImuData : mvlQueueImuData)
            lQueueImuData.push_back(imuMeasurement);
    }

    void IMU_Manager::preintegrateIMU(uint preIntId, const EvFramePtr& mpCurrentFrame, const IMU::Bias& lastBias)
    {
        //cout << "start preintegration" << endl;
        if (preIntId < 0 || preIntId >= mvlQueueImuData.size()) {

            DLOG(ERROR) << "IMU_Manager::preintegrateIMU: Wrong Preintegration ID: " << preIntId << endl;
            return;
        }

        EvFramePtr pPrevFrame = mpCurrentFrame->getPrevFrame();

        if(!pPrevFrame)
        {
            Verbose::PrintMess("non prev frame ", Verbose::VERBOSITY_NORMAL);
            mpCurrentFrame->setIntegrated();
            return;
        }

        // cout << "start loop. Total meas:" << mlQueueImuData.size() << endl;

        mvImuFromLastFrame.clear();
        mvImuFromLastFrame.reserve(mvlQueueImuData[preIntId].size());
        if(mvlQueueImuData[preIntId].empty())
        {
            Verbose::PrintMess("Not IMU data in mlQueueImuData!!", Verbose::VERBOSITY_NORMAL);
            mpCurrentFrame->setIntegrated();
            return;
        }

        while(true)
        {
            bool bSleep = false;
            {
                unique_lock<mutex> lock(mMutexImuQueue);
                if(!mvlQueueImuData[preIntId].empty())
                {
                    IMU::Point* m = &mvlQueueImuData[preIntId].front();
                    cout.precision(17);
                    if(m->t < pPrevFrame->mTimeStamp - 0.001l)
                    {
                        mvlQueueImuData[preIntId].pop_front();
                    }
                    else if(m->t < mpCurrentFrame->mTimeStamp - 0.001l)
                    {
                        mvImuFromLastFrame.push_back(*m);
                        mvlQueueImuData[preIntId].pop_front();
                    }
                    else
                    {
                        mvImuFromLastFrame.push_back(*m);
                        break;
                    }
                }
                else
                {
                    break;
                    bSleep = true;
                }
            }
            if(bSleep)
                std::this_thread::sleep_for(std::chrono::microseconds(500));
        }

        // Check if firstAcc empty & fill it
        // TODO: Can even average a few Acc readings
        if (!mvImuFromLastFrame.empty()) {
            mpCurrentFrame->setFirstAcc(mvImuFromLastFrame.back());
            if (pPrevFrame->getFirstAcc().empty()) {
                pPrevFrame->setFirstAcc(mvImuFromLastFrame[0]);
            }
        }

        const int n = static_cast<int>(mvImuFromLastFrame.size())-1;
        IMU::Preintegrated* pImuPreintegratedFromLastFrame = new IMU::Preintegrated(lastBias, mpCurrentFrame->mImuCalib);

        for(int i=0; i<n; i++)
        {
            float tstep;
            cv::Point3f acc, angVel;
            if((i==0) && (i<(n-1)))
            {
                float tab = mvImuFromLastFrame[i+1].t - mvImuFromLastFrame[i].t;
                float tini = mvImuFromLastFrame[i].t - pPrevFrame->mTimeStamp;
                acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a-
                       (mvImuFromLastFrame[i+1].a-mvImuFromLastFrame[i].a)*(tini/tab))*0.5f;
                angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w-
                          (mvImuFromLastFrame[i+1].w-mvImuFromLastFrame[i].w)*(tini/tab))*0.5f;
                tstep = mvImuFromLastFrame[i+1].t - pPrevFrame->mTimeStamp;
            }
            else if(i<(n-1))
            {
                acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a)*0.5f;
                angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w)*0.5f;
                tstep = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
            }
            else if((i>0) && (i==(n-1)))
            {
                float tab = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
                float tend = mvImuFromLastFrame[i+1].t - mpCurrentFrame->mTimeStamp;
                acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a-
                       (mvImuFromLastFrame[i+1].a-mvImuFromLastFrame[i].a)*(tend/tab))*0.5f;
                angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w-
                          (mvImuFromLastFrame[i+1].w-mvImuFromLastFrame[i].w)*(tend/tab))*0.5f;
                tstep = mpCurrentFrame->mTimeStamp - mvImuFromLastFrame[i].t;
            }
            else if((i==0) && (i==(n-1)))
            {
                acc = mvImuFromLastFrame[i].a;
                angVel = mvImuFromLastFrame[i].w;
                tstep = mpCurrentFrame->mTimeStamp - pPrevFrame->mTimeStamp;
            }

            if (!mvpImuPreintegratedFromLastKF[preIntId])
                cout << "mpImuPreintegratedFromLastKF does not exist" << endl;
            mvpImuPreintegratedFromLastKF[preIntId]->IntegrateNewMeasurement(acc,angVel,tstep);
            pImuPreintegratedFromLastFrame->IntegrateNewMeasurement(acc,angVel,tstep);
        }

        mpCurrentFrame->mpImuPreintegratedFrame = pImuPreintegratedFromLastFrame;
        mpCurrentFrame->mpImuPreintegrated = mvpImuPreintegratedFromLastKF[preIntId];
        //mpCurrentFrame->mpLastKeyFrame = mpLastKeyFrame;

        mpCurrentFrame->setIntegrated();

        Verbose::PrintMess("Preintegration is finished!! ", Verbose::VERBOSITY_DEBUG);
    }

    bool IMU_Manager::isInitializing() {

        //std::unique_lock<mutex> lock1(mMtxInitImu);
        return mbInitializing == true;
    }

    // TODO: Stuff related to mpTracker, KeyFrame management and some local vars are removed.
    // Localized vars: bInitializing, mFirstTs, mTinit, infoInertial, mbNewInit, mnKFs
    void IMU_Manager::initializeIMU(EvAsynchTracker* pTracker, ORB_SLAM3::Atlas* pAtlas, ORB_SLAM3::KeyFrame* pCurrKF,
                                    double& mFirstTs, double& mTinit, const bool bMonocular, const float priorG,
                                    const float priorA, const bool bFIBA)
    {
        //if (mbResetRequested)
        //    return;
        Map* pCurrMap = pCurrKF->GetMap();

        float minTime;
        int nMinKF;
        if (bMonocular)
        {
            minTime = 2.0;
            nMinKF = 10;
        }
        else
        {
            minTime = 1.0;
            nMinKF = 10;
        }

        if(pAtlas->KeyFramesInMap()<nMinKF)
            return;

        // Retrieve all keyframe in temporal order (only in current map)
        list<KeyFrame*> lpKF;
        KeyFrame* pKF = pCurrKF;
        while(pKF->mPrevKF && pCurrMap == pKF->mPrevKF->GetMap())
        {
            lpKF.push_front(pKF);
            pKF = pKF->mPrevKF;
        }
        lpKF.push_front(pKF);
        vector<KeyFrame*> vpKF(lpKF.begin(),lpKF.end());

        if(vpKF.size()<nMinKF)
            return;

        mFirstTs = vpKF.front()->mTimeStamp;
        if(pCurrKF->mTimeStamp-mFirstTs<minTime)
            return;

        mbInitializing.set(true);

//        while(CheckNewKeyFrames())
//        {
//            ProcessNewKeyFrame();
//            vpKF.push_back(mpCurrentKeyFrame);
//            lpKF.push_back(mpCurrentKeyFrame);
//        }

        const int N = vpKF.size();

        Eigen::Matrix3d mRwg;
        double scale = 1;//mpInitInfoIMU->getLastScale();

        Eigen::Vector3d mba, mbg;
        Eigen::MatrixXd infoInertial;

        {
            std::unique_lock<std::mutex> lock1(mMtxInfoIMU);
            InitInfoIMU::toImuBias(mpInitInfoIMU->getLastImuBias(), mbg, mba);
            infoInertial = mpInitInfoIMU->getInfoInertial();
        }

        // Compute and KF velocities mRwg estimation
        if (!pCurrKF->GetMap()->isImuInitialized())
        {
            cv::Mat cvRwg;
            cv::Mat dirG = cv::Mat::zeros(3,1,CV_32F);
            for(vector<KeyFrame*>::iterator itKF = vpKF.begin(); itKF!=vpKF.end(); itKF++)
            {
                if (!(*itKF)->mpImuPreintegrated)
                    continue;
                if (!(*itKF)->mPrevKF)
                    continue;

                // TODO: How??? dirG -= Rwb(j-1) * dVw(j-1)(j)
                dirG -= (*itKF)->mPrevKF->GetImuRotation()*(*itKF)->mpImuPreintegrated->GetUpdatedDeltaVelocity();
                cv::Mat _vel = ((*itKF)->GetImuPosition() - (*itKF)->mPrevKF->GetImuPosition())/(*itKF)->mpImuPreintegrated->dT;
                (*itKF)->SetVelocity(_vel);
                (*itKF)->mPrevKF->SetVelocity(_vel);
            }

            dirG = dirG/cv::norm(dirG);
            cv::Mat gI = (cv::Mat_<float>(3,1) << 0.0f, 0.0f, -1.0f);
            cv::Mat v = gI.cross(dirG);
            const float nv = cv::norm(v);
            const float cosg = gI.dot(dirG);
            const float ang = acos(cosg);
            cv::Mat vzg = v*ang/nv;
            cvRwg = Converter::ExpSO3(vzg);
            mRwg = Converter::toMatrix3d(cvRwg);
            mTinit = static_cast<float>(pCurrKF->mTimeStamp-mFirstTs);
        }
        else
        {
            mRwg = Eigen::Matrix3d::Identity();
            mbg = Converter::toVector3d(pCurrKF->GetGyroBias());
            mba = Converter::toVector3d(pCurrKF->GetAccBias());
        }

        // double mInitTime = mpTracker->mpLastFrame->mTimeStamp - vpKF.front()->mTimeStamp;
        if (mbAbortInit == true) {
            mbInitializing.set(false);
            return;
        }
        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        Optimizer::InertialOptimization(pAtlas->GetCurrentMap(), mRwg, scale, mbg, mba,
                                        bMonocular, infoInertial, false, false, priorG, priorA);
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        //cout << "scale after inertial-only optimization: " << mScale << endl;
        //cout << "bg after inertial-only optimization: " << mbg << endl;
        //cout << "ba after inertial-only optimization: " << mba << endl;

        if (scale<1e-1)
        {
            cout << "scale too small" << endl;
            mbInitializing.set(false);
            return;
        }

        // Before this line we are not changing the map

        if (mbAbortInit == true) {
            mbInitializing.set(false);
            return;
        }
        unique_lock<mutex> lock(pAtlas->GetCurrentMap()->mMutexMapUpdate);
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        if ((fabs(scale-1.f)>0.00001)||!bMonocular)
        {
            LOG(INFO) << "IMU_Manager::initializeIMU: Aquiring tracker update lock...\n";
            std::unique_lock<mutex> lock1(pTracker->mMtxUpdateState); // avoid possible collisions
            LOG(INFO) << "IMU_Manager::initializeIMU: Lock Aquired successfully!\n";
            pAtlas->GetCurrentMap()->ApplyScaledRotation(Converter::toCvMat(mRwg).t(), static_cast<float>(scale),true);
            pTracker->updateFrameIMU(static_cast<float>(scale),vpKF[0]->GetImuBias(),pCurrKF);
        }
        std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

        // Check if initialization OK
        if (!pAtlas->isImuInitialized())
            for(int i=0;i<N;i++)
            {
                KeyFrame* pKF2 = vpKF[i];
                pKF2->bImu = true;
            }

        //cout << "Before GIBA: " << endl;
        //cout << "ba: " << mpCurrentKeyFrame->GetAccBias() << endl;
        //cout << "bg: " << mpCurrentKeyFrame->GetGyroBias() << endl;

        std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
        if (bFIBA && mbAbortInit == false)
        {
            if (priorA!=0.f)
                Optimizer::FullInertialBA(pAtlas->GetCurrentMap(), 100, false, 0, NULL, true, priorG, priorA);
            else
                Optimizer::FullInertialBA(pAtlas->GetCurrentMap(), 100, false, 0, NULL, false);
        }

        std::chrono::steady_clock::time_point t5 = std::chrono::steady_clock::now();

        // Update last state
        {
            std::unique_lock<std::mutex> lock1(mMtxInfoIMU);
            mpInitInfoIMU->setLastImuBias(mbg, mba);
            mpInitInfoIMU->setInfoInertial(infoInertial);
            if (!pAtlas->isImuInitialized())
                mpInitInfoIMU->setIniRwg(Converter::toCvMat(mRwg));
            mpInitInfoIMU->setLastRwg(mpInitInfoIMU->getIniRwg());
            mpInitInfoIMU->setLastScale(scale);
        }

        // If initialization is OK
        {
            LOG(INFO) << "IMU_Manager::initializeIMU: Aquiring tracker update lock...\n";
            std::unique_lock<mutex> lock1(pTracker->mMtxUpdateState); // avoid possible collisions
            LOG(INFO) << "IMU_Manager::initializeIMU: Lock Aquired successfully!\n";
            pTracker->updateFrameIMU(1.0,vpKF[0]->GetImuBias(),pCurrKF);
        }
        if (!pAtlas->isImuInitialized())
        {
            cout << "IMU in Map " << pAtlas->GetCurrentMap()->GetId() << " is initialized" << endl;
            pAtlas->SetImuInitialized();
            //mpTracker->t0IMU = mpTracker->mpCurrentFrame->mTimeStamp;
            pCurrKF->bImu = true;
        }

        bool mbNewInit=true;
        int mnKFs=vpKF.size();
        //mIdxInit++;

//        for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
//        {
//            (*lit)->SetBadFlag();
//            delete *lit;
//        }
//        mlNewKeyFrames.clear();

        //mpTracker->mState=Tracking::OK;
        mbInitializing.set(false);

//        cout << "After GIBA: " << endl;
//        cout << "ba: " << mpCurrentKeyFrame->GetAccBias() << endl;
//        cout << "bg: " << mpCurrentKeyFrame->GetGyroBias() << endl;
//        double t_inertial_only = std::chrono::duration_cast<std::chrono::duration<double> >(t1 - t0).count();
//        double t_update = std::chrono::duration_cast<std::chrono::duration<double> >(t3 - t2).count();
//        double t_viba = std::chrono::duration_cast<std::chrono::duration<double> >(t5 - t4).count();
//        cout << t_inertial_only << ", " << t_update << ", " << t_viba << endl;

        pCurrKF->GetMap()->IncreaseChangeIndex();
    }

    void IMU_Manager::scaleRefinement(EvAsynchTracker* pTracker, ORB_SLAM3::Atlas* pAtlas, ORB_SLAM3::KeyFrame* pCurrKF,
                                      const bool bMonocular)
    {
        Map* pCurrMap = pCurrKF->GetMap();
//        if (mbResetRequested)
//            return;

        // Retrieve all keyframes in temporal order
        list<KeyFrame*> lpKF;
        KeyFrame* pKF = pCurrKF;
        while(pKF->mPrevKF && pCurrMap == pKF->mPrevKF->GetMap())
        {
            lpKF.push_front(pKF);
            pKF = pKF->mPrevKF;
        }
        lpKF.push_front(pKF);
        vector<KeyFrame*> vpKF(lpKF.begin(),lpKF.end());

//        while(CheckNewKeyFrames())
//        {
//            ProcessNewKeyFrame();
//            vpKF.push_back(mpCurrentKeyFrame);
//            lpKF.push_back(mpCurrentKeyFrame);
//        }

        const int N = vpKF.size();

        Eigen::Matrix3d mRwg = Eigen::Matrix3d::Identity();
        double mScale = 1.0;

        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        Optimizer::InertialOptimization(pAtlas->GetCurrentMap(), mRwg, mScale);
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        if (mScale<1e-1) // 1e-1
        {
            cout << "scale too small" << endl;
            mbInitializing=false;
            return;
        }

        // Before this line we are not changing the map
        unique_lock<mutex> lock(pAtlas->GetCurrentMap()->mMutexMapUpdate);
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        if ((fabs(mScale-1.f)>0.00001)||!bMonocular)
        {
            LOG(INFO) << "IMU_Manager::scaleRefinement: Aquiring tracker update lock...\n";
            std::unique_lock<mutex> lock1(pTracker->mMtxUpdateState); // avoid possible collisions
            LOG(INFO) << "IMU_Manager::scaleRefinement: Lock Aquired successfully!\n";
            pAtlas->GetCurrentMap()->ApplyScaledRotation(Converter::toCvMat(mRwg).t(),mScale,true);
            pTracker->updateFrameIMU(static_cast<float>(mScale), pCurrKF->GetImuBias(), pCurrKF);
        }
        std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

//        for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
//        {
//            (*lit)->SetBadFlag();
//            delete *lit;
//        }
//        mlNewKeyFrames.clear();

        double t_inertial_only = std::chrono::duration_cast<std::chrono::duration<double> >(t1 - t0).count();

        // To perform pose-inertial opt w.r.t. last keyframe
        pCurrKF->GetMap()->IncreaseChangeIndex();
    }

    bool IMU_Manager::predictStateIMU(const uint intId, KeyFrame* pLastKeyFrame, Frame* pCurrentFrame, const bool bMapUpdated)
    {
        if(!pCurrentFrame->mpPrevFrame)
        {
            Verbose::PrintMess("No last frame", Verbose::VERBOSITY_NORMAL);
            return false;
        }

        if(bMapUpdated && pLastKeyFrame)
        {
            const cv::Mat twb1 = pLastKeyFrame->GetImuPosition();
            const cv::Mat Rwb1 = pLastKeyFrame->GetImuRotation();
            const cv::Mat Vwb1 = pLastKeyFrame->GetVelocity();

            const cv::Mat Gz = (cv::Mat_<float>(3,1) << 0,0,-IMU::GRAVITY_VALUE);
            const float t12 = mvpImuPreintegratedFromLastKF[intId]->dT;

            cv::Mat Rwb2 = Converter::NormalizeRotation(Rwb1*mvpImuPreintegratedFromLastKF[intId]->GetDeltaRotation(pLastKeyFrame->GetImuBias()));
            cv::Mat twb2 = twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mvpImuPreintegratedFromLastKF[intId]->GetDeltaPosition(pLastKeyFrame->GetImuBias());
            cv::Mat Vwb2 = Vwb1 + t12*Gz + Rwb1*mvpImuPreintegratedFromLastKF[intId]->GetDeltaVelocity(pLastKeyFrame->GetImuBias());
            pCurrentFrame->SetImuPoseVelocity(Rwb2, twb2, Vwb2);
            pCurrentFrame->mPredRwb = Rwb2.clone();
            pCurrentFrame->mPredtwb = twb2.clone();
            pCurrentFrame->mPredVwb = Vwb2.clone();
            pCurrentFrame->mImuBias = pLastKeyFrame->GetImuBias();
            pCurrentFrame->mPredBias = pCurrentFrame->mImuBias;
            return true;
        }
        else
            cout << "no IMU prediction!!" << endl;

        return false;
    }

    bool IMU_Manager::predictStateIMU(Frame* pLastFrame, Frame* pCurrentFrame, const bool bMapUpdated)
    {
        if(!pCurrentFrame->mpPrevFrame)
        {
            Verbose::PrintMess("No last frame", Verbose::VERBOSITY_NORMAL);
            return false;
        }

        if(!bMapUpdated)
        {
            const cv::Mat twb1 = pLastFrame->GetImuPosition();
            const cv::Mat Rwb1 = pLastFrame->GetImuRotation();
            const cv::Mat Vwb1 = pLastFrame->mVw;
            const cv::Mat Gz = (cv::Mat_<float>(3,1) << 0,0,-IMU::GRAVITY_VALUE);
            const float t12 = pCurrentFrame->mpImuPreintegratedFrame->dT;

            cv::Mat Rwb2 = Converter::NormalizeRotation(Rwb1 * pCurrentFrame->mpImuPreintegratedFrame->GetDeltaRotation(pLastFrame->mImuBias));
            cv::Mat twb2 = twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1 * pCurrentFrame->mpImuPreintegratedFrame->GetDeltaPosition(pLastFrame->mImuBias);
            cv::Mat Vwb2 = Vwb1 + t12*Gz + Rwb1 * pCurrentFrame->mpImuPreintegratedFrame->GetDeltaVelocity(pLastFrame->mImuBias);

            pCurrentFrame->SetImuPoseVelocity(Rwb2, twb2, Vwb2);
            pCurrentFrame->mPredRwb = Rwb2.clone();
            pCurrentFrame->mPredtwb = twb2.clone();
            pCurrentFrame->mPredVwb = Vwb2.clone();
            pCurrentFrame->mImuBias = pLastFrame->mImuBias;
            pCurrentFrame->mPredBias = pCurrentFrame->mImuBias;
            return true;
        }
        else
            cout << "no IMU prediction!!" << endl;

        return false;
    }

    ORB_SLAM3::IMU::Point IMU_Manager::getFirstImu(uint id) {

        if (id >= 0 && id < mvlQueueImuData.size()) {
            return mvlQueueImuData[id].front();
        }
        return {0,0,0,0,0,0,0.0};
    }

    ORB_SLAM3::IMU::Point IMU_Manager::getLastImu(uint id) {

        if (id >= 0 && id < mvlQueueImuData.size()) {
            return mvlQueueImuData[id].back();
        }
        return {0,0,0,0,0,0,0.0};
    }

    ORB_SLAM3::IMU::Preintegrated *IMU_Manager::getImuPreintegratedFromLastKF(uint id) {

        if (id >= 0 && id < mvpImuPreintegratedFromLastKF.size()) {
            return mvpImuPreintegratedFromLastKF[id];
        }
        return nullptr;
    }

    void IMU_Manager::updateState(const InitInfoImuPtr& pImuInfo) {

        std::unique_lock<std::mutex> lock1(mMtxInfoIMU);
        mpInitInfoIMU->updateState(pImuInfo);
    }

    void IMU_Manager::reset() {

        std::unique_lock<std::mutex> lock1(mMtxInfoIMU);
        mpInitInfoIMU = make_shared<InitInfoIMU>();
    }

    void IMU_Manager::resetAll() {

        this->reset();

        // reset queue and other data
        {
            unique_lock<mutex> lock(mMutexImuQueue);
            for (std::list<ORB_SLAM3::IMU::Point>& imuQueue : mvlQueueImuData) {

                imuQueue.clear();
            }
        }

        {
            std::unique_lock<std::mutex> lock1(mMtxInfoIMU);
            for (ORB_SLAM3::IMU::Preintegrated *&pPreint : mvpImuPreintegratedFromLastKF) {

                pPreint = new IMU::Preintegrated(mpInitInfoIMU->getLastImuBias(), *mpImuCalib);
            }
        }

        mvImuFromLastFrame.clear();
    }

    InitInfoImuPtr IMU_Manager::getInitInfoIMU() {

        std::unique_lock<mutex> lock1(mMtxInfoIMU);
        return mpInitInfoIMU;
    }

    void IMU_Manager::updateLastImuBias(const ORB_SLAM3::IMU::Bias &bIMU)  {

        std::unique_lock<mutex> lock1(mMtxInfoIMU);
        mpInitInfoIMU->setLastImuBias(bIMU);
    }

    ORB_SLAM3::IMU::Bias IMU_Manager::getLastImuBias()  {

        std::unique_lock<mutex> lock1(mMtxInfoIMU);
        return mpInitInfoIMU->getLastImuBias();
    }

    cv::Mat IMU_Manager::getIniRwg()  {

        std::unique_lock<mutex> lock1(mMtxInfoIMU);
        return mpInitInfoIMU->getIniRwg();
    }

    cv::Mat IMU_Manager::getLastRwg()  {

        std::unique_lock<mutex> lock1(mMtxInfoIMU);
        return mpInitInfoIMU->getLastRwg();
    }

    /* ============================================================================================================== */

    InitInfoIMU::InitInfoIMU() : mLastBias(), mScale(1.0), mInfoInertial(Eigen::MatrixXd::Zero(9,9)),
            mInfoA(1.f), mInfoG(1.f), mInfoRwg(1.f), mInfoS(1.f) {

        mRwg0 = cv::Mat::eye(3,3,CV_32FC1);
        mLastRwg = mRwg0.clone();
    }

    InitInfoIMU::InitInfoIMU(const ORB_SLAM3::IMU::Bias& bIMU, const cv::Mat &Rwg0, double scale) : InitInfoIMU() {

        setLastScale(scale);
        setLastImuBias(bIMU);
        setIniRwg(Rwg0);
        setLastRwg(Rwg0);
    }

    void InitInfoIMU::initRwg(const cv::Mat& firstAcc) {

        if (!firstAcc.empty()){

            // Always remember: Acc measures a in reverse direction -> so multiply by -1
            const float a_a = static_cast<float>(cv::norm(firstAcc));
            cv::Mat a_u = -firstAcc / a_a;
            cv::Mat g = cv::Mat::zeros(3,1,CV_32FC1);
            g.at<float>(2) = -1;

            cv::Mat v = g.cross(a_u);
            const float s = static_cast<float>(cv::norm(v));
            const float cosg = static_cast<float>(g.dot(a_u));
            const float ang = acos(cosg);
            cv::Mat vzg = v*ang/s;

            // Rwg == Rbw (w -> gw)
            mRwg0 = Converter::ExpSO3(vzg);
            mLastRwg = mRwg0.clone();

            double err = cv::norm(mRwg0 * g - a_u);
            DLOG(INFO) << "IMU_Manager::getIniRwg: err(Rwg*g-au) = " << err << endl;
        }
    }

    void InitInfoIMU::updateState(const shared_ptr<InitInfoIMU> &pImuInfo) {

        setLastScale(pImuInfo->getLastScale());
        setLastImuBias(pImuInfo->getLastImuBias());
        setIniRwg(pImuInfo->getIniRwg());
        setLastRwg(pImuInfo->getLastRwg());
        setInfoInertial(pImuInfo->getInfoInertial());
    }

    ORB_SLAM3::IMU::Bias InitInfoIMU::toImuBias(const Eigen::Vector3d &mbg, const Eigen::Vector3d &mba) {

        return ORB_SLAM3::IMU::Bias(mba.x(), mba.y(), mba.z(), mbg.x(), mbg.y(), mbg.z());
    }

    void InitInfoIMU::toImuBias(const IMU::Bias &bIMU, Eigen::Vector3d &mbg, Eigen::Vector3d &mba) {

        mbg << bIMU.bwx, bIMU.bwy, bIMU.bwz;
        mba << bIMU.bax, bIMU.bay, bIMU.baz;
    }
}