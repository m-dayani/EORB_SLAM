//
// Created by root on 8/6/21.
//

#include "EvLocalMapping.h"

#include <utility>

#include "Optimizer.h"
#include "MyOptimizer.h"

#include "EvAsynchTracker.h"

//#include <opencv2/sfm.hpp>


using namespace ORB_SLAM3;

namespace EORB_SLAM {

    EvLocalMapping::EvLocalMapping(EvAsynchTracker* pEvTracker, std::shared_ptr<ORB_SLAM3::Atlas> pAtlas,
            SensorConfigPtr pSensor, const CamParamsPtr& pCamParams, std::shared_ptr<IMU_Manager> pImuManager) :
                mpCurrKF(nullptr), mbStop(false), mbProcFlag(false), mbOptimize(true), mbLocalMappingEnabled(true),
                mbAbortBA(false), mbImuInitialized(false), mpTracker(pEvTracker), mpEvAtlas(std::move(pAtlas)),
                mpSensor(std::move(pSensor)), mpImuManager(std::move(pImuManager)), mFirstTs(0.f), mTinit(0.f),
                mbMonocular(mpSensor->isMonocular()), mbInertial(mpSensor->isInertial()), mK(pCamParams->mK.clone()),
                mbFarPoints(false), mThFarPoints(pCamParams->mThFarPoints)
    {
        if (mThFarPoints != 0)
            mbFarPoints = true;
    }

    bool EvLocalMapping::isReadyForNewKeyFrame() {
        return mqpKeyFrames <= 1;
    }

    void EvLocalMapping::insertNewKeyFrame(ORB_SLAM3::KeyFrame *pCurrKF) {

        mqpKeyFrames.push(pCurrKF);
    }

    // TODO: This seems to have small logical problems:
    // TODO: Once initialized, try to use ref. KF instead of init. frame.
    void EvLocalMapping::localMapping(KeyFrame* pKFcur) {

        // Maybe refining map (discarding bad points) or even key frames????
        // Because we're not using descriptors here, this does not seem to be a good idea
        //int nRemoved = this->mapPointCullingCovis(pKFcur->mnId);
        int nRemoved = EvLocalMapping::mapPointCullingSTD(mpEvAtlas);
        DLOG(INFO) << "* L2 localMapping, removed " << nRemoved << endl;

        KeyFrame* pReferenceKF = pKFcur->GetParent();

        // Triangulate and add new map points
        //vector<int> vMatches = mpCurrFrame->getMatches();
        //vector<MapPoint*> mIniMapPoints = mpIniFrame->getAllMapPointsMono();
        //vector<bool> vbOutliers = mpIniFrame->getAllOutliersMono();

        vector<MapPoint*> mIniMapPoints = pReferenceKF->GetMapPointMatches();
        int nMPts = mIniMapPoints.size();
        vector<bool> vbOutliers(nMPts, true);
        vector<int> vMatches(nMPts, -1);

        for (int i = 0; i < nMPts; i++) {
            if (mIniMapPoints[i]) {
                vbOutliers[i] = false;
                vMatches[i] = i;
            }
        }

        //unsigned nNewP3D =
        int nNewPts = EvLocalMapping::createNewMapPoints(mpEvAtlas, pReferenceKF,
                                                         pKFcur, mIniMapPoints, vbOutliers, mIniMapPoints, vMatches);
        DLOG(INFO) << "* L2 localMapping, added " << nNewPts << " new map points\n";

        //mpIniFrame->setAllMapPointsMono(mIniMapPoints);
        //mpIniFrame->setAllMPOutliers(vbOutliers);

        //for (size_t i = 0; i < vMatches.size(); i++) {
        //    if (vMatches[i] >= 0) {
                //mpCurrFrame->setMapPoint(i, mIniMapPoints[i]);
                //mpCurrFrame->setMPOutlier(i, vbOutliers[i]);
        //    }
        //}

        //mpTracker->updateRefMapPoints(mIniMapPoints, vbOutliers);
    }

    // This is local because we have lots of tiny maps
    void EvLocalMapping::localBA() {

        // The first pose in current map is always fixed (unlike GBA)
        Optimizer::GlobalBundleAdjustemnt(mpEvAtlas->GetCurrentMap());
    }

    void EvLocalMapping::Run() {

        while (!mbStop) {

            if (mqpKeyFrames > 0) {

                mbProcFlag.set(true);

                mpCurrKF = mqpKeyFrames.front();
                mqpKeyFrames.pop();

                if (!mpCurrKF) {
                    LOG(WARNING) << "EvLocalMapping::Run: Null KeyFrame! Abort...\n";
                    mbProcFlag.set(false);
                    continue;
                }

                // Local Mapping (Cull/Create Map Points)
                if (mbLocalMappingEnabled && mpCurrKF->GetMap()->GetAllKeyFrames().size() > 2) {
                    // Local Mapping (Add/Remove Map Points)
                    this->localMapping(mpCurrKF);
                }

                mbAbortBA = false;
                if (mbInertial && mpImuManager)
                    mpImuManager->abortInit(false);

                // LBA
                if(mpEvAtlas->KeyFramesInMap() > 2)
                {
                    bool b_doneLBA = false;
                    int num_FixedKF_BA = 0;
                    int num_OptKF_BA = 0;
                    int num_MPs_BA = 0;
                    int num_edges_BA = 0;

                    if(mbInertial && mpCurrKF->GetMap()->isImuInitialized())
                    {
                        float dist = cv::norm(mpCurrKF->mPrevKF->GetCameraCenter() - mpCurrKF->GetCameraCenter()) +
                                     cv::norm(mpCurrKF->mPrevKF->mPrevKF->GetCameraCenter() - mpCurrKF->mPrevKF->GetCameraCenter());

                        if(dist>0.05)
                            mTinit += mpCurrKF->mTimeStamp - mpCurrKF->mPrevKF->mTimeStamp;
                        if(!mpCurrKF->GetMap()->GetIniertialBA2())
                        {
                            if((mTinit<10.f) && (dist<0.02))
                            {
                                cout << "Not enough motion for initializing. Reseting..." << endl;
//                                unique_lock<mutex> lock(mMutexReset);
//                                mbResetRequestedActiveMap = true;
//                                mpMapToReset = mpCurrentKeyFrame->GetMap();
//                                mbBadImu = true;
                            }
                        }

                        int nMatchesInliers = mpTracker->getMatchesInliers();
                        bool bLarge = false; //((nMatchesInliers>75)&&mbMonocular)||((nMatchesInliers>100)&&!mbMonocular);

                        //bool bLarge = ((mpTracker->GetMatchesInliers()>75)&&mbMonocular)||((mpTracker->GetMatchesInliers()>100)&&!mbMonocular);
                        if (mbOptimize == true) {
                            //Optimizer::LocalInertialBA(mpCurrentKeyFrame, &mbAbortBA, mpCurrentKeyFrame->GetMap(), bLarge, !mpCurrentKeyFrame->GetMap()->GetIniertialBA2());
                            MyOptimizer::LocalInertialBA(mpCurrKF, &mbAbortBA, mpCurrKF->GetMap(),
                                                         num_FixedKF_BA, num_OptKF_BA, num_MPs_BA, num_edges_BA,
                                                         10, 10, bLarge, !mpCurrKF->GetMap()->GetIniertialBA2());
                        }
                    }
                    else
                    {
                        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
                        MyOptimizer::LocalBundleAdjustment(mpCurrKF, &mbAbortBA, mpCurrKF->GetMap(),
                                                           num_FixedKF_BA,num_OptKF_BA,num_MPs_BA,num_edges_BA);
                        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                    }

                    b_doneLBA = true;
                }

                // Initialize IMU here
                // TODO: What about updating IMU manager's last bias
                if(mbInertial && mpImuManager && !mpCurrKF->GetMap()->isImuInitialized() && mbOptimize == true)
                {
                    if (mbMonocular)
                        mpImuManager->initializeIMU(mpTracker, mpEvAtlas.get(), mpCurrKF, mFirstTs, mTinit, mbMonocular, 1e2, 1e10, true);
                    else
                        mpImuManager->initializeIMU(mpTracker, mpEvAtlas.get(), mpCurrKF, mFirstTs, mTinit, mbMonocular, 1e2, 1e5, true);

                    this->updateInertialStateTracker();
                }

                // Check redundant local Keyframes
                // This is disabled because it has bad effects on event-based config.
                this->keyFrameCulling();

                // Further IMU Init. & Scale Refinement
                if ((mTinit<100.0f) && mbInertial && mpImuManager && mbOptimize == true)
                {
                    if(mpCurrKF->GetMap()->isImuInitialized() && mpTracker->isTracking()) // Enter here everytime local-mapping is called
                    {
                        if(!mpCurrKF->GetMap()->GetIniertialBA1()){
                            if (mTinit>5.0f)
                            {
                                cout << "start VIBA 1" << endl;
                                mpCurrKF->GetMap()->SetIniertialBA1();

                                mpImuManager->initializeIMU(mpTracker, mpEvAtlas.get(), mpCurrKF, mFirstTs, mTinit, mbMonocular, 1.f, 1e5, true); // 1.f, 1e5

                                cout << "end VIBA 1" << endl;
                            }
                        }
                        //else if (mbNotBA2){
                        else if(!mpCurrKF->GetMap()->GetIniertialBA2()){
                            if (mTinit>15.0f){ // 15.0f
                                cout << "start VIBA 2" << endl;
                                mpCurrKF->GetMap()->SetIniertialBA2();

                                mpImuManager->initializeIMU(mpTracker, mpEvAtlas.get(), mpCurrKF, mFirstTs, mTinit, mbMonocular, 0.f, 0.f, true);

                                cout << "end VIBA 2" << endl;
                            }
                        }

                        //this->updateInertialStateTracker();

                        // scale refinement
                        if (((mpEvAtlas->KeyFramesInMap())<=100) &&
                            ((mTinit>25.0f && mTinit<25.5f)||
                             (mTinit>35.0f && mTinit<35.5f)||
                             (mTinit>45.0f && mTinit<45.5f)||
                             (mTinit>55.0f && mTinit<55.5f)||
                             (mTinit>65.0f && mTinit<65.5f)||
                             (mTinit>75.0f && mTinit<75.5f))){
                            cout << "start scale ref" << endl;
                            if (mbMonocular)
                                mpImuManager->scaleRefinement(mpTracker, mpEvAtlas.get(), mpCurrKF, mbMonocular);
                            cout << "end scale ref" << endl;
                        }

                        //this->updateInertialStateTracker();
                    }
                }

                mbProcFlag.set(false);
            }
            else {
                // Sleep
                std::this_thread::sleep_for(std::chrono::milliseconds (1));
            }
        }
    }

    void EvLocalMapping::resetAll() {

        mqpKeyFrames.clear();
        mpCurrKF = nullptr;

        mbProcFlag = false;
        mbAbortBA = false;
        mbImuInitialized = false;

        mFirstTs = 0.0;
        mTinit = 0.0;
    }

    void EvLocalMapping::updateInertialStateTracker() {

        bool imuInitState = false;

        if (mpEvAtlas->isImuInitialized()) {

            imuInitState = true;
//            mpTracker->updateFrameIMU(static_cast<float>(mpImuManager->getInitInfoIMU()->getLastScale()),
//                                      mpImuManager->getLastImuBias(), mpCurrKF);
        }

        mbImuInitialized = imuInitState;
        mpTracker->setImuInitialized(imuInitState);
    }

    void EvLocalMapping::processNewKeyFrame(std::shared_ptr<ORB_SLAM3::Atlas>& mpAtlas, ORB_SLAM3::KeyFrame* mpCurrentKeyFrame,
                                            list<ORB_SLAM3::MapPoint*>& mlpRecentAddedMapPoints)
    {
        // Compute Bags of Words structures
        //mpCurrentKeyFrame->ComputeBoW();

        // Associate MapPoints to the new keyframe and update normal and descriptor
        const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

        for(size_t i=0; i<vpMapPointMatches.size(); i++)
        {
            MapPoint* pMP = vpMapPointMatches[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(!pMP->IsInKeyFrame(mpCurrentKeyFrame))
                    {
                        pMP->AddObservation(mpCurrentKeyFrame, i);
                        pMP->UpdateNormalAndDepth();
                        //pMP->ComputeDistinctiveDescriptors();
                    }
                    else // this can only happen for new stereo points inserted by the Tracking
                    {
                        mlpRecentAddedMapPoints.push_back(pMP);
                    }
                }
            }
        }

        // Update links in the Covisibility Graph
        mpCurrentKeyFrame->UpdateConnections();

        // Insert Keyframe in Map
        mpAtlas->AddKeyFrame(mpCurrentKeyFrame);
    }

    void EvLocalMapping::keyFrameCulling()
    {
        // Check redundant keyframes (only local keyframes)
        // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
        // in at least other 3 keyframes (in the same or finer scale)
        // We only consider close stereo points
        const int Nd = 21; // MODIFICATION_STEREO_IMU 20 This should be the same than that one from LIBA
        mpCurrKF->UpdateBestCovisibles();
        vector<KeyFrame*> vpLocalKeyFrames = mpCurrKF->GetVectorCovisibleKeyFrames();

        float redundant_th;
        if(!mbInertial)
            redundant_th = 0.9;
        else if (mbMonocular)
            redundant_th = 0.9;
        else
            redundant_th = 0.5;

        const bool bInitImu = mpEvAtlas->isImuInitialized();
        int count=0;

        // Compoute last KF from optimizable window:
        unsigned int last_ID;
        if (mbInertial)
        {
            int count = 0;
            KeyFrame* aux_KF = mpCurrKF;
            while(count<Nd && aux_KF->mPrevKF)
            {
                aux_KF = aux_KF->mPrevKF;
                count++;
            }
            last_ID = aux_KF->mnId;
        }



        for(auto pKF : vpLocalKeyFrames)
        {
            count++;
            if((pKF->mnId==pKF->GetMap()->GetInitKFid()) || pKF->isBad())
                continue;
            const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

            int nObs = 3;
            const int thObs=nObs;
            int nRedundantObservations=0;
            int nMPs=0;
            for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
            {
                MapPoint* pMP = vpMapPoints[i];
                if(pMP)
                {
                    if(!pMP->isBad())
                    {
                        if(!mbMonocular)
                        {
                            if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0)
                                continue;
                        }

                        nMPs++;
                        if(pMP->Observations()>thObs)
                        {
                            const int &scaleLevel = (pKF -> numAllKPtsLeft() == -1) ? pKF->getKPtLevelMono(i)
                                                                                    : (i < pKF -> numAllKPtsLeft()) ? pKF -> getDistKPtMono(i).octave
                                                                                                                    : pKF -> getKPtRight(i).octave;
                            const map<KeyFrame*, tuple<int,int>> observations = pMP->GetObservations();
                            int nObs=0;
                            for(const auto & observation : observations)
                            {
                                KeyFrame* pKFi = observation.first;
                                if(pKFi==pKF)
                                    continue;
                                tuple<int,int> indexes = observation.second;
                                int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);
                                int scaleLeveli = -1;
                                if(pKFi -> numAllKPtsLeft() == -1)
                                    scaleLeveli = pKFi->getKPtLevelMono(leftIndex);
                                else {
                                    if (leftIndex != -1) {
                                        scaleLeveli = pKFi->getDistKPtMono(leftIndex).octave;
                                    }
                                    if (rightIndex != -1) {
                                        int rightLevel = pKFi->getKPtRight(rightIndex - pKFi->numAllKPtsLeft()).octave;
                                        scaleLeveli = (scaleLeveli == -1 || scaleLeveli > rightLevel) ? rightLevel
                                                                                                      : scaleLeveli;
                                    }
                                }

                                if(scaleLeveli<=scaleLevel+1)
                                {
                                    nObs++;
                                    if(nObs>thObs)
                                        break;
                                }
                            }
                            if(nObs>thObs)
                            {
                                nRedundantObservations++;
                            }
                        }
                    }
                }
            }

            if(nRedundantObservations>redundant_th*nMPs)
            {
                if (mbInertial)
                {
                    if (mpEvAtlas->KeyFramesInMap()<=Nd)
                        continue;

                    if(pKF->mnId>(mpCurrKF->mnId-2))
                        continue;

                    if(pKF->mPrevKF && pKF->mNextKF)
                    {
                        const float t = pKF->mNextKF->mTimeStamp-pKF->mPrevKF->mTimeStamp;

                        if((bInitImu && (pKF->mnId<last_ID) && t<3.) || (t<0.5))
                        {
                            pKF->mNextKF->mpImuPreintegrated->MergePrevious(pKF->mpImuPreintegrated);
                            pKF->mNextKF->mPrevKF = pKF->mPrevKF;
                            pKF->mPrevKF->mNextKF = pKF->mNextKF;
                            pKF->mNextKF = NULL;
                            pKF->mPrevKF = NULL;
                            pKF->SetBadFlag();
                        }
                        else if(!mpCurrKF->GetMap()->GetIniertialBA2() &&
                                (cv::norm(pKF->GetImuPosition()-pKF->mPrevKF->GetImuPosition())<0.02) && (t<3))
                        {
                            pKF->mNextKF->mpImuPreintegrated->MergePrevious(pKF->mpImuPreintegrated);
                            pKF->mNextKF->mPrevKF = pKF->mPrevKF;
                            pKF->mPrevKF->mNextKF = pKF->mNextKF;
                            pKF->mNextKF = NULL;
                            pKF->mPrevKF = NULL;
                            pKF->SetBadFlag();
                        }
                    }
                }
                else
                {
                    pKF->SetBadFlag();
                }
            }
            if((count > 20 && mbAbortBA) || count>100) // MODIFICATION originally 20 for mbabortBA check just 10 keyframes
            {
                break;
            }
        }
    }

    cv::Mat EvLocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
    {
        cv::Mat R1w = pKF1->GetRotation();
        cv::Mat t1w = pKF1->GetTranslation();
        cv::Mat R2w = pKF2->GetRotation();
        cv::Mat t2w = pKF2->GetTranslation();

        cv::Mat R12 = R1w*R2w.t();
        cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;

        cv::Mat t12x = SkewSymmetricMatrix(t12);

        const cv::Mat &K1 = pKF1->mpCamera->toK();
        const cv::Mat &K2 = pKF2->mpCamera->toK();


        return K1.t().inv()*t12x*R12*K2.inv();
    }

    cv::Mat EvLocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
    {
        return (cv::Mat_<float>(3,3) <<             0, -v.at<float>(2), v.at<float>(1),
                v.at<float>(2),               0,-v.at<float>(0),
                -v.at<float>(1),  v.at<float>(0),              0);
    }

    bool EvLocalMapping::triangulateAndCheckMP(const cv::KeyPoint& kp1, const cv::KeyPoint& kp2,
            KeyFrame* pKF1, KeyFrame* pKF2, const cv::Mat& Tcw1, const cv::Mat& Rcw1, const cv::Mat& Rwc1,
            const cv::Mat& tcw1, const cv::Mat& Ow1, const cv::Mat& Tcw2, const cv::Mat& Rcw2, const cv::Mat& Rwc2,
            const cv::Mat& tcw2, const cv::Mat& Ow2, const float fThCosParr, const float ratioFactor, cv::Mat& x3D,
            const bool checkKPtLevelInfo) {

        // Check parallax between rays
        cv::Mat xn1 = pKF1->mpCamera->unprojectMat(kp1.pt);
        cv::Mat xn2 = pKF2->mpCamera->unprojectMat(kp2.pt);

        cv::Mat ray1 = Rwc1*xn1;
        cv::Mat ray2 = Rwc2*xn2;
        const float cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2));

        //float fThCosParr = 0.9998;
        //if (mpSensor->isInertial()) fThCosParr = 0.9998;

        if(cosParallaxRays>0 && cosParallaxRays<fThCosParr)
        {
            //TwoViewReconstruction::Triangulate(kp1, kp2, Tcw1, Tcw2, x3D);
            // Linear Triangulation Method
            cv::Mat A(4,4,CV_32F);
            A.row(0) = xn1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
            A.row(1) = xn1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
            A.row(2) = xn2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
            A.row(3) = xn2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

            cv::Mat w,u,vt;
            cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

            x3D = vt.row(3).t();

            if(x3D.at<float>(3) <= 1e-12)
                return false;

            // Euclidean coordinates
            x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
        }
        else {
            return false; //No stereo and very low parallax
        }

        cv::Mat x3Dt = x3D.t();

        if(x3Dt.empty()) return false;
        //Check triangulation in front of cameras
        float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2);
        if(z1<=0)
            return false;

        float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);
        if(z2<=0)
            return false;

        //Check reprojection error in first keyframe
        const float &sigmaSquare1 = pKF1->getORBLevelSigma2(kp1.octave);
        const float x1 = Rcw1.row(0).dot(x3Dt)+tcw1.at<float>(0);
        const float y1 = Rcw1.row(1).dot(x3Dt)+tcw1.at<float>(1);
        const float invz1 = 1.0/z1;

        cv::Point2f uv1 = pKF1->mpCamera->project(cv::Point3f(x1,y1,z1));
        float errX1 = uv1.x - kp1.pt.x;
        float errY1 = uv1.y - kp1.pt.y;

        if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)
            return false;

        //Check reprojection error in second keyframe
        const float sigmaSquare2 = pKF2->getORBLevelSigma2(kp2.octave);
        const float x2 = Rcw2.row(0).dot(x3Dt)+tcw2.at<float>(0);
        const float y2 = Rcw2.row(1).dot(x3Dt)+tcw2.at<float>(1);
        const float invz2 = 1.0/z2;

        cv::Point2f uv2 = pKF2->mpCamera->project(cv::Point3f(x2,y2,z2));
        float errX2 = uv2.x - kp2.pt.x;
        float errY2 = uv2.y - kp2.pt.y;
        if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
            return false;

        //Check scale consistency
        cv::Mat normal1 = x3D-Ow1;
        float dist1 = cv::norm(normal1);

        cv::Mat normal2 = x3D-Ow2;
        float dist2 = cv::norm(normal2);

        if(dist1==0 || dist2==0)
            return false;

        //if(mbFarPoints && (dist1>=mThFarPoints||dist2>=mThFarPoints)) // MODIFICATION
        //    continue;

        const float ratioDist = dist2/dist1;
        const float ratioOctave = pKF1->getORBScaleFactor(kp1.octave)/pKF2->getORBScaleFactor(kp2.octave);

        return !(ratioDist * ratioFactor < ratioOctave || ratioDist > ratioOctave * ratioFactor);
    }

    void EvLocalMapping::getCameraPoseInfo(ORB_SLAM3::KeyFrame* pKF, cv::Mat& Tcw1, cv::Mat& Rcw1, cv::Mat& Rwc1,
                                            cv::Mat& tcw1, cv::Mat& Ow1) {

        Rcw1 = pKF->GetRotation();
        Rwc1 = Rcw1.t();
        tcw1 = pKF->GetTranslation();
        Rcw1.copyTo(Tcw1.colRange(0,3));
        tcw1.copyTo(Tcw1.col(3));
        Ow1 = pKF->GetCameraCenter();
    }

    // Addes new map points based on previously triangulated P3Ds
    unsigned EvLocalMapping::addNewMapPoints(const shared_ptr<Atlas>& mpAtlas, KeyFrame* pKFini, KeyFrame* pKFcur,
                                             const vector<int>& vMatches12, const vector<cv::Point3f>& vP3D) {

        assert(mpAtlas && pKFini && pKFcur && vMatches12.size() == vP3D.size());
        // might also check if max(vMatches12) be less than size of vpMPs and ...

        unsigned nNewMPs = 0;
        const int nPtsCur = pKFcur->numAllKPts();

        for(size_t i=0; i < vMatches12.size(); i++)
        {
            const int matchedIdx = vMatches12[i];
            if(matchedIdx<0 || matchedIdx >= nPtsCur)
                continue;

            //Create MapPoint.
            cv::Mat worldPos(vP3D[i]);
            // refKeyFrame in MapPoint(...) is used in map culling!
            auto* pMP = new MapPoint(worldPos, pKFcur, mpAtlas->GetCurrentMap());

            pKFini->AddMapPoint(pMP,i);
            pKFcur->AddMapPoint(pMP,matchedIdx);

            pMP->AddObservation(pKFini,i);
            pMP->AddObservation(pKFcur,matchedIdx);

            pMP->UpdateNormalAndDepth();

            //Add to Map
            mpAtlas->AddMapPoint(pMP);

            nNewMPs++;
        }
        return nNewMPs;
    }

    // Cull map point based on 3 Sigma of map point depth
    int EvLocalMapping::mapPointCullingSTD(std::shared_ptr<ORB_SLAM3::Atlas>& pEvAtlas) {

        int nRemoved = 0;

        vector<MapPoint*> allMPs = pEvAtlas->GetAllMapPoints();

        cv::Mat vDepths = cv::Mat::ones(allMPs.size(), 1, CV_32F);
        // Calculate STD of Depths
        for (size_t i = 0; i < allMPs.size(); i++) {

            MapPoint* pMP = allMPs[i];
            if (pMP) {
                cv::Mat curPos = pMP->GetWorldPos();
                vDepths.at<float>(i, 0) = curPos.at<float>(2,0);
            }
        }
        cv::Scalar dMean, dSTD;
        cv::meanStdDev(vDepths, dMean, dSTD);
        const float depthSTD = dSTD[0];
        const float depthMean = dMean[0];

        for (auto* pMP : allMPs) {

            if (pMP && abs(pMP->GetWorldPos().at<float>(2,0) - depthMean) > 3 * depthSTD) {

                pMP->SetBadFlag();
                nRemoved++;
            }
        }

        return nRemoved;
    }

    int EvLocalMapping::createNewMapPoints(const shared_ptr<Atlas>& mpAtlas, KeyFrame* pKFini, KeyFrame* pKFcur,
            vector<MapPoint*>& vpMPs, vector<bool>& vbOutliers, const vector<MapPoint*>& vpIniMPs,
            const vector<int>& vMatches12, const float medDepth) {

        assert(mpAtlas && pKFini && pKFcur && vpMPs.size() == vbOutliers.size() && vpMPs.size() == vpIniMPs.size());

        vector<cv::KeyPoint> vIniKPts = pKFini->getAllUndistKPtsMono();
        vector<cv::KeyPoint> vKPts = pKFcur->getAllUndistKPtsMono();

        assert(vIniKPts.size() == vMatches12.size());

        int nNewPts = 0;
        const float ratioFactor = 1.5f*pKFcur->getORBScaleFactor();
        float fThCosParr = 0.9998;

        cv::Mat Rcw1, Rwc1, tcw1, Ow1, Tcw1(3,4,CV_32F);
        getCameraPoseInfo(pKFcur, Tcw1, Rcw1, Rwc1, tcw1, Ow1);

        cv::Mat Rcw2, Rwc2, tcw2, Ow2, Tcw2(3,4,CV_32F);
        getCameraPoseInfo(pKFini, Tcw2, Rcw2, Rwc2, tcw2, Ow2);

        cv::Mat vBaseline = Ow2-Ow1;
        const float baseline = cv::norm(vBaseline);

        float medianDepthKF2 = medDepth;
        if (medDepth < 0) {
            medianDepthKF2 = pKFini->ComputeSceneMedianDepth(2);
        }
        const float ratioBaselineDepth = baseline/medianDepthKF2;

        if(ratioBaselineDepth<0.01)
            return nNewPts;

        // Compute Fundamental Matrix
        //cv::Mat F12 = ComputeF12(pKFcur,pKFini);

        // Triangulate each match
        const int nRefKpts = vMatches12.size();
        const int nCurKpts = vKPts.size();

        for(int i0=0; i0<nRefKpts; i0++) {

            const int i1 = vMatches12[i0];
            if(i1<0 || i1 >= nCurKpts)
                continue;

            MapPoint* pMP = vpIniMPs[i1];
            if (pMP) {

                pMP->AddObservation(pKFcur,i1);
                pKFcur->AddMapPoint(pMP,i1);
                pMP->UpdateNormalAndDepth();

                vpMPs[i1] = pMP;
                vbOutliers[i1] = false;

                continue;
            }

            const cv::KeyPoint &kp1 = vKPts[i1];
            const cv::KeyPoint &kp2 = vIniKPts[i0];

            cv::Mat x3D;
            bool res = triangulateAndCheckMP(kp1, kp2, pKFcur, pKFini, Tcw1, Rcw1, Rwc1, tcw1, Ow1,
                                                   Tcw2, Rcw2, Rwc2, tcw2, Ow2, fThCosParr, ratioFactor, x3D);

            if (!res)
                continue;

            // Triangulation is succesfull
            pMP = new MapPoint(x3D,pKFcur,mpAtlas->GetCurrentMap());

            pMP->AddObservation(pKFcur,i1);
            pMP->AddObservation(pKFini,i0);

            pKFcur->AddMapPoint(pMP,i1);
            pKFini->AddMapPoint(pMP,i0);

            //pMP->ComputeDistinctiveDescriptors();

            pMP->UpdateNormalAndDepth();

            mpAtlas->AddMapPoint(pMP);
            //mlpRecentAddedMapPoints.push_back(pMP);

            // All matched map points, new and old go here
            vpMPs[i1] = pMP;
            vbOutliers[i1] = false;

            nNewPts++;
        }

        return nNewPts;
    }

    void EvLocalMapping::scalePoseAndMap(vector<KeyFrame*>& vpKFs, vector<MapPoint*>& vpAllMapPoints,
                                         const float& medianDepth, const bool isInertial) {

        assert(medianDepth != 0.f);

        float invMedianDepth;
        if(isInertial)
            invMedianDepth = 4.0f/medianDepth; // 4.0f
        else
            invMedianDepth = 1.0f/medianDepth;

        // Scale initial baseline
        for (KeyFrame* kf : vpKFs) {
            cv::Mat Tcw = kf->GetPose();
            Tcw.col(3).rowRange(0, 3) = Tcw.col(3).rowRange(0, 3) * invMedianDepth;
            kf->SetPose(Tcw);
        }

        // Scale points
        for(auto & pMapPoint : vpAllMapPoints)
        {
            if(pMapPoint)
            {
                MapPoint* pMP = pMapPoint;
                pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    void EvLocalMapping::scalePoseAndMap(std::vector<EvFramePtr>& vpFrs, std::vector<cv::Point3f>& vPts3D,
                                         const float& medianDepth, const bool isInertial) {

        assert(medianDepth != 0.f);

        float invMedianDepth;
        if(isInertial)
            invMedianDepth = 4.0f/medianDepth; // 4.0f
        else
            invMedianDepth = 1.0f/medianDepth;

        // Scale initial baseline
        for (EvFramePtr& pFri : vpFrs) {
            cv::Mat Tcw = pFri->mTcw.clone();
            Tcw.col(3).rowRange(0, 3) = Tcw.col(3).rowRange(0, 3) * invMedianDepth;
            pFri->SetPose(Tcw);
        }

        // Scale points
        for(auto& pt3d : vPts3D) {

            pt3d *= invMedianDepth;
        }
    }

    void EvLocalMapping::mapPointCullingCovis(shared_ptr<Atlas>& mpAtlas, KeyFrame* mpCurrentKeyFrame,
            const EvFramePtr& pFrCur, list<ORB_SLAM3::MapPoint*>& mlpRecentAddedMapPoints, const bool mbMonocular)
    {
        // Check Recent Added MapPoints
        auto lit = mlpRecentAddedMapPoints.begin();
        const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

        int nThObs;
        if(mbMonocular)
            nThObs = 2;
        else
            nThObs = 3; // MODIFICATION_STEREO_IMU here 3
        const int cnThObs = nThObs;

        int borrar = mlpRecentAddedMapPoints.size();

        while(lit!=mlpRecentAddedMapPoints.end())
        {
            MapPoint* pMP = *lit;

            if(pMP->isBad())
                lit = mlpRecentAddedMapPoints.erase(lit);
            else if(pMP->GetFoundRatio()<0.25f)
            {
                pMP->SetBadFlag();
                lit = mlpRecentAddedMapPoints.erase(lit);
            }
            else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs)
            {
                pMP->SetBadFlag();
                lit = mlpRecentAddedMapPoints.erase(lit);
            }
            else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3)
                lit = mlpRecentAddedMapPoints.erase(lit);
            else
            {
                lit++;
                borrar--;
            }
        }
        //cout << "erase MP: " << borrar << endl;
    }

    void EvLocalMapping::createNewMapPoints(shared_ptr<Atlas>& mpAtlas, KeyFrame* mpCurrentKeyFrame,
                                            list<ORB_SLAM3::MapPoint*>& mlpRecentAddedMapPoints)
    {
        auto* pTracker = dynamic_cast<EvAsynchTrackerU*>(mpTracker);
        if (!pTracker) {
            return;
        }

        EvFramePtr pFrCur = pTracker->getFrameById(mpCurrentKeyFrame->mnFrameId);
        if (!pFrCur) {
            return;
        }

        // Retrieve neighbor keyframes in covisibility graph
        int nn = 10;
        // For stereo inertial case
        if(mbMonocular)
            nn=20;
        vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

        if (mbInertial)
        {
            KeyFrame* pKF = mpCurrentKeyFrame;
            int count=0;
            while((vpNeighKFs.size()<=nn)&&(pKF->mPrevKF)&&(count++<nn))
            {
                vector<KeyFrame*>::iterator it = std::find(vpNeighKFs.begin(), vpNeighKFs.end(), pKF->mPrevKF);
                if(it==vpNeighKFs.end())
                    vpNeighKFs.push_back(pKF->mPrevKF);
                pKF = pKF->mPrevKF;
            }
        }

        cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
        cv::Mat Rwc1 = Rcw1.t();
        cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
        cv::Mat Tcw1(3,4,CV_32F);
        Rcw1.copyTo(Tcw1.colRange(0,3));
        tcw1.copyTo(Tcw1.col(3));
        cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

        const float &fx1 = mpCurrentKeyFrame->fx;
        const float &fy1 = mpCurrentKeyFrame->fy;
        const float &cx1 = mpCurrentKeyFrame->cx;
        const float &cy1 = mpCurrentKeyFrame->cy;
        const float &invfx1 = mpCurrentKeyFrame->invfx;
        const float &invfy1 = mpCurrentKeyFrame->invfy;

        float fORBScaleFactor = mpCurrentKeyFrame->getORBScaleFactor();
        float ratioFactor = 1.5f*fORBScaleFactor;

        // Search matches with epipolar restriction and triangulate
        for(size_t i=0; i<vpNeighKFs.size(); i++)
        {
            //if(i>0 && CheckNewKeyFrames())// && (mnMatchesInliers>50))
            //    return;

            KeyFrame* pKF2 = vpNeighKFs[i];

            GeometricCamera* pCamera1 = mpCurrentKeyFrame->mpCamera, *pCamera2 = pKF2->mpCamera;

            // Check first that baseline is not too short
            cv::Mat Ow2 = pKF2->GetCameraCenter();
            cv::Mat vBaseline = Ow2-Ow1;
            const float baseline = cv::norm(vBaseline);

            if(!mbMonocular)
            {
                if(baseline<pKF2->mb)
                    continue;
            }
            else
            {
                const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
                const float ratioBaselineDepth = baseline/medianDepthKF2;

                if(ratioBaselineDepth<0.01)
                    continue;
            }

            // Search matches using feature tracks
            //vector<pair<size_t,size_t> > vMatchedIndices;
            EvFramePtr pFr2 = pTracker->getFrameById(pKF2->mnFrameId);
            if (!pFr2) {
                continue;
            }
            FeatureTrack::matchFeatureTracks(pFrCur, pFr2);
            vector<int> vMatches12 = pFr2->getMatches();

            const int nKPts2 = pKF2->numAllKPts();

            cv::Mat Rcw2 = pKF2->GetRotation();
            cv::Mat Rwc2 = Rcw2.t();
            cv::Mat tcw2 = pKF2->GetTranslation();
            cv::Mat Tcw2(3,4,CV_32F);
            Rcw2.copyTo(Tcw2.colRange(0,3));
            tcw2.copyTo(Tcw2.col(3));

            const float &fx2 = pKF2->fx;
            const float &fy2 = pKF2->fy;
            const float &cx2 = pKF2->cx;
            const float &cy2 = pKF2->cy;
            const float &invfx2 = pKF2->invfx;
            const float &invfy2 = pKF2->invfy;

            // Triangulate each match
            const int nmatches = vMatches12.size();
            for(int ikp=0; ikp<nmatches; ikp++)
            {
                const int &idx1 = ikp; //vMatchedIndices[ikp].first;
                const int &idx2 = vMatches12[ikp];//vMatchedIndices[ikp].second;

                if (idx2 < 0 || idx2 >= nKPts2)
                    continue;

                MapPoint* pMpt2 = pFr2->getMapPoint(idx2);
                // Don't duplicate map points
                if (pMpt2 && !pMpt2->isBad())
                    continue;

                const cv::KeyPoint &kp1 = (mpCurrentKeyFrame->numAllKPtsLeft() == -1) ? mpCurrentKeyFrame->getUndistKPtMono(idx1)
                                                                                        : (idx1 < mpCurrentKeyFrame->numAllKPtsLeft()) ? mpCurrentKeyFrame->getDistKPtMono(idx1)
                                                                                                                                         : mpCurrentKeyFrame->getKPtRight(idx1 - mpCurrentKeyFrame->numAllKPtsLeft());
                const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1];
                bool bStereo1 = (!mpCurrentKeyFrame->mpCamera2 && kp1_ur>=0);
                const bool bRight1 = !(mpCurrentKeyFrame->numAllKPtsLeft() == -1 || idx1 < mpCurrentKeyFrame->numAllKPtsLeft());

                const cv::KeyPoint &kp2 = (pKF2 -> numAllKPtsLeft() == -1) ? pKF2->getUndistKPtMono(idx2)
                                                                           : (idx2 < pKF2->numAllKPtsLeft()) ? pKF2->getDistKPtMono(idx2)
                                                                                                               : pKF2->getKPtRight(idx2 - pKF2->numAllKPtsLeft());

                const float kp2_ur = pKF2->mvuRight[idx2];
                bool bStereo2 = (!pKF2->mpCamera2 && kp2_ur>=0);
                const bool bRight2 = !(pKF2->numAllKPtsLeft() == -1 || idx2 < pKF2->numAllKPtsLeft());

                if(mpCurrentKeyFrame->mpCamera2 && pKF2->mpCamera2){
                    if(bRight1 && bRight2){
                        Rcw1 = mpCurrentKeyFrame->GetRightRotation();
                        Rwc1 = Rcw1.t();
                        tcw1 = mpCurrentKeyFrame->GetRightTranslation();
                        Tcw1 = mpCurrentKeyFrame->GetRightPose();
                        Ow1 = mpCurrentKeyFrame->GetRightCameraCenter();

                        Rcw2 = pKF2->GetRightRotation();
                        Rwc2 = Rcw2.t();
                        tcw2 = pKF2->GetRightTranslation();
                        Tcw2 = pKF2->GetRightPose();
                        Ow2 = pKF2->GetRightCameraCenter();

                        pCamera1 = mpCurrentKeyFrame->mpCamera2;
                        pCamera2 = pKF2->mpCamera2;
                    }
                    else if(bRight1 && !bRight2){
                        Rcw1 = mpCurrentKeyFrame->GetRightRotation();
                        Rwc1 = Rcw1.t();
                        tcw1 = mpCurrentKeyFrame->GetRightTranslation();
                        Tcw1 = mpCurrentKeyFrame->GetRightPose();
                        Ow1 = mpCurrentKeyFrame->GetRightCameraCenter();

                        Rcw2 = pKF2->GetRotation();
                        Rwc2 = Rcw2.t();
                        tcw2 = pKF2->GetTranslation();
                        Tcw2 = pKF2->GetPose();
                        Ow2 = pKF2->GetCameraCenter();

                        pCamera1 = mpCurrentKeyFrame->mpCamera2;
                        pCamera2 = pKF2->mpCamera;
                    }
                    else if(!bRight1 && bRight2){
                        Rcw1 = mpCurrentKeyFrame->GetRotation();
                        Rwc1 = Rcw1.t();
                        tcw1 = mpCurrentKeyFrame->GetTranslation();
                        Tcw1 = mpCurrentKeyFrame->GetPose();
                        Ow1 = mpCurrentKeyFrame->GetCameraCenter();

                        Rcw2 = pKF2->GetRightRotation();
                        Rwc2 = Rcw2.t();
                        tcw2 = pKF2->GetRightTranslation();
                        Tcw2 = pKF2->GetRightPose();
                        Ow2 = pKF2->GetRightCameraCenter();

                        pCamera1 = mpCurrentKeyFrame->mpCamera;
                        pCamera2 = pKF2->mpCamera2;
                    }
                    else{
                        Rcw1 = mpCurrentKeyFrame->GetRotation();
                        Rwc1 = Rcw1.t();
                        tcw1 = mpCurrentKeyFrame->GetTranslation();
                        Tcw1 = mpCurrentKeyFrame->GetPose();
                        Ow1 = mpCurrentKeyFrame->GetCameraCenter();

                        Rcw2 = pKF2->GetRotation();
                        Rwc2 = Rcw2.t();
                        tcw2 = pKF2->GetTranslation();
                        Tcw2 = pKF2->GetPose();
                        Ow2 = pKF2->GetCameraCenter();

                        pCamera1 = mpCurrentKeyFrame->mpCamera;
                        pCamera2 = pKF2->mpCamera;
                    }
                }

                // Check parallax between rays
                cv::Mat xn1 = pCamera1->unprojectMat(kp1.pt);
                cv::Mat xn2 = pCamera2->unprojectMat(kp2.pt);

                cv::Mat ray1 = Rwc1*xn1;
                cv::Mat ray2 = Rwc2*xn2;
                const float cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2));

                float cosParallaxStereo = cosParallaxRays+1;
                float cosParallaxStereo1 = cosParallaxStereo;
                float cosParallaxStereo2 = cosParallaxStereo;

                if(bStereo1)
                    cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1]));
                else if(bStereo2)
                    cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));

                cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);

                cv::Mat x3D;
                if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 ||
                                                                              (cosParallaxRays<0.9998 && mbInertial) || (cosParallaxRays<0.9998 && !mbInertial)))
                {
                    // Linear Triangulation Method
                    cv::Mat A(4,4,CV_32F);
                    A.row(0) = xn1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                    A.row(1) = xn1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
                    A.row(2) = xn2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
                    A.row(3) = xn2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

                    cv::Mat w,u,vt;
                    cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

                    x3D = vt.row(3).t();

                    if(x3D.at<float>(3)==0)
                        continue;

                    // Euclidean coordinates
                    x3D = x3D.rowRange(0,3)/x3D.at<float>(3);

                }
                else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)
                {
                    x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);
                }
                else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)
                {
                    x3D = pKF2->UnprojectStereo(idx2);
                }
                else
                {
                    continue; //No stereo and very low parallax
                }

                cv::Mat x3Dt = x3D.t();

                if(x3Dt.empty()) continue;
                //Check triangulation in front of cameras
                float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2);
                if(z1<=0)
                    continue;

                float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);
                if(z2<=0)
                    continue;

                //Check reprojection error in first keyframe
                const float &sigmaSquare1 = mpCurrentKeyFrame->getKPtLevelSigma2(idx1);
                const float x1 = Rcw1.row(0).dot(x3Dt)+tcw1.at<float>(0);
                const float y1 = Rcw1.row(1).dot(x3Dt)+tcw1.at<float>(1);
                const float invz1 = 1.0/z1;

                if(!bStereo1)
                {
                    cv::Point2f uv1 = pCamera1->project(cv::Point3f(x1,y1,z1));
                    float errX1 = uv1.x - kp1.pt.x;
                    float errY1 = uv1.y - kp1.pt.y;

                    if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)
                        continue;

                }
                else
                {
                    float u1 = fx1*x1*invz1+cx1;
                    float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1;
                    float v1 = fy1*y1*invz1+cy1;
                    float errX1 = u1 - kp1.pt.x;
                    float errY1 = v1 - kp1.pt.y;
                    float errX1_r = u1_r - kp1_ur;
                    if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1)
                        continue;
                }

                //Check reprojection error in second keyframe
                const float sigmaSquare2 = pKF2->getKPtLevelSigma2(idx2);
                const float x2 = Rcw2.row(0).dot(x3Dt)+tcw2.at<float>(0);
                const float y2 = Rcw2.row(1).dot(x3Dt)+tcw2.at<float>(1);
                const float invz2 = 1.0/z2;
                if(!bStereo2)
                {
                    cv::Point2f uv2 = pCamera2->project(cv::Point3f(x2,y2,z2));
                    float errX2 = uv2.x - kp2.pt.x;
                    float errY2 = uv2.y - kp2.pt.y;
                    if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                        continue;
                }
                else
                {
                    float u2 = fx2*x2*invz2+cx2;
                    float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
                    float v2 = fy2*y2*invz2+cy2;
                    float errX2 = u2 - kp2.pt.x;
                    float errY2 = v2 - kp2.pt.y;
                    float errX2_r = u2_r - kp2_ur;
                    if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)
                        continue;
                }

                //Check scale consistency
                cv::Mat normal1 = x3D-Ow1;
                float dist1 = cv::norm(normal1);

                cv::Mat normal2 = x3D-Ow2;
                float dist2 = cv::norm(normal2);

                if(dist1==0 || dist2==0)
                    continue;

                if(mbFarPoints && (dist1>=mThFarPoints||dist2>=mThFarPoints)) // MODIFICATION
                    continue;

                const float ratioDist = dist2/dist1;
                const float ratioOctave = mpCurrentKeyFrame->getKPtScaleFactor(idx1)/pKF2->getKPtScaleFactor(idx2);

                if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
                    continue;

                // Triangulation is succesfull
                MapPoint* pMP = new MapPoint(x3D,mpCurrentKeyFrame,mpAtlas->GetCurrentMap());

                pMP->AddObservation(mpCurrentKeyFrame,idx1);
                pMP->AddObservation(pKF2,idx2);

                mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
                pKF2->AddMapPoint(pMP,idx2);

                //pMP->ComputeDistinctiveDescriptors();

                pMP->UpdateNormalAndDepth();

                mpAtlas->AddMapPoint(pMP);
                mlpRecentAddedMapPoints.push_back(pMP);
            }
        }
    }

    // Note: This is so sensitive to noise, consider doing something else!
    void EvLocalMapping::createNewMapPoints(std::shared_ptr<ORB_SLAM3::Atlas> &mpAtlas, std::vector<FeatureTrackPtr> &vpFtTracks,
            const ulong minFrId, const ulong maxFrId, std::list<ORB_SLAM3::MapPoint *> &mlpRecentAddedMapPoints) {

        auto* pTracker = dynamic_cast<EvAsynchTrackerU*>(mpTracker);
        if (!pTracker) {
            return;
        }

        // retrieve the point tracks structure
        vector<cv::Mat> pointTracks;
        FeatureTrack::getTracksReconstSFM(vpFtTracks, minFrId, maxFrId, pointTracks);

        google::ShutdownGoogleLogging();
        
        // sfm reconstruction
        bool is_projective = true;
        vector<cv::Mat> Rs_est, ts_est, points3d_estimated;
        cv::Mat K = mK.clone();

        cv::sfm::reconstruct(pointTracks, Rs_est, ts_est, mK, points3d_estimated, is_projective);

        // retrieve the corresponding KFs
        vector<KeyFrame*> vpCurKFs, vpKFs = mpAtlas->GetAllKeyFrames();
        const size_t nKFs = vpKFs.size();
        vpCurKFs.reserve(nKFs);

        for (size_t i = 0; i < nKFs; i++) {

            KeyFrame* pKFi = vpKFs[i];
            ulong curFrId = pKFi->mnFrameId;

            if (curFrId >= minFrId && curFrId <= maxFrId)
                vpCurKFs.push_back(pKFi);
        }

        // create and add map points if reconst. successful
        ORB_SLAM3::Map* pMap = mpAtlas->GetCurrentMap();
        for (size_t i = 0; i < vpFtTracks.size(); i++) {

            FeatureTrackPtr pTrack = vpFtTracks[i];
            if (!pTrack || !pTrack->isValid())
                continue;

            // if current track already has a map point, don't process it
            cv::KeyPoint kpt;
            MapPoint* pMpt;
            pTrack->getFeatureAndMapPoint(pTrack->getFrameIdMax(), kpt, pMpt);
            if (pMpt)
                continue;

            bool firstKF = true;

            for (auto pKFi : vpCurKFs) {

                ulong curFrId = pKFi->mnFrameId;

                if (!pTrack->validFrame(curFrId))
                    continue;

                if (firstKF) {
                    pMpt = new ORB_SLAM3::MapPoint(points3d_estimated[i], pKFi, pMap);

                    pTrack->updateMapPoint(curFrId, pMpt);

                    mpAtlas->AddMapPoint(pMpt);
                    mlpRecentAddedMapPoints.push_back(pMpt);

                    firstKF = false;
                }

                EvFramePtr pFri = pTracker->getFrameById(curFrId);

                if (pFri) {
                    const int idx = pTrack->findTrackById(pFri->getAllFeatureTracks());

                    if (idx >= 0) {
                        pKFi->AddMapPoint(pMpt, idx);
                        pMpt->AddObservation(pKFi, idx);

                        pMpt->UpdateNormalAndDepth();
                    }
                }
            }
        }
    }

} // EORB_SLAM

