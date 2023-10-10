/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/



#include <exception>
#include <chrono>
#include <thread>
#include <utility>
#include <iomanip>

#include <openssl/md5.h>

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/string.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

#include <glog/logging.h>

#include <pangolin/pangolin.h>

#include "System.h"
#include "Converter.h"
#include "MyOptimizer.h"


namespace ORB_SLAM3
{

Verbose::eLevel Verbose::th = Verbose::VERBOSITY_NONE;

void printWelcome() {
    cout << endl <<
         "ORB-SLAM3 Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza." << endl <<
         "ORB-SLAM2 Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza." << endl <<
         "This program comes with ABSOLUTELY NO WARRANTY;" << endl  <<
         "This is free software, and you are welcome to redistribute it" << endl <<
         "under certain conditions. See LICENSE.txt." << endl << endl;
}

System::System(const string &strVocFile, EORB_SLAM::SensorConfigPtr  pSensor,
               const EORB_SLAM::CamParamsPtr& pCamParams, const EORB_SLAM::PairCalibPtr& pCalib,
               const EORB_SLAM::MixedFtsParamsPtr& pORBParams, const EORB_SLAM::IMUParamsPtr& pIMUParams,
               const EORB_SLAM::ViewerParamsPtr& pViewerParams,
               const int initFr, const string &strSequence, const string &strLoadingFile):
    mpViewer(static_cast<Viewer*>(nullptr)), mbReset(false), mbResetActiveMap(false), mbActivateLocalizationMode(false),
    mbDeactivateLocalizationMode(false), mpSensor(std::move(pSensor))
{
    // Output welcome message
    //printWelcome();

    cout << "Input sensor was set to: " << mpSensor->toStr() << endl;

    bool loadedAtlas = false;
    mpCamParams = pCamParams;

    //----
    //Load ORB Vocabulary
    cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;

    mpVocabulary = new ORBVocabulary();
    if (mpSensor->isImage()) {
        bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
        if (!bVocLoad) {
            cerr << "Wrong path to vocabulary. " << endl;
            cerr << "Falied to open at: " << strVocFile << endl;
            exit(-1);
        }
        cout << "Vocabulary loaded!" << endl << endl;
    }
    else {
        cout << "Vocabulary not loaded!" << endl << endl;
    }

    //Create KeyFrame Database
    mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

    //Create the Atlas
    //mpMap = new Map();
    mpAtlas = new Atlas(0);
    mpGlobalAtlas = new Atlas(0);
    //----

    /*if(strLoadingFile.empty())
    {
        //Load ORB Vocabulary
        cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;

        mpVocabulary = new ORBVocabulary();
        bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
        if(!bVocLoad)
        {
            cerr << "Wrong path to vocabulary. " << endl;
            cerr << "Falied to open at: " << strVocFile << endl;
            exit(-1);
        }
        cout << "Vocabulary loaded!" << endl << endl;

        //Create KeyFrame Database
        mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

        //Create the Atlas
        //mpMap = new Map();
        mpAtlas = new Atlas(0);
    }
    else
    {
        //Load ORB Vocabulary
        cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;

        mpVocabulary = new ORBVocabulary();
        bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
        if(!bVocLoad)
        {
            cerr << "Wrong path to vocabulary. " << endl;
            cerr << "Falied to open at: " << strVocFile << endl;
            exit(-1);
        }
        cout << "Vocabulary loaded!" << endl << endl;

        cout << "Load File" << endl;

        // Load the file with an earlier session
        //clock_t start = clock();
        bool isRead = LoadAtlas(strLoadingFile,BINARY_FILE);

        if(!isRead)
        {
            cout << "Error to load the file, please try with other session file or vocabulary file" << endl;
            exit(-1);
        }
        mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

        mpAtlas->SetKeyFrameDababase(mpKeyFrameDatabase);
        mpAtlas->SetORBVocabulary(mpVocabulary);
        mpAtlas->PostLoad();
        //cout << "KF in DB: " << mpKeyFrameDatabase->mnNumKFs << "; words: " << mpKeyFrameDatabase->mnNumWords << endl;

        loadedAtlas = true;

        mpAtlas->CreateNewMap();

        //clock_t timeElapsed = clock() - start;
        //unsigned msElapsed = timeElapsed / (CLOCKS_PER_SEC / 1000);
        //cout << "Binary file read in " << msElapsed << " ms" << endl;

        //usleep(10*1000*1000);
    }*/

    if (mpSensor->isInertial())
        mpAtlas->SetInertialSensor();

    //Create Drawers. These are used by the Viewer
    mpFrameDrawer = new EORB_SLAM::MyFrameDrawer(mpAtlas);
    mpMapDrawer = new MapDrawer(mpAtlas, pViewerParams);

    //Initialize the Tracking thread
    //(it will live in the main thread of execution, the one that called this constructor)
    //cout << "Seq. Name: " << strSequence << endl;
    mpTracker = new Tracking(this, mpVocabulary, mpFrameDrawer, mpMapDrawer, mpAtlas, mpKeyFrameDatabase,
            mpSensor, mpCamParams, pCalib, pORBParams, pIMUParams, strSequence);

    //Initialize the Local Mapping thread and launch
    mpLocalMapper = new LocalMapping(this, mpAtlas, mpSensor->isMonocular(), mpSensor->isInertial(), strSequence);
    mptLocalMapping = new thread(&ORB_SLAM3::LocalMapping::Run,mpLocalMapper);
    mpLocalMapper->mInitFr = initFr;

    mpLocalMapper->mThFarPoints = mpCamParams->mThFarPoints;
    if(mpLocalMapper->mThFarPoints!=0)
    {
        cout << "Discard points further than " << mpLocalMapper->mThFarPoints << " m from current camera" << endl;
        mpLocalMapper->mbFarPoints = true;
    }
    else
        mpLocalMapper->mbFarPoints = false;

    //Initialize the Loop Closing thread and launch
    // mpSensor!=MONOCULAR && mpSensor!=IMU_MONOCULAR
    mpLoopCloser = new LoopClosing(mpAtlas, mpKeyFrameDatabase, mpVocabulary,
            *(mpSensor)!=EORB_SLAM::MySensorConfig::MONOCULAR); // mpSensor!=MONOCULAR);
    mptLoopClosing = new thread(&ORB_SLAM3::LoopClosing::Run, mpLoopCloser);

    //Initialize the Viewer thread and launch
    if(pViewerParams && pViewerParams->mbUseViewer)
    {
        mpViewer = new Viewer(this, mpFrameDrawer, mpMapDrawer, mpTracker, pViewerParams);
        mptViewer = new thread(&Viewer::Run, mpViewer);
        mpTracker->SetViewer(mpViewer);
        mpLoopCloser->mpViewer = mpViewer;
        //mpViewer->both = mpFrameDrawer->both;

        DLOG(INFO) << "Using viewer\n";
    }

    //Set pointers between threads
    mpTracker->SetLocalMapper(mpLocalMapper);
    mpTracker->SetLoopClosing(mpLoopCloser);

    mpLocalMapper->SetTracker(mpTracker);
    mpLocalMapper->SetLoopCloser(mpLoopCloser);

    mpLoopCloser->SetTracker(mpTracker);
    mpLoopCloser->SetLocalMapper(mpLocalMapper);

    // Fix verbosity
    //Verbose::SetTh(Verbose::VERBOSITY_NONE);

}

System::System(const string &strVocFile, const EORB_SLAM::SensorConfigPtr& pSensor,
               const EORB_SLAM::CamParamsPtr& pCamParams, const EORB_SLAM::PairCalibPtr& pCalib,
               const EORB_SLAM::MixedFtsParamsPtr& pORBParams, const EORB_SLAM::IMUParamsPtr& pIMUParams,
               const EORB_SLAM::ViewerParamsPtr& pViewerParams, const EORB_SLAM::EvParamsPtr& pEvParams, const int initFr,
               const string &strSequence, const string &strLoadingFile) :
        System::System(strVocFile, pSensor, pCamParams, pCalib, pORBParams,
                pIMUParams, pViewerParams, initFr, strSequence, strLoadingFile) {

    if (mpSensor->isEvent()) {

        //Initialize event-based tracker
        if (mpSensor->isInertial()) {
            mpEvTracker = std::unique_ptr<EORB_SLAM::EvTrackManager>(new EORB_SLAM::EvTrackManager(pEvParams,
                    pCamParams, pORBParams, mpSensor, mpVocabulary, mpViewer, pIMUParams));
        }
        else {
            mpEvTracker = std::unique_ptr<EORB_SLAM::EvTrackManager>(new EORB_SLAM::EvTrackManager(pEvParams,
                    pCamParams, pORBParams, mpSensor, mpVocabulary, mpViewer));
        }

        mpEvTracker->setOrbSlamSystem(this);
        //mpEvTracker->setMapDrawer(mpMapDrawer);

        //If we have both event and image tracking, spawn a new thread
        //Otherwise, the event tracking is in control of main thread
        if (mpSensor->isImage()) {

            mpEvTracker->setSeparateThread(true);
            mptEvTracking = std::unique_ptr<std::thread>(
                    new thread(&EORB_SLAM::EvTrackManager::Track, mpEvTracker.get()));

            //if (mpViewer) {
            //    mpViewer->setEvImDisplay(mpEvTracker->getImageDisplay());
            //}

            DLOG(INFO) << "New thread is created for event tracking\n";
        }
        //else {
            // Change MapDrawer's Atlas to Events' Atlas
        //    if (mpMapDrawer) {
        //        mpMapDrawer->mpAtlas = mpEvTracker->getEventsAtlas().get();
        //    }
        //}
    }
}

System::~System() {

//    if (mpCamParams->isPinhole()) {
//        delete dynamic_cast<Pinhole*>(mpCamera);
//    }
//    else if (mpCamParams->isFisheye()) {
//        delete dynamic_cast<KannalaBrandt8*>(mpCamera);
//        if (mpCamParams->mLinkedCam)
//            delete dynamic_cast<KannalaBrandt8*>(mpCamera2);
//    }
    // Delete all global key frames
    for (auto& kfRow : mGlobalPoseChain) {
        //delete kfRow.second;
        kfRow.second = nullptr;
    }
}

cv::Mat System::TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp,
        const vector<IMU::Point>& vImuMeas, string filename)
{
    if(!mpSensor->isStereo())
    {
        cerr << "ERROR: you called TrackStereo but input sensor was not set to Stereo nor Stereo-Inertial." << endl;
        exit(-1);
    }   

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
        unique_lock<mutex> lock(mMutexReset);
        if(mbReset)
        {
            mpTracker->Reset();
            cout << "Reset stereo..." << endl;
            mbReset = false;
            mbResetActiveMap = false;
        }
        else if(mbResetActiveMap)
        {
            mpTracker->ResetActiveMap();
            mbResetActiveMap = false;
        }
    }

    if (mpSensor->isInertial())
        for(size_t i_imu = 0; i_imu < vImuMeas.size(); i_imu++)
            mpTracker->GrabImuData(vImuMeas[i_imu]);

    // std::cout << "start GrabImageStereo" << std::endl;
    cv::Mat Tcw = mpTracker->GrabImageStereo(imLeft,imRight,timestamp,filename);

    // std::cout << "out grabber" << std::endl;

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mpCurrentFrame->getAllMapPointsMono();
    mTrackedKeyPointsUn = mpTracker->mpCurrentFrame->getAllUndistKPtsMono();

    return Tcw;
}

cv::Mat System::TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp, string filename)
{
    if(!mpSensor->isRGBD())
    {
        cerr << "ERROR: you called TrackRGBD but input sensor was not set to RGBD." << endl;
        exit(-1);
    }    

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
        unique_lock<mutex> lock(mMutexReset);
        if(mbReset)
        {
            mpTracker->Reset();
            mbReset = false;
            mbResetActiveMap = false;
        }
        else if(mbResetActiveMap)
        {
            mpTracker->ResetActiveMap();
            mbResetActiveMap = false;
        }
    }


    cv::Mat Tcw = mpTracker->GrabImageRGBD(im,depthmap,timestamp,filename);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mpCurrentFrame->getAllMapPointsMono();
    mTrackedKeyPointsUn = mpTracker->mpCurrentFrame->getAllUndistKPtsMono();
    return Tcw;
}

cv::Mat System::TrackMonocular(const cv::Mat &im, const double &timestamp, const vector<IMU::Point>& vImuMeas, string filename)
{
    if(!mpSensor->isMonocular())
    {
        cerr << "ERROR: you called TrackMonocular but input sensor was not set to Monocular nor Monocular-Inertial." << endl;
        exit(-1);
    }

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
        unique_lock<mutex> lock(mMutexReset);
        if(mbReset)
        {
            mpTracker->Reset();
            mbReset = false;
            mbResetActiveMap = false;
        }
        else if(mbResetActiveMap)
        {
            cout << "SYSTEM-> Reseting active map in monocular case" << endl;
            mpTracker->ResetActiveMap();
            mbResetActiveMap = false;
        }
    }

    if (mpSensor->isInertial())
        for(const auto & vImuMea : vImuMeas)
            mpTracker->GrabImuData(vImuMea);

    cv::Mat Tcw = mpTracker->GrabImageMonocular(im,timestamp,std::move(filename));

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mpCurrentFrame->getAllMapPointsMono();
    mTrackedKeyPointsUn = mpTracker->mpCurrentFrame->getAllUndistKPtsMono();

    return Tcw;
}


void System::ActivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbActivateLocalizationMode = true;
}

void System::DeactivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbDeactivateLocalizationMode = true;
}

bool System::MapChanged()
{
    static int n=0;
    int curn = mpAtlas->GetLastBigChangeIdx();
    if(n<curn)
    {
        n=curn;
        return true;
    }
    else
        return false;
}

void System::Reset()
{
    unique_lock<mutex> lock(mMutexReset);
    mbReset = true;
}

void System::ResetActiveMap()
{
    unique_lock<mutex> lock(mMutexReset);
    mbResetActiveMap = true;
}

void System::Shutdown()
{
    mpLocalMapper->RequestFinish();
    mpLoopCloser->RequestFinish();
    if(mpViewer)
    {
        mpViewer->RequestFinish();
        while(!mpViewer->isFinished())
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    // Wait until all thread have effectively stopped
    while(!mpLocalMapper->isFinished() || !mpLoopCloser->isFinished() || mpLoopCloser->isRunningGBA())
    {
        if(!mpLocalMapper->isFinished())
            cout << "mpLocalMapper is not finished" << endl;
        if(!mpLoopCloser->isFinished())
            cout << "mpLoopCloser is not finished" << endl;
        if(mpLoopCloser->isRunningGBA()){
            cout << "mpLoopCloser is running GBA" << endl;
            cout << "break anyway..." << endl;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    if(mpViewer)
        pangolin::BindToContext("ORB-SLAM2: Map Viewer");

    if (mpEvTracker) {
        mpEvTracker->shutdown();
    }
}


void System::SaveTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving TUM camera trajectory to " << filename << " ..." << endl;
    if(*(mpSensor) == EORB_SLAM::MySensorConfig::MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryTUM cannot be used for monocular." << endl;
        return;
    }

    EORB_SLAM::Visualization::saveAllFramePoses(mpAtlas, mpTracker->mFrameInfo, filename, false, false, 1.0, 6);

    cout << endl << "trajectory saved!" << endl;
}

void System::SaveKeyFrameTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving TUM keyframe trajectory to " << filename << " ..." << endl;

    EORB_SLAM::Visualization::saveAllKeyFramePoses(mpAtlas, filename, false, false, 1.0, 6, 7);

    cout << endl << "trajectory saved!" << endl;
}

void System::SaveTrajectoryEuRoC(const string &filename, const double tsc, const int ts_prec)
{

    cout << "# Saving EuRoC trajectory to " << filename << " ..." << endl;
    /*if(mpSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryEuRoC cannot be used for monocular." << endl;
        return;
    }*/

    EORB_SLAM::Visualization::saveAllFramePoses(mpAtlas, mpTracker->mFrameInfo, filename, true, mpSensor->isInertial(), tsc, ts_prec);

    cout << endl << "End of saving trajectory to " << filename << " ..." << endl;
}

void System::SaveKeyFrameTrajectoryEuRoC(const string &filename, const double tsc, const int ts_prec)
{
    cout << "# Saving EuRoC keyframe trajectory to " << filename << " ..." << endl;

    EORB_SLAM::Visualization::saveAllKeyFramePoses(mpAtlas, filename, true, mpSensor->isInertial(), tsc, ts_prec);

    cout << endl << "End of saving trajectory to " << filename << " ..." << endl;
}

void System::SaveTrajectoryKITTI(const string &filename)
{
    cout << endl << "Saving KITTI camera trajectory to " << filename << " ..." << endl;
    if(*(mpSensor)==EORB_SLAM::MySensorConfig::MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular." << endl;
        return;
    }

    vector<KeyFrame*> vpKFs = mpAtlas->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.
    list<cv::Mat> mlRelativeFramePoses;
    list<KeyFrame*> mlpReferences;
    list<double> mlFrameTimes;
    list<bool> mlbLost;
    mpTracker->mFrameInfo.getAllState(mlRelativeFramePoses, mlpReferences, mlFrameTimes, mlbLost);

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM3::KeyFrame*>::iterator lRit = mlpReferences.begin();
    list<double>::iterator lT = mlFrameTimes.begin();
    for(list<cv::Mat>::iterator lit=mlRelativeFramePoses.begin(), lend=mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++)
    {
        ORB_SLAM3::KeyFrame* pKF = *lRit;

        cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);

        while(pKF->isBad())
        {
            Trw = Trw*pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw*pKF->GetPose()*Two;

        cv::Mat Tcw = (*lit)*Trw;
        cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
        cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

        f << setprecision(9) << Rwc.at<float>(0,0) << " " << Rwc.at<float>(0,1)  << " " << Rwc.at<float>(0,2) << " "  << twc.at<float>(0) << " " <<
             Rwc.at<float>(1,0) << " " << Rwc.at<float>(1,1)  << " " << Rwc.at<float>(1,2) << " "  << twc.at<float>(1) << " " <<
             Rwc.at<float>(2,0) << " " << Rwc.at<float>(2,1)  << " " << Rwc.at<float>(2,2) << " "  << twc.at<float>(2) << endl;
    }

    f.close();
}


void System::SaveDebugData(const int &initIdx)
{
    // 0. Save initialization trajectory
    SaveTrajectoryEuRoC("init_FrameTrajectoy_" +to_string(mpLocalMapper->mInitSect)+ "_" + to_string(initIdx)+".txt");

    // 1. Save scale
    ofstream f;
    f.open("init_Scale_" + to_string(mpLocalMapper->mInitSect) + ".txt", ios_base::app);
    f << fixed;
    f << mpLocalMapper->mScale << endl;
    f.close();

    // 2. Save gravity direction
    f.open("init_GDir_" +to_string(mpLocalMapper->mInitSect)+ ".txt", ios_base::app);
    f << fixed;
    f << mpLocalMapper->mRwg(0,0) << "," << mpLocalMapper->mRwg(0,1) << "," << mpLocalMapper->mRwg(0,2) << endl;
    f << mpLocalMapper->mRwg(1,0) << "," << mpLocalMapper->mRwg(1,1) << "," << mpLocalMapper->mRwg(1,2) << endl;
    f << mpLocalMapper->mRwg(2,0) << "," << mpLocalMapper->mRwg(2,1) << "," << mpLocalMapper->mRwg(2,2) << endl;
    f.close();

    // 3. Save computational cost
    f.open("init_CompCost_" +to_string(mpLocalMapper->mInitSect)+ ".txt", ios_base::app);
    f << fixed;
    f << mpLocalMapper->mCostTime << endl;
    f.close();

    // 4. Save biases
    f.open("init_Biases_" +to_string(mpLocalMapper->mInitSect)+ ".txt", ios_base::app);
    f << fixed;
    f << mpLocalMapper->mbg(0) << "," << mpLocalMapper->mbg(1) << "," << mpLocalMapper->mbg(2) << endl;
    f << mpLocalMapper->mba(0) << "," << mpLocalMapper->mba(1) << "," << mpLocalMapper->mba(2) << endl;
    f.close();

    // 5. Save covariance matrix
    f.open("init_CovMatrix_" +to_string(mpLocalMapper->mInitSect)+ "_" +to_string(initIdx)+".txt", ios_base::app);
    f << fixed;
    for(int i=0; i<mpLocalMapper->mcovInertial.rows(); i++)
    {
        for(int j=0; j<mpLocalMapper->mcovInertial.cols(); j++)
        {
            if(j!=0)
                f << ",";
            f << setprecision(15) << mpLocalMapper->mcovInertial(i,j);
        }
        f << endl;
    }
    f.close();

    // 6. Save initialization time
    f.open("init_Time_" +to_string(mpLocalMapper->mInitSect)+ ".txt", ios_base::app);
    f << fixed;
    f << mpLocalMapper->mInitTime << endl;
    f.close();
}


int System::GetTrackingState()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackingState;
}

vector<MapPoint*> System::GetTrackedMapPoints()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackedMapPoints;
}

vector<cv::KeyPoint> System::GetTrackedKeyPointsUn()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackedKeyPointsUn;
}

double System::GetTimeFromIMUInit()
{
    double aux = mpLocalMapper->GetCurrKFTime()-mpLocalMapper->mFirstTs;
    if ((aux>0.) && mpAtlas->isImuInitialized())
        return mpLocalMapper->GetCurrKFTime()-mpLocalMapper->mFirstTs;
    else
        return 0.f;
}

bool System::isLost()
{
    if (!mpAtlas->isImuInitialized())
        return false;
    else
    {
        if ((mpTracker->mState==Tracking::LOST)) //||(mpTracker->mState==Tracking::RECENTLY_LOST))
            return true;
        else
            return false;
    }
}


bool System::isFinished()
{
    return (GetTimeFromIMUInit()>0.1);
}

void System::ChangeDataset()
{
    if (mpSensor->isImage()) {

        if (mpAtlas->GetCurrentMap()->KeyFramesInMap() < 12) {

            mpTracker->ResetActiveMap();
        }
        else {
            mpTracker->CreateMapInAtlas();
        }

        mpTracker->NewDataset();
    }

    if (mpEvTracker) {
        mpEvTracker->reset();
    }
}

// Masoud's contribution

    // System is ready when none of its child modules is jammed!
    bool System::IsReady() {

        bool mEvTrackingReady = true;
        if (mpEvTracker) {
            mEvTrackingReady = mpEvTracker->isReady();
        }
        return mEvTrackingReady; // && mpTracker->isReady()
    }

    void System::insertGlobalKeyFrame(KeyFrame *pKf) {

        mGlobalPoseChain.insert(pair<double, KeyFrame*>(pKf->mTimeStamp, pKf));
    }

    cv::Mat
    System::TrackEvent(const vector<EORB_SLAM::EventData> &evs, const vector<IMU::Point> &vImuMeas, const string& filename) {

        if(!(mpSensor->isMonocular() && mpSensor->isEvent() && !mpSensor->isImage()))
        {
            LOG(FATAL) << "You called TrackEvent but input sensor was not set to Event-only or Event-Inertial" << endl;
        }

        // Check mode change
        {
            unique_lock<mutex> lock(mMutexMode);
            if(mbActivateLocalizationMode)
            {
                mpLocalMapper->RequestStop();

                // Wait until Local Mapping has effectively stopped
                while(!mpLocalMapper->isStopped())
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }

                mpTracker->InformOnlyTracking(true);
                mbActivateLocalizationMode = false;
            }
            if(mbDeactivateLocalizationMode)
            {
                mpTracker->InformOnlyTracking(false);
                mpLocalMapper->Release();
                mbDeactivateLocalizationMode = false;
            }
        }

        // Check reset
        {
            unique_lock<mutex> lock(mMutexReset);
            if(mbReset)
            {
                mpTracker->Reset();
                mbReset = false;
                mbResetActiveMap = false;
            }
            else if(mbResetActiveMap)
            {
                cout << "SYSTEM-> Reseting active map in monocular case" << endl;
                mpTracker->ResetActiveMap();
                mbResetActiveMap = false;
            }
        }

        if (mpSensor->isInertial())
            for(const auto & vImuMea : vImuMeas)
                mpEvTracker->grabImuData(vImuMea);

        mpEvTracker->fillBuffer(evs);
        // If the event tracking is not running on separate thread already
        // trigger it manually.
        if (!mptEvTracking) {
            mpEvTracker->Track();
        }

//        unique_lock<mutex> lock2(mMutexState);
//        mTrackingState = mpTracker->mState;
//        mTrackedMapPoints = mpTracker->mpCurrentFrame->getAllMapPointsMono();
//        mTrackedKeyPointsUn = mpTracker->mpCurrentFrame->getAllUndistKPtsMono();

        return cv::Mat(); //Tcw
    }

    cv::Mat System::TrackEvMono(const vector<EORB_SLAM::EventData> &evs, const cv::Mat &im, const double &timestamp,
                                const vector<IMU::Point> &vImuMeas, string filename) {

        if(!mpSensor->isMonocular())
        {
            LOG(FATAL) << "You called TrackMonocular but input sensor was not set to Monocular configuration." << endl;
        }

        // Check mode change
        {
            unique_lock<mutex> lock(mMutexMode);
            if(mbActivateLocalizationMode)
            {
                mpLocalMapper->RequestStop();

                // Wait until Local Mapping has effectively stopped
                while(!mpLocalMapper->isStopped())
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }

                mpTracker->InformOnlyTracking(true);
                mbActivateLocalizationMode = false;
            }
            if(mbDeactivateLocalizationMode)
            {
                mpTracker->InformOnlyTracking(false);
                mpLocalMapper->Release();
                mbDeactivateLocalizationMode = false;
            }
        }

        // Check reset
        {
            unique_lock<mutex> lock(mMutexReset);
            if(mbReset)
            {
                mpTracker->Reset();
                mbReset = false;
                mbResetActiveMap = false;
            }
            else if(mbResetActiveMap)
            {
                cout << "SYSTEM-> Reseting active map in monocular case" << endl;
                mpTracker->ResetActiveMap();
                mbResetActiveMap = false;
            }
        }

        if (mpSensor->isInertial()) {
            for (const auto & imuMea : vImuMeas) {

                mpTracker->GrabImuData(imuMea);
                // Do this for L1 image builder
                mpEvTracker->grabImuData(imuMea);
            }

        }

        mpEvTracker->fillBuffer(evs);

        cv::Mat Tcw = mpTracker->GrabImageMonocular(im,timestamp,filename);

        // Note: Event tracking with images is alwayes in separate thread

        unique_lock<mutex> lock2(mMutexState);
        mTrackingState = mpTracker->mState;
        mTrackedMapPoints = mpTracker->mpCurrentFrame->getAllMapPointsMono();
        mTrackedKeyPointsUn = mpTracker->mpCurrentFrame->getAllUndistKPtsMono();

        return Tcw;
    }

    // NOTE: All pose writers use Twc now!
    void System::SaveTrajectoryEvent(const std::string& fileName, const double tsc, const int ts_prec) {

        cout << "# Saving Event trajectory to " << fileName << " ...\n";

        if (mpEvTracker) {
            mpEvTracker->saveFrameTrajectory(fileName, tsc, ts_prec);

            cout << endl << "End of saving trajectory to " << fileName << " ..." << endl;
        }
    }

    void System::SaveKeyFrameTrajectoryEvent(const std::string& fileName, const double tsc, const int ts_prec) {

        cout << "# Saving Event keyframe trajectory to " << fileName << " ...\n";
        if (mpEvTracker) {
            mpEvTracker->saveTrajectory(fileName, tsc, ts_prec);

            cout << endl << "End of saving trajectory to " << fileName << " ..." << endl;
        }
    }

    void System::SaveTrajectoryEvIm(const std::string &fileName, const double tsc, const int ts_prec) {

        if (!mpGlobalAtlas) {
            LOG(WARNING) << "System::SaveTrajectoryEvIm: Cannot find merged atlas, abort...\n";
            return;
        }
        cout << "# Saving keyframe trajectory to " << fileName << " ...\n";

        vector<Map*> vpMaps = mpGlobalAtlas->GetAllMaps();
        Map* pBiggerMap;
        int numMaxKFs = 0;
        for(Map* pMap :vpMaps)
        {
            if(pMap->GetAllKeyFrames().size() > numMaxKFs)
            {
                numMaxKFs = pMap->GetAllKeyFrames().size();
                pBiggerMap = pMap;
            }
        }

        if (numMaxKFs <= 0) {
            LOG(WARNING) << "System::SaveTrajectoryEvIm: Cannot retrieve map.\n";
            return;
        }

        vector<KeyFrame*> vpKFs = pBiggerMap->GetAllKeyFrames();
        sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

        // Transform all keyframes so that the first keyframe is at the origin.
        // After a loop closure the first keyframe might not be at the origin.
        ofstream f;
        f.open(fileName.c_str());
        f << fixed;

        for(auto pKF : vpKFs)
        {
            // pKF->SetPose(pKF->GetPose()*Two);

            if(pKF->isBad())
                continue;
            // currently, inertial case is not supported!
//            if (mpSensor->isInertial())
//            {
//                cv::Mat Rwb = pKF->GetImuRotation();
//                cv::Mat twb = pKF->GetImuPosition();
//
//                f << setprecision(6) << tsc*pKF->mTimeStamp  << " " << Converter::toStringQuatRaw(twb, Rwb) << endl;
//            }
//            else
//            {
            cv::Mat Rwc = pKF->GetRotation().t();
            cv::Mat twc = pKF->GetCameraCenter();

            f << setprecision(ts_prec) << tsc*pKF->mTimeStamp << " " << Converter::toStringQuatRaw(twc, Rwc) << endl;
//            }
        }
        f.close();
    }

    void System::FuseEventORB() {

        assert(mpSensor->isEvent() && mpSensor->isImage() && mpGlobalAtlas);

        // Get all event constraints
        vector<KeyFrame*> vpEvKFs, vpEvKFsFused;
        this->GetEventConstraints(vpEvKFs);

        // Merge event key frames internally
        EORB_SLAM::EvTrackManager::fuseEventTracks(vpEvKFs, vpEvKFsFused);

        // Fuse all constraints into a single map and optimize
        EORB_SLAM::MyOptimizer::MergeVisualEvent(mpAtlas, vpEvKFsFused, mpGlobalAtlas);

        // We can also exchange local ORB Atlas with Global Atlas
    }

    void System::GetEventConstraints(vector<KeyFrame*>& vpEvKFs) {

        if (mpEvTracker) {

            mpEvTracker->getEventConstraints(vpEvKFs);
        }
    }

    void System::PrepareNextIeration() {

        if (mpSensor->isImage()) {

            if (mpAtlas->GetCurrentMap()->KeyFramesInMap() < 12) {

                mpTracker->ResetActiveMap();
            } else {
                mpTracker->CreateMapInAtlas();
            }
        }

        if (mpEvTracker) {
            mpEvTracker->reset();
        }
    }

    void System::ConsumeEvents() {

        if (mpEvTracker) {
            // Event-Image config.
            if (mptEvTracking) {
                while (!mpEvTracker->allInputsEmpty()) {
                    DLOG_EVERY_N(INFO, 10000) << "System::ConsumeEvents: Waiting for event tracker to exhaust buffers.\n";
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
            }
            else {
                while (!mpEvTracker->allInputsEmpty()) {
                    mpEvTracker->Track();
                    DLOG_EVERY_N(INFO, 10000) << "System::ConsumeEvents: Waiting for event tracker to exhaust buffers.\n";
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
            }
        }
    }

    unsigned System::GetL1EvWinSize() {
        if (mpEvTracker) {
            return mpEvTracker->getL1ChunkSize();
        }
        return DEF_L1_CHUNK_SIZE;
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    void System::SetInitEvFrameSynch(Frame *pFrame) {

        if (mpEvTracker) {
            mpEvTracker->setInitEvFrameSynch(pFrame);
        }
    }

    int System::TrackAndOptEvFrameSynch(Frame* pFrame) {

        if (mpEvTracker) {
            return mpEvTracker->trackAndOptEvFrameSynch(pFrame);
        }
        return 0;
    }

    void System::TrackEvFrameSynch(Frame *pFrame) {

        if (mpEvTracker) {
            mpEvTracker->trackEvFrameSynch(pFrame);
        }
    }

    void System::SetRefEvKeyFrameSynch(KeyFrame *pKFcur) {

        if (mpEvTracker) {
            mpEvTracker->setRefEvKeyFrameSynch(pKFcur);
        }
    }

    void System::TrackEvKeyFrameSynch(KeyFrame *pKFcur) {

        if (mpEvTracker) {
            mpEvTracker->trackEvKeyFrameSynch(pKFcur);
        }
    }

    void System::EventImageInitOptimization(Map* pMap, const int nIterations) {

        if (mpEvTracker) {
            return mpEvTracker->eventImageInitOptimization(pMap, nIterations);
        }
    }

    void System::EventImageInitOptimization(Map* pMap, KeyFrame* pKFini, KeyFrame* pKFcur, const int nIterations) {

        if (mpEvTracker) {
            mpEvTracker->eventImageInitOptimization(pMap, pKFini, pKFcur, nIterations);
        }
    }

    bool System::EvImReconst2ViewsSynch(GeometricCamera* pCamera, const Frame* pFrame1, const Frame* pFrame2,
            const std::vector<int> &vMatches12, cv::Mat &R21, cv::Mat &t21,
            std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated) {

        if (mpEvTracker) {
            return mpEvTracker->evImReconst2ViewsSynch(pCamera, pFrame1, pFrame2, vMatches12,
                                                       R21, t21, vP3D, vbTriangulated);
        }
        // Do not call this when sensor config is not event!
        return false;
    }

    void System::SaveMap(const string& fileName, const int stat) {

        EORB_SLAM::Visualization::saveMapPoints(this->mpAtlas->GetCurrentMap()->GetAllMapPoints(), fileName);
    }

/*void System::SaveAtlas(int type){
    cout << endl << "Enter the name of the file if you want to save the current Atlas session. To exit press ENTER: ";
    string saveFileName;
    getline(cin,saveFileName);
    if(!saveFileName.empty())
    {
        //clock_t start = clock();

        // Save the current session
        mpAtlas->PreSave();
        mpKeyFrameDatabase->PreSave();

        string pathSaveFileName = "./";
        pathSaveFileName = pathSaveFileName.append(saveFileName);
        pathSaveFileName = pathSaveFileName.append(".osa");

        string strVocabularyChecksum = CalculateCheckSum(mStrVocabularyFilePath,TEXT_FILE);
        std::size_t found = mStrVocabularyFilePath.find_last_of("/\\");
        string strVocabularyName = mStrVocabularyFilePath.substr(found+1);

        if(type == TEXT_FILE) // File text
        {
            cout << "Starting to write the save text file " << endl;
            std::remove(pathSaveFileName.c_str());
            std::ofstream ofs(pathSaveFileName, std::ios::binary);
            boost::archive::text_oarchive oa(ofs);

            oa << strVocabularyName;
            oa << strVocabularyChecksum;
            oa << mpAtlas;
            oa << mpKeyFrameDatabase;
            cout << "End to write the save text file" << endl;
        }
        else if(type == BINARY_FILE) // File binary
        {
            cout << "Starting to write the save binary file" << endl;
            std::remove(pathSaveFileName.c_str());
            std::ofstream ofs(pathSaveFileName, std::ios::binary);
            boost::archive::binary_oarchive oa(ofs);
            oa << strVocabularyName;
            oa << strVocabularyChecksum;
            oa << mpAtlas;
            oa << mpKeyFrameDatabase;
            cout << "End to write save binary file" << endl;
        }

        //clock_t timeElapsed = clock() - start;
        //unsigned msElapsed = timeElapsed / (CLOCKS_PER_SEC / 1000);
        //cout << "Binary file saved in " << msElapsed << " ms" << endl;
    }
}

bool System::LoadAtlas(string filename, int type)
{
    string strFileVoc, strVocChecksum;
    bool isRead = false;

    if(type == TEXT_FILE) // File text
    {
        cout << "Starting to read the save text file " << endl;
        std::ifstream ifs(filename, std::ios::binary);
        if(!ifs.good())
        {
            cout << "Load file not found" << endl;
            return false;
        }
        boost::archive::text_iarchive ia(ifs);
        ia >> strFileVoc;
        ia >> strVocChecksum;
        ia >> mpAtlas;
        //ia >> mpKeyFrameDatabase;
        cout << "End to load the save text file " << endl;
        isRead = true;
    }
    else if(type == BINARY_FILE) // File binary
    {
        cout << "Starting to read the save binary file"  << endl;
        std::ifstream ifs(filename, std::ios::binary);
        if(!ifs.good())
        {
            cout << "Load file not found" << endl;
            return false;
        }
        boost::archive::binary_iarchive ia(ifs);
        ia >> strFileVoc;
        ia >> strVocChecksum;
        ia >> mpAtlas;
        //ia >> mpKeyFrameDatabase;
        cout << "End to load the save binary file" << endl;
        isRead = true;
    }

    if(isRead)
    {
        //Check if the vocabulary is the same
        string strInputVocabularyChecksum = CalculateCheckSum(mStrVocabularyFilePath,TEXT_FILE);

        if(strInputVocabularyChecksum.compare(strVocChecksum) != 0)
        {
            cout << "The vocabulary load isn't the same which the load session was created " << endl;
            cout << "-Vocabulary name: " << strFileVoc << endl;
            return false; // Both are differents
        }

        return true;
    }
    return false;
}

string System::CalculateCheckSum(string filename, int type)
{
    string checksum = "";

    unsigned char c[MD5_DIGEST_LENGTH];

    std::ios_base::openmode flags = std::ios::in;
    if(type == BINARY_FILE) // Binary file
        flags = std::ios::in | std::ios::binary;

    ifstream f(filename.c_str(), flags);
    if ( !f.is_open() )
    {
        cout << "[E] Unable to open the in file " << filename << " for Md5 hash." << endl;
        return checksum;
    }

    MD5_CTX md5Context;
    char buffer[1024];

    MD5_Init (&md5Context);
    while ( int count = f.readsome(buffer, sizeof(buffer)))
    {
        MD5_Update(&md5Context, buffer, count);
    }

    f.close();

    MD5_Final(c, &md5Context );

    for(int i = 0; i < MD5_DIGEST_LENGTH; i++)
    {
        char aux[10];
        sprintf(aux,"%02x", c[i]);
        checksum = checksum + aux;
    }

    return checksum;
}*/

} //namespace ORB_SLAM


