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


#ifndef SYSTEM_H
#define SYSTEM_H

//#define SAVE_TIMES

#include <unistd.h>
#include<cstdio>
#include<cstdlib>
#include<string>
#include<thread>
#include<opencv2/core/core.hpp>

#include "Tracking.h"
#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "Atlas.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "KeyFrameDatabase.h"
#include "ORBVocabulary.h"
#include "Viewer.h"
#include "include/IMU/ImuTypes.h"
#include "GeometricCamera.h"

#include "EventData.h"
#include "MyDataTypes.h"
#include "MyParameters.h"
#include "EvTrackManager.h"
#include "MyFrameDrawer.h"

namespace ORB_SLAM3
{

class Verbose
{
public:
    enum eLevel
    {
        VERBOSITY_NONE=-1,
        VERBOSITY_QUIET=0,
        VERBOSITY_NORMAL=1,
        VERBOSITY_VERBOSE=2,
        VERBOSITY_VERY_VERBOSE=3,
        VERBOSITY_DEBUG=4
    };

    static eLevel th;

public:
    static void PrintMess(std::string str, eLevel lev)
    {
        if(lev <= th)
            cout << str << endl;
    }

    static void SetTh(eLevel _th)
    {
        th = _th;
    }
};

class Viewer;
class FrameDrawer;
class Atlas;
class Tracking;
class LocalMapping;
class LoopClosing;

class System
{
public:
    // Input sensor
    /*enum eSensor{
        MONOCULAR=0,
        STEREO=1,
        RGBD=2,
        IMU_MONOCULAR=3,
        IMU_STEREO=4,
        EVENT_ONLY,
        EVENT_MONO,
        EVENT_IMU,
        EVENT_IMU_MONO,
        IDLE
    };*/

    // File type
    enum eFileType{
        TEXT_FILE=0,
        BINARY_FILE=1,
    };

public:

    // Initialize the SLAM system. It launches the Local Mapping, Loop Closing and Viewer threads.
    System(const string &strVocFile, EORB_SLAM::SensorConfigPtr  pSensor,
           const EORB_SLAM::CamParamsPtr& pCamParams, const EORB_SLAM::PairCalibPtr& pCalib,
           const EORB_SLAM::MixedFtsParamsPtr& pORBParams, const EORB_SLAM::IMUParamsPtr& pIMUParams,
           const EORB_SLAM::ViewerParamsPtr& pViewerParams, const int initFr = 0,
           const string &strSequence = std::string(), const string &strLoadingFile = std::string());

    // Event Constructor
    System(const string &strVocFile, const EORB_SLAM::SensorConfigPtr& pSensor,
           const EORB_SLAM::CamParamsPtr& pCamParams, const EORB_SLAM::PairCalibPtr& pCalib,
           const EORB_SLAM::MixedFtsParamsPtr& pORBParams, const EORB_SLAM::IMUParamsPtr& pIMUParams,
           const EORB_SLAM::ViewerParamsPtr& pViewerParams, const EORB_SLAM::EvParamsPtr& pEvParams,
           const int initFr = 0, const string &strSequence = std::string(),
           const string &strLoadingFile = std::string());

    ~System();

    // Proccess the given stereo frame. Images must be synchronized and rectified.
    // Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Returns the camera pose (empty if tracking fails).
    cv::Mat TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp,
            const vector<IMU::Point>& vImuMeas = vector<IMU::Point>(), string filename="");

    // Process the given rgbd frame. Depthmap must be registered to the RGB frame.
    // Input image: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Input depthmap: Float (CV_32F).
    // Returns the camera pose (empty if tracking fails).
    cv::Mat TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp, string filename="");

    // Proccess the given monocular frame and optionally imu data
    // Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Returns the camera pose (empty if tracking fails).
    cv::Mat TrackMonocular(const cv::Mat &im, const double &timestamp,
            const vector<IMU::Point>& vImuMeas = vector<IMU::Point>(), string filename="");


    // This stops local mapping thread (map building) and performs only camera tracking.
    void ActivateLocalizationMode();
    // This resumes local mapping thread and performs SLAM again.
    void DeactivateLocalizationMode();

    // Returns true if there have been a big map change (loop closure, global BA)
    // since last call to this function
    bool MapChanged();

    // Reset the system (clear Atlas or the active map)
    void Reset();
    void ResetActiveMap();

    // All threads will be requested to finish.
    // It waits until all threads have finished.
    // This function must be called before saving the trajectory.
    void Shutdown();

    // Save camera trajectory in the TUM RGB-D dataset format.
    // Only for stereo and RGB-D. This method does not work for monocular.
    // Call first Shutdown()
    // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
    void SaveTrajectoryTUM(const string &filename);

    // Save keyframe poses in the TUM RGB-D dataset format.
    // This method works for all sensor input.
    // Call first Shutdown()
    // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
    void SaveKeyFrameTrajectoryTUM(const string &filename);

    void SaveTrajectoryEuRoC(const string &filename, const double tsc = 1e9, const int ts_prec = 6);
    void SaveKeyFrameTrajectoryEuRoC(const string &filename, const double tsc = 1e9, const int ts_prec = 6);

    // Save data used for initialization debug
    void SaveDebugData(const int &iniIdx);

    // Save camera trajectory in the KITTI dataset format.
    // Only for stereo and RGB-D. This method does not work for monocular.
    // Call first Shutdown()
    // See format details at: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
    void SaveTrajectoryKITTI(const string &filename);

    // TODO: Save/Load functions
    // SaveMap(const string &filename);
    // LoadMap(const string &filename);

    // Information from most recent processed frame
    // You can call this right after TrackMonocular (or stereo or RGBD)
    int GetTrackingState();
    std::vector<MapPoint*> GetTrackedMapPoints();
    std::vector<cv::KeyPoint> GetTrackedKeyPointsUn();

    // For debugging
    double GetTimeFromIMUInit();
    bool isLost();
    bool isFinished();

    void ChangeDataset();

    //void SaveAtlas(int type);

    // Masoud's contribution

    bool IsReady();

    void insertGlobalKeyFrame(KeyFrame* pKf);

    // Event SLAM (Tracking)
    cv::Mat TrackEvent(const vector<EORB_SLAM::EventData>& evs,
            const vector<IMU::Point>& vImuMeas = vector<IMU::Point>(), const string& filename="");
    cv::Mat TrackEvMono(const vector<EORB_SLAM::EventData>& evs, const cv::Mat &im, const double &timestamp,
            const vector<IMU::Point>& vImuMeas = vector<IMU::Point>(), string filename="");

    void SaveTrajectoryEvent(const std::string& fileName, const double tsc = 1.0, const int ts_prec = 9);
    void SaveKeyFrameTrajectoryEvent(const string &fileName, const double tsc = 1.0, const int ts_prec = 9);

    void SaveTrajectoryEvIm(const std::string& fileName, const double tsc = 1.0, const int ts_prec = 9);

    void FuseEventORB();

    void GetEventConstraints(std::vector<KeyFrame*>& vpEvKFs);

    //Atlas* GetNewAtlas();

    void PrepareNextIeration();

    void ConsumeEvents();

    unsigned GetL1EvWinSize();

    void EventSynchPreProcessing(FramePtr& pFrame);
    void EventSynchPostProcessing();

    void SetInitEvFrameSynch(FramePtr& pFrame);
    int TrackAndOptEvFrameSynch(FramePtr& pFrame);
    void TrackEvFrameSynch(FramePtr& pFrame);

    //void SetRefEvKeyFrameSynch(KeyFrame* pKFini);
    void TrackEvKeyFrameSynch(KeyFrame* pKFcur);
    //void EventImageInitOptimization(Map* pMap, int nIterations = 20);
    //void EventImageInitOptimization(Map* pMap, KeyFrame* pKFini, KeyFrame* pKFcur, int nIterations = 20);
    int EventLocalMappingSynch(ORB_SLAM3::KeyFrame* pEvKFcur);

    bool ResolveEventMapInit(FramePtr& pFrIni, KeyFrame* pKFini, FramePtr& pFrCurr, KeyFrame* pKFcur, int nIteration);

    bool EvImReconst2ViewsSynch(const FramePtr& pFrame1, FramePtr& pFrame2,
            const std::vector<int> &vMatches12, cv::Mat &R21, cv::Mat &t21,
            std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated);

    void ApplyScaleAndRotationEvSynch(const cv::Mat& R, float scale, bool bScaleVel, const cv::Mat& t);
    void UpdateEvFrameImuSynch(float s, const IMU::Bias& b, KeyFrame* pOrbKFcur);

    void SaveMap(const std::string& fileName, int stat = 0);

private:

    //bool LoadAtlas(string filename, int type);

    //string CalculateCheckSum(string filename, int type);

    // ORB vocabulary used for place recognition and feature matching.
    ORBVocabulary* mpVocabulary;

    // KeyFrame database for place recognition (relocalization and loop detection).
    KeyFrameDatabase* mpKeyFrameDatabase;

    // Map structure that stores the pointers to all KeyFrames and MapPoints.
    //Map* mpMap;
    Atlas* mpAtlas;

    // Tracker. It receives a frame and computes the associated camera pose.
    // It also decides when to insert a new keyframe, create some new MapPoints and
    // performs relocalization if tracking fails.
    Tracking* mpTracker;

    // Local Mapper. It manages the local map and performs local bundle adjustment.
    LocalMapping* mpLocalMapper;

    // Loop Closer. It searches loops with every new keyframe. If there is a loop it performs
    // a pose graph optimization and full bundle adjustment (in a new thread) afterwards.
    LoopClosing* mpLoopCloser;

    // The viewer draws the map and the current camera pose. It uses Pangolin.
    Viewer* mpViewer;

    EORB_SLAM::MyFrameDrawer* mpFrameDrawer;
    MapDrawer* mpMapDrawer;

    // System threads: Local Mapping, Loop Closing, Viewer.
    // The Tracking thread "lives" in the main execution thread that creates the System object.
    std::thread* mptLocalMapping;
    std::thread* mptLoopClosing;
    std::thread* mptViewer;

    // Reset flag
    std::mutex mMutexReset;
    bool mbReset;
    bool mbResetActiveMap;

    // Change mode flags
    std::mutex mMutexMode;
    bool mbActivateLocalizationMode;
    bool mbDeactivateLocalizationMode;

    // Tracking state
    int mTrackingState{};
    std::vector<MapPoint*> mTrackedMapPoints;
    std::vector<cv::KeyPoint> mTrackedKeyPointsUn;
    std::mutex mMutexState;

    // Masoud's contribution
    // Input sensor
    const EORB_SLAM::SensorConfigPtr mpSensor;
    // Event Track Manager
    unique_ptr<EORB_SLAM::EvTrackManager> mpEvTracker;
    unique_ptr<std::thread> mptEvTracking;

    // Camera Objects
    EORB_SLAM::CamParamsPtr mpCamParams;
    //GeometricCamera *mpCamera, *mpCamera2;

    // Global Pose Chain Container
    std::mutex mMutexGPCh;
    std::map<double, KeyFrame*> mGlobalPoseChain;
    Atlas* mpGlobalAtlas;
};

}// namespace ORB_SLAM

#endif // SYSTEM_H
