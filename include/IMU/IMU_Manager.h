//
// Created by root on 7/28/21.
//

#ifndef ORB_SLAM3_IMU_MANAGER_H
#define ORB_SLAM3_IMU_MANAGER_H

#include <iostream>
#include <thread>
#include <memory>

#include "MyParameters.h"

#include "ImuTypes.h"
#include "Frame.h"
#include "KeyFrame.h"
#include "Atlas.h"


namespace EORB_SLAM {

    class EvAsynchTracker;

    class InitInfoIMU {
    public:

        InitInfoIMU();
        InitInfoIMU(const ORB_SLAM3::IMU::Bias& bIMU, const cv::Mat& Rwg0, double scale);

        void setLastImuBias(const ORB_SLAM3::IMU::Bias& bIMU) { mLastBias = bIMU; }
        void setLastImuBias(const Eigen::Vector3d& mbg, const Eigen::Vector3d& mba) { mLastBias = toImuBias(mbg, mba); }
        ORB_SLAM3::IMU::Bias getLastImuBias() const { return mLastBias; }

        void setLastScale(const double scale) { mScale = scale; }
        double getLastScale() const { return mScale; }

        void initRwg(const cv::Mat& firstAcc);
        cv::Mat getIniRwg() const { return mRwg0.clone(); }
        cv::Mat getLastRwg() const { return mLastRwg.clone(); }
        void setIniRwg(const cv::Mat& Rwg0) { mRwg0 = Rwg0.clone(); }
        void setLastRwg(const cv::Mat& Rwg) { mLastRwg = Rwg.clone(); }


        Eigen::MatrixXd getInfoInertial() const { return mInfoInertial; }
        void setInfoInertial(const Eigen::MatrixXd& infoInertial) { mInfoInertial = infoInertial; }

        void updateState(const std::shared_ptr<InitInfoIMU>& pImuInfo);

        static ORB_SLAM3::IMU::Bias toImuBias(const Eigen::Vector3d& mbg, const Eigen::Vector3d& mba);
        static void toImuBias(const ORB_SLAM3::IMU::Bias& bIMU, Eigen::Vector3d& mbg, Eigen::Vector3d& mba);

        // Last Bias Estimation (at keyframe creation)
        ORB_SLAM3::IMU::Bias mLastBias;

        cv::Mat mLastRwg, mRwg0;

        double mScale;
        Eigen::MatrixXd mInfoInertial;
        float mInfoA, mInfoG, mInfoRwg, mInfoS;
    };

    typedef std::shared_ptr<InitInfoIMU> InitInfoImuPtr;

    /* ============================================================================================================== */

    class IMU_Manager {
    public:
        explicit IMU_Manager(const IMUParamsPtr& pIMUParams);
        ~IMU_Manager();

        uint setNewIntegrationChannel();

        void refreshPreintFromLastKF(uint preIntId);

        void grabImuData(const ORB_SLAM3::IMU::Point &imuMeasurement);

        void preintegrateIMU(uint preIntId, const EvFramePtr& mpCurrentFrame,
                             const ORB_SLAM3::IMU::Bias& lastBias);

        bool isInitializing();

        void abortInit(bool state) { mbAbortInit = state; }

        void initializeIMU(EvAsynchTracker* pTracker, ORB_SLAM3::Atlas* pAtlas, ORB_SLAM3::KeyFrame* pCurrKF,
                           double& mFirstTs, double& mTinit, bool bMonocular, float priorG = 1e2, float priorA = 1e6,
                           bool bFirst = false);
        void scaleRefinement(EvAsynchTracker* pTracker, ORB_SLAM3::Atlas* pAtlas, ORB_SLAM3::KeyFrame* pCurrKF, bool bMonocular);

        bool predictStateIMU(uint intId, ORB_SLAM3::KeyFrame* pLastKeyFrame, ORB_SLAM3::Frame* pCurrentFrame, bool bMapUpdated = true);
        static bool predictStateIMU(ORB_SLAM3::Frame* pLastFrame, ORB_SLAM3::Frame* pCurrentFrame, bool bMapUpdated = false);
        //void ResetFrameIMU();
        //void UpdateFrameIMU(const float s, const IMU::Bias &b, KeyFrame* pCurrentKeyFrame);

        ORB_SLAM3::IMU::Calib *getImuCalib() { return mpImuCalib; }

        ORB_SLAM3::IMU::Point getFirstImu(uint id);
        ORB_SLAM3::IMU::Point getLastImu(uint id);

        ORB_SLAM3::IMU::Preintegrated * getImuPreintegratedFromLastKF(uint id);

        InitInfoImuPtr getInitInfoIMU();
        void updateState(const InitInfoImuPtr& pImuInfo);
        void updateLastImuBias(const ORB_SLAM3::IMU::Bias& bIMU);

        ORB_SLAM3::IMU::Bias getLastImuBias();
        cv::Mat getIniRwg();
        cv::Mat getLastRwg();

        void reset();
        void resetAll();

    private:

        // IMU I/O

        // Queue of IMU measurements between frames
        std::vector<std::list<ORB_SLAM3::IMU::Point>> mvlQueueImuData;

        // Imu preintegration from last frame
        std::vector<ORB_SLAM3::IMU::Preintegrated *> mvpImuPreintegratedFromLastKF;

        // Vector of IMU measurements from previous to current frame (to be filled by preintegrateIMU)
        std::vector<ORB_SLAM3::IMU::Point> mvImuFromLastFrame;
        std::mutex mMutexImuQueue;

        // IMU State

        SharedFlag mbInitializing;

        SharedFlag mbAbortInit;

        // Imu calibration parameters
        ORB_SLAM3::IMU::Calib *mpImuCalib;

        InitInfoImuPtr mpInitInfoIMU;
        std::mutex mMtxInfoIMU;

        //int mnFirstImuFrameId;
        //int mnFramesToResetIMU;
    };


}


#endif //ORB_SLAM3_IMU_MANAGER_H
