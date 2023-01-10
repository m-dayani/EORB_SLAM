//
// Created by root on 1/17/21.
//

#ifndef ORB_SLAM3_EVBASETRACKER_H
#define ORB_SLAM3_EVBASETRACKER_H

#include "compiler_options.h"

#include <chrono>

#include "MyDataTypes.h"
#include "EventData.h"
#include "KLT_Tracker.h"
#include "EventFrame.h"
#include "IMU_Manager.h"

#include "GeometricCamera.h"
#include "ORBextractor.h"
#include "Converter.h"
#include "MyFrameDrawer.h"


namespace EORB_SLAM {

#ifndef DEF_MATCHES_WIN_NAME
#define DEF_MATCHES_WIN_NAME "Matched Pairs"
#endif


    class EvBaseTracker {
    public:

        EvBaseTracker();
        EvBaseTracker(EvParamsPtr evParams, CamParamsPtr camParams);
        virtual ~EvBaseTracker();

        void setSeparateThread(bool flag) { this->sepThread.set(flag); }

        void fillEvents(const std::vector<EventData>& evs);
        void fillImage(const PoseImagePtr& pPoseImage);
        void fillIMU(const std::vector<ImuData>& imuMeas);

        virtual bool isTrackerReady() = 0;
        virtual bool isTrackerSaturated() = 0;

        virtual void Track() = 0;

        virtual bool isStopped() = 0;
        virtual void stop() = 0;
        virtual void resetAll();

        virtual void makeFrame(const PoseImagePtr& pImage, unsigned fId, EvFramePtr& frame);

        virtual bool isInputGood() = 0;

        virtual bool isProcessing() { return mbIsProcessing == true; }

        //void setImuManager(const std::shared_ptr<IMU_Manager>& pImuManager) { mpImuManager = pImuManager; }
        virtual void setImuManagerAndChannel(const std::shared_ptr<IMU_Manager>& pImuManager);
        uint getImuId() const { return mnImuID; }

        static void copyConnectedEvFrames(const std::vector<EvFramePtr>& vpInFrames, std::vector<EvFramePtr>& vpOutFrames);

        void setFrameDrawerChannel(MyFrameDrawer* pFrDrawer);
        //virtual void updateFrameDrawer();

        //virtual float getAverageTrackingTime() { return 0.0; }
        virtual std::string getTrackingTimeStat() { return mTrackingTimer.getCommentedTimeStat(); }

    protected:

        bool isImageBuffGood();
        bool isImageBuffSaturated(std::size_t maxSize);
        PoseImagePtr frontImage();
        void popImage();

        static bool isReconstDepthGood(const std::vector<std::vector<cv::Point3f>>& pts3d, std::vector<int>& vMatches12);

        virtual void initFeatureExtractors(int ftDtMode, const ORB_SLAM3::ORBxParams& parORB);

        virtual void makeFrame(const PoseImagePtr& pImage, unsigned fId, std::unique_ptr<ELK_Tracker>& pLKTracker,
                               EvFramePtr& frame) = 0;
        virtual void makeFrame(const PoseImagePtr& pImage, unsigned fId, const std::vector<cv::KeyPoint>& p1,
                               EvFramePtr& frame) = 0;
        virtual unsigned trackAndFrame(const PoseImagePtr& pImage, unsigned fId,
                                       std::unique_ptr<ELK_Tracker>& pLKTracker, EvFramePtr& evFrame) = 0;

        static void getValidInitPts3D(const std::vector<EvFramePtr>& vpFrames, std::vector<int>& vValidPts3D);

        //virtual void reset() = 0;

        // For multi-threaded operation,
        // we need thread-safe state vars & input/output buffers

        // States
        SharedFlag sepThread;   // Higher authority
        SharedFlag mbIsProcessing;
        unsigned mCurrIdx;

        //std::mutex mMtxImBuff;

        // Inputs
        SharedQueue<std::vector<EventData>> mqEvChunks;
        SharedQueue<PoseImagePtr> mqImageBuff;
        SharedQueue<std::vector<ImuData>> mqImuMeas;

        // Outputs
        std::vector<int> mvMatchesCnt;
        std::mutex mMtxPts3D;
        std::vector<cv::Point3f> mvPts3D;

        // Utils

        // Params
        // Since we access this across multiple threads, it must be thread-safe
        // Better to copy the parameters that we need
        MySharedPtr<EvParams> mpEvParams;
        MySharedPtr<MyCamParams> mpCamParams;

        const bool mbContTracking;
        const bool isEvsRectified;
        int mnEvPts;
        const int mThFAST;
        int mFtDtMode;

        const int imWidth;
        const int imHeight;
        float mImSigma;

        cv::Mat mDistCoefs;
        cv::Mat mK;

        const std::string mMchWindow;
        unsigned int mIdxImSave;

        const std::string mRootImPath;

        uint mnImuID;
        std::shared_ptr<IMU_Manager> mpImuManager;

        //cv::Ptr<cv::FastFeatureDetector> mpFastDetector;
        std::shared_ptr<ORB_SLAM3::ORBextractor> mpORBDetector;

        ORB_SLAM3::GeometricCamera* mpCamera;

        int mFrDrawerId;
        MyFrameDrawer* mpFrameDrawer;

        MySmartTimer mTrackingTimer;
        MySmartWatchDog mTrackingWD;

    private:

//        ORB_SLAM3::GeometricCamera* mpCamera;


    };

} // EORB_SLAM

#endif //ORB_SLAM3_EVBASETRACKER_H
