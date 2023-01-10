//
// Created by root on 12/4/20.
//

#ifndef ORB_SLAM3_MYPARAMETERS_H
#define ORB_SLAM3_MYPARAMETERS_H

#include <string>
#include <iostream>
#include <memory>
//#include <bits/refwrap.h>

#include <opencv2/core/core.hpp>

#include <glog/logging.h>

#include "GeometricCamera.h"

//using namespace std;

namespace EORB_SLAM {

#define DEF_FT_IMG_MARGIN 30

    struct MyCamParams {

        ~MyCamParams() { delete mLinkedCam; }

        bool leftCam = true;

        float fx;
        float fy;
        float cx;
        float cy;

        float k1, k2, k3 = 0.f, k4 = 0.f;
        float p1, p2, p3 = 0.f, p4 = 0.f;

        cv::Mat mK;
        cv::Mat mDistCoef;
        cv::Mat mR; // Rectification matrix
        cv::Mat mP; // Projection matrix

        int mImWidth, mImHeight;

        int mbRGB = 0;

        float mbf = 0.f;

        int leftLappingBegin = -1, leftLappingEnd = -1;
        int rightLappingBegin = -1, rightLappingEnd = -1;

        cv::Mat mTlr{};

        float fps = 30.0f;
        float mMinFrames = 0;
        float mMaxFrames = 30.0f;

        float mThDepth = 0.f;
        float mDepthMapFactor;
        float mThFarPoints;

        std::string camType;
        bool hasK3 = false;
        bool missParams = false;

        MyCamParams* mLinkedCam = nullptr;

        bool isFisheye() const;
        bool isPinhole() const;
        void initK();
        void initDistCoefs();

        std::string printPinhole() const;
        std::string printFisheye(const std::string& camName = "Camera") const;
        std::string printStereo() const;
        std::string print() const;
    };

    struct MixedFtsParams {

        int detMode = 0;    // ORB

        int imMargin = DEF_FT_IMG_MARGIN;

        int nFeatures = 1000, nLevels = 8, initThFast = 20, minThFast = 10;
        float scaleFactor = 1.2;

        int nFeaturesAK = 1000, nOctaves = 2, nOctaveLayers = 10;
        float iniThAK = 1e-3, minThAK = 1e-6;

        bool missParams = false;

        std::string printParams() const;
    };

    struct MyIMUSettings {

        float sf;
        float freq, Ng, Na, Ngw, Naw;
        cv::Mat Tbc;

        bool missParams = false;

        std::string printParams() const;
    };

    struct MyViewerSettings {

        bool mbUseViewer = false;

        // 1/fps in ms
        double mT;
        float mImageWidth, mImageHeight;
        float mViewpointX, mViewpointY, mViewpointZ, mViewpointF;

        float mKeyFrameSize;
        float mKeyFrameLineWidth;
        float mGraphLineWidth;
        float mPointSize;
        float mCameraSize;
        float mCameraLineWidth;

        bool missParams = false;

        std::string printParams() const;
    };

    typedef std::shared_ptr<MyCamParams> CamParamsPtr;
    typedef std::shared_ptr<MixedFtsParams> MixedFtsParamsPtr;
    typedef std::shared_ptr<MyIMUSettings> IMUParamsPtr;
    typedef std::shared_ptr<MyViewerSettings> ViewerParamsPtr;

    class MyParameters {
    public:
        static bool parseCameraParams(const cv::FileStorage &fSettings,
                MyCamParams &camParams, bool bIsStereo = false, bool bIsRgbd = false);

        static ORB_SLAM3::GeometricCamera* dummyCamera;
        static void createCameraObject(const MyCamParams& camParams, ORB_SLAM3::GeometricCamera*& pCamera,
                ORB_SLAM3::GeometricCamera*& pCamera2 = dummyCamera, bool isCalibrated = false);
        static void createCameraObject(const MyCamParams& camParams, ORB_SLAM3::TwoViewReconstruction* _tvr,
                ORB_SLAM3::GeometricCamera*& pCamera, ORB_SLAM3::GeometricCamera*& pCamera2 = dummyCamera,
                bool isCalibrated = false);

        static bool parseFeatureParams(const cv::FileStorage &fSettings, MixedFtsParams &orbSettings);
        static bool parseIMUParams(const cv::FileStorage &fSettings, MyIMUSettings &imuSettings);
        static bool parseViewerParams(const cv::FileStorage &fSettings, MyViewerSettings &viewerSettings);

    protected:
        static bool parseCamIntrinsics(const cv::FileStorage &fSettings, MyCamParams &camParams,
                const std::string& camName = "Camera", std::vector<std::string> paramNames = std::vector<std::string>());
        static bool parsePinholeDist(const cv::FileStorage &fSettings, MyCamParams &camParams,
                const std::string& camName = "Camera", std::vector<std::string> paramNames = std::vector<std::string>());
        static bool parseFisheyeDist(const cv::FileStorage &fSettings, MyCamParams &camParams,
                const std::string& camName = "Camera", std::vector<std::string> paramNames = std::vector<std::string>());
        static bool parseImageSize(const cv::FileStorage &fSettings, MyCamParams &camParams,
                                   const std::string& camName = "Camera");
        static void parseCamRectProjMats(const cv::FileStorage &fSettings, MyCamParams &camParams,
                                         const std::string& camName = "Camera");
        static void parseFisheyeLapping(const cv::FileStorage &fSettings, MyCamParams &camParams,
                                const std::string& camName = "Camera", bool left = true);
        static void getDefIntrinsicsNames(std::vector<std::string> &out);
        static void getDefPinholeDistNames(std::vector<std::string> &out);
        static void getDefFisheyeDistNames(std::vector<std::string> &out);

        static void copyIntrinsics(const MyCamParams &srcCam, MyCamParams &dstCam);
        static void copyDistortionParams(const MyCamParams &srcCam, MyCamParams &dstCam);
    };

    /*void createCameraObject(const MyCamParams& camParams, ORB_SLAM3::GeometricCamera* pCamera,
                            ORB_SLAM3::GeometricCamera* pCamera2 = nullptr);*/

} // EORB_SLAM

#endif //ORB_SLAM3_MYPARAMETERS_H
