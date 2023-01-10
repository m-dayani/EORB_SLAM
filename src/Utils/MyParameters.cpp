//
// Created by root on 12/4/20.
//

#include "MyParameters.h"

#include "Pinhole.h"
#include "KannalaBrandt8.h"

using namespace std;


namespace EORB_SLAM {

    bool MyCamParams::isFisheye() const {
        return camType == "fisheye";
    }

    bool MyCamParams::isPinhole() const {
        return camType == "pinhole";
    }

    void MyCamParams::initK() {

        mK = cv::Mat::eye(3,3,CV_32F);
        mK.at<float>(0,0) = fx;
        mK.at<float>(1,1) = fy;
        mK.at<float>(0,2) = cx;
        mK.at<float>(1,2) = cy;
    }

    void MyCamParams::initDistCoefs() {

        mDistCoef = cv::Mat::zeros(4,1,CV_32F);
        if (isFisheye()) {
            mDistCoef.at<float>(0) = k1;
            mDistCoef.at<float>(1) = k2;
            mDistCoef.at<float>(2) = k3;
            mDistCoef.at<float>(3) = k4;
        }
        else {
            mDistCoef.at<float>(0) = k1;
            mDistCoef.at<float>(1) = k2;
            mDistCoef.at<float>(2) = p1;
            mDistCoef.at<float>(3) = p2;
            if (hasK3) {
                mDistCoef.resize(5);
                mDistCoef.at<float>(4) = k3;
            }
        }
    }

    string MyCamParams::printPinhole() const {

        ostringstream oss;

        oss << "- Camera: Pinhole" << std::endl;
        oss << "- fx: " << fx << std::endl;
        oss << "- fy: " << fy << std::endl;
        oss << "- cx: " << cx << std::endl;
        oss << "- cy: " << cy << std::endl;
        oss << "- k1: " << k1 << std::endl;
        oss << "- k2: " << k2 << std::endl;


        oss << "- p1: " << p1 << std::endl;
        oss << "- p2: " << p2 << std::endl;

        oss << "- k3: " << k3 << std::endl;

        return oss.str();
    }

    string MyCamParams::printFisheye(const string& camName) const {

        ostringstream oss;

        oss << "- " << camName << ": Fisheye" << std::endl;
        oss << "- fx: " << fx << std::endl;
        oss << "- fy: " << fy << std::endl;
        oss << "- cx: " << cx << std::endl;
        oss << "- cy: " << cy << std::endl;
        oss << "- k1: " << k1 << std::endl;
        oss << "- k2: " << k2 << std::endl;
        oss << "- k3: " << k3 << std::endl;
        oss << "- k4: " << k4 << std::endl;

        return oss.str();
    }

    string MyCamParams::printStereo() const {

        ostringstream oss;

        oss << "- Camera: Stereo Configuration" << std::endl;

        oss << printFisheye("Camera1");
        oss << "- Camera1 Lapping: " << leftLappingBegin << ", " << leftLappingEnd << std::endl;

        if (mLinkedCam) {
            oss << mLinkedCam->printFisheye("Camera2");
        }
        oss << "- Camera2 Lapping: " << rightLappingBegin << ", " << rightLappingEnd << std::endl;

        oss << "- mTlr: \n" << mTlr << std::endl;

        return oss.str();
    }

    string MyCamParams::print() const {

        ostringstream oss;

        if (this->isPinhole()) {
            oss << this->printPinhole();
        }
        else if (this->isFisheye()) {
            oss << this->printFisheye();
        }
        else {
            oss << "* Camera model is not supported.\n";
        }

        oss << "- Frame Rate: " << fps << " (fps)\n";
        oss << "- RGB Color order? " << ((mbRGB) ? "Yes" : "No") << endl;

        return oss.str();
    }

    string MixedFtsParams::printParams() const {

        ostringstream oss;

        oss << "-- Feature Extraction Mode (Regular Images): ";
        if (detMode == 1) {
            oss << "AKAZE";
        }
        else if (detMode == 2) {
            oss << "MIXED";
        }
        else {
            oss << "ORB";
        }
        oss << endl << "-- Image Margin for Feature Detection: " << imMargin << endl;
        oss << "-- ORB Extractor Parameters: " << endl;
        oss << "- Number of Features: " << nFeatures << endl;
        oss << "- Scale Levels: " << nLevels << endl;
        oss << "- Scale Factor: " << scaleFactor << endl;
        oss << "- Initial Fast Threshold: " << initThFast << endl;
        oss << "- Minimum Fast Threshold: " << minThFast << endl;

        oss << endl << "-- AKAZE Extractor Parameters: " << endl;
        oss << "- Number of Features: " << nFeaturesAK << endl;
        oss << "- Number of Octaves: " << nOctaves << endl;
        oss << "- Number of Octave Layers: " << nOctaveLayers << endl;
        oss << "- Initial Threshold: " << iniThAK << endl;
        oss << "- Minimum Threshold: " << minThAK << endl;

        return oss.str();
    }

    string MyIMUSettings::printParams() const {

        ostringstream oss;

        oss << "- Left camera to Imu Transform (Tbc): " << endl << Tbc << endl;
        oss << "- IMU frequency: " << freq << " Hz" << endl;
        oss << "- IMU gyro noise: " << Ng << " rad/s/sqrt(Hz)" << endl;
        oss << "- IMU gyro walk: " << Ngw << " rad/s^2/sqrt(Hz)" << endl;
        oss << "- IMU accelerometer noise: " << Na << " m/s^2/sqrt(Hz)" << endl;
        oss << "- IMU accelerometer walk: " << Naw << " m/s^3/sqrt(Hz)" << endl;

        return oss.str();
    }

    string MyViewerSettings::printParams() const {

        ostringstream oss;

        oss << "-- Viewer: \n";
        oss << "- Frame time: " << mT << endl;
        oss << "- View point X: " << mViewpointX << endl;
        oss << "- View point Y: " << mViewpointY << endl;
        oss << "- View point Z: " << mViewpointZ << endl;
        oss << "- View point F: " << mViewpointF << endl;

        oss << "- Viewer image width: " << mImageWidth << endl;
        oss << "- Viewer image height: " << mImageHeight << endl;

        oss << "-- Map-Drawer: \n";
        oss << "- Key frame size: " << mKeyFrameSize << endl;
        oss << "- Key frame line width: " << mKeyFrameLineWidth << endl;
        oss << "- Graph line width: " << mGraphLineWidth << endl;
        oss << "- Point size: " << mPointSize << endl;
        oss << "- Camera size: " << mCameraSize << endl;
        oss << "- Camera line width: " << mCameraLineWidth << endl;

        return oss.str();
    }

    bool MyParameters::parseCamIntrinsics(const cv::FileStorage &fSettings, MyCamParams &camParams, const string& camName,
                                          vector<string> paramNames) {
        bool b_miss_params = false;
        if (paramNames.empty())
            MyParameters::getDefIntrinsicsNames(paramNames);

        string strParam = camName+'.'+paramNames[0];
        cv::FileNode node = fSettings[strParam];
        if(!node.empty() && node.isReal())
        {
            camParams.fx = node.real();
        }
        else
        {
            std::cerr << "*" << strParam << " parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        strParam = camName+'.'+paramNames[1];
        node = fSettings[strParam];
        if(!node.empty() && node.isReal())
        {
            camParams.fy = node.real();
        }
        else
        {
            std::cerr << "*" << strParam <<" parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        strParam = camName+'.'+paramNames[2];
        node = fSettings[strParam];
        if(!node.empty() && node.isReal())
        {
            camParams.cx = node.real();
        }
        else
        {
            std::cerr << "*" << strParam << " parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        strParam = camName+'.'+paramNames[3];
        node = fSettings[strParam];
        if(!node.empty() && node.isReal())
        {
            camParams.cy = node.real();
        }
        else
        {
            std::cerr << "*" << strParam << " parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        return !b_miss_params;
    }

    bool MyParameters::parsePinholeDist(const cv::FileStorage &fSettings, MyCamParams &camParams, const string& camName,
                                        vector<string> paramNames) {

        bool b_miss_params = false;
        if (paramNames.empty())
            getDefPinholeDistNames(paramNames);

        string strParam = camName+'.'+paramNames[0];
        cv::FileNode node = fSettings[strParam];
        if(!node.empty() && node.isReal())
        {
            camParams.k1 = node.real();
        }
        else
        {
            std::cerr << "*" << strParam << " parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        strParam = camName+'.'+paramNames[1];
        node = fSettings[strParam];
        if(!node.empty() && node.isReal())
        {
            camParams.k2 = node.real();
        }
        else
        {
            std::cerr << "*" << strParam << " parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        strParam = camName+'.'+paramNames[2];
        node = fSettings[strParam];
        if(!node.empty() && node.isReal())
        {
            camParams.p1 = node.real();
        }
        else
        {
            std::cerr << "*"+strParam+" parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        strParam = camName+'.'+paramNames[3];
        node = fSettings[strParam];
        if(!node.empty() && node.isReal())
        {
            camParams.p2 = node.real();
        }
        else
        {
            std::cerr << "*"+strParam+" parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        strParam = camName+'.'+paramNames[4];
        node = fSettings[strParam];
        if(!node.empty() && node.isReal())
        {
            camParams.k3 = node.real();
            camParams.hasK3 = true;
        }

        parseCamRectProjMats(fSettings, camParams, camName);

        return !b_miss_params;
    }

    bool MyParameters::parseFisheyeDist(const cv::FileStorage &fSettings, MyCamParams &camParams, const string& camName,
                                        vector<string> paramNames) {

        bool b_miss_params = false;
        if (paramNames.empty())
            getDefFisheyeDistNames(paramNames);

        string strParam = camName+'.'+paramNames[0];
        cv::FileNode node = fSettings[strParam];
        if(!node.empty() && node.isReal())
        {
            camParams.k1 = node.real();
        }
        else
        {
            std::cerr << "*"+strParam+" parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        strParam = camName+'.'+paramNames[1];
        node = fSettings[strParam];
        if(!node.empty() && node.isReal())
        {
            camParams.k2 = node.real();
        }
        else
        {
            std::cerr << "*"+strParam+" parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        strParam = camName+'.'+paramNames[2];
        node = fSettings[strParam];
        if(!node.empty() && node.isReal())
        {
            camParams.k3 = node.real();
        }
        else
        {
            std::cerr << "*"+strParam+" parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        strParam = camName+'.'+paramNames[3];
        node = fSettings[strParam];
        if(!node.empty() && node.isReal())
        {
            camParams.k4 = node.real();
        }
        else
        {
            std::cerr << "*"+strParam+" parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        parseCamRectProjMats(fSettings, camParams, camName);

        return !b_miss_params;
    }

    bool MyParameters::parseImageSize(const cv::FileStorage &fSettings, MyCamParams &camParams, const string &camName) {

        bool b_miss_params = false;

        cv::FileNode node = fSettings[camName+".width"];
        if(!node.empty() && node.isInt())
        {
            camParams.mImWidth = node.operator int();
        }
        else
        {
            std::cerr << "*"+camName+".width parameter doesn't exist or is not an int number*" << std::endl;
            b_miss_params = true;
        }
        node = fSettings[camName+".height"];
        if(!node.empty() && node.isInt())
        {
            camParams.mImHeight = node.operator int();
        }
        else
        {
            std::cerr << "*"+camName+".height parameter doesn't exist or is not an int number*" << std::endl;
            b_miss_params = true;
        }

        return !b_miss_params;
    }

    void MyParameters::parseCamRectProjMats(const cv::FileStorage &fSettings, MyCamParams &camParams, const std::string &camName) {

        // FishEye camera might also have a rectification and projection matrix
        //cv::FileNode node = fSettings[camName+".R"];
        cv::Mat mR, mP;
        fSettings[camName+".R"] >> mR;
        fSettings[camName+".P"] >> mP;

        if(!mR.empty())
        {
            //cv::Mat mR = node.mat();
            if(mR.rows != 3 || mR.cols != 3)
            {
                camParams.mR = cv::Mat();
                DLOG(INFO) << "*Rectification matrix have to be a 3x3 transformation matrix*" << std::endl;
            }
            else {
                camParams.mR = mR.clone();
            }
        }
        else {
            DLOG(INFO) << "*Rectification matrix doesn't exist*" << std::endl;
        }

        //node = fSettings[camName+".P"];
        if(!mP.empty())
        {
            //cv::Mat mP = node.mat();
            if(mP.rows != 3 || !(mP.cols == 4 || mP.cols == 3))
            {
                camParams.mP = camParams.mK.clone();
                DLOG(INFO) << "*Rectification matrix have to be a 3x3 or 3x4 transformation matrix*" << std::endl;
            }
            else {
                camParams.mP = mP.clone();
            }
        }
        else {
            DLOG(INFO) << "*Projection matrix doesn't exist*" << std::endl;
            camParams.mP = camParams.mK.clone();
        }
    }

    void MyParameters::parseFisheyeLapping(const cv::FileStorage &fSettings, MyCamParams &camParams,
                                           const string& camName, bool left) {

        string strParam = camName+'.'+"lappingBegin";
        cv::FileNode node = fSettings[strParam];
        if(!node.empty() && node.isInt())
        {
            if (left)
                camParams.leftLappingBegin = node.operator int();
            else
                camParams.rightLappingBegin = node.operator int();
        }
        else
        {
            std::cout << "WARNING: "+strParam+" not correctly defined" << std::endl;
        }

        strParam = camName+'.'+"lappingEnd";
        node = fSettings[strParam];
        if(!node.empty() && node.isInt())
        {
            if (left)
                camParams.leftLappingEnd = node.operator int();
            else
                camParams.rightLappingEnd = node.operator int();
        }
        else
        {
            std::cout << "WARNING: "+strParam+" not correctly defined" << std::endl;
        }
    }

    void MyParameters::getDefIntrinsicsNames(vector<string> &out) {
        out = {"fx", "fy", "cx", "cy"};
    }

    void MyParameters::getDefPinholeDistNames(vector<string> &out) {
        out = {"k1", "k2", "p1", "p2", "k3"};
    }

    void MyParameters::getDefFisheyeDistNames(vector<string> &out) {
        out = {"k1", "k2", "k3", "k4"};
    }

    void MyParameters::copyIntrinsics(const MyCamParams &srcCam, MyCamParams &dstCam) {

        dstCam.fx = srcCam.fx;
        dstCam.fy = srcCam.fy;
        dstCam.cx = srcCam.cx;
        dstCam.cy = srcCam.cy;

        if (!srcCam.mK.empty()) {
            dstCam.mK = srcCam.mK.clone();
        }
    }

    void MyParameters::copyDistortionParams(const MyCamParams &srcCam, MyCamParams &dstCam) {

        assert(srcCam.camType == dstCam.camType);

        if (srcCam.isPinhole()) {

            dstCam.k1 = srcCam.k1;
            dstCam.k2 = srcCam.k2;
            dstCam.p1 = srcCam.p1;
            dstCam.p2 = srcCam.p3;
            dstCam.k3 = srcCam.k3;
            dstCam.hasK3 = srcCam.hasK3;
        }
        else if (srcCam.isFisheye()) {

            dstCam.k1 = srcCam.k1;
            dstCam.k2 = srcCam.k2;
            dstCam.k3 = srcCam.k3;
            dstCam.k4 = srcCam.k4;
        }

        if (!srcCam.mDistCoef.empty()) {
            dstCam.mDistCoef = srcCam.mDistCoef.clone();
        }
        if (!srcCam.mR.empty()) {
            dstCam.mR = srcCam.mR.clone();
        }
        if (!srcCam.mP.empty()) {
            dstCam.mP = srcCam.mP.clone();
        }
    }

    bool MyParameters::parseCameraParams(const cv::FileStorage &fSettings,
                                         EORB_SLAM::MyCamParams &camParams, bool bIsStereo, bool bIsRgbd) {

        // Parse Left Camera and general parameters
        bool miss_intr = false, miss_dist = false, miss_im_sz = false;

        // Camera calibration parameters
        miss_intr = !parseCamIntrinsics(fSettings, camParams);

        // Image width and height parameters
        miss_im_sz = !parseImageSize(fSettings, camParams);

        string sCameraName = fSettings["Camera.type"];
        if(sCameraName == "PinHole")
        {
            camParams.camType = "pinhole";

            // Distortion parameters
            miss_dist = !parsePinholeDist(fSettings, camParams);
        }
        else if(sCameraName == "KannalaBrandt8")
        {
            camParams.camType = "fisheye";

            // Distortion parameters
            miss_dist = !parseFisheyeDist(fSettings, camParams);
        }
        else
        {
            std::cerr << "*Not Supported Camera Sensor*" << std::endl;
            std::cerr << "Check an example configuration file with the desired sensor" << std::endl;
        }

        bool b_miss_params = miss_intr || miss_im_sz || miss_dist;

        if(b_miss_params)
        {
            camParams.missParams = b_miss_params;
            return false;
        }
        else
        {
            camParams.initK();
            camParams.initDistCoefs();
        }

        float fps = fSettings["Camera.fps"];
        if(fps==0)
            fps=30;
        camParams.fps = fps;

        // Max/Min Frames to insert keyframes and to check relocalisation
        camParams.mMaxFrames = fps;

        int nRGB = fSettings["Camera.RGB"];
        camParams.mbRGB = nRGB;

        if(bIsStereo){
            // Parse Right Camera
            auto* camParams1 = new MyCamParams();
            camParams1->leftCam = false;
            camParams1->camType = camParams.camType;

            // Camera calibration parameters
            miss_intr = !parseCamIntrinsics(fSettings, *(camParams1), "Camera2");
            if (miss_intr) {
                copyIntrinsics(camParams, *camParams1);
            }
            else {
                camParams1->initK();
            }

            // Distortion parameters
            if (camParams.isFisheye()) {
                miss_dist = !parseFisheyeDist(fSettings, *(camParams1), "Camera2");
            }
            else if (camParams.isPinhole()) {
                miss_dist = !parsePinholeDist(fSettings, *(camParams1), "Camera2");
            }
            if (miss_dist) {
                copyDistortionParams(camParams, *camParams1);
            }
            else {
                camParams1->initDistCoefs();
            }

            // Image Size
            miss_im_sz = !parseImageSize(fSettings, *(camParams.mLinkedCam), "Camera2");
            if (miss_im_sz) {
                camParams1->mImWidth = camParams.mImWidth;
                camParams1->mImHeight = camParams.mImHeight;
            }

            // Lapping for both cameras
            parseFisheyeLapping(fSettings, camParams);
            parseFisheyeLapping(fSettings, *(camParams1), "Camera2", false);

            camParams.mLinkedCam = camParams1;

            cv::FileNode node = fSettings["Camera.bf"];
            if(!node.empty() && node.isReal())
            {
                camParams.mbf = node.real();
            }
            else
            {
                std::cerr << "*Camera.bf parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            // Left to Right Transform
            node = fSettings["Tlr"];
            bool miss_tlr = false;
            if(!node.empty())
            {
                cv::Mat mTlr = node.mat();
                if(mTlr.rows != 3 || mTlr.cols != 4)
                {
                    camParams.mTlr = cv::Mat();
                    std::cerr << "*Tlr matrix have to be a 3x4 transformation matrix*" << std::endl;
                    miss_tlr = true;
                }
                else {
                    camParams.mTlr = mTlr;
                }
            }
            else
            {
                std::cerr << "*Tlr matrix doesn't exist*" << std::endl;
                miss_tlr = true;
            }

            // Tlr is required for FishEye (also Camera2 params)
            if (camParams.isFisheye() && miss_tlr)
                b_miss_params = true;
        }

        if(bIsStereo || bIsRgbd)
        {
            //float fx = mpCamera->getParameter(0);
            cv::FileNode node = fSettings["ThDepth"];
            if(!node.empty()  && node.isReal())
            {
                float mThDepth = node.real();
                mThDepth = camParams.mbf*mThDepth/camParams.fx;
                camParams.mThDepth = mThDepth;
                cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
            }
            else
            {
                std::cerr << "*ThDepth parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }
        }

        if(bIsRgbd)
        {
            cv::FileNode node = fSettings["DepthMapFactor"];
            if(!node.empty() && node.isReal())
            {
                float mDepthMapFactor = node.real();
                if(fabs(mDepthMapFactor)<1e-5)
                    mDepthMapFactor=1;
                else
                    mDepthMapFactor = 1.0f/mDepthMapFactor;
                camParams.mDepthMapFactor = mDepthMapFactor;
            }
            else
            {
                std::cerr << "*DepthMapFactor parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }
        }

        camParams.mThFarPoints = fSettings["thFarPoints"];

        camParams.missParams = b_miss_params;
        return !b_miss_params;
    }

    ORB_SLAM3::GeometricCamera* MyParameters::dummyCamera = nullptr;
    
    void MyParameters::createCameraObject(const MyCamParams &camParams, ORB_SLAM3::GeometricCamera*& pCamera,
            ORB_SLAM3::GeometricCamera*& pCamera2, const bool isCalibrated) {

        if (camParams.isPinhole() || isCalibrated) {

            vector<float> vCamCalib{camParams.fx,camParams.fy,camParams.cx,camParams.cy};

            pCamera = new ORB_SLAM3::Pinhole(vCamCalib);
            //mpAtlas->AddCamera(mpCamera);
            pCamera2 = static_cast<ORB_SLAM3::GeometricCamera*>(nullptr);
        }
        else if (camParams.isFisheye()) {

            vector<float> vCamCalib{camParams.fx,camParams.fy,camParams.cx,camParams.cy,
                                    //0, 0, 0, 0};
                                    camParams.k1, camParams.k2, camParams.k3, camParams.k4};

            // Some datasets might also have R & P matrices
            pCamera = new ORB_SLAM3::KannalaBrandt8(vCamCalib, camParams.mR, camParams.mP, nullptr);

            if (camParams.mLinkedCam) { //isStereo
                dynamic_cast<ORB_SLAM3::KannalaBrandt8*>(pCamera)->mvLappingArea[0] = camParams.leftLappingBegin;
                dynamic_cast<ORB_SLAM3::KannalaBrandt8*>(pCamera)->mvLappingArea[1] = camParams.leftLappingEnd;

                //mpFrameDrawer->both = true;

                vector<float> vCamCalib2{camParams.mLinkedCam->fx,camParams.mLinkedCam->fy,
                                         camParams.mLinkedCam->cx,camParams.mLinkedCam->cy,
                                         camParams.mLinkedCam->k1,camParams.mLinkedCam->k2,
                                         camParams.mLinkedCam->k3,camParams.mLinkedCam->k4};

                pCamera2 = new ORB_SLAM3::KannalaBrandt8(vCamCalib2, camParams.mLinkedCam->mR, camParams.mLinkedCam->mP, nullptr);

                dynamic_cast<ORB_SLAM3::KannalaBrandt8*>(pCamera2)->mvLappingArea[0] = camParams.rightLappingBegin;
                dynamic_cast<ORB_SLAM3::KannalaBrandt8*>(pCamera2)->mvLappingArea[1] = camParams.rightLappingEnd;
            }
            else {
                pCamera2 = static_cast<ORB_SLAM3::GeometricCamera*>(nullptr);
            }
        }
        else {
            cerr << "* MyParameters::createCameraObject: Camera type is not supported.\n";
            exit(-1);
        }
    }

    void MyParameters::createCameraObject(const MyCamParams &camParams, ORB_SLAM3::TwoViewReconstruction* _tvr,
            ORB_SLAM3::GeometricCamera*& pCamera, ORB_SLAM3::GeometricCamera*& pCamera2, const bool isCalibrated) {

        if (camParams.isPinhole() || isCalibrated) {
            vector<float> vCamCalib{camParams.fx,camParams.fy,camParams.cx,camParams.cy};
            pCamera = new ORB_SLAM3::Pinhole(vCamCalib, _tvr);
            //mpAtlas->AddCamera(mpCamera);
            pCamera2 = static_cast<ORB_SLAM3::GeometricCamera*>(nullptr);
        }
        else if (camParams.isFisheye()) {
            vector<float> vCamCalib{camParams.fx,camParams.fy,camParams.cx,camParams.cy,
                                    camParams.k1,camParams.k2,camParams.k3,camParams.k4};

            pCamera = new ORB_SLAM3::KannalaBrandt8(vCamCalib, camParams.mR, camParams.mP, _tvr);

            if (camParams.mLinkedCam) { //isStereo
                dynamic_cast<ORB_SLAM3::KannalaBrandt8*>(pCamera)->mvLappingArea[0] = camParams.leftLappingBegin;
                dynamic_cast<ORB_SLAM3::KannalaBrandt8*>(pCamera)->mvLappingArea[1] = camParams.leftLappingEnd;

                //mpFrameDrawer->both = true;

                vector<float> vCamCalib2{camParams.mLinkedCam->fx,camParams.mLinkedCam->fy,
                                         camParams.mLinkedCam->cx,camParams.mLinkedCam->cy,
                                         camParams.mLinkedCam->k1,camParams.mLinkedCam->k2,
                                         camParams.mLinkedCam->k3,camParams.mLinkedCam->k4};

                pCamera2 = new ORB_SLAM3::KannalaBrandt8(vCamCalib2, camParams.mLinkedCam->mR, camParams.mLinkedCam->mP, _tvr);

                dynamic_cast<ORB_SLAM3::KannalaBrandt8*>(pCamera2)->mvLappingArea[0] = camParams.rightLappingBegin;
                dynamic_cast<ORB_SLAM3::KannalaBrandt8*>(pCamera2)->mvLappingArea[1] = camParams.rightLappingEnd;
            }
            else {
                pCamera2 = static_cast<ORB_SLAM3::GeometricCamera*>(nullptr);
            }
            //mpAtlas->AddCamera(mpCamera);
            //mpAtlas->AddCamera(mpCamera2);
        }
        else {
            cerr << "* MyParameters::createCameraObject: Camera type is not supported.\n";
            exit(-1);
        }
    }

    /*void createCameraObject(const MyCamParams &camParams, ORB_SLAM3::GeometricCamera *pCamera,
                                          ORB_SLAM3::GeometricCamera *pCamera2) {

        if (camParams.isPinhole()) {
            vector<float> vCamCalib{camParams.fx,camParams.fy,camParams.cx,camParams.cy};
            pCamera = new ORB_SLAM3::Pinhole(vCamCalib);
            //mpAtlas->AddCamera(mpCamera);
        }
        else if (camParams.isFisheye()) {
            vector<float> vCamCalib{camParams.fx,camParams.fy,camParams.cx,camParams.cy,
                                    camParams.k1,camParams.k2,camParams.k3,camParams.k4};
            pCamera = new ORB_SLAM3::KannalaBrandt8(vCamCalib);

            if (camParams.mLinkedCam) { //isStereo
                dynamic_cast<ORB_SLAM3::KannalaBrandt8*>(pCamera)->mvLappingArea[0] = camParams.leftLappingBegin;
                dynamic_cast<ORB_SLAM3::KannalaBrandt8*>(pCamera)->mvLappingArea[1] = camParams.leftLappingEnd;

                //mpFrameDrawer->both = true;

                vector<float> vCamCalib2{camParams.mLinkedCam->fx,camParams.mLinkedCam->fy,
                                         camParams.mLinkedCam->cx,camParams.mLinkedCam->cy,
                                         camParams.mLinkedCam->k1,camParams.mLinkedCam->k2,
                                         camParams.mLinkedCam->k3,camParams.mLinkedCam->k4};
                pCamera2 = new ORB_SLAM3::KannalaBrandt8(vCamCalib2);

                dynamic_cast<ORB_SLAM3::KannalaBrandt8*>(pCamera2)->mvLappingArea[0] = camParams.rightLappingBegin;
                dynamic_cast<ORB_SLAM3::KannalaBrandt8*>(pCamera2)->mvLappingArea[1] = camParams.rightLappingEnd;
            }
            //mpAtlas->AddCamera(mpCamera);
            //mpAtlas->AddCamera(mpCamera2);
        }
        else {
            cerr << "* MyParameters::createCameraObject: Camera type is not supported.\n";
            exit(-1);
        }
    }*/

    bool MyParameters::parseFeatureParams(const cv::FileStorage &fSettings, EORB_SLAM::MixedFtsParams &orbSettings) {

        bool b_miss_params = false;
//        int nFeatures, nLevels, fIniThFAST, fMinThFAST;
//        float fScaleFactor;

        cv::FileNode node = fSettings["Features.mode"];
        if(!node.empty() && node.isInt())
        {
            orbSettings.detMode = node.operator int();
        }
        else
        {
            std::cerr << "*Features.mode parameter doesn't exist or is not an integer*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Features.imMargin"];
        if(!node.empty() && node.isInt())
        {
            orbSettings.imMargin = node.operator int();
        }

        node = fSettings["ORBextractor.nFeatures"];
        if(!node.empty() && node.isInt())
        {
            orbSettings.nFeatures = node.operator int();
        }
        else
        {
            std::cerr << "*ORBextractor.nFeatures parameter doesn't exist or is not an integer*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["ORBextractor.scaleFactor"];
        if(!node.empty() && node.isReal())
        {
            orbSettings.scaleFactor = node.real();
        }
        else
        {
            std::cerr << "*ORBextractor.scaleFactor parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["ORBextractor.nLevels"];
        if(!node.empty() && node.isInt())
        {
            orbSettings.nLevels = node.operator int();
        }
        else
        {
            std::cerr << "*ORBextractor.nLevels parameter doesn't exist or is not an integer*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["ORBextractor.iniThFAST"];
        if(!node.empty() && node.isInt())
        {
            orbSettings.initThFast = node.operator int();
        }
        else
        {
            std::cerr << "*ORBextractor.iniThFAST parameter doesn't exist or is not an integer*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["ORBextractor.minThFAST"];
        if(!node.empty() && node.isInt())
        {
            orbSettings.minThFast = node.operator int();
        }
        else
        {
            std::cerr << "*ORBextractor.minThFAST parameter doesn't exist or is not an integer*" << std::endl;
            b_miss_params = true;
        }

        // AKAZE & Mixed Params
        if (orbSettings.detMode > 0) {

            node = fSettings["AKAZEextractor.nFeatures"];
            if(!node.empty() && node.isInt())
            {
                orbSettings.nFeaturesAK = node.operator int();
            }
            else
            {
                std::cerr << "*AKAZEextractor.nFeatures parameter doesn't exist or is not an integer*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["AKAZEextractor.iniTh"];
            if(!node.empty() && node.isReal())
            {
                orbSettings.iniThAK = node.real();
            }
            else
            {
                std::cerr << "*AKAZEextractor.iniTh parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["AKAZEextractor.minTh"];
            if(!node.empty() && node.isReal())
            {
                orbSettings.minThAK = node.real();
            }
            else
            {
                std::cerr << "*AKAZEextractor.minTh parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["AKAZEextractor.nOctaves"];
            if(!node.empty() && node.isInt())
            {
                orbSettings.nOctaves = node.operator int();
            }
            else
            {
                std::cerr << "*AKAZEextractor.nOctaves parameter doesn't exist or is not an integer*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["AKAZEextractor.nOctaveLayers"];
            if(!node.empty() && node.isInt())
            {
                orbSettings.nOctaveLayers = node.operator int();
            }
            else
            {
                std::cerr << "*AKAZEextractor.nOctaveLayers parameter doesn't exist or is not an integer*" << std::endl;
                b_miss_params = true;
            }
        }

        orbSettings.missParams = b_miss_params;
        return !b_miss_params;

        //mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

        //if(mpSensor==System::STEREO || mpSensor==System::IMU_STEREO)
        //    mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

        //if(mpSensor==System::MONOCULAR || mpSensor==System::IMU_MONOCULAR)
        //    mpIniORBextractor = new ORBextractor(5*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

        //orbSettings.printParams();

    }

    bool MyParameters::parseIMUParams(const cv::FileStorage &fSettings, EORB_SLAM::MyIMUSettings &imuSettings)
    {
        bool b_miss_params = false;

        cv::Mat Tbc;
        cv::FileNode node = fSettings["Tbc"];
        if(!node.empty())
        {
            Tbc = node.mat();
            if(Tbc.rows != 4 || Tbc.cols != 4)
            {
                std::cerr << "*Tbc matrix have to be a 4x4 transformation matrix*" << std::endl;
                b_miss_params = true;
            }
            else {
                imuSettings.Tbc = Tbc;
            }
        }
        else
        {
            std::cerr << "*Tbc matrix doesn't exist*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["IMU.Frequency"];
        if(!node.empty() && node.isInt())
        {
            imuSettings.freq = node.operator int();
            imuSettings.sf = sqrt(imuSettings.freq);
        }
        else
        {
            std::cerr << "*IMU.Frequency parameter doesn't exist or is not an integer*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["IMU.NoiseGyro"];
        if(!node.empty() && node.isReal())
        {
            imuSettings.Ng = node.real();
        }
        else
        {
            std::cerr << "*IMU.NoiseGyro parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["IMU.NoiseAcc"];
        if(!node.empty() && node.isReal())
        {
            imuSettings.Na = node.real();
        }
        else
        {
            std::cerr << "*IMU.NoiseAcc parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["IMU.GyroWalk"];
        if(!node.empty() && node.isReal())
        {
            imuSettings.Ngw = node.real();
        }
        else
        {
            std::cerr << "*IMU.GyroWalk parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["IMU.AccWalk"];
        if(!node.empty() && node.isReal())
        {
            imuSettings.Naw = node.real();
        }
        else
        {
            std::cerr << "*IMU.AccWalk parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        imuSettings.missParams = b_miss_params;
        return !b_miss_params;

        //imuSettings.printParams();

        //mpImuCalib = new IMU::Calib(Tbc,Ng*sf,Na*sf,Ngw/sf,Naw/sf);

        //mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
    }

    bool MyParameters::parseViewerParams(const cv::FileStorage &fSettings, EORB_SLAM::MyViewerSettings &viewerSettings) {

        bool b_miss_params = false;

        // Viewer
        float fps = fSettings["Camera.fps"];
        if(fps<1)
            fps=30;
        viewerSettings.mT = 1e3/fps;

        cv::FileNode node = fSettings["Camera.width"];
        if(!node.empty())
        {
            viewerSettings.mImageWidth = node.real();
        }
        else
        {
            std::cerr << "*Camera.width parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.height"];
        if(!node.empty())
        {
            viewerSettings.mImageHeight = node.real();
        }
        else
        {
            std::cerr << "*Camera.height parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Viewer.UseViewer"];
        if(!node.empty())
        {
            int useViewer = node.operator int();
            if (useViewer)
                viewerSettings.mbUseViewer = true;
            else
                viewerSettings.mbUseViewer = false;
        }
        else
        {
            std::cerr << "*Camera.height parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Viewer.ViewpointX"];
        if(!node.empty())
        {
            viewerSettings.mViewpointX = node.real();
        }
        else
        {
            std::cerr << "*Viewer.ViewpointX parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Viewer.ViewpointY"];
        if(!node.empty())
        {
            viewerSettings.mViewpointY = node.real();
        }
        else
        {
            std::cerr << "*Viewer.ViewpointY parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Viewer.ViewpointZ"];
        if(!node.empty())
        {
            viewerSettings.mViewpointZ = node.real();
        }
        else
        {
            std::cerr << "*Viewer.ViewpointZ parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Viewer.ViewpointF"];
        if(!node.empty())
        {
            viewerSettings.mViewpointF = node.real();
        }
        else
        {
            std::cerr << "*Viewer.ViewpointF parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        // Map Drawer
        node = fSettings["Viewer.KeyFrameSize"];
        if(!node.empty())
        {
            viewerSettings.mKeyFrameSize = node.real();
        }
        else
        {
            std::cerr << "*Viewer.KeyFrameSize parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Viewer.KeyFrameLineWidth"];
        if(!node.empty())
        {
            viewerSettings.mKeyFrameLineWidth = node.real();
        }
        else
        {
            std::cerr << "*Viewer.KeyFrameLineWidth parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Viewer.GraphLineWidth"];
        if(!node.empty())
        {
            viewerSettings.mGraphLineWidth = node.real();
        }
        else
        {
            std::cerr << "*Viewer.GraphLineWidth parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Viewer.PointSize"];
        if(!node.empty())
        {
            viewerSettings.mPointSize = node.real();
        }
        else
        {
            std::cerr << "*Viewer.PointSize parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Viewer.CameraSize"];
        if(!node.empty())
        {
            viewerSettings.mCameraSize = node.real();
        }
        else
        {
            std::cerr << "*Viewer.CameraSize parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Viewer.CameraLineWidth"];
        if(!node.empty())
        {
            viewerSettings.mCameraLineWidth = node.real();
        }
        else
        {
            std::cerr << "*Viewer.CameraLineWidth parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        viewerSettings.missParams = b_miss_params;
        return !b_miss_params;
    }

}
