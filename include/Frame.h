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


#ifndef FRAME_H
#define FRAME_H

//#define SAVE_TIMES

#include<vector>
#include <mutex>

#include <opencv2/opencv.hpp>

#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#include "include/IMU/ImuTypes.h"
#include "ORBVocabulary.h"

#include "MyCalibrator.h"


//namespace EORB_SLAM {
//    class EvFrame;
//}

namespace ORB_SLAM3
{
#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

class MapPoint;
class KeyFrame;
class ConstraintPoseImu;
class GeometricCamera;
class ORBextractor;

class Frame
{
public:
    Frame();

    // Copy constructor.
    Frame(const Frame &frame);

    // Constructor for stereo cameras.
    Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft,
            ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf,
            const float &thDepth, GeometricCamera* pCamera, EORB_SLAM::PairCalibPtr ppCalib,
            Frame* pPrevF = static_cast<Frame*>(nullptr), const IMU::Calib &ImuCalib = IMU::Calib());

    Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft,
          ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef,
          const float &bf, const float &thDepth, GeometricCamera* pCamera, GeometricCamera* pCamera2,
          cv::Mat& Tlr,Frame* pPrevF = static_cast<Frame*>(nullptr), const IMU::Calib &ImuCalib = IMU::Calib());

    // Constructor for RGB-D cameras.
    Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,
            ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth,
            GeometricCamera* pCamera,Frame* pPrevF = static_cast<Frame*>(nullptr), const IMU::Calib &ImuCalib = IMU::Calib());

    // Constructor for Monocular cameras.
    Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor, ORBVocabulary* voc,
            GeometricCamera* pCamera, const EORB_SLAM::MyCalibPtr& pCalib, cv::Mat &distCoef, const float &bf, const float &thDepth,
            Frame* pPrevF = static_cast<Frame*>(nullptr), const IMU::Calib &ImuCalib = IMU::Calib(), bool isMixed = false);

    // Destructor
    virtual ~Frame();

    // Extract ORB on the image. 0 for left image and 1 for right image.
    void ExtractFeatures(int flag, const cv::Mat &im, int x0, int x1, bool extractDesc = true);
    virtual void ExtractORB(int flag, const cv::Mat &im, int x0, int x1, bool extractDesc = true);

    // Compute Bag of Words representation.
    virtual void ComputeBoW();

    // Set the camera pose. (Imu pose is not modified!)
    void SetPose(cv::Mat Tcw);
    void GetPose(cv::Mat &Tcw);

    // Set IMU velocity
    void SetVelocity(const cv::Mat &Vwb);

    // Set IMU pose and velocity (implicitly changes camera pose)
    void SetImuPoseVelocity(const cv::Mat &Rwb, const cv::Mat &twb, const cv::Mat &Vwb);


    // Computes rotation, translation and camera center matrices from the camera pose.
    void UpdatePoseMatrices();

    // Returns the camera center.
    inline cv::Mat GetCameraCenter(){
        return mOw.clone();
    }

    // Returns inverse of rotation
    inline cv::Mat GetRotationInverse(){
        return mRwc.clone();
    }

    cv::Mat GetImuPosition();
    cv::Mat GetImuRotation();
    cv::Mat GetImuPose();

    void SetNewBias(const IMU::Bias &b);

    // Check if a MapPoint is in the frustum of the camera
    // and fill variables of the MapPoint to be used by the tracking
    bool isInFrustum(MapPoint* pMP, float viewingCosLimit);

    bool ProjectPointDistort(MapPoint* pMP, cv::Point2f &kp, float &u, float &v);

    cv::Mat inRefCoordinates(cv::Mat pCw);

    // Compute the cell of a keypoint (return false if outside the grid)
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

    vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r, int minLevel=-1,
            int maxLevel=-1, bool bRight = false) const;

    // Search a match for each keypoint in the left image to a keypoint in the right image.
    // If there is a match, depth is computed and the right coordinate associated to the left keypoint is stored.
    void ComputeStereoMatches();

    // Associate a "right" coordinate to a keypoint if there is valid depth in the depthmap.
    void ComputeStereoFromRGBD(const cv::Mat &imDepth);

    // Backprojects a keypoint (if stereo/depth info available) into 3D world coordinates.
    cv::Mat UnprojectStereo(const int &i);

    bool imuIsPreintegrated();
    void setIntegrated();

    //Stereo fisheye
    void ComputeStereoFishEyeMatches();

    bool isInFrustumChecks(MapPoint* pMP, float viewingCosLimit, bool bRight = false);

    cv::Mat UnprojectStereoFishEye(const int &i);

    void PrintPointDistribution(){
        int left = 0, right = 0;
        int Nlim = (Nleft != -1) ? Nleft : N;
        for(int i = 0; i < N; i++){
            if(mvpMapPoints[i] && !mvbOutlier[i]){
                if(i < Nlim) left++;
                else right++;
            }
        }
        cout << "Point distribution in Frame: left-> " << left << " --- right-> " << right << endl;
    }

    /* ============================================================================================================== */

    // It is not recommended to use virtual method in constructor

    int numAllKPts() const { return this->numAllKPtsVir(); }
    // These are only for stereo, not applicable in my current version
    int numKPtsLeft() const { return this->Nleft; }
    int numKPtsRight() const { return this->Nright; }

    std::vector<cv::KeyPoint>& getAllORBUndistKPtsMono() { return this->mvKeysUn; }
    std::vector<cv::KeyPoint> getAllORBUndistKPtsMono() const { return this->mvKeysUn; }
    std::vector<cv::KeyPoint>& getAllORBDistKPtsMono() { return this->mvKeys; }
    std::vector<cv::KeyPoint> getAllORBDistKPtsMono() const { return this->mvKeys; }
    std::vector<cv::KeyPoint>& getAllORBKPtsRight() { return this->mvKeysRight; }
    std::vector<cv::KeyPoint> getAllORBKPtsRight() const { return this->mvKeysRight; }

    // Note: Methods that get all things are very dangrous and must be handled with care
    virtual std::vector<cv::KeyPoint>& getAllUndistKPtsMono() { return this->getAllORBUndistKPtsMono(); }
    virtual std::vector<cv::KeyPoint> getAllUndistKPtsMono() const { return this->getAllORBUndistKPtsMono(); }
    //virtual std::vector<cv::KeyPoint>& getAllDistKPtsMono() { return this->getAllORBDistKPtsMono(); }
    virtual std::vector<cv::KeyPoint> getAllDistKPtsMono() const { return this->getAllORBDistKPtsMono(); }
    //virtual std::vector<cv::KeyPoint>& getAllKPtsRight() { return this->getAllORBKPtsRight(); }
    virtual std::vector<cv::KeyPoint> getAllKPtsRight() const { return this->getAllORBKPtsRight(); }

    virtual cv::KeyPoint getUndistKPtMono(int idx) const;
    virtual cv::KeyPoint getDistKPtMono(int idx) const;
    virtual cv::KeyPoint getKPtRight(int idx) const;

    cv::Mat& getAllORBDescMono() { return mDescriptors; }
    cv::Mat getAllORBDescMono() const { return mDescriptors.clone(); }
    virtual cv::Mat getORBDescriptor(int idx);
    // WARNING!! Never use this for matching!!
    virtual cv::Mat getDescriptorMono(int idx);

    int numAllMPsMono() const { return this->numAllMPsMonoVir();}
    std::vector<MapPoint*> getAllMapPointsMono() const { return this->getAllMapPointsMonoVir(); }
    std::vector<bool> getAllOutliersMono() const { return this->getAllOutliersMonoVir(); }

    virtual MapPoint* getMapPoint(int idx);
    virtual MapPoint* getMapPoint(int idx) const;
    virtual void setMapPoint(int idx, MapPoint* pMP);
    virtual void setAllMapPointsMono(const std::vector<MapPoint*>& vpMPs);
    virtual void resetAllMapPointsMono() {
        std::fill(this->mvpMapPoints.begin(), this->mvpMapPoints.end(), static_cast<MapPoint*>(nullptr));
    }

    virtual bool getMPOutlier(int idx) const;
    virtual void setMPOutlier(int idx, bool flag);
    virtual void setAllMPOutliers(const std::vector<bool>& vbOutliers);

    int checkORBLevel(const int orbLevel) const {
        if (orbLevel < 0)
            return 0;
        if (orbLevel >= mnScaleLevels)
            return mnScaleLevels-1;
        return orbLevel;
    }
    int getORBNLevels() const { return this->mnScaleLevels; }
    float getORBScaleFactor() const { return this->mfScaleFactor; }
    float getORBLogScaleFactor() const { return this->mfLogScaleFactor; }

    std::vector<float> getAllORBScaleFactors() const { return this->mvScaleFactors; }
    float getORBScaleFactor(const int level) const {
        //assert(level >= 0 && level < mnScaleLevels);
        return this->mvScaleFactors[this->checkORBLevel(level)];
    }
    std::vector<float> getAllORBInvScaleFactors() const { return this->mvInvScaleFactors; }
    //float getORBInvScaleFactor(const int level) const;// { return this->mvInvScaleFactors[level]; }
    std::vector<float> getAllORBLevelSigma2() const { return this->mvLevelSigma2; }
    //float getORBLevelSigma2(const int level) const;// { return this->mvLevelSigma2[level]; }
    std::vector<float> getAllORBInvLevelSigma2() const { return this->mvInvLevelSigma2; }
    float getORBInvLevelSigma2(const int level) const {
        //assert(level >= 0 && level < mnScaleLevels);
        return this->mvInvLevelSigma2[this->checkORBLevel(level)];
    }

    //virtual float getLevelScaleFactor(const int level) const { return this->getORBScaleFactor(level); }
    //virtual int getMaxLevels(const int idxPt) const { return this->getORBNLevels(); }
    virtual int getKPtLevelMono(int idx) const;
    virtual float getKPtScaleFactor(int idx) const;
    //virtual float getKPtInvScaleFactor(const int idx) const;// { return this->mvInvScaleFactors[this->getUndistKPtMono(idx).octave]; }
    virtual float getKPtLevelSigma2(int idx) const;
    virtual float getKPtInvLevelSigma2(int idx) const;

    void setPosePrior(const cv::Mat& priorTcw) { mPriorTcw = priorTcw.clone(); }
    cv::Mat getPosePrior() { return mPriorTcw.clone(); }

    void setMatches(const std::vector<int>& vMatches12) { mvMatches12 = vMatches12; }
    std::vector<int>& getMatches() { return mvMatches12; }
    int getMatches(int idx);

    void setNumMatches(const uint nMatches) { mnMatches = nMatches; }
    uint getNumMatches() const { return mnMatches; }

protected:
    virtual int numAllKPtsVir() const { return N; }
    //virtual int numKPtsLeftVir() const { return Nleft; }
    //virtual int numKPtsRightVir() const { return Nright; }

    virtual int numAllMPsMonoVir() const { return mvpMapPoints.size(); }
    virtual std::vector<MapPoint*> getAllMapPointsMonoVir() const { return mvpMapPoints; }
    virtual std::vector<bool> getAllOutliersMonoVir() const { return mvbOutlier; }

    /* ============================================================================================================== */

    // Undistort keypoints given OpenCV distortion parameters.
    // Only for the RGB-D case. Stereo must be already rectified!
    // (called in the constructor).
    void UndistortKeyPoints();

    // Computes image bounds for the undistorted image (called in the constructor).
    void ComputeImageBounds(const cv::Mat &imLeft);

    // Assign keypoints to the grid for speed up feature matching (called in the constructor).
    void AssignFeaturesToGrid();

public:
    ConstraintPoseImu* mpcpi;

    cv::Mat mRwc;
    cv::Mat mOw;

    // Vocabulary used for relocalization.
    ORBVocabulary* mpORBvocabulary;

    // Feature extractor. The right is used only in the stereo case.
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;

    // Frame timestamp.
    double mTimeStamp;

    // Calibration matrix and OpenCV distortion parameters.
    cv::Mat mK;
    static float fx;
    static float fy;
    static float cx;
    static float cy;
    static float invfx;
    static float invfy;
    cv::Mat mDistCoef;

    // Stereo baseline multiplied by fx.
    float mbf;

    // Stereo baseline in meters.
    float mb;

    // Threshold close/far points. Close points are inserted from 1 view.
    // Far points are inserted as in the monocular case from 2 views.
    float mThDepth;

    // Corresponding stereo coordinate and depth for each keypoint.
    // "Monocular" keypoints have a negative value.
    // TODO: Should we protect these too????
    std::vector<float> mvuRight;
    std::vector<float> mvDepth;
    int mnCloseMPs;

    map<long unsigned int, cv::Point2f> mmProjectPoints;
    map<long unsigned int, cv::Point2f> mmMatchedInImage;

    // Bag of Words Vector structures.
    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;

    // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
    static float mfGridElementWidthInv;
    static float mfGridElementHeightInv;
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    // Camera pose.
    cv::Mat mTcw;

    // IMU linear velocity
    cv::Mat mVw;

    cv::Mat mPredRwb, mPredtwb, mPredVwb;
    IMU::Bias mPredBias;

    // IMU bias
    IMU::Bias mImuBias;

    // Imu calibration
    IMU::Calib mImuCalib;

    // Imu preintegration from last keyframe
    IMU::Preintegrated* mpImuPreintegrated;
    KeyFrame* mpLastKeyFrame;

    // Pointer to previous frame
    Frame* mpPrevFrame;
    IMU::Preintegrated* mpImuPreintegratedFrame;

    // Current and Next Frame id.
    static long unsigned int nNextId;
    long unsigned int mnId;

    // Reference Keyframe.
    KeyFrame* mpReferenceKF;

    // Undistorted Image Bounds (computed once).
    static float mnMinX;
    static float mnMaxX;
    static float mnMinY;
    static float mnMaxY;

    static bool mbInitialComputations;

    string mNameFile;

    int mnDataset;

    double mTimeStereoMatch;
    double mTimeORB_Ext;

    GeometricCamera* mpCamera, *mpCamera2;

    //For stereo matching
    std::vector<int> mvLeftToRightMatch, mvRightToLeftMatch;

    //For stereo fisheye matching
    static cv::BFMatcher BFmatcher;

    //Triangulated stereo observations using as reference the left camera. These are
    //computed during ComputeStereoFishEyeMatches
    std::vector<cv::Mat> mvStereo3Dpoints;

    //Grid for the right image
    std::vector<std::size_t> mGridRight[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    cv::Mat mTlr, mRlr, mtlr, mTrl;

    cv::Mat imgLeft, imgRight;

    std::weak_ptr<Frame> mpEvFrame; // Frame* mpEvFrame??

protected:
    // Important members that should not be public (for further extension of this class)

    // Number of KeyPoints.
    int N;
    //Number of KeyPoints extracted in the left and right images
    int Nleft, Nright;
    //Number of Non Lapping Keypoints
    int monoLeft, monoRight;

    // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
    // In the stereo case, mvKeysUn is redundant as images must be rectified.
    // In the RGB-D case, RGB images can be distorted.
    std::vector<cv::KeyPoint> mvKeys, mvKeysRight;
    std::vector<cv::KeyPoint> mvKeysUn;

    // ORB descriptor, each row associated to a keypoint.
    cv::Mat mDescriptors, mDescriptorsRight;

    // Scale pyramid info.
    int mnScaleLevels;
    float mfScaleFactor;
    float mfLogScaleFactor;
    vector<float> mvScaleFactors;
    vector<float> mvInvScaleFactors;
    vector<float> mvLevelSigma2;
    vector<float> mvInvLevelSigma2;

    // MapPoints associated to keypoints, NULL pointer if no association.
    // Map point matches and outliers
    std::vector<MapPoint*> mvpMapPoints;
    // Flag to identify outlier associations.
    std::vector<bool> mvbOutlier;

    // Rotation, translation and camera center
    cv::Mat mRcw;
    cv::Mat mtcw;
    //==mtwc

    bool mbImuPreintegrated;

    cv::Mat mPriorTcw; // Used as a holder for L1 Pose estimates

    std::vector<int> mvMatches12;
    uint mnMatches;

//private:
    std::mutex *mpMutexImu;

};

typedef std::shared_ptr<Frame> FramePtr;

}// namespace ORB_SLAM

#endif // FRAME_H
