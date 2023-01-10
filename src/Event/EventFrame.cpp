//
// Created by root on 1/8/21.
//

#include "EventFrame.h"

#include <utility>
#include "ORBextractor.h"
#include "GeometricCamera.h"
#include "Pinhole.h"
#include "MapPoint.h"
#include "FeatureTrack.h"


namespace EORB_SLAM {

    //long unsigned int EvFrame::nNextId=0;
    /*bool EvFrame::mbInitialComputations=true;
    float EvFrame::cx, EvFrame::cy, EvFrame::fx, EvFrame::fy, EvFrame::invfx, EvFrame::invfy;
    float EvFrame::mnMinX, EvFrame::mnMinY, EvFrame::mnMaxX, EvFrame::mnMaxY;
    float EvFrame::mfGridElementWidthInv, EvFrame::mfGridElementHeightInv;*/

    // response comparison, for list sorting
    bool response_comparator(const cv::KeyPoint& first, const cv::KeyPoint& second)
    {
        return first.response > second.response;
    }

    // Protected constructor for init. vars and avoid repetition
    EvFrame::EvFrame(unsigned long kfId, const cv::Mat &evImage, const double &timeStamp,
            ORB_SLAM3::GeometricCamera *pCamera, const cv::Mat &distCoef, const std::shared_ptr<EvFrame> &pPrevF,
            const ORB_SLAM3::IMU::Calib &ImuCalib) : EvFrame()
    {
        this->setPrevFrame(pPrevF);

        mTimeStamp = timeStamp;
        // Frame ID
        mnId = kfId;

        mpORBvocabulary = nullptr;
        mpORBextractorRight = static_cast<ORB_SLAM3::ORBextractor*>(nullptr);
        mpORBextractorLeft = static_cast<ORB_SLAM3::ORBextractor*>(nullptr);
        mpCamera = pCamera;
        mpCamera2 = static_cast<ORB_SLAM3::GeometricCamera*>(nullptr);

        mK = static_cast<ORB_SLAM3::Pinhole*>(mpCamera)->toK();
        mDistCoef = distCoef.clone();
        mImuCalib = ImuCalib;
        mbf = 0.f;
        mThDepth = 0.f;
        mTimeStereoMatch = 0;
        mTimeORB_Ext = 0;

        // Scale Level Info
        mnScaleLevels = 1;
        mfScaleFactor = 1.f;
        mfLogScaleFactor = log(mfScaleFactor);
        mvScaleFactors = vector<float>(mnScaleLevels, 1.f);
        mvInvScaleFactors = vector<float>(mnScaleLevels, 1.f);
        mvLevelSigma2 = vector<float>(mnScaleLevels, 1.f);
        mvInvLevelSigma2 = vector<float>(mnScaleLevels, 1.f);

        mnCloseMPs = 0;

        mmProjectPoints.clear();// = map<long unsigned int, cv::Point2f>(N, static_cast<cv::Point2f>(NULL));
        mmMatchedInImage.clear();

        // This is done only for the first Frame (or after a change in the calibration)
        if(mbInitialComputations)
        {
            ComputeImageBounds(evImage);

            mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
            mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

            fx = mK.at<float>(0,0);
            fy = mK.at<float>(1,1);
            cx = mK.at<float>(0,2);
            cy = mK.at<float>(1,2);
            invfx = 1.0f/fx;
            invfy = 1.0f/fy;

            mbInitialComputations=false;
        }

        mb = mbf/fx;

        //Set no stereo fisheye information
        Nleft = -1;
        Nright = -1;
        mvLeftToRightMatch = vector<int>(0);
        mvRightToLeftMatch = vector<int>(0);
        mTlr = cv::Mat(3,4,CV_32F);
        mTrl = cv::Mat(3,4,CV_32F);
        mvStereo3Dpoints = vector<cv::Mat>(0);
        monoLeft = -1;
        monoRight = -1;

        // mVw = cv::Mat::zeros(3,1,CV_32F);
        if(pPrevF)
        {
            if(!pPrevF->mVw.empty())
                mVw = pPrevF->mVw.clone();
        }
        else
        {
            mVw = cv::Mat::zeros(3,1,CV_32F);
        }

        mpMutexImu = new std::mutex();

/*
#ifdef SAVE_TIMES
        std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
        fastDet->detect(evImage, mvKeys);

        // Choose only the best points
        if (mvKeys.size() > nMaxPts) {

            sort(mvKeys.begin(), mvKeys.end(), response_comparator);
            mvKeys.resize(nMaxPts);
        }

#ifdef SAVE_TIMES
        std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif

        N = mvKeys.size();
        if(mvKeys.empty())
            return;

        //mvMatchCnt = vector<int>(N, 1);
        //mvMatches12 = vector<int>(N, -1);

        // Event frames are already undistorted, do not do this!
        //UndistortKeyPoints();
        //pCalib->undistKeyPoints(mvKeys, mvKeysUn);
        mvKeysUn = mvKeys;

        // Set no stereo information
        mvuRight = vector<float>(N,-1);
        mvDepth = vector<float>(N,-1);

        mvpMapPoints = vector<ORB_SLAM3::MapPoint*>(N,static_cast<ORB_SLAM3::MapPoint*>(nullptr));
        mvbOutlier = vector<bool>(N,false);

        AssignFeaturesToGrid();
        */
    }

    EvFrame::EvFrame(unsigned long kfId, const cv::Mat &evImage, const double &timeStamp,
            const cv::Ptr<cv::FastFeatureDetector>& fastDet, int nMaxPts, ORB_SLAM3::GeometricCamera *pCamera,
            const cv::Mat &distCoef,  const std::shared_ptr<EvFrame>& pPrevF, const ORB_SLAM3::IMU::Calib &ImuCalib) :
            EvFrame(kfId, evImage, timeStamp, pCamera, distCoef, pPrevF, ImuCalib) {

#ifdef SAVE_TIMES
        std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
        fastDet->detect(evImage, mvKeys);

        // Choose only the best points
        if (mvKeys.size() > nMaxPts) {

            sort(mvKeys.begin(), mvKeys.end(), response_comparator);
            mvKeys.resize(nMaxPts);
        }

#ifdef SAVE_TIMES
        std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif

        N = mvKeys.size();
        if(mvKeys.empty())
            return;

        //mvMatchCnt = vector<int>(N, 1);
        //mvMatches12 = vector<int>(N, -1);

        // Event frames are already undistorted, do not do this!
        //UndistortKeyPoints();
        //pCalib->undistKeyPoints(mvKeys, mvKeysUn);
        mvKeysUn = mvKeys;

        // Set no stereo information
        mvuRight = vector<float>(N,-1);
        mvDepth = vector<float>(N,-1);

        mvpMapPoints = vector<ORB_SLAM3::MapPoint*>(N,static_cast<ORB_SLAM3::MapPoint*>(nullptr));
        mvbOutlier = vector<bool>(N,false);

        AssignFeaturesToGrid();
    }

    EvFrame::EvFrame(unsigned long kfId, const cv::Mat &evImage, const double &timeStamp, ORB_SLAM3::ORBextractor *extractor,
                     ORB_SLAM3::GeometricCamera *pCamera, const cv::Mat &distCoef, const bool extDesc,
                     const std::shared_ptr<EvFrame>& pPrevF, const ORB_SLAM3::IMU::Calib &ImuCalib) :
            EvFrame(kfId, evImage, timeStamp, pCamera, distCoef, pPrevF, ImuCalib) {

        mpORBextractorLeft = extractor;

        // Scale Level Info
        mnScaleLevels = mpORBextractorLeft->GetLevels();
        mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
        mfLogScaleFactor = log(mfScaleFactor);
        mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
        mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
        mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
        mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

        // ORB extraction
#ifdef SAVE_TIMES
        std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
        // Event Frames have no descriptors
        ExtractFeatures(0,evImage,0,1000, extDesc);

#ifdef SAVE_TIMES
        std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif

        N = mvKeys.size();
        if(mvKeys.empty())
            return;

        //mvMatchCnt = vector<int>(N, 1);
        //mvMatches12 = vector<int>(N, -1);

        //UndistortKeyPoints();
        //pCalib->undistKeyPoints(mvKeys, mvKeysUn);
        mvKeysUn = mvKeys;

        // Set no stereo information
        mvuRight = vector<float>(N,-1);
        mvDepth = vector<float>(N,-1);

        mvpMapPoints = vector<ORB_SLAM3::MapPoint*>(N,static_cast<ORB_SLAM3::MapPoint*>(nullptr));
        mvbOutlier = vector<bool>(N,false);

        AssignFeaturesToGrid();
    }

    EvFrame::EvFrame(unsigned long kfId, const cv::Mat &evImage, const double &timeStamp,
                     const cv::Ptr<cv::FastFeatureDetector>& fastDet, ORB_SLAM3::ORBextractor* extractorORB,
                     int nMaxPts, ORB_SLAM3::GeometricCamera *pCamera,
                     const cv::Mat &distCoef,  const std::shared_ptr<EvFrame>& pPrevF, const ORB_SLAM3::IMU::Calib &ImuCalib) :
            EvFrame(kfId, evImage, timeStamp, pCamera, distCoef, pPrevF, ImuCalib) {

        mpORBextractorLeft = extractorORB;

        // Scale Level Info
        mnScaleLevels = mpORBextractorLeft->GetLevels();
        mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
        mfLogScaleFactor = log(mfScaleFactor);
        mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
        mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
        mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
        mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

#ifdef SAVE_TIMES
        std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
        fastDet->detect(evImage, mvKeys);

        // Choose only the best points
        int nFAST = mvKeys.size();
        if (nFAST > nMaxPts) {

            sort(mvKeys.begin(), mvKeys.end(), response_comparator);
            mvKeys.resize(nMaxPts);
        }

        // If we have not enough points, detect points in other levels
        if (nFAST < nMaxPts) {

            vector<cv::KeyPoint> vCurrKPts = mvKeys;
            vCurrKPts.reserve(nMaxPts);

            ExtractFeatures(0, evImage, 0, 1000, false);

            for (size_t currIdx = nFAST, orbIdx = 0; currIdx < nMaxPts && orbIdx < mvKeys.size(); orbIdx++) {

                if (mvKeys[orbIdx].octave > 0) {
                    vCurrKPts.push_back(mvKeys[orbIdx]);
                    currIdx++;
                }
            }
            mvKeys = vCurrKPts;
        }

#ifdef SAVE_TIMES
        std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif

        N = mvKeys.size();
        if(mvKeys.empty())
            return;

        //mvMatchCnt = vector<int>(N, 1);
        //mvMatches12 = vector<int>(N, -1);

        // Event frames are already undistorted, do not do this!
        //UndistortKeyPoints();
        //pCalib->undistKeyPoints(mvKeys, mvKeysUn);
        mvKeysUn = mvKeys;

        // Set no stereo information
        mvuRight = vector<float>(N,-1);
        mvDepth = vector<float>(N,-1);

        mvpMapPoints = vector<ORB_SLAM3::MapPoint*>(N,static_cast<ORB_SLAM3::MapPoint*>(nullptr));
        mvbOutlier = vector<bool>(N,false);

        AssignFeaturesToGrid();
    }

    EvFrame::EvFrame(unsigned long kfId, const cv::Mat &evImage, const double &timeStamp, const std::vector<cv::KeyPoint>& kpts,
                     ORB_SLAM3::GeometricCamera *pCamera, const cv::Mat &distCoef,  const std::shared_ptr<EvFrame>& pPrevF,
                     const ORB_SLAM3::IMU::Calib &ImuCalib) :
            EvFrame(kfId, evImage, timeStamp, pCamera, distCoef, pPrevF, ImuCalib) {

        // All mvKeys, mvKeysUn, mvTrackedPts, mvTrackedPtsUn are initialized
        // the same way. If you want new points be detected, do this explicitly.
        N = kpts.size();
        mnTrackedPts = N;
        if(mnTrackedPts <= 0)
            return;

        //Better to do this outside
        //mvMatchCnt = vector<int>(mnTrackedPts, 1);
        //mvMatches12 = vector<int>(N, -1);

        // Input key points are alwayes rectified!
        mvKeys = kpts;
        mvKeysUn = kpts;
        mvTrackedPts = kpts;
        mvTrackedPtsUn = kpts;

        // Set no stereo information
        mvuRight = vector<float>(mnTrackedPts,-1);
        mvDepth = vector<float>(mnTrackedPts,-1);
        mnCloseMPs = 0;

        mvpMapPoints = vector<ORB_SLAM3::MapPoint*>(mnTrackedPts,static_cast<ORB_SLAM3::MapPoint*>(nullptr));
        mvbOutlier = vector<bool>(mnTrackedPts,false);

        AssignFeaturesToGrid();
    }

    EvFrame::EvFrame(const EvFrame& evFrame) : ORB_SLAM3::Frame(*(&evFrame)), mnTrackedPts(evFrame.mnTrackedPts),
            mvTrackedPts(evFrame.mvTrackedPts), mvTrackedPtsUn(evFrame.mvTrackedPtsUn),
            mvMchTrackedPts(evFrame.mvMchTrackedPts), mpRefFrame(evFrame.mpRefFrame), mpPrevFrame(evFrame.mpPrevFrame),
            mMedPxDisp(evFrame.mMedPxDisp), mFirstAcc(evFrame.mFirstAcc.clone())
    {}

    EvFrame::EvFrame(ulong id, const PoseImagePtr &pImage, const std::vector<cv::KeyPoint> &kpts,
            const std::vector<FeatureTrackPtr> &vFtTracks, ORB_SLAM3::GeometricCamera *pCamera,
            const cv::Mat &distCoef, const std::shared_ptr<EvFrame> &pPrevF, const ORB_SLAM3::IMU::Calib &ImuCalib) :
            EvFrame(id, pImage->mImage, pImage->ts, kpts, pCamera, distCoef, pPrevF, ImuCalib)
    {
        assert(kpts.size() == vFtTracks.size());
        mvpFeatureTracks = vFtTracks;
    }

    /*EvFrame::~EvFrame() = default; {

        //Frame::~Frame();
    }*/

    void EvFrame::setPrevFrame(const std::shared_ptr<EvFrame> &pEvFrame) {

        mpPrevFrame = pEvFrame;
        Frame::mpPrevFrame = pEvFrame.get();
    }

    std::shared_ptr<EvFrame> EvFrame::getPrevFrame() {

        if (std::shared_ptr<EvFrame> pPreFrame = mpPrevFrame.lock())
            return pPreFrame;
        return nullptr;
    }

    void EvFrame::setRefFrame(const std::shared_ptr<EvFrame> &pEvFrame) {

        mpRefFrame = pEvFrame;
    }

    std::shared_ptr<EvFrame> EvFrame::getRefFrame() {

        std::shared_ptr<EvFrame> pRefFrame = mpRefFrame.lock();
        return pRefFrame;
    }

    /*void EvFrame::resetAllDistKPtsMono(const std::vector<cv::KeyPoint>& vKpts, const bool distorted) {

        N = mvKeys.size();
        mnTrackedPts = N;

        mvKeys = vKpts;

        if (distorted) {
            if (dynamic_cast<ORB_SLAM3::Pinhole*>(mpCamera)) {
                MyCalibrator::undistKeyPointsPinhole(mvTrackedPts, mvTrackedPtsUn, mK, mDistCoef, cv::Mat(), mK);
            }
            else {
                MyCalibrator::undistKeyPointsFishEye(mvTrackedPts, mvTrackedPtsUn, mK, mDistCoef, cv::Mat(), mK);
            }
        }
        else {
            mvKeysUn = vKpts;
        }

        mvTrackedPts = mvKeys;
        mvTrackedPtsUn = mvKeysUn;

        // Set no stereo information
        mvuRight = vector<float>(mnTrackedPts,-1);
        mvDepth = vector<float>(mnTrackedPts,-1);
        mnCloseMPs = 0;

        mvpMapPoints.resize(N, static_cast<ORB_SLAM3::MapPoint*>(nullptr));
        mvbOutlier.resize(N, false);

        AssignFeaturesToGrid();
    }*/

    void EvFrame::setDistTrackedPts(const std::vector<cv::KeyPoint> &trackedPts) {

        mnTrackedPts = trackedPts.size();
        mvTrackedPts = trackedPts;

        if (dynamic_cast<ORB_SLAM3::Pinhole*>(mpCamera)) {
            MyCalibrator::undistKeyPointsPinhole(mvTrackedPts, mvTrackedPtsUn, mK, mDistCoef, cv::Mat(), mK);
        }
        else {
            MyCalibrator::undistKeyPointsFishEye(mvTrackedPts, mvTrackedPtsUn, mK, mDistCoef, cv::Mat(), mK);
        }
    }

    /*void EvFrame::ExtractORB(int flag, const cv::Mat &im, const int x0, const int x1)
    {
        vector<int> vLapping = {x0,x1};
        if(flag==0)
            monoLeft = (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,vLapping);
        else
            monoRight = (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,vLapping);
    }

    void EvFrame::UndistortTrackedPts() {

        if(abs(mDistCoef.at<float>(0)-0.f) < 1e-12)
        {
            mvTrackedPtsUn = mvTrackedPts;
            return;
        }

        // Fill matrix with points
        cv::Mat mat(mnTrackedPts,2,CV_32F);

        for(int i=0; i<mnTrackedPts; i++)
        {
            mat.at<float>(i,0)=mvTrackedPts[i].pt.x;
            mat.at<float>(i,1)=mvTrackedPts[i].pt.y;
        }

        // Undistort points
        mat=mat.reshape(2);
        cv::undistortPoints(mat, mat, mK,mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        // Fill undistorted keypoint vector
        mvTrackedPtsUn.resize(mnTrackedPts);
        for(int i=0; i<mnTrackedPts; i++)
        {
            cv::KeyPoint kp = mvTrackedPts[i];
            kp.pt.x=mat.at<float>(i,0);
            kp.pt.y=mat.at<float>(i,1);
            mvTrackedPtsUn[i]=kp;
        }
    }*/

    float calcPtDist(const cv::KeyPoint& kp1, const cv::KeyPoint& kp2) {

        return sqrtf(powf(kp1.pt.x - kp2.pt.x, 2) + powf(kp1.pt.y - kp2.pt.y, 2));
    }

    unsigned EvFrame::connectNewAndTrackedPts(const float& threshDist) {

        assert(!mvKeysUn.empty() && !mvTrackedPtsUn.empty());

        mvMchTrackedPts.resize(mvKeysUn.size(), -1);

        unsigned nMatches = 0;
        for (size_t i = 0; i < mvKeysUn.size(); i++) {

            const cv::KeyPoint& currPt = mvKeysUn[i];

            map<float, int> mTrackedDist;
            for (size_t j = 0; j < mvTrackedPtsUn.size(); j++) {

                mTrackedDist.insert(make_pair(calcPtDist(mvTrackedPtsUn[j], currPt), j));
            }
            if (mTrackedDist.begin()->first < threshDist) {
                mvMchTrackedPts[i] = mTrackedDist.begin()->second;
                nMatches++;
            }
        }
        return nMatches;
    }

    // Attention!!! This is not thread safe!
    float EvFrame::computeSceneMedianDepth(const int q)
    {
        vector<ORB_SLAM3::MapPoint*> vpMapPoints;
        cv::Mat Tcw_;
        {
            //unique_lock<mutex> lock(mMutexFeatures);
            //unique_lock<mutex> lock2(mMutexPose);
            vpMapPoints = this->getAllMapPointsMono();
            Tcw_ = mTcw.clone();
        }

        vector<float> vDepths;
        vDepths.reserve(N);
        cv::Mat Rcw2 = Tcw_.row(2).colRange(0,3);
        Rcw2 = Rcw2.t();
        float zcw = Tcw_.at<float>(2,3);
        for(int i=0; i<N; i++)
        {
            if(vpMapPoints[i])
            {
                ORB_SLAM3::MapPoint* pMP = vpMapPoints[i];
                cv::Mat x3Dw = pMP->GetWorldPos();
                float z = static_cast<float>(Rcw2.dot(x3Dw))+zcw;
                vDepths.push_back(z);
            }
        }

        sort(vDepths.begin(),vDepths.end());

        return vDepths[(vDepths.size()-1)/q];
    }

    float EvFrame::computeSceneMedianDepth(const int q, const std::vector<cv::Point3f>& pts3d)
    {
        cv::Mat Tcw_ = mTcw.clone();
        vector<int> vMatches = mvMatches12;

        //assert(pts3d.size() == vMatches.size());
        if (pts3d.size() != vMatches.size()) {
            DLOG(ERROR) << "EvFrame::computeSceneMedianDepth: P3D size missmatch: P3Dsz: "
                        << pts3d.size() << ", vMatchesSz: " << vMatches.size() << endl;
            return 1.f;
        }

        vector<float> vDepths;
        vDepths.reserve(N);
        cv::Mat Rcw2 = Tcw_.row(2).colRange(0,3);
        Rcw2 = Rcw2.t();
        float zcw = Tcw_.at<float>(2,3);
        for(int i=0; i<N; i++)
        {
            if(vMatches[i] > 0)
            {
                cv::Mat pt3d = cv::Mat(pts3d[i]);
                float z = static_cast<float>(Rcw2.dot(pt3d))+zcw;
                vDepths.push_back(z);
            }
        }

        sort(vDepths.begin(),vDepths.end());

        return vDepths[(vDepths.size()-1)/q];
    }

    float EvFrame::computeSceneMedianDepth(const cv::Mat& Tcw_, const std::vector<ORB_SLAM3::MapPoint*>& vpMPs, int q)
    {
        if (Tcw_.empty() || vpMPs.empty()) {
            return 1.f;
        }

        int nMPs = vpMPs.size();

        vector<float> vDepths;
        vDepths.reserve(nMPs);

        cv::Mat Rcw2 = Tcw_.row(2).colRange(0,3);
        Rcw2 = Rcw2.t();
        float zcw = Tcw_.at<float>(2,3);

        for(int i=0; i<nMPs; i++)
        {
            ORB_SLAM3::MapPoint* pMP = vpMPs[i];
            if(pMP && !pMP->isBad())
            {
                cv::Mat pt3d = pMP->GetWorldPos();
                float z = static_cast<float>(Rcw2.dot(pt3d))+zcw;
                vDepths.push_back(z);
            }
        }

        if (!vDepths.empty()) {
            sort(vDepths.begin(), vDepths.end());
            return vDepths[(vDepths.size() - 1) / q];
        }
        else {
            return 1.f;
        }
    }

    void EvFrame::setFirstAcc(const ORB_SLAM3::IMU::Point &imuMea) {

        mFirstAcc = cv::Mat::zeros(3, 1, CV_32FC1);
        mFirstAcc.at<float>(0) = imuMea.a.x;
        mFirstAcc.at<float>(1) = imuMea.a.y;
        mFirstAcc.at<float>(2) = imuMea.a.z;
    }

    FeatureTrackPtr EvFrame::getFeatureTrack(const size_t &idx) {

        if (idx >= 0 && idx < mvpFeatureTracks.size()) {
            return mvpFeatureTracks[idx];
        }
        return nullptr;
    }

    /*void EvFrame::setAllFeatureTracks(const std::vector<FeatureTrackPtr> &vpFtTracks) {

        // Only set feature tracks for the first time and
        // assert its size matches num. key points
        if (mvpFeatureTracks.empty() && vpFtTracks.size() == N) {
            mvpFeatureTracks = vpFtTracks;
        }
        else if (vpFtTracks.size() != N) {
            LOG(ERROR) << "EvFrame::setAllFeatureTracks: Num. Feature Tracks mismatch: "
                       << vpFtTracks.size() << " != " << N << endl;
        }
    }*/

} // EORB_SLAM
