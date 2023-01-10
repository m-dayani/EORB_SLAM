//
// Created by root on 12/10/20.
//

#ifndef ORB_SLAM3_MIXEDFRAME_H
#define ORB_SLAM3_MIXEDFRAME_H

#include <vector>

#include <opencv2/core/core.hpp>

#include "Frame.h"


namespace EORB_SLAM {

    struct MixedFeatures {

        int mnMix;
        std::vector<std::vector<cv::KeyPoint>> mvKeyPts;
        std::vector<std::vector<cv::Mat>> mvDescriptors;
        std::vector<int> mvNfeatures;
        std::vector<size_t> mvFtsIdxBegin;
    };

    // Wrapper around cv::AKAZE extractor
    class AKAZEextractor {
    public:
        AKAZEextractor(int nFeatures, int nOctaves, int nOctaveLayers, float iniTh, float minTh);

        void detectAndCompute(const cv::Mat& image, const cv::Mat& mask, std::vector<cv::KeyPoint>& vKPts, cv::Mat& desc);

        int getNFeatures() const { return mnFeatures; }
        int getNOctaves() const { return mnOctaves; }
        int getNOctaveLayers() const { return mnOctaveLayers; }
        int getNLevels() const { return mnLevelsAK; }
        float getIniThreshold() const { return mIniTh; }
        float getMinThreshold() const { return mMinTh; }
        float getScaleFactor() const { return mfScaleFactorAK; }
        float getLogScaleFactor() const { return mfLogScaleFactorAK; }

        vector<vector<float>> getScaleFactors() { return mvScaleFactorsAK; }
        vector<vector<float>> getInvScaleFactors() { return mvInvScaleFactorsAK; }
        vector<vector<float>> getAllLevelSigma2() { return mvLevelSigma2AK; }
        vector<vector<float>> getAllInvLevelSigma2() { return mvInvLevelSigma2AK; }

    private:
        const int mnFeatures, mnOctaves, mnOctaveLayers, mnLevelsAK;
        const float mIniTh, mMinTh;
        float mfScaleFactorAK, mfLogScaleFactorAK;

        std::vector<std::vector<float>> mvScaleFactorsAK;
        std::vector<std::vector<float>> mvInvScaleFactorsAK;
        std::vector<std::vector<float>> mvLevelSigma2AK;
        std::vector<std::vector<float>> mvInvLevelSigma2AK;

        cv::Ptr<cv::AKAZE> mpAKAZEextractor;
    };

    class MixedFrame : public ORB_SLAM3::Frame {
    public:

        enum MixedFrameType {
            ORB,
            AKAZE,
            MIXED
        };

        MixedFrame();

        MixedFrame(const cv::Mat &imGray, const double &timeStamp, ORB_SLAM3::ORBextractor* extractor,
                const std::shared_ptr<AKAZEextractor>& extractorAK, ORB_SLAM3::ORBVocabulary* voc,
                ORB_SLAM3::GeometricCamera* pCamera, const MyCalibPtr& pCalib, cv::Mat &distCoef,
                const float &bf, const float &thDepth, Frame* pPrevF = static_cast<Frame*>(nullptr),
                const ORB_SLAM3::IMU::Calib &ImuCalib = ORB_SLAM3::IMU::Calib());

        // Pure AKAZE constructor -> only if we have bag of AKAZE words!
//        MixedFrame(const cv::Mat &imGray, const double &timeStamp, const cv::Ptr<cv::AKAZE>& extractorAK,
//                const int &nPtsAK, ORB_SLAM3::ORBVocabulary* voc, ORB_SLAM3::GeometricCamera* pCamera,
//                cv::Mat &distCoef, const float &bf, const float &thDepth, const float &minThAK,
//                Frame* pPrevF = static_cast<Frame*>(nullptr), const ORB_SLAM3::IMU::Calib &ImuCalib = ORB_SLAM3::IMU::Calib());

        explicit MixedFrame(MixedFrame* mixedFrame);

        //~MixedFrame() override;

        MixedFrameType getFrameType() { return mFrameType; }

        int numAllORBFts() const { return mnFtsORB; }
        int numAllMixedFts() const { return mnFtsAK; }

        std::vector<std::vector<cv::KeyPoint>> getAllUndistMixedPoints() { return mvMixedKPtsUn; }
        std::vector<std::vector<cv::KeyPoint>> getAllDistMixedPoints() { return mvMixedKPts; }

        std::vector<cv::Mat> getMixedDescriptors() { return mvMixedDesc; }
        bool isORBDescValid(const int idx) const { return idx < mnFtsORB; }

        int getAKAZENOctaves() const { return this->mnOctaves; }
        int getAKAZENOctaveLayers() const { return this->mnOctaveLayers; }
        int getAKAZENLevels() const { return this->mnLevelsAK; }
        float getAKAZEIniThreshold() const { return this->mfThAK; }
        float getAKAZEMinThreshold() const { return this->mfMinThAK; }
        float getAKAZEScaleFactor() const { return this->mfScaleFactorAK; }
        float getAKAZELogScaleFactor() const { return this->mfLogScaleFactorAK; }

        std::vector<std::vector<float>> getAllAKAZEScaleFactors() const { return this->mvScaleFactorsAK; }
        std::vector<std::vector<float>> getAllAKAZEInvScaleFactors() const { return this->mvInvScaleFactorsAK; }
        std::vector<std::vector<float>> getAllAKAZELevelSigma2() const { return this->mvLevelSigma2AK; }
        std::vector<std::vector<float>> getAllAKAZEInvLevelSigma2() const { return this->mvInvLevelSigma2AK; }

        std::shared_ptr<AKAZEextractor>& getAKAZEextractor() { return mpAKAZEextractorLeft; }

        /* ---------------------------------------------------------------------------------------------------------- */

        //std::vector<cv::KeyPoint>& getAllUndistKPtsMono() override { return this->mvKeysUn; }
        //std::vector<cv::KeyPoint> getAllUndistKPtsMono() const override { return this->mvKeysUn; }
        //virtual std::vector<cv::KeyPoint>& getAllDistKPtsMono();
        //std::vector<cv::KeyPoint> getAllDistKPtsMono() const override { return this->mvKeys; }
        //virtual std::vector<cv::KeyPoint>& getAllKPtsRight();
        //std::vector<cv::KeyPoint> getAllKPtsRight() const override;

        //cv::KeyPoint getUndistKPtMono(int idx) const override;
        //cv::KeyPoint getDistKPtMono(int idx) const override;
        //cv::KeyPoint getKPtRight(const int idx) const override { return mvKeysRight[idx]; }

        void ComputeBoW() override;
        //cv::Mat getDescriptorMono(int idxMix, int idxDesc);
        cv::Mat getDescriptors(int idxMix);
        cv::Mat getORBDescriptor(int idx) override;
        cv::Mat getDescriptorMono(int idx) override;

        //ORB_SLAM3::MapPoint* getMapPoint(const int idx) override { return mvpMapPoints[idx]; }
        //ORB_SLAM3::MapPoint* getMapPoint(const int idx) const override { return mvpMapPoints[idx]; }
        //void setMapPoint(const int idx, ORB_SLAM3::MapPoint* pMP) override { this->mvpMapPoints[idx] = pMP; }
        //void setAllMapPointsMono(const std::vector<ORB_SLAM3::MapPoint*>& vpMPs) override { this->mvpMapPoints = vpMPs; }
        //void resetAllMapPointsMono() override {
        //    std::fill(this->mvpMapPoints.begin(), this->mvpMapPoints.end(), static_cast<ORB_SLAM3::MapPoint*>(nullptr));
        //}

        //bool getMPOutlier(const int idx) const override { return mvbOutlier[idx]; }
        //void setMPOutlier(const int idx, bool flag) override { this->mvbOutlier[idx] = flag; }
        //void setAllMPOutliers(const std::vector<bool>& vbOutliers) override { this->mvbOutlier = vbOutliers; }

        //float getLevelScaleFactor(int orbLevel) const override;
        float getAKAZEScaleFactor(int orbLevel) const;
        //int getMaxLevels(int idxPt) const override;
        int getKPtLevelMono(int idx) const override;
        float getKPtScaleFactor(int idx) const override;
        //float getKPtInvScaleFactor(const int idx) const override { return this->mvInvScaleFactors[this->getUndistKPtMono(idx).octave]; }
        float getKPtLevelSigma2(int idx) const override;
        float getKPtInvLevelSigma2(int idx) const override;

        static int toLevelORB(int nOctave, int nOctaveLayer, int nOctaveLayers);
        static void toOctaveLayerAK(int levelORB, int &nOctave, int &nOctaveLayer, int nOctaveLayers);
        static void toOctaveLayerAK(int levelORB, int &nOctave, int &nOctaveLayer, int nLevelsAK, int nOctaves, int nOctaveLayers);

        void printClassIdAKAZE();
        void printOctaveAKAZE();

    protected:

        // No need to rewrite, the default works fine
        //int numAllKPtsVir() const override { return N; }

        //int numAllMPsMonoVir() const override { return mvpMapPoints.size(); }
        //std::vector<ORB_SLAM3::MapPoint*> getAllMapPointsMonoVir() const override { return mvpMapPoints; }
        //std::vector<bool> getAllOutliersMonoVir() const override { return mvbOutlier; }

        /* ---------------------------------------------------------------------------------------------------------- */

        static void sortFeaturesResponse(std::vector<cv::KeyPoint>& vKPts, cv::Mat& desc);
        static void sortFeaturesResponse(std::vector<cv::KeyPoint>& vKPts, std::vector<cv::KeyPoint>& vKPtsUn, cv::Mat& desc);

        //void undistortKeyPoints(const std::vector<cv::KeyPoint>& vKPts, std::vector<cv::KeyPoint>& vKPtsUn);

        void resolveNumMixedPts();

        void fillAllKPtsContainer();

        void assignMixedFeaturesToGrid();

    private:
        MixedFrameType mFrameType;
        int nMix{2};

        // vector of vector of cv_kpts (other than ORB)
        //std::vector<cv::KeyPoint> mvAllKPts, mvAllKPtsUn;
        std::vector<std::vector<cv::KeyPoint>> mvMixedKPts, mvMixedKPtsUn;
        std::vector<cv::Mat> mvMixedDesc;

        // Although we can have a vector of lots of feature types, currently only ORB & AKAZE are used
        int mnFtsORB;
        int mnFtsAK;
        //std::vector<int> vNmix;

        // Scale pyramid info.
        int mnOctaveLayers, mnOctaves, mnLevelsAK;
        float mfThAK, mfMinThAK, mfScaleFactorAK, mfLogScaleFactorAK;

        std::vector<std::vector<float>> mvScaleFactorsAK;
        std::vector<std::vector<float>> mvInvScaleFactorsAK;
        std::vector<std::vector<float>> mvLevelSigma2AK;
        std::vector<std::vector<float>> mvInvLevelSigma2AK;

        //std::vector<ORB_SLAM3::MapPoint*> mvpMapPoints;
        //std::vector<bool> mvbOutlier;

        std::shared_ptr<AKAZEextractor> mpAKAZEextractorLeft;
    };

} //EORB_SLAM

#endif //ORB_SLAM3_MIXEDFRAME_H
