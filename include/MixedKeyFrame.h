//
// Created by root on 2/23/21.
//

#ifndef ORB_SLAM3_MIXEDKEYFRAME_H
#define ORB_SLAM3_MIXEDKEYFRAME_H

#include "MixedFrame.h"
#include "KeyFrame.h"

namespace EORB_SLAM {

    class MixedKeyFrame : public ORB_SLAM3::KeyFrame {
    public:
        MixedKeyFrame(MixedFrame* F, ORB_SLAM3::Map* pMap, ORB_SLAM3::KeyFrameDatabase* pKFDB);

        //~MixedKeyFrame() override;

        /* ---------------------------------------------------------------------------------------------------------- */

        //std::vector<cv::KeyPoint> getAllUndistKPtsMono() const override { return this->mvAllKPtsUn; }
        //cv::KeyPoint getUndistKPtMono(int idx) const override;
        //std::vector<cv::KeyPoint> getAllDistKPtsMono() const override { return this->mvAllKPts; }
        //cv::KeyPoint getDistKPtMono(int idx) const override;
        //std::vector<cv::KeyPoint> getAllKPtsRight() const override { return this->mvKeysRight; }
        //cv::KeyPoint getKPtRight(const int idx) const override { return this->mvKeysRight[idx]; }

        void ComputeBoW() override;
        bool isORBDescValid(const int idx) const { return idx < mnFtsORB; }
        cv::Mat getORBDescriptor(int idx) const override;
        cv::Mat getDescriptors(int idxMix);
        cv::Mat getDescriptorMono(int idx) override;

        int getAKAZENLevels() const { return this->mnLevelsAK; }
        float getAKAZEScaleFactor() const { return this->mfScaleFactorAK; }
        float getAKAZELogScaleFactor() const { return this->mfLogScaleFactorAK; }
        std::vector<std::vector<float>> getAllAKAZEScaleFactors() { return this->mvScaleFactorsAK; }
        float getAKAZEScaleFactor(int levelORB) const;
        float getAKAZEInvLevelSigma2(int levelORB) const;

        int getKPtLevelMono(int idx) const override;
        float getKPtScaleFactor(int idx) const override;
        float getKPtLevelSigma2(int idx) const override;
        float getKPtInvLevelSigma2(int idx) const override;

    protected:

        //int numAllKPtsVir() const override { return this->N; }

    private:
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
        std::vector<std::vector<float>> mvLevelSigma2AK;
        std::vector<std::vector<float>> mvInvLevelSigma2AK;

    };

} // EORB_SLAM

#endif //ORB_SLAM3_MIXEDKEYFRAME_H
