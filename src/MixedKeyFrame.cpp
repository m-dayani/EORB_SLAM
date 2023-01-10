//
// Created by root on 2/23/21.
//

#include "MixedKeyFrame.h"
#include "Converter.h"


namespace EORB_SLAM {

    MixedKeyFrame::MixedKeyFrame(MixedFrame *mixedF, ORB_SLAM3::Map *pMap, ORB_SLAM3::KeyFrameDatabase *pKFDB) :
        ORB_SLAM3::KeyFrame(*(mixedF), pMap, pKFDB)
    {
        //auto* mixedF = dynamic_cast<MixedFrame*>(F);
        //assert(mixedF);

        //mvAllKPtsUn = mixedF->getAllUndistKPtsMono();
        //mvAllKPts = mixedF->getAllDistKPtsMono();

        mvMixedKPts = mixedF->getAllDistMixedPoints();
        mvMixedKPtsUn = mixedF->getAllUndistMixedPoints();

        //mvMixedDesc = mixedF->getMixedDescriptors();
        nMix = mvMixedKPts.size();
        mvMixedDesc.resize(nMix);
        for (size_t i = 0; i < nMix; i++)
            mvMixedDesc[i] = mixedF->getDescriptors(i).clone();

        mnFtsORB = mixedF->numAllORBFts();
        mnFtsAK = mixedF->numAllMixedFts();

        assert(N == mnFtsORB + mnFtsAK);

        mnOctaves = mixedF->getAKAZENOctaves();
        mnOctaveLayers = mixedF->getAKAZENOctaveLayers();
        mnLevelsAK = mixedF->getAKAZENLevels();
        mfThAK = mixedF->getAKAZEIniThreshold();
        mfMinThAK = mixedF->getAKAZEMinThreshold();
        mfScaleFactorAK = mixedF->getAKAZEScaleFactor();
        mfLogScaleFactorAK = mixedF->getAKAZELogScaleFactor();

        mvScaleFactorsAK = mixedF->getAllAKAZEScaleFactors();
        mvLevelSigma2AK = mixedF->getAllAKAZELevelSigma2();
        mvInvLevelSigma2AK = mixedF->getAllAKAZEInvLevelSigma2();
    }

    /*MixedKeyFrame::~MixedKeyFrame() {

        KeyFrame::~KeyFrame();
    }*/

    /*cv::KeyPoint MixedKeyFrame::getUndistKPtMono(const int idx) const {

        return (idx < mnFtsORB) ? mvKeysUn[idx] : mvMixedKPtsUn[0][idx];
    }

    cv::KeyPoint MixedKeyFrame::getDistKPtMono(const int idx) const {

        return (idx < mnFtsORB) ? mvKeys[idx] : mvMixedKPts[0][idx];
    }*/

    void MixedKeyFrame::ComputeBoW() {

        if(mBowVec.empty() || mFeatVec.empty()) {

            vector<cv::Mat> vCurrentDesc = ORB_SLAM3::Converter::toDescriptorVector(this->getDescriptors(0));
            // Feature vector associate features with nodes in the 4th level (from leaves up)
            // We assume the vocabulary tree has 6 levels, change the 4 otherwise
            mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
        }
    }

    cv::Mat MixedKeyFrame::getDescriptors(const int idxMix) {

        assert(idxMix >= 0 && idxMix < mvMixedDesc.size());
        return mvMixedDesc[idxMix];
    }

    cv::Mat MixedKeyFrame::getDescriptorMono(const int idx) {

        assert(idx >= 0 && idx < N);
        if (idx < mnFtsORB)
            return mvMixedDesc[0].row(idx);
        int idxAK = idx - mnFtsORB;
        if (idxAK >= 0 && idxAK < mnFtsAK)
            return mvMixedDesc[1].row(idxAK);
        return cv::Mat();
    }

    cv::Mat MixedKeyFrame::getORBDescriptor(const int idx) const {

        assert(idx >= 0 && idx < N);
        if (idx < mnFtsORB)
            return mvMixedDesc[0].row(idx);
        return cv::Mat();
    }

    float MixedKeyFrame::getAKAZEScaleFactor(const int levelORB) const {

        int nOctave = 0, nOctaveLayer = 0;
        MixedFrame::toOctaveLayerAK(levelORB, nOctave, nOctaveLayer, mnLevelsAK, mnOctaves, mnOctaveLayers);
        assert(nOctave >= 0 && nOctave < mnOctaves && nOctaveLayer >= 0 && nOctaveLayer < mnOctaveLayers);
        return mvScaleFactorsAK[nOctave][nOctaveLayer];
    }

    float MixedKeyFrame::getAKAZEInvLevelSigma2(const int levelORB) const {

        int nOctave = 0, nOctaveLayer = 0;
        MixedFrame::toOctaveLayerAK(levelORB, nOctave, nOctaveLayer, mnLevelsAK, mnOctaves, mnOctaveLayers);
        assert(nOctave >= 0 && nOctave < mnOctaves && nOctaveLayer >= 0 && nOctaveLayer < mnOctaveLayers);
        return mvInvLevelSigma2AK[nOctave][nOctaveLayer];
    }

    int MixedKeyFrame::getKPtLevelMono(const int idx) const {

        assert(idx >= 0 && idx < N);
        if (idx < mnFtsORB)
            return KeyFrame::getKPtLevelMono(idx);
        cv::KeyPoint kpt = this->getDistKPtMono(idx);
        assert(kpt.class_id >= 0);
        return kpt.class_id;
    }

    float MixedKeyFrame::getKPtScaleFactor(const int idx) const {

        assert(idx >= 0 && idx < N);
        if (idx < mnFtsORB)
            return KeyFrame::getKPtScaleFactor(idx);
        cv::KeyPoint kpt = this->getDistKPtMono(idx);
        assert(kpt.class_id >= 0);
        int nOctave = 0, nOctaveLayer = 0;
        MixedFrame::toOctaveLayerAK(kpt.class_id, nOctave, nOctaveLayer, mnLevelsAK, mnOctaves, mnOctaveLayers);
        assert(kpt.octave >= 0 && kpt.octave < mnOctaves && nOctaveLayer >= 0 && nOctaveLayer < mnOctaveLayers);
        return mvScaleFactorsAK[kpt.octave][nOctaveLayer];
    }

    float MixedKeyFrame::getKPtLevelSigma2(const int idx) const {

        assert(idx >= 0 && idx < N);
        if (idx < mnFtsORB)
            return KeyFrame::getKPtLevelSigma2(idx);
        cv::KeyPoint kpt = this->getDistKPtMono(idx);
        assert(kpt.class_id >= 0);
        int nOctave = 0, nOctaveLayer = 0;
        MixedFrame::toOctaveLayerAK(kpt.class_id, nOctave, nOctaveLayer, mnLevelsAK, mnOctaves, mnOctaveLayers);
        assert(kpt.octave >= 0 && kpt.octave < mnOctaves && nOctaveLayer >= 0 && nOctaveLayer < mnOctaveLayers);
        return mvLevelSigma2AK[kpt.octave][nOctaveLayer];
    }

    float MixedKeyFrame::getKPtInvLevelSigma2(const int idx) const {

        assert(idx >= 0 && idx < N);
        if (idx < mnFtsORB)
            return KeyFrame::getKPtInvLevelSigma2(idx);
        cv::KeyPoint kpt = this->getDistKPtMono(idx);
        assert(kpt.class_id >= 0);
        int nOctave = 0, nOctaveLayer = 0;
        MixedFrame::toOctaveLayerAK(kpt.class_id, nOctave, nOctaveLayer, mnLevelsAK, mnOctaves, mnOctaveLayers);
        assert(kpt.octave >= 0 && kpt.octave < mnOctaves && nOctaveLayer >= 0 && nOctaveLayer < mnOctaveLayers);
        return mvInvLevelSigma2AK[kpt.octave][nOctaveLayer];
    }

} // EORB_SLAM
