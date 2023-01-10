//
// Created by root on 12/10/20.
//


#include <thread>

#include "MixedFrame.h"
#include "ORBextractor.h"
#include "Pinhole.h"
#include "Converter.h"

namespace EORB_SLAM {

    AKAZEextractor::AKAZEextractor(const int nFeatures, const int nOctaves, const int nOctaveLayers,
            const float iniTh, const float minTh) :
            mnFeatures(nFeatures), mnOctaves(nOctaves), mnOctaveLayers(nOctaveLayers),
            mnLevelsAK(nOctaves*nOctaveLayers), mIniTh(iniTh), mMinTh(minTh) {

        mpAKAZEextractor = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0,
                3, mIniTh, mnOctaves, mnOctaveLayers,cv::KAZE::DIFF_PM_G2);

        mfScaleFactorAK = powf(2.f, 1.f / float(mnOctaveLayers));
        mfLogScaleFactorAK = logf(mfScaleFactorAK);

        // Build up AKAZE pyramid info.
        mvScaleFactorsAK.resize(mnOctaves);
        mvInvScaleFactorsAK.resize(mnOctaves);
        mvLevelSigma2AK.resize(mnOctaves);
        mvInvLevelSigma2AK.resize(mnOctaves);

        for (int i = 0; i < mnOctaves; i++) {

            mvScaleFactorsAK[i].resize(mnOctaveLayers);
            mvInvScaleFactorsAK[i].resize(mnOctaveLayers);
            mvLevelSigma2AK[i].resize(mnOctaveLayers);
            mvInvLevelSigma2AK[i].resize(mnOctaveLayers);

            for (int j = 0; j < mnOctaveLayers; j++) {

                float levelSigma = powf(2.f, float(i) + float(j) / float(mnOctaveLayers));
                float invLevelSigma = 1.f / levelSigma;

                mvScaleFactorsAK[i][j] = levelSigma;
                mvInvScaleFactorsAK[i][j] = invLevelSigma;
                mvLevelSigma2AK[i][j] = levelSigma * levelSigma;
                mvInvLevelSigma2AK[i][j] = invLevelSigma * invLevelSigma;
            }
        }
    }

    void AKAZEextractor::detectAndCompute(const cv::Mat &image, const cv::Mat &mask,
            std::vector<cv::KeyPoint> &vKPts, cv::Mat &desc) {

        if (mask.empty()) {
            mpAKAZEextractor->detectAndCompute(image, cv::noArray(), vKPts, desc);
        }
        else {
            mpAKAZEextractor->detectAndCompute(image, mask, vKPts, desc);
        }
    }

    /* ============================================================================================================== */

    MixedFrame::MixedFrame() : Frame(), mFrameType(MIXED), mnFtsORB(0), mnFtsAK(0), mnOctaveLayers(0), mnOctaves(0),
            mnLevelsAK(0), mfThAK(0), mfMinThAK(0), mfScaleFactorAK(1.f), mfLogScaleFactorAK(0.f) { N = 0; }

    MixedFrame::MixedFrame(const cv::Mat &imGray, const double &timeStamp, ORB_SLAM3::ORBextractor *extractor,
            const std::shared_ptr<AKAZEextractor>& extractorAK, ORB_SLAM3::ORBVocabulary *voc,
            ORB_SLAM3::GeometricCamera *pCamera, const MyCalibPtr& pCalib, cv::Mat &distCoef, const float &bf,
            const float &thDepth, Frame *pPrevF, const ORB_SLAM3::IMU::Calib &ImuCalib) :

            ORB_SLAM3::Frame(imGray, timeStamp, extractor, voc, pCamera, pCalib, distCoef, bf, thDepth, pPrevF, ImuCalib, true),
            mFrameType(MIXED), mnFtsORB(extractor->GetNumFeatures() - extractorAK->getNFeatures()),
            mnFtsAK(extractorAK->getNFeatures()), mnOctaveLayers(extractorAK->getNOctaveLayers()),
            mnOctaves(extractorAK->getNOctaves()), mnLevelsAK(extractorAK->getNLevels()),
            mfThAK(extractorAK->getIniThreshold()), mfMinThAK(extractorAK->getMinThreshold()),
            mfScaleFactorAK(extractorAK->getScaleFactor()), mfLogScaleFactorAK(extractorAK->getLogScaleFactor()),
            mpAKAZEextractorLeft(extractorAK) {

        // Fill scale info.
        mvScaleFactorsAK = mpAKAZEextractorLeft->getScaleFactors();
        mvInvScaleFactorsAK = mpAKAZEextractorLeft->getInvScaleFactors();
        mvLevelSigma2AK = mpAKAZEextractorLeft->getAllLevelSigma2();
        mvInvLevelSigma2AK = mpAKAZEextractorLeft->getAllInvLevelSigma2();

        mvMixedKPts.resize(2);
        mvMixedKPtsUn.resize(2);
        mvMixedDesc.resize(2);

        // Detect and compute ORB and AKAZE simultaneously
#ifdef SAVE_TIMES
        std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
        // ORB extraction
        // All ORB Features are detected (for BoW) but only the first Norb - Nakaze are used!
        std::thread threadORB(&Frame::ExtractORB,this,0,imGray,0,1000,true);
        // AKAZE extraction
        std::thread threadAKAZE(&AKAZEextractor::detectAndCompute,mpAKAZEextractorLeft.get(),imGray,cv::Mat(),
                                std::ref(mvMixedKPts[1]),std::ref(mvMixedDesc[1]));
        threadORB.join();
        threadAKAZE.join();
#ifdef SAVE_TIMES
        std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

        mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif

        N = mvKeys.size();
        if (mvKeys.empty())
            return;

        // Also sort ORB key points based on response
        int nDetectedPtsORB = N;
        if (nDetectedPtsORB > mnFtsORB) {

            // Sort key points and descriptors based on response
            sortFeaturesResponse(mvKeys, mDescriptors);
        }
        // Move detected ORB features to mixed containers
        mvMixedKPts[0] = mvKeys;
        mvMixedDesc[0] = mDescriptors.clone();

        // If we detected more than AKAZE features, select only the best responses
        int nDetectedPtsAK = mvMixedKPts[1].size();
        if (nDetectedPtsAK > mnFtsAK) {

            // Sort key points and descriptors based on response
            sortFeaturesResponse(mvMixedKPts[1], mvMixedDesc[1]);
        }

        //UndistortKeyPoints();
        pCalib->undistKeyPoints(mvKeys, mvKeysUn);
        mvMixedKPtsUn[0] = mvKeysUn;
        //LOG_EVERY_N(INFO, 1000) << "Frame: mvKeys[0].pt: " << mvKeys[0].pt << ", mvKeysUn[0].pt: " << mvKeysUn[0].pt << endl;
        // Undistort key points
        if (nDetectedPtsAK > 0) {
            pCalib->undistKeyPoints(mvMixedKPts[1], mvMixedKPtsUn[1]);
        }

        // Update key point counts
        this->resolveNumMixedPts();

        // Set no stereo information
        mvuRight = vector<float>(N, -1);
        mvDepth = vector<float>(N, -1);

        mvpMapPoints = vector<ORB_SLAM3::MapPoint *>(N, static_cast<ORB_SLAM3::MapPoint *>(nullptr));
        mvbOutlier = vector<bool>(N, false);

        this->fillAllKPtsContainer();
        // Accessing this is so dangerous now -> clear it
        //mDescriptors = cv::Mat();

        //this->assignMixedFeaturesToGrid();
        // Since we are using original containers for all KPts, we can use original utils
        this->AssignFeaturesToGrid();
    }

    MixedFrame::MixedFrame(MixedFrame *mixedFrame) :
        Frame(*mixedFrame)
    {
        //auto* currMixedFrame = dynamic_cast<MixedFrame*>(mixedFrame);
        //assert(currMixedFrame);
//        if (!mixedFrame) {
//            cerr << "MixedFrame::MixedFrame(...): Input frame is not mixed!\n";
//            return;
//        }

        mFrameType = mixedFrame->mFrameType;

        mvMixedKPtsUn = mixedFrame->getAllUndistMixedPoints();
        mvMixedKPts = mixedFrame->getAllDistMixedPoints();

        nMix = mvMixedKPts.size();
        mvMixedDesc.resize(nMix);
        for (size_t i = 0; i < nMix; i++)
            mvMixedDesc[i] = mixedFrame->getDescriptors(i).clone();

        //mvAllKPtsUn = currMixedFrame->getAllUndistKPtsMono();
        //mvAllKPts = currMixedFrame->getAllDistKPtsMono();

        mnFtsORB = mixedFrame->numAllORBFts();
        mnFtsAK = mixedFrame->numAllMixedFts();

        assert(N == mnFtsORB + mnFtsAK);

        mnOctaves = mixedFrame->getAKAZENOctaves();
        mnOctaveLayers = mixedFrame->getAKAZENOctaveLayers();
        mnLevelsAK = mixedFrame->getAKAZENLevels();
        mfThAK = mixedFrame->getAKAZEIniThreshold();
        mfMinThAK = mixedFrame->getAKAZEMinThreshold();
        mfScaleFactorAK = mixedFrame->getAKAZEScaleFactor();
        mfLogScaleFactorAK = mixedFrame->getAKAZELogScaleFactor();

        mvScaleFactorsAK = mixedFrame->getAllAKAZEScaleFactors();
        mvInvScaleFactorsAK = mixedFrame->getAllAKAZEInvScaleFactors();
        mvLevelSigma2AK = mixedFrame->getAllAKAZELevelSigma2();
        mvInvLevelSigma2AK = mixedFrame->getAllAKAZEInvLevelSigma2();

        mpAKAZEextractorLeft = mixedFrame->getAKAZEextractor();
    }

    /*MixedFrame::~MixedFrame() {



        Frame::~Frame();
    }*/

    void MixedFrame::sortFeaturesResponse(std::vector<cv::KeyPoint> &vKPts, cv::Mat &desc) {

        assert(vKPts.size() == desc.rows);
        multimap<float, pair<cv::KeyPoint, cv::Mat>, greater<float>> mResp;
        int nPts = vKPts.size();
        for (size_t i = 0; i < nPts; i++) {
            mResp.insert(make_pair(vKPts[i].response, make_pair(vKPts[i], desc.row(i).clone())));
        }
        int i = 0;
        for (const auto& currResp : mResp) {
            vKPts[i] = currResp.second.first;
            currResp.second.second.copyTo(desc.row(i));
            i++;
        }
    }

    void MixedFrame::sortFeaturesResponse(std::vector<cv::KeyPoint> &vKPts, std::vector<cv::KeyPoint>& vKPtsUn, cv::Mat &desc) {

        assert(vKPts.size() == vKPtsUn.size() && vKPts.size() == desc.rows);
        multimap<float, pair<pair<cv::KeyPoint, cv::KeyPoint>, cv::Mat>, greater<float>> mResp;
        int nPts = vKPts.size();
        for (size_t i = 0; i < nPts; i++) {
            mResp.insert(make_pair(vKPts[i].response, make_pair(make_pair(vKPts[i], vKPtsUn[i]), desc.row(i).clone())));
        }
        int i = 0;
        for (const auto& currResp : mResp) {
            vKPts[i] = currResp.second.first.first;
            vKPtsUn[i] = currResp.second.first.second;
            currResp.second.second.copyTo(desc.row(i));
            i++;
        }
    }

    /*void MixedFrame::undistortKeyPoints(const std::vector<cv::KeyPoint>& vKPts, std::vector<cv::KeyPoint>& vKPtsUn)
    {
        if(mDistCoef.at<float>(0)==0.0)
        {
            vKPtsUn = vKPts;
            return;
        }

        int nPts = vKPts.size();
        // Fill matrix with points
        cv::Mat mat(nPts,2, CV_32F);

        for(int i=0; i<nPts; i++)
        {
            mat.at<float>(i,0)=vKPts[i].pt.x;
            mat.at<float>(i,1)=vKPts[i].pt.y;
        }

        // Undistort points
        mat=mat.reshape(2);
        cv::undistortPoints(mat, mat, static_cast<ORB_SLAM3::Pinhole*>(mpCamera)->toK(),
                mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);


        // Fill undistorted keypoint vector
        vKPtsUn.resize(nPts);
        for(int i=0; i<nPts; i++)
        {
            cv::KeyPoint kp = vKPts[i];
            kp.pt.x=mat.at<float>(i,0);
            kp.pt.y=mat.at<float>(i,1);
            vKPtsUn[i]=kp;
        }

    }*/

    void MixedFrame::resolveNumMixedPts() {

        assert(mvMixedKPts.size() > 1 && mpORBextractorLeft && mpAKAZEextractorLeft);
        int nDetectedORB = mvMixedKPts[0].size();
        int nDetectedAK = mvMixedKPts[1].size();
        int nDetected = nDetectedORB + nDetectedAK;

        int nDesired = mpORBextractorLeft->GetNumFeatures();
        int nDesiredAK = mpAKAZEextractorLeft->getNFeatures();
        int nDesiredORB = nDesired - nDesiredAK;

        if (nDetected > nDesired) {

            int nDiff = nDetected - nDesired;
            if (nDetectedORB > nDesiredORB && nDetectedAK > nDesiredAK) {
                mnFtsORB = nDesiredORB;
                mnFtsAK = nDesiredAK;
            }
            else if (nDetectedORB > nDesiredORB) {
                mnFtsORB = nDetectedORB-nDiff;
                mnFtsAK = min(nDetectedAK, nDesiredAK);
            }
            else if (nDetectedAK > nDesiredAK) {
                mnFtsORB = min(nDetectedORB, nDesiredORB);
                mnFtsAK = nDetectedAK-nDiff;
            }
            else {
                cerr << "MixedFrame::resolveNumMixedPts -> nDetected ?><? nDesired: "
                     << nDetected << ", " << nDesired << endl;
            }
        }
        else {
            mnFtsORB = nDetectedORB;
            mnFtsAK = nDetectedAK;
        }
        N = mnFtsAK+mnFtsORB;
    }

    void MixedFrame::fillAllKPtsContainer() {

        assert(N == mnFtsORB + mnFtsAK);
        // Origianl KeyPoints can still be found in mixed container
        mvKeys.resize(N);
        mvKeysUn.resize(N);

        for (int i = 0; i < N; i++) {

            if (i < mnFtsORB) {
                mvKeysUn[i] = mvMixedKPtsUn[0][i];
                mvKeys[i] = mvMixedKPts[0][i];
            }
            else if (i - mnFtsORB < mnFtsAK){
                int idxAK = i - mnFtsORB;
                mvKeysUn[i] = mvMixedKPtsUn[1][idxAK];
                mvKeys[i] = mvMixedKPts[1][idxAK];
            }
        }
    }

    void MixedFrame::assignMixedFeaturesToGrid() {

        // Fill matrix with points
        const int nCells = FRAME_GRID_COLS*FRAME_GRID_ROWS;

        int nReserve = 0.5*N/(nCells);

        for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
            for (unsigned int j=0; j<FRAME_GRID_ROWS;j++){
                mGrid[i][j].reserve(nReserve);
                if(Nleft != -1){
                    mGridRight[i][j].reserve(nReserve);
                }
            }

        // ORB Features are already assigned to the grid
        // This has been changed and only supports monocular camera
        for (int i = 0; i < N; i++) {

            const cv::KeyPoint &kp = mvKeysUn[i];

            int nGridPosX, nGridPosY;
            if (PosInGrid(kp, nGridPosX, nGridPosY)) {
                mGrid[nGridPosX][nGridPosY].push_back(i);
            }
        }
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    /*cv::KeyPoint MixedFrame::getUndistKPtMono(const int idx) const {

        if (idx < mnFtsORB)
            return mvKeysUn[idx];
        int idxAK = idx - mnFtsORB;
        if (idxAK >= 0 && idxAK < mnFtsAK)
            return mvMixedKPtsUn[0][idxAK];
    }

    cv::KeyPoint MixedFrame::getDistKPtMono(const int idx) const {

        if (idx < mnFtsORB)
            return mvKeys[idx];
        int idxAK = idx - mnFtsORB;
        if (idxAK >= 0 && idxAK < mnFtsAK)
            return mvMixedKPts[0][idxAK];
    }*/

    void MixedFrame::ComputeBoW() {

        if(mBowVec.empty())
        {
            vector<cv::Mat> vCurrentDesc = ORB_SLAM3::Converter::toDescriptorVector(this->getDescriptors(0));
            mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
        }
    }

    cv::Mat MixedFrame::getDescriptors(const int idxMix) {

        assert(idxMix >= 0 && idxMix < mvMixedDesc.size());
        return mvMixedDesc[idxMix];
    }

    cv::Mat MixedFrame::getORBDescriptor(const int idx) {

        assert(idx >= 0 && idx < N);
        if (idx < mnFtsORB)
            return mvMixedDesc[0].row(idx);
        return cv::Mat();
    }

    cv::Mat MixedFrame::getDescriptorMono(const int idx) {

        assert(idx >= 0 && idx < N);
        if (idx < mnFtsORB)
            return mvMixedDesc[0].row(idx);
        int idxAK = idx - mnFtsORB;
        if (idxAK >= 0 && idxAK < mnFtsAK)
            return mvMixedDesc[1].row(idxAK);
        return cv::Mat();
    }

    /*float MixedFrame::getLevelScaleFactor(const int orbLevel) const {

        int nOctave = 0, nOctaveLayer = 0;
        toOctaveLayerAK(orbLevel, nOctave, nOctaveLayer, mnLevelsAK, mnOctaves, mnOctaveLayers);
        assert(nOctave >= 0 && nOctave < mnOctaves && nOctaveLayer >= 0 && nOctaveLayer < mnOctaveLayers);
        return mvScaleFactorsAK[nOctave][nOctaveLayer];
    }*/

    float MixedFrame::getAKAZEScaleFactor(const int orbLevel) const {

        int nOctave = 0, nOctaveLayer = 0;
        toOctaveLayerAK(orbLevel, nOctave, nOctaveLayer, mnLevelsAK, mnOctaves, mnOctaveLayers);
        assert(nOctave >= 0 && nOctave < mnOctaves && nOctaveLayer >= 0 && nOctaveLayer < mnOctaveLayers);
        return mvScaleFactorsAK[nOctave][nOctaveLayer];
    }

    int MixedFrame::getKPtLevelMono(const int idx) const {

        assert(idx >= 0 && idx < N);
        if (idx < mnFtsORB)
            return Frame::getKPtLevelMono(idx);
        cv::KeyPoint kpt = this->getDistKPtMono(idx);
        assert(kpt.class_id >= 0);
        return kpt.class_id;
    }

    float MixedFrame::getKPtScaleFactor(const int idx) const {

        assert(idx >= 0 && idx < N);
//        if (!(idx >= 0 || idx < N)) {
//            cerr << "MixedFrame::getKPtScaleFactor -> Bad idx: " << idx << ", N = " << N << endl;
//            return mfScaleFactor;
//        }
        if (idx < mnFtsORB)
            return Frame::getKPtScaleFactor(idx);
        cv::KeyPoint kpt = this->getDistKPtMono(idx);
        assert(kpt.class_id >= 0);
//        if (kpt.class_id < 0 || kpt.class_id >= mnLevelsAK) {
//            cerr << "MixedFrame::getKPtScaleFactor -> Bad KPt, idx: " << idx << ", class_id = " << kpt.class_id << endl;
//            return kpt.octave;
//        }
        int nOctave = 0, nOctaveLayer = 0;
        toOctaveLayerAK(kpt.class_id, nOctave, nOctaveLayer, mnLevelsAK, mnOctaves, mnOctaveLayers);
        assert(kpt.octave >= 0 && kpt.octave < mnOctaves && nOctaveLayer >= 0 && nOctaveLayer < mnOctaveLayers);
//        if (!(kpt.octave >= 0 && kpt.octave < mnOctaves && nOctaveLayer >= 0 && nOctaveLayer < mnOctaveLayers)) {
//            cerr << "MixedFrame::getKPtScaleFactor -> Bad KPt, idx: " << idx << ", N = " << N << endl;
//            return mfScaleFactorAK;
//        }
        return mvScaleFactorsAK[kpt.octave][nOctaveLayer];
    }

    float MixedFrame::getKPtLevelSigma2(const int idx) const {

        assert(idx >= 0 && idx < N);
        if (idx < mnFtsORB)
            return Frame::getKPtLevelSigma2(idx);
        cv::KeyPoint kpt = this->getDistKPtMono(idx);
        assert(kpt.class_id >= 0);
        int nOctave = 0, nOctaveLayer = 0;
        toOctaveLayerAK(kpt.class_id, nOctave, nOctaveLayer, mnLevelsAK, mnOctaves, mnOctaveLayers);
        assert(kpt.octave >= 0 && kpt.octave < mnOctaves && nOctaveLayer >= 0 && nOctaveLayer < mnOctaveLayers);
        return mvLevelSigma2AK[kpt.octave][nOctaveLayer];
    }

    float MixedFrame::getKPtInvLevelSigma2(const int idx) const {

        assert(idx >= 0 && idx < N);
        if (idx < mnFtsORB)
            return Frame::getKPtInvLevelSigma2(idx);
        cv::KeyPoint kpt = this->getDistKPtMono(idx);
        assert(kpt.class_id >= 0);
        int nOctave = 0, nOctaveLayer = 0;
        toOctaveLayerAK(kpt.class_id, nOctave, nOctaveLayer, mnLevelsAK, mnOctaves, mnOctaveLayers);
        assert(kpt.octave >= 0 && kpt.octave < mnOctaves && nOctaveLayer >= 0 && nOctaveLayer < mnOctaveLayers);
        return mvInvLevelSigma2AK[kpt.octave][nOctaveLayer];
    }

    // Big Note: Seem like cv::KeyPoint's class_id for AKAZE holds equivalent ORB level

    int MixedFrame::toLevelORB(const int nOctave, const int nOctaveLayer, const int nOctaveLayers) {

        return nOctave * nOctaveLayers + nOctaveLayer;
    }

    void MixedFrame::toOctaveLayerAK(const int levelORB, int &nOctave, int &nOctaveLayer, const int nOctaveLayers) {

        nOctave = levelORB / nOctaveLayers;
        nOctaveLayer = levelORB - nOctave * nOctaveLayers;
    }

    void MixedFrame::toOctaveLayerAK(const int levelORB, int &nOctave, int &nOctaveLayer,
            const int nLevelsAK, const int nOctaves, const int nOctaveLayers) {

        int level = levelORB;
        if (level >= nLevelsAK) {
            level = nLevelsAK-1;
        }

        nOctave = level / nOctaveLayers;
        if (nOctave < 0)
            nOctave = 0;
        if (nOctave >= nOctaves)
            nOctave = nOctaves-1;

        nOctaveLayer = level - nOctave * nOctaveLayers;
        if (nOctaveLayer < 0)
            nOctaveLayer = 0;
        if (nOctaveLayer >= nOctaveLayers)
            nOctaveLayer = nOctaveLayers-1;
    }


    void MixedFrame::printClassIdAKAZE() {

        if (mvMixedKPts.size() > 1) {

            cout << "AKAZE class_ids: [";
            for (size_t i = 0; i < mvMixedKPts[1].size(); i++) {
                cout << mvMixedKPts[1][i].class_id;
                if (i == mvMixedKPts[1].size()-1) {
                    cout << "]\n";
                }
                else {
                    cout << ", ";
                }
            }
        }
    }

    void MixedFrame::printOctaveAKAZE() {

        if (mvMixedKPts.size() > 1) {

            cout << "AKAZE class_ids: [";
            for (size_t i = 0; i < mvMixedKPts[1].size(); i++) {
                cout << mvMixedKPts[1][i].octave;
                if (i == mvMixedKPts[1].size()-1) {
                    cout << "]\n";
                }
                else {
                    cout << ", ";
                }
            }
        }
    }

} //EORB_SLAM