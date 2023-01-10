//
// Created by root on 11/25/20.
//

#include "KLT_Tracker.h"


using namespace cv;
using namespace std;


namespace EORB_SLAM
{
    ELK_Tracker::ELK_Tracker(const shared_ptr<EvParams>&  params) : //mpEvParams(std::move(params)),
            mPatchSz(params->kltWinSize), mMaxLevel(params->maxLevel) {

        mLKCriteria = TermCriteria((TermCriteria::COUNT) +
                        (TermCriteria::EPS), params->kltMaxItr, params->kltEps);
    }

    void ELK_Tracker::setRefImage(const cv::Mat& image, const vector<KeyPoint>& refPts) {

        assert(!image.empty() && !refPts.empty());

        this->mRefFrame = image.clone();

        int nPts = refPts.size();
        mRefKPoints.resize(nPts);
        mLastTrackedKPts.resize(nPts);
        mRefPoints.resize(nPts);
        mLastTrackedPts.resize(nPts);

        // Convert key points
        for (size_t i = 0; i < nPts; i++) {

            cv::KeyPoint kpt = refPts[i];
            cv::Point2f pt = kpt.pt;

            mRefKPoints[i] = kpt;
            mLastTrackedKPts[i] = kpt;
            mRefPoints[i] = pt;
            mLastTrackedPts[i] = pt;
        }

        //dummyCntMatches.clear();
        //dummyPxDisps.clear();
    }

    void ELK_Tracker::trackCurrImage(const cv::Mat& currImage,
                                     vector<Point2f> &kpts, vector<uchar> &status, vector<float> &err) {

        if (mRefPoints.empty() || mRefFrame.empty()) {

            cerr << "ELK_Tracker::trackCurrImage: First set reference image and key points.\n";
            return;
        }

        assert(!currImage.empty());

        // calculate optical flow
        try {
            if (kpts.size() == mRefPoints.size()){
                calcOpticalFlowPyrLK(this->mRefFrame, currImage, mRefPoints, kpts, status, err,
                        Size(mPatchSz, mPatchSz), mMaxLevel, mLKCriteria, OPTFLOW_USE_INITIAL_FLOW);
            }
            else {
                kpts.clear();
                calcOpticalFlowPyrLK(this->mRefFrame, currImage, mRefPoints, kpts, status, err,
                                     Size(mPatchSz, mPatchSz), mMaxLevel, mLKCriteria);
            }
        }
        catch (exception& ex) {
            cout << ex.what() << endl;
        }
    }

    // Use previously tracked points for faster performance (with OPTFLOW_USE_INITIAL_FLOW flag)
    void ELK_Tracker::trackCurrImage(const cv::Mat &currImage, const std::vector<cv::KeyPoint> &initPts,
            std::vector<cv::Point2f> &kpts, std::vector<uchar> &status, std::vector<float> &err) {

        assert(!mRefPoints.empty() && !mRefFrame.empty() && !currImage.empty() && initPts.size() == mRefPoints.size());

        // Convert key points
        size_t nPts = mRefPoints.size();
        kpts.resize(nPts);
        for (size_t i = 0; i < nPts; i++) {
            kpts[i] = initPts[i].pt;
        }
        // calculate optical flow
        try {
            calcOpticalFlowPyrLK(this->mRefFrame, currImage, mRefPoints, kpts, status, err,
                    Size(mPatchSz, mPatchSz), mMaxLevel, mLKCriteria, OPTFLOW_USE_INITIAL_FLOW);
        }
        catch (exception& ex) {
            cout << ex.what() << endl;
        }
        //this->mCurrPoints = kpts;
    }

    bool isInImage(const cv::Point2f& pt, int imWidth, int imHeight) {

        return pt.x >= 0 && pt.x < (float)imWidth && pt.y >= 0 && pt.y < (float)imHeight;
    }

    unsigned ELK_Tracker::refineTrackedPts(const vector<Point2f>& currPts, const vector<uchar>& status,
            const vector<float>& err, vector<KeyPoint>& p1, vector<int>& vMatches12,
            vector<int>& vCntMatches, std::vector<float>& vPxDisp) {

        assert(currPts.size() == status.size() && currPts.size() == err.size());
        assert(!mRefPoints.empty() && mRefPoints.size() == currPts.size());

        size_t nRefKpts = mRefPoints.size();

        if (vCntMatches.empty()) {
            vCntMatches.resize(nRefKpts, 1);
        }
        else {
            assert(vCntMatches.size() == nRefKpts);
        }
        if (vPxDisp.empty()) {
            vPxDisp.reserve(nRefKpts);
        }
        else {
            //assert(vPxDisp.size() == nRefKpts);
            DLOG(WARNING) << "ELK_Tracker::refineTrackedPts: vPxDisp vector is not empty!\n";
            vPxDisp.clear();
            vPxDisp.reserve(nRefKpts);
        }

        p1.resize(nRefKpts);
        vMatches12.resize(nRefKpts, -1);
        unsigned nMatches = 0;
        int M = mRefFrame.cols;
        int N = mRefFrame.rows;

        for (uint i = 0; i < nRefKpts; i++) {

            // Convert points
            Point2f currPt = currPts[i];
            KeyPoint prePt = mRefKPoints[i];

            p1[i] = KeyPoint(currPt, prePt.size, prePt.angle, prePt.response, prePt.octave, prePt.class_id);

            if (status[i] == 1 && isInImage(currPt, M, N)) {

                vCntMatches[i]++;
                vMatches12[i] = i;
                nMatches++;

                // Calculate pixel displacement only for matches.
                vPxDisp.push_back(sqrtf(powf((currPt.x - prePt.pt.x), 2)+powf((currPt.y - prePt.pt.y), 2)));
            }
        }
        return nMatches;
    }

    unsigned ELK_Tracker::refineFirstOctaveLevel(std::vector<cv::KeyPoint> &trackedKPts, std::vector<int> &vMatches12,
            unsigned& nMatches, std::vector<int> &vCntMatches) {

        int nTrackedPts = trackedKPts.size();
        int nvMatches = vMatches12.size();
        assert(!trackedKPts.empty() && nvMatches == mRefKPoints.size() && nvMatches == vCntMatches.size());

        for (size_t i = 0; i < nvMatches; i++) {

            int currIdx = vMatches12[i];

            if (currIdx >= 0 && currIdx < nTrackedPts) {

                cv::KeyPoint refKpt = mRefKPoints[i];
                //cv::KeyPoint currKpt = trackedKPts[vMatches12[i]];

                // Seems only the level for refs must be zero
                if (refKpt.octave > 0) { // || currKpt.octave > 0) {
                    vMatches12[i] = -1;
                    vCntMatches[i]--;
                    nMatches--;
                }
            }
        }
        return nMatches;
    }

    unsigned ELK_Tracker::refineFirstOctaveLevel(const std::vector<cv::KeyPoint>& refKPts, const std::vector<cv::KeyPoint> &trackedKPts,
                                                 std::vector<int> &vMatches12, unsigned& nMatches) {

        int nTrackedPts = trackedKPts.size();
        int nvMatches = vMatches12.size();
        assert(!trackedKPts.empty() && nvMatches == refKPts.size());

        for (size_t i = 0; i < nvMatches; i++) {

            int currIdx = vMatches12[i];

            if (currIdx >= 0 && currIdx < nTrackedPts) {

                cv::KeyPoint refKpt = refKPts[i];
                //cv::KeyPoint currKpt = trackedKPts[currIdx];

                // Seems only the level for refs must be zero
                if (refKpt.octave > 0) { // || currKpt.octave > 0) {
                    vMatches12[i] = -1;
                    nMatches--;
                }
            }
        }
        return nMatches;
    }

    //vector<int> ELK_Tracker::dummyCntMatches = vector<int>();
    //vector<float> ELK_Tracker::dummyPxDisps = vector<float>();

    // Track and match using last tracked kpts
    // This automatically retains last state and tracks using last tracked points
    unsigned int ELK_Tracker::trackAndMatchCurrImage(const cv::Mat &image, std::vector<cv::KeyPoint> &trackedKPts,
            std::vector<int> &vMatches12, std::vector<int> &vCntMatches, std::vector<float> &vPxDisp) {

        if (mRefKPoints.empty() || mRefFrame.empty()) {
            LOG(WARNING) << "trackAndMatchCurrImage: No reference info., did you forget to init. tracker??\n";
            return 0;
        }

        // LK Optical Flow matching
        vector<uchar> status;
        vector<float> err;
        this->trackCurrImage(image, mLastTrackedPts, status, err);

        // Reject outliers using LK method status
        unsigned nMatches = this->refineTrackedPts(mLastTrackedPts, status, err, trackedKPts, vMatches12,
                                                   vCntMatches, vPxDisp);
        mLastTrackedKPts = trackedKPts;

        return nMatches;
    }

    unsigned int ELK_Tracker::trackAndMatchCurrImage(const cv::Mat &image, std::vector<cv::KeyPoint> &trackedKPts,
            std::vector<int> &vMatches12) {

        vector<int> vCntMatches;
        vector<float> vPxDisp;
        return this->trackAndMatchCurrImage(image, trackedKPts, vMatches12, vCntMatches, vPxDisp);
    }

    // This only matches the key points in first level
    unsigned int ELK_Tracker::trackAndMatchCurrImageInit(const cv::Mat &image, std::vector<cv::KeyPoint> &trackedKPts,
            std::vector<int> &vMatches12, std::vector<int>& vCntMatches, std::vector<float>& vPxDisp) {

        unsigned nmatches = this->trackAndMatchCurrImage(image, trackedKPts, vMatches12, vCntMatches, vPxDisp);

        return this->refineFirstOctaveLevel(trackedKPts, vMatches12, nmatches, vCntMatches);
    }

    void ELK_Tracker::setLastTrackedPts(const std::vector<cv::KeyPoint> &currTrackedPts) {

        int nPts = currTrackedPts.size();
        mLastTrackedKPts.resize(nPts);
        mLastTrackedPts.resize(nPts);

        for (size_t i = 0; i < nPts; i++) {

            mLastTrackedKPts[i] = currTrackedPts[i];
            mLastTrackedPts[i] = currTrackedPts[i].pt;
        }
    }

}
