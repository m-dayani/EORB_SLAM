//
// Created by root on 12/24/20.
//

#include "EventData.h"

//using namespace boost::filesystem;
//using namespace cv;
using namespace std;


namespace EORB_SLAM {

    double calcEventGenRate(const std::vector<EventData>& l1Evs, const int imWidth, const int imHeight) {

        double evTspan = l1Evs.back().ts-l1Evs[0].ts;
        double evGenRate = ((double) l1Evs.size()) / (evTspan * imWidth * imHeight);
        return evGenRate;
    }

    bool EvParams::parseParams(const std::string& settingsFile) {

        cv::FileStorage fSettings(settingsFile, cv::FileStorage::READ);
        if (!fSettings.isOpened())
        {
            cerr << "failed to open " << settingsFile << endl;
            return false;
        }
        return this->parseParams(fSettings);
    }

    bool EvParams::parseParams(const cv::FileStorage& fSettings) {

        bool b_miss_params = false;

        cv::FileNode node = fSettings["Camera.width"];
        if (!node.empty() && node.isInt()) {
            this->imWidth = (int) node;
        }
        else {
            cerr << "Cannot read image width\n";
            b_miss_params = true;
        }
        node = fSettings["Camera.height"];
        if (!node.empty() && node.isInt()) {
            this->imHeight = (int) node;
        }
        else {
            cerr << "Cannot read image height\n";
            b_miss_params = true;
        }
        node = fSettings["Event.data.isRectified"];
        if (!node.empty() && node.isString()) {
            if (node.string().compare("false") == 0) {
                this->isRectified = false;
                //cout << "isRectified = " << node.string() << endl;
            }
            else if (node.string().compare("true") == 0) {
                this->isRectified = true;
                //cout << "isRectified = " << node.string() << endl;
            }
        }
        else {
            cerr << "Cannot read event isRectified stat\n";
            b_miss_params = true;
        }
        node = fSettings["Event.l2TrackMode"];
        if (!node.empty() && node.isInt()) {
            this->l2TrackMode = (int) node;
        }
        else {
            cerr << "Cannot read l2 tracking mode\n";
            b_miss_params = true;
        }
        node = fSettings["Event.contTracking"];
        if (!node.empty() && node.isString()) {
            if (node.string().compare("false") == 0) {
                this->continTracking = false;
            }
            else if (node.string().compare("true") == 0) {
                this->continTracking = true;
            }
        }
        node = fSettings["Event.trackTinyFrames"];
        if (!node.empty() && node.isString()) {
            if (node.string().compare("false") == 0) {
                this->trackTinyFrames = false;
            }
            else if (node.string().compare("true") == 0) {
                this->trackTinyFrames = true;
            }
        }
        node = fSettings["Event.data.l1FixedWin"];
        if (!node.empty() && node.isString()) {
            if (node.string().compare("false") == 0) {
                this->l1FixedWinSz = false;
            }
            else if (node.string().compare("true") == 0) {
                this->l1FixedWinSz = true;
            }
        }
        else {
            cerr << "Cannot read fixedWin stat\n";
            b_miss_params = true;
        }
        node = fSettings["Event.data.l1ChunkSize"];
        if (!node.empty() && node.isInt()) {
            this->l1ChunkSize = (int) node;
        }
        else {
            cerr << "Cannot read event l1 chunk size\n";
            b_miss_params = true;
        }
        node = fSettings["Event.data.l1WinOverlap"];
        if (!node.empty() && node.isReal()) {
            this->l1WinOverlap = node.real();
        }
        node = fSettings["Event.data.l1NumLoop"];
        if (!node.empty() && node.isInt()) {
            this->l1NumLoop = (int) node;
        }
        else {
            cerr << "Cannot read event l1 num. loop\n";
            b_miss_params = true;
        }
        node = fSettings["Event.data.minEvGenRate"];
        if (!node.empty() && node.isReal()) {
            this->minEvGenRate = node.real();
        }
        else {
            cerr << "Cannot read l1 min event generation rate \n";
            b_miss_params = true;
        }
        node = fSettings["Event.data.maxEvGenRate"];
        if (!node.empty() && node.isReal()) {
            this->maxEvGenRate = node.real();
        }
        else {
            cerr << "Cannot read l1 max event event generation rate\n";
            b_miss_params = true;
        }
        node = fSettings["Event.data.maxPixelDisp"];
        if (!node.empty() && node.isReal()) {
            this->maxPixelDisp = node.real();
        }
        else {
            cerr << "Cannot read l1 max med. pixel displacement \n";
            b_miss_params = true;
        }
        node = fSettings["Event.image.l1Sigma"];
        if (!node.empty() && node.isReal()) {
            this->l1ImSigma = node.real();
        }
        else {
            cerr << "Cannot read l1 event image sigma\n";
            b_miss_params = true;
        }
        node = fSettings["Event.image.l2Sigma"];
        if (!node.empty() && node.isReal()) {
            this->l2ImSigma = node.real();
        }
        else {
            cerr << "Cannot read l2 event image sigma\n";
            b_miss_params = true;
        }
        node = fSettings["Event.fts.detMode"];
        if (!node.empty() && node.isInt()) {
            this->detMode = (int) node;
        }
        else {
            cerr << "Cannot read fts detection mode \n";
            b_miss_params = true;
        }
        node = fSettings["Event.fts.fastTh"];
        if (!node.empty() && node.isInt()) {
            this->fastTh = (int) node;
        }
        else {
            cerr << "Cannot read fts fastTh \n";
            b_miss_params = true;
        }
        node = fSettings["Event.fts.maxNumPts"];
        if (!node.empty() && node.isInt()) {
            this->maxNumPts = (int) node;
        }
        else {
            cerr << "Cannot read fts max num. points \n";
            b_miss_params = true;
        }
        if (!b_miss_params && this->detMode != 0) {

            node = fSettings["Event.fts.l1NLevels"];
            if (!node.empty() && node.isInt()) {
                this->l1NLevels = (int) node;
            }
            else {
                cerr << "Cannot read l1 nLevels \n";
                b_miss_params = true;
            }
            node = fSettings["Event.fts.l2NLevels"];
            if (!node.empty() && node.isInt()) {
                this->l2NLevels = (int) node;
            }
            else {
                cerr << "Cannot read l2 nLevels \n";
                b_miss_params = true;
            }
            node = fSettings["Event.fts.l1ScaleFactor"];
            if (!node.empty() && node.isReal()) {
                this->l1ScaleFactor = node.real();
            }
            else {
                cerr << "Cannot read l1 scale factor \n";
                b_miss_params = true;
            }
            node = fSettings["Event.fts.l2ScaleFactor"];
            if (!node.empty() && node.isReal()) {
                this->l2ScaleFactor = node.real();
            }
            else {
                cerr << "Cannot read l2 scale factor \n";
                b_miss_params = true;
            }
        }
        node = fSettings["Event.klt.maxLevel"];
        if (!node.empty() && node.isInt()) {
            this->maxLevel = (int) node;
        }
        else {
            cerr << "Cannot read klt max level \n";
            b_miss_params = true;
        }
        node = fSettings["Event.klt.winSize"];
        if (!node.empty() && node.isInt()) {
            this->kltWinSize = (int) node;
        }
        else {
            cerr << "Cannot read klt win size \n";
            b_miss_params = true;
        }
        node = fSettings["Event.klt.maxIter"];
        if (!node.empty() && node.isInt()) {
            this->kltMaxItr = (int) node;
        }
        else {
            cerr << "Cannot read klt max iteration \n";
            b_miss_params = true;
        }
        node = fSettings["Event.klt.eps"];
        if (!node.empty() && node.isReal()) {
            this->kltEps = node.real();
        }
        else {
            cerr << "Cannot read klt epsilon \n";
            b_miss_params = true;
        }
        node = fSettings["Event.klt.maxThRefreshPoints"];
        if (!node.empty() && node.isReal()) {
            this->kltMaxThRefreshPts = node.real();
        }
        else {
            cerr << "Cannot read klt max thresh refresh points \n";
            b_miss_params = true;
        }
        node = fSettings["Event.klt.distRefreshPoints"];
        if (!node.empty() && node.isReal()) {
            this->kltMaxDistRefreshPts = node.real();
        }
        else {
            cerr << "Cannot read klt max dist refresh points \n";
            b_miss_params = true;
        }

        this->missParams = b_miss_params;
        return !b_miss_params;
    }

    string EvParams::printParams() const {

        ostringstream oss;

        if (isRectified)
            oss << "- Event coordinates are rectified\n";
        else
            oss << "- Event coordinates are not rectified\n";

        oss << "- Level 2 Tracking Mode: ";
        switch (l2TrackMode) {
            case 1:
                oss << "Tracking local map\n";
                break;
            case 2:
                oss << "Tracking local map with ref. change\n";
                break;
            default:
                oss << "Pure odometry\n";
                break;
        }

        if (continTracking)
            oss << "- Continuous Tracking (Ultimate SLAM?)\n";
        else
            oss << "- ORB-SLAM Style Tracking\n";

        oss << "- Use Tiny Frames? " << ((trackTinyFrames) ? "Yes" : "No") << endl;

        oss << "- Fixed Event Window Length? " << ((l1FixedWinSz) ? "Yes" : "No") << endl;
        oss << "- Level 1 Num. Events: " << l1ChunkSize << endl;
        oss << "- Level 1 Window Overlap (pct): " << l1WinOverlap << endl;
        oss << "- Level 1 Num. Loops: " << l1NumLoop << endl;

        oss << "- Level 1 Min Event Gen. Rate: " << minEvGenRate << endl;
        oss << "- Level 1 Max Event Gen. Rate: " << maxEvGenRate << endl;
        oss << "- Level 1 Max Pixel Displacement " << maxPixelDisp << endl;

        oss << "- Level 1 Image reconstruction std: " << l1ImSigma << endl;
        oss << "- Level 2 Image reconstruction std: " << l2ImSigma << endl;

        oss << "- Feature detection mode: ";
        switch (detMode) {
            case 1:
                oss << "ORB";
                break;
            case 2:
                oss << "Mixed";
                break;
            case 0:
            default:
                oss << "FAST";
                break;
        }
        oss << endl;
        oss << "- Fast Threshold: " << fastTh << endl;
        oss << "- Max. Num. Points: " << maxNumPts << endl;
        oss << "- Level 1 Num. Levels: " << l1NLevels << endl;
        oss << "- Level 1 Scale Factor: " << l1ScaleFactor << endl;
        oss << "- Level 2 Num. Levels: " << l2NLevels << endl;
        oss << "- Level 2 Scale Factor: " << l2ScaleFactor << endl;

        oss << "- KLT Epsilon: " << kltEps << endl;
        oss << "- KLT Max Level: " << maxLevel << endl;
        oss << "- KLT Win Size: " << kltWinSize << endl;
        oss << "- KLT Zero-based Max Iterations: " << kltMaxItr << endl;
        oss << "- KLT Max thresh refresh points: " << kltMaxThRefreshPts << endl;
        oss << "- KLT Max distance refresh points: " << kltMaxDistRefreshPts << endl;

        oss << "- Image width (image reconstruction): " << imWidth << endl;
        oss << "- Image height (image reconstruction): " << imHeight << endl;

        return oss.str();
    }

    /* ------------------------------------------------------------------------------------------------------------- */

    unsigned long EventQueue::consumeBegin(unsigned long chunkSize, vector<EventData> &evs, vector<EventData> &accEvs) {

        std::unique_lock<std::mutex> lock1(mMutexBuffer);
        if (mQueue.empty())
            return 0;

        unsigned nEvs = min(chunkSize, mQueue.size());
        evs.resize(nEvs);

        for (unsigned i = 0; i < nEvs; i++) {
            EventData ev = mQueue.front();
            evs[i] = ev;
            accEvs.push_back(ev);
            mQueue.pop();
        }
        return nEvs;
    }

} //EORB_SLAM
