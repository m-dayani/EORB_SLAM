//
// Created by root on 11/19/21.
//

#include "MyFrameDrawer.h"

using namespace std;

namespace EORB_SLAM {


    void FrameDrawFilter::drawTextLines(cv::Mat &textIm, const std::vector<std::stringstream>&msg, int cols) {

        int nMsgs = msg.size();
        if (nMsgs <= 0 || cols <= 0)
            return;

        int baseline=0;
        cv::Size textSize = cv::getTextSize(msg[0].str(),cv::FONT_HERSHEY_PLAIN,1,1,&baseline);

        textIm = cv::Mat::zeros(nMsgs*(textSize.height+5)+5,cols,CV_8UC3);

        for (int i = 0; i < nMsgs; i++) {
            cv::putText(textIm, msg[i].str(), cv::Point(5, (i+1)*(textSize.height+5)),cv::FONT_HERSHEY_PLAIN,
                    1,cv::Scalar(255, 255, 255), 1, 8);
        }
    }

    void FrameDrawFilter::drawKeyPoint(cv::Mat &im, const cv::KeyPoint &kpt, float r, const cv::Scalar& color) {

        cv::Point2f pt1,pt2;
        pt1.x=kpt.pt.x-r;
        pt1.y=kpt.pt.y-r;
        pt2.x=kpt.pt.x+r;
        pt2.y=kpt.pt.y+r;

        cv::rectangle(im,pt1,pt2,color);
        cv::circle(im,kpt.pt,2,color,-1);
    }

    void FrameDrawFilter::drawKeyPoints(cv::Mat &im, const vector<cv::KeyPoint> &vCurrKeyPts) {

        for (const cv::KeyPoint& kpt : vCurrKeyPts) {

            drawKeyPoint(im, kpt, 5.f, cv::Scalar(0, 255, 0));
        }
    }

    void FrameDrawFilter::drawKeyPointMatches(cv::Mat &im, const vector<cv::KeyPoint> &vIniKeyPts,
                                              const vector<cv::KeyPoint> &vCurrKeyPts, const vector<int> &vMatches12) {

        for(unsigned int i=0; i<vMatches12.size(); i++) {

            if(vMatches12[i]>=0) {

                cv::line(im, vIniKeyPts[i].pt, vCurrKeyPts[vMatches12[i]].pt, cv::Scalar(0,255,0));
            }
        }
    }

    void FrameDrawFilter::drawMapPointMatches(cv::Mat &im, const vector<cv::KeyPoint> &vCurrKeyPts,
                                              const vector<int> &vMatches12, const vector<bool> &vbOutlierMPts) {

        for(int mchIdx : vMatches12) {

            if(mchIdx>=0) {

                if (vbOutlierMPts[mchIdx]) {
                    drawKeyPoint(im, vCurrKeyPts[mchIdx], 5.f, cv::Scalar(0, 0, 255));
                }
                else {
                    drawKeyPoint(im, vCurrKeyPts[mchIdx], 5.f, cv::Scalar(0, 255, 0));
                }
            }
        }
    }

    void FrameDrawFilter::makeNoImage(cv::Mat &im, int rows, int cols) {

        im = cv::Mat::zeros(rows, cols, CV_8UC3);

        if(im.channels()<3)
            cvtColor(im,im,CV_GRAY2BGR);

        const string text = "No Image";
        int baseline=0;
        double fscale = 3;
        int thickness = 3;
        cv::Size textSize = cv::getTextSize(text,cv::FONT_HERSHEY_PLAIN,fscale,thickness,&baseline);

        cv::putText(im, text, cv::Point((cols - textSize.width)/2, (rows - textSize.height)/2),
                cv::FONT_HERSHEY_PLAIN, fscale,cv::Scalar(255, 255, 255), thickness, 8);
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    bool MyFrameDrawer::checkChannelId(const int chId) const {

        return chId >= 0 && chId < mnChannels;
    }

    int MyFrameDrawer::requestNewChannel(const FrameDrawFilterPtr& filter) {

        std::unique_lock<mutex> lock1(mMtxChannel);

        mnChannels = mvStats.size();

        // Only up to 4 channels supported
        if (mnChannels >= 4)
            return -1;

        mnChannels++;

        mvStats.resize(mnChannels);
        mvpMtxUpdate.reserve(mnChannels);
        mvpMtxUpdate.push_back(unique_ptr<mutex>(new mutex()));
        mvqCurrFrames.resize(mnChannels);
        mvIniFrames.resize(mnChannels);
        mvLastFrames.resize(mnChannels);
        mvFilters.resize(mnChannels);
        mvFilters[mnChannels-1] = filter;

        return mnChannels-1;
    }

    void MyFrameDrawer::setChannelDrawFilter(const int chId, const FrameDrawFilterPtr& filter) {

        if (checkChannelId(chId)) {
            unique_lock<mutex> lock1(*(mvpMtxUpdate[chId]));
            mvFilters[chId] = filter;
        }
    }

    void MyFrameDrawer::updateAtlas(ORB_SLAM3::Atlas *pAtlas)  { mpAtlas = pAtlas; }

    void MyFrameDrawer::updateLastTrackerState(int chId, FrameDrawerState state) {

        if (checkChannelId(chId)) {
            unique_lock<mutex> lock1(*(mvpMtxUpdate[chId]));
            mvStats[chId] = state;
        }
    }

    void MyFrameDrawer::updateIniFrame(int chId, const ORB_SLAM3::FramePtr& pIniFr) {

        if (checkChannelId(chId)) {
            unique_lock<mutex> lock1(*(mvpMtxUpdate[chId]));
            mvIniFrames[chId] = pIniFr;
        }
    }

    void MyFrameDrawer::pushNewFrame(int chId, const ORB_SLAM3::FramePtr& pCurrFr) {

        if (checkChannelId(chId)) {
            unique_lock<mutex> lock1(*(mvpMtxUpdate[chId]));
            mvqCurrFrames[chId].push(pCurrFr);
        }
    }

    void MyFrameDrawer::draw(cv::Mat &toShow) {

        //if (mnChannels <= 0)
        //    return;

        // Retrieve & draw each image according to channel criteria
        vector<cv::Mat> vImagesToShow(mnChannels);

        for (int ch = 0; ch < mnChannels; ch++) {

            unique_lock<mutex> lock1(*(mvpMtxUpdate[ch]));

            if (!mvqCurrFrames[ch].empty()) {
                mvLastFrames[ch] = mvqCurrFrames[ch].front();
                mvqCurrFrames[ch].pop();
            }

            mvFilters[ch]->drawFrame(vImagesToShow[ch], mvStats[ch], mvLastFrames[ch], mvIniFrames[ch]);
        }

        // A final touch and combine all images into a master image
        int maxRows = 0, maxCols = 0;
        for (int i = 0; i < mnChannels; i++) {

            cv::Mat currIm = vImagesToShow[i];
            if (!currIm.empty()) {
                if (maxRows < currIm.rows)
                    maxRows = currIm.rows;
                if (maxCols < currIm.cols)
                    maxCols = currIm.cols;
            }
        }

        if (mnChannels > 2) {
            toShow = cv::Mat(maxRows*2, maxCols*2, vImagesToShow[0].type());

            for (int i = 0; i < mnChannels; i++) {

                cv::Mat currIm = vImagesToShow[i];
                int currRows = currIm.rows, currCols = currIm.cols;
                currIm.copyTo(toShow.rowRange(i*maxRows, i*maxRows+currRows).colRange(i*maxCols, i*maxCols+currCols));
            }
        }
        else {
            if (mnChannels == 1) {
                toShow = vImagesToShow[0].clone();
            }
            else if (mnChannels == 2){
                toShow = cv::Mat(maxRows, maxCols*2, vImagesToShow[0].type());

                cv::Mat im0 = vImagesToShow[0], im1 = vImagesToShow[1];
                int rows0 = im0.rows, cols0 = im0.cols, rows1 = im1.rows, cols1 = im1.cols;

                vImagesToShow[0].copyTo(toShow.rowRange(0, rows0).colRange(0, cols0));
                vImagesToShow[1].copyTo(toShow.rowRange(0, rows1).colRange(maxCols, maxCols+cols1));
            }
            else {
                FrameDrawFilter::makeNoImage(toShow, 144, 240);
            }
        }

        // Put general info under the master image
        int nMaps = mpAtlas->CountMaps();
        ulong nKFs = mpAtlas->KeyFramesInMap();
        ulong nMPs = mpAtlas->MapPointsInMap();

        vector<stringstream> vMsgs(1);
        vMsgs[0] << "Maps: " << nMaps << ", KFs: " << nKFs << ", MPs: " << nMPs;

        cv::Mat txtIm;
        FrameDrawFilter::drawTextLines(txtIm, vMsgs, toShow.cols);

        cv::vconcat(toShow, txtIm, toShow);
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    void SimpleFrameDrawFilter::drawFrame(cv::Mat &im, FrameDrawerState state, const ORB_SLAM3::FramePtr &pCurrFr,
                                          const ORB_SLAM3::FramePtr &pIniFr) {

        // Copy important variables
        vector<cv::KeyPoint> vIniKeyPts, vCurrKeyPts;
        vector<ORB_SLAM3::MapPoint*> vCurrMPts;
        vector<bool> vbOutlierMPts;
        vector<int> vMatches12;

        {
            std::unique_lock<mutex> lock1(mMtxDraw);

            if (!pCurrFr || pCurrFr->imgLeft.empty()) {

                makeNoImage(im, 180, 240);
                return;
            }
            if (pCurrFr) {
                im = pCurrFr->imgLeft.clone();
                vCurrKeyPts = pCurrFr->getAllDistKPtsMono();
                vCurrMPts = pCurrFr->getAllMapPointsMono();
                vbOutlierMPts = pCurrFr->getAllOutliersMono();
                vMatches12 = pCurrFr->getMatches();
            }
            if (pIniFr) {
                vIniKeyPts = pIniFr->getAllDistKPtsMono();
            }
        }

        if(im.channels()<3) //this should be always true
            cvtColor(im,im,CV_GRAY2BGR);

        // Draw the image based on current state
        switch (state) {
            case FrameDrawerState::INIT_TRACK:
                drawKeyPoints(im, vCurrKeyPts);
                break;
            case FrameDrawerState::INIT_MAP:
                drawKeyPointMatches(im, vIniKeyPts, vCurrKeyPts, vMatches12);
                break;
            case FrameDrawerState::TRACKING:
                drawMapPointMatches(im, vCurrKeyPts, vMatches12, vbOutlierMPts);
                break;
            case FrameDrawerState::IDLE:
            default:
                makeNoImage(im, 180, 240);
                DLOG(INFO) << "SimpleFrameDrawFilter::drawFrame: IDLE state, no image yet\n";
                break;
        }

        // put text info
        this->drawFrameInfo(im, state, pCurrFr);
    }

    void SimpleFrameDrawFilter::drawFrameInfo(cv::Mat &im, FrameDrawerState state, const ORB_SLAM3::FramePtr &pFr) {

        const int imCols = im.cols;

        double imTs = 0;
        ulong nKPts = 0;

        {
            std::unique_lock<mutex> lock1(mMtxDraw);
            imTs = pFr->mTimeStamp;
            nKPts = pFr->numAllKPts();
        }

        string statStr;
        switch (state) {
            case FrameDrawerState::INIT_TRACK:
                statStr = "Init. Tracker";
                break;
            case FrameDrawerState::INIT_MAP:
                statStr = "Init. Map";
                break;
            case FrameDrawerState::TRACKING:
                statStr = "Tracking";
                break;
            case FrameDrawerState::IDLE:
            default:
                statStr = "Idle";
                break;
        }

        vector<stringstream> vMsgs;

        vMsgs.resize(1);
        vMsgs[0].precision(6);
        vMsgs[0] << "ts: " << imTs << ", nKPts: " << nKPts << ", " << statStr;

        cv::Mat textIm;
        drawTextLines(textIm, vMsgs, imCols);

        cv::vconcat(im, textIm, im);
    }
} // EORB_SLAM