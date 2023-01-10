//
// Created by root on 11/19/21.
//

#ifndef ORB_SLAM3_MYFRAMEDRAWER_H
#define ORB_SLAM3_MYFRAMEDRAWER_H

#include <vector>
#include <queue>
#include <memory>
#include <opencv2/core.hpp>

#include "Atlas.h"
#include "Frame.h"


namespace EORB_SLAM {

    enum FrameDrawerState {
        IDLE,
        INIT_TRACK,
        INIT_MAP,
        TRACKING
    };

    class FrameDrawFilter {
    public:
        FrameDrawFilter() = default;

        virtual void drawFrame(cv::Mat& im, FrameDrawerState state, const ORB_SLAM3::FramePtr& pCurrFr, const ORB_SLAM3::FramePtr& pIniFr) = 0;
        virtual void drawFrameInfo(cv::Mat& im, FrameDrawerState state, const ORB_SLAM3::FramePtr& pFr) = 0;

        static void drawKeyPoint(cv::Mat& im, const cv::KeyPoint& kpt, float r, const cv::Scalar& color);
        static void drawKeyPoints(cv::Mat& im, const std::vector<cv::KeyPoint>& vCurrKeyPts);
        static void drawKeyPointMatches(cv::Mat& im, const std::vector<cv::KeyPoint>& vIniKeyPts,
                const std::vector<cv::KeyPoint>& vCurrKeyPts, const std::vector<int>& vMatches12);
        static void drawMapPointMatches(cv::Mat& im, const std::vector<cv::KeyPoint>& vCurrKeyPts,
                const std::vector<int>& vMatches12, const std::vector<bool>& vbOutlierMPts);

        static void drawTextLines(cv::Mat& textIm, const std::vector<std::stringstream>& msg, int cols);

        static void makeNoImage(cv::Mat& im, int rows, int cols);

    protected:
        std::mutex mMtxDraw;

    private:

    };

    typedef std::shared_ptr<FrameDrawFilter> FrameDrawFilterPtr;

    class MyFrameDrawer {
    public:
        explicit MyFrameDrawer(ORB_SLAM3::Atlas* pAtlas) : mnChannels(0), mbOnlyTracking(false), mpAtlas(pAtlas) {}
        ~MyFrameDrawer() { mpAtlas = nullptr; }

        int requestNewChannel(const FrameDrawFilterPtr& filter);

        void setChannelDrawFilter(int chId, const FrameDrawFilterPtr& filter);

        void updateAtlas(ORB_SLAM3::Atlas* pAtlas);
        void updateLastTrackerState(int chId, FrameDrawerState state);
        void updateIniFrame(int chId, const ORB_SLAM3::FramePtr& pIniFr);

        void pushNewFrame(int chId, const ORB_SLAM3::FramePtr& pIniFr);

        void draw(cv::Mat& toShow);

    protected:
        bool checkChannelId(int chId) const;

    private:
        int mnChannels;
        std::vector<FrameDrawerState> mvStats;
        bool mbOnlyTracking;

        std::mutex mMtxChannel;
        std::vector<std::unique_ptr<std::mutex>> mvpMtxUpdate;
        std::vector<std::queue<ORB_SLAM3::FramePtr>> mvqCurrFrames;
        std::vector<ORB_SLAM3::FramePtr> mvIniFrames, mvLastFrames;
        std::vector<FrameDrawFilterPtr> mvFilters;

        ORB_SLAM3::Atlas* mpAtlas;
    };

    class SimpleFrameDrawFilter : public FrameDrawFilter {
    public:

        void drawFrame(cv::Mat &im, FrameDrawerState state, const ORB_SLAM3::FramePtr &pCurrFr,
                       const ORB_SLAM3::FramePtr &pIniFr) override;

        void drawFrameInfo(cv::Mat &im, FrameDrawerState state, const ORB_SLAM3::FramePtr &pFr) override;

    private:
    };

} // EORB_SLAM


#endif //ORB_SLAM3_MYFRAMEDRAWER_H
