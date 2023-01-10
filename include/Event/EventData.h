//
// Created by root on 12/24/20.
//

#ifndef ORB_SLAM3_EVENTDATA_H
#define ORB_SLAM3_EVENTDATA_H

#include <vector>
#include <queue>
#include <memory>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "DataStore.h"


namespace EORB_SLAM {

#define DEF_L1_CHUNK_SIZE 2000
#define DEF_L1_NUM_LOOP 5
#define DEF_IMG_SIGMA 1.0f
#define DEF_EV_WIN_MIN_SZ 2000
#define DEF_TH_MIN_KPTS 100
#define DEF_TH_MAX_KPTS 200
#define DEF_TH_MIN_MATCHES 50

#define DEF_LOW_FTS_THRESH 0.9f
#define DEF_TH_EV_FAST_DET 1
#define DEF_LK_ITR_MAX_CNT 10
#define DEF_LK_TERM_EPS 0.03f
#define DEF_LK_WIN_SIZE 15
#define DEF_LK_MAX_LEVEL 1
#define DEF_DIST_REFRESH_PTS 1.f

    struct EventData {

        EventData() = default;

        EventData(double ts, float x, float y, bool p) :
                ts(ts), x(x), y(y), p(p) {}

        void print() const {
            std::cout << "[ts, (x,y), pol]: [" << ts << ", (" << x << ", " << y << "), " << p << "]\n";
        }

        /*static bool isInImage(const EventData& ev, const float imWidth, const float imHeight) {
            return ev.x >= 0 && ev.x < imWidth && ev.y >= 0 && ev.y < imHeight;
        }
        bool isInImage(const float imWidth, const float imHeight) const {
            return x >= 0 && x < imWidth && y >= 0 && y < imHeight;
        }*/

        double ts = 0.0;
        float x = 0.f;
        float y = 0.f;
        bool p = false;
    };

    double calcEventGenRate(const std::vector<EventData>& l1Chunk, int imWidth, int imHeight);

    struct EventStat {

        unsigned int mnCurrFileIdx;
        unsigned long mnNextByteIdx;

        double evRate;
        double avgSpDistr;
        double avgVoxDistr;
        std::vector<int> vSpDistr;
        std::vector<int> vVoxDistr;
        cv::Point3f mGrid;
    };

    struct EvParams {

        EvParams() : l2TrackMode(0), l1FixedWinSz(true), l1ChunkSize(DEF_L1_CHUNK_SIZE), l1NumLoop(DEF_L1_NUM_LOOP),
                    minEvGenRate(1.f), maxEvGenRate(1.f), maxPixelDisp(3),
                    l1ImSigma(DEF_IMG_SIGMA), l2ImSigma(DEF_IMG_SIGMA), detMode(0), fastTh(DEF_TH_EV_FAST_DET),
                    maxNumPts(DEF_TH_MAX_KPTS), maxLevel(DEF_LK_MAX_LEVEL), kltWinSize(DEF_LK_WIN_SIZE),
                    kltEps(DEF_LK_TERM_EPS), kltMaxItr(DEF_LK_ITR_MAX_CNT), kltMaxThRefreshPts(DEF_LOW_FTS_THRESH),
                    kltMaxDistRefreshPts(DEF_DIST_REFRESH_PTS), isRectified(false), missParams(false) {}

        int l2TrackMode;
        bool continTracking = false;
        bool trackTinyFrames = false;
        bool l1FixedWinSz;
        unsigned int l1ChunkSize;
        float l1WinOverlap = 0.5f;
        unsigned int l1NumLoop;
        float minEvGenRate;
        float maxEvGenRate;
        float maxPixelDisp;
        unsigned int imWidth;
        unsigned int imHeight;
        float l1ImSigma;
        float l2ImSigma;

        int detMode;
        int fastTh;
        int maxNumPts;
        float l1ScaleFactor = 1.f;
        float l2ScaleFactor = 1.f;
        int l1NLevels = 1;
        int l2NLevels = 1;

        int maxLevel;
        int kltWinSize;
        float kltEps;
        int kltMaxItr;
        float kltMaxThRefreshPts;
        float kltMaxDistRefreshPts;

        bool isRectified;

        bool missParams;

        void updateL1ChunkSize(unsigned newChSize) { this->l1ChunkSize = newChSize; }

        unsigned long getL2ChunkSize() const { return this->l1ChunkSize * this->l1NumLoop; }

        bool parseParams(const std::string& settingsPath);
        bool parseParams(const cv::FileStorage& settingsPath);

        std::string printParams() const;
    };

    typedef std::shared_ptr<EvParams> EvParamsPtr;

    class EventQueue : public SharedQueue<EventData> {
    public:
        using SharedQueue<EventData>::fillBuffer;
        using SharedQueue<EventData>::consumeBegin;//(unsigned long chunkSize, std::vector<EventData>& evs);
        unsigned long consumeBegin(unsigned long chunkSize, std::vector<EventData>& evs, std::vector<EventData>& accEvs);

    private:
        //std::queue<EventData> mEvBuffer;
        //std::mutex mMutexBuffer;
    };



} //EORB_SLAM

#endif //ORB_SLAM3_EVENTDATA_H
