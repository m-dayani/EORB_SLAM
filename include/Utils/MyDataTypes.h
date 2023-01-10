//
// Created by root on 12/24/20.
//

#ifndef ORB_SLAM3_MYDATATYPES_H
#define ORB_SLAM3_MYDATATYPES_H

#include <iostream>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>
#include <queue>
#include <set>
#include <map>
#include <memory>
#include <list>

#include "include/IMU/ImuTypes.h"


namespace ORB_SLAM3 {
    class Frame;
    class KeyFrame;
}

namespace EORB_SLAM {

    class EvFrame;
    typedef std::shared_ptr<EvFrame> EvFramePtr;

    struct MySmartTimer {

        explicit MySmartTimer(std::string name);
        MySmartTimer() : MySmartTimer("Tracker") {}

        void reserve(const std::size_t& n);
        void setName(const std::string& name) { mName = name; }

        void tic();
        void toc();
        void push();

        double getAverageTime();

        std::string getCommentedTimeStat();

        float getLastDtime() { return mvDeltaTimes.back(); }

        std::size_t numDtimes() const { return mvDeltaTimes.size(); }

        void reset();

        std::string mName;
        std::chrono::steady_clock::time_point mt0, mt1;
        std::vector<double> mvDeltaTimes;
    };

    struct MySmartWatchDog {

        MySmartWatchDog(std::string  name, float waitTimeSec, ulong tStepUsec = 500);

        void step();

        void reset();

        void setName(const std::string& newName) { mName = newName; }
        void setWaitTimeSec(float waitTimeSec);

        float getWaitTimeSec() const;

        std::string mName;
        float mWaitTimeSec;

    private:
        const ulong mTimeStepUsec;
        ulong mCount;
        ulong mWaitCount{};
    };

    class MyDataTypes {
    public:
    };

    struct ImageData {

        ImageData() : ts(0.0), uri(std::string()) {}

        ImageData(double ts, std::string s) : ts(ts), uri(std::move(s)) {}

        virtual void print() const {
            std::cout << "[ts, imageName]: [" << ts << ", " << uri << "]\n";
        }

        virtual std::string printStr() const {
            std::ostringstream oss;
            oss << "[ts, imageName]: [" << ts << ", " << uri << "]\n";
            return oss.str();
        }

        double ts;
        std::string uri;
    };

    struct ImageTs : public ImageData {

        ImageTs(double _ts, const cv::Mat& image, const std::string&  _uri):
            ImageData(_ts, _uri), mImage(image.clone()) {}

        cv::Mat mImage;
    };

    typedef std::shared_ptr<ImageTs> ImageTsPtr;

    struct PoseImage : public ImageTs {

        PoseImage(double _ts, const cv::Mat& image, const cv::Mat& Tcw, const std::string&  _uri) :
                ImageTs(_ts, image, _uri), mTcw(Tcw.clone()), mReconstStat(-1) {}

        PoseImage(double _ts, const cv::Mat& image, const cv::Mat& Tcw, const int rStat, const std::string&  _uri) :
            ImageTs(_ts, image, _uri), mTcw(Tcw.clone()), mReconstStat(rStat) {}

        cv::Mat mTcw;
        int mReconstStat;
    };

    typedef std::shared_ptr<PoseImage> PoseImagePtr;

    struct ImuData {

        ImuData() = default;

        ImuData(double ts, float gx, float gy, float gz, float ax, float ay, float az) : ts(ts) {
            accel[0] = ax;
            accel[1] = ay;
            accel[2] = az;
            gyro[0] = gx;
            gyro[1] = gy;
            gyro[2] = gz;
        }

        ImuData(double ts, const float g[3], const float a[3]) : ts(ts) {
            for (unsigned char i; i < 3; i++) {
                gyro[i] = g[i];
                accel[i] = a[i];
            }
        }

        ORB_SLAM3::IMU::Point toImuPoint(double tsFactor = 1);

        void print() const {
            std::cout << "[ts, gx, gy, gz, ax, ay, az]: [" << ts;
            for (unsigned char i; i < 3; i++)
                std::cout << ", " << gyro[i];
            for (unsigned char i; i < 3; i++)
                std::cout << ", " << accel[i];
            std::cout << "]\n";
        }

        double ts = 0.0;
        float accel[3] = {}; //ax, ay, az
        float gyro[3] = {};  //gx, gy, gz
    };

    struct GtDataQuat {

        GtDataQuat() = default;

        GtDataQuat(double ts, float px, float py, float pz, float qw, float qx, float qy, float qz) : ts(ts) {
            q[0] = qw;
            q[1] = qx;
            q[2] = qy;
            q[3] = qz;
            p[0] = px;
            p[1] = py;
            p[2] = pz;
        }

        GtDataQuat(double ts, const float _p[3], const float _q[4]) : ts(ts) {
            for (unsigned char i; i < 3; i++) {
                q[i] = _q[i];
                p[i] = _p[i];
            }
            q[3] = _q[3];
        }

        void print() const {
            std::cout << "[ts, px, py, pz, qw, qx, qy, qz]: [" << ts;
            for (unsigned char i; i < 3; i++)
                std::cout << ", " << p[i];
            for (unsigned char i; i < 4; i++)
                std::cout << ", " << q[i];
            std::cout << "]\n";
        }

        double ts = 0.0;
        float q[4] = {};    //q_w, q_x, q_y, q_z
        float p[3] = {};    //px, py, pz
    };

    class MySensorConfig {
    public:
        enum SensorConfig {
            MONOCULAR = 0,
            STEREO = 1,
            RGBD = 2,
            IMU_MONOCULAR = 3,
            IMU_STEREO = 4,
            EVENT_ONLY = 5,
            EVENT_MONO = 6,
            EVENT_IMU = 7,
            EVENT_IMU_MONO = 8,
            IDLE = 9
        };

        MySensorConfig() : mSensor(IDLE) {}
        explicit MySensorConfig(SensorConfig sConf) : mSensor(sConf) {}

        static SensorConfig mapConfig(const std::string &dsConfig);
        static std::string mapConfig(SensorConfig dsConfig);

        // Convert Sensor config. types
        std::string toStr() const;
        std::string toDsStr() const;
        //static eSensor convertSensorConfig(EORB_SLAM::MyDataTypes::SensorConfig sConf);

        // Sensor configurations:
        // 3 camera config.
        bool isMonocular() const;
        bool isStereo() const;
        bool isRGBD() const;
        // 3 types of sensors
        bool isImage() const;
        bool isEvent() const;
        bool isInertial() const;
        bool isEventOnly() const;
        bool isImageOnly() const;

        bool operator==(SensorConfig sConf);
        bool operator!=(SensorConfig sConf);

        SensorConfig getConfig() const { return this->mSensor; }
        void setConfig(SensorConfig sConf) { this->mSensor = sConf; }
    private:
        SensorConfig mSensor;
    };

    typedef std::shared_ptr<MySensorConfig> SensorConfigPtr;
    typedef std::unique_ptr<ImageData> ImDataUnPtr;
    typedef std::unique_ptr<ImuData> ImuDataUnPtr;
    typedef std::unique_ptr<GtDataQuat> GtDataUnPtr;

    class SharedFlag {
    public:
        SharedFlag() : flag(false) {}
        explicit SharedFlag(bool state) : flag(state) {}

        explicit operator bool() {
            std::unique_lock<std::mutex> lock(mMtxState);
            return flag;
        }
        bool operator!() {
            std::unique_lock<std::mutex> lock(mMtxState);
            return !flag;
        }
        SharedFlag& operator=(bool state) {
            std::unique_lock<std::mutex> lock(mMtxState);
            this->flag = state;
            return *this;
        }
        bool operator==(const bool _flag) {
            std::unique_lock<std::mutex> lock(mMtxState);
            return flag == _flag;
        }
        bool operator!=(const bool _flag) {
            std::unique_lock<std::mutex> lock(mMtxState);
            return flag != _flag;
        }
        void set(bool state) {
            std::unique_lock<std::mutex> lock(mMtxState);
            this->flag = state;
        }
    private:
        bool flag;
        std::mutex mMtxState;
    };

    template<typename T>
    class SharedState {
    public:
        SharedState() : mState(T()) {}
        explicit SharedState(T state) : mState(state) {}

        T getState() {
            std::unique_lock<std::mutex> lock(mMutexState);
            return mState;
        }

        T getState() const { return mState; }

        //bool isIdle();
        bool operator==(T tState) {
            std::unique_lock<std::mutex> lock(mMutexState);
            return mState == tState;
        }

        bool operator==(SharedState& tState) {
            std::unique_lock<std::mutex> lock(mMutexState);
            return mState == tState.getState();
        }

        void update(T tState) {
            std::unique_lock<std::mutex> lock1(mMutexState);
            this->mState = tState;
        }

    private:
        T mState;
        std::mutex mMutexState;
    };

    template<typename T>
    class SharedQueue {
    public:
        bool isEmpty() {
            std::unique_lock<std::mutex> lock(mMutexBuffer);
            return mQueue.empty();
        }

        bool operator<(unsigned long nEls) {
            std::unique_lock<std::mutex> lock(mMutexBuffer);
            return mQueue.size() < nEls;
        }

        bool operator<=(unsigned long nEls) {
            std::unique_lock<std::mutex> lock(mMutexBuffer);
            return mQueue.size() <= nEls;
        }

        bool operator>(unsigned long nEls) {
            std::unique_lock<std::mutex> lock(mMutexBuffer);
            return mQueue.size() > nEls;
        }

        bool operator>=(unsigned long nEls) {
            std::unique_lock<std::mutex> lock(mMutexBuffer);
            return mQueue.size() >= nEls;
        }

        void fillBuffer(const std::vector<T>& vEls) {

            if (vEls.empty())
                return;

            std::unique_lock<std::mutex> lock1(mMutexBuffer);
            for (const T& ev : vEls) {
                mQueue.push(ev);
            }
        }

        unsigned long consumeBegin(unsigned long chunkSize, std::vector<T>& vEls) {

            std::unique_lock<std::mutex> lock1(mMutexBuffer);
            if (mQueue.empty())
                return 0;

            unsigned nEvs = std::min(chunkSize, mQueue.size());
            vEls.resize(nEvs);

            for (unsigned i = 0; i < nEvs; i++) {
                vEls[i] = mQueue.front();
                mQueue.pop();
            }
            return nEvs;
        }

        void push(T el) {
            std::unique_lock<std::mutex> lock1(mMutexBuffer);
            mQueue.push(el);
        }

        T front() {
            std::unique_lock<std::mutex> lock1(mMutexBuffer);
            return mQueue.front();
        }

        void pop() {
            std::unique_lock<std::mutex> lock1(mMutexBuffer);
            if (!mQueue.empty())
                mQueue.pop();
        }

        size_t size() {
            std::unique_lock<std::mutex> lock1(mMutexBuffer);
            return mQueue.size();
        }

        void clear() {
            std::unique_lock<std::mutex> lock1(mMutexBuffer);

            while(!mQueue.empty()) {
                mQueue.pop();
            }
        }
    protected:
        std::queue<T> mQueue;
        std::mutex mMutexBuffer;
    };

    template<typename T>
    class SharedVector {
    public:
        bool isEmpty() {
            std::unique_lock<std::mutex> lock(mMutexBuffer);
            return mVector.empty();
        }

        bool operator<(unsigned long nEls) {
            std::unique_lock<std::mutex> lock(mMutexBuffer);
            return mVector.size() < nEls;
        }

        bool operator<=(unsigned long nEls) {
            std::unique_lock<std::mutex> lock(mMutexBuffer);
            return mVector.size() <= nEls;
        }

        bool operator>(unsigned long nEls) {
            std::unique_lock<std::mutex> lock(mMutexBuffer);
            return mVector.size() > nEls;
        }

        bool operator>=(unsigned long nEls) {
            std::unique_lock<std::mutex> lock(mMutexBuffer);
            return mVector.size() >= nEls;
        }

        void fillBuffer(const std::vector<T>& vEls) {

            if (vEls.empty())
                return;

            std::unique_lock<std::mutex> lock1(mMutexBuffer);
            for (const T& ev : vEls) {
                mVector.push_back(ev);
            }
        }

        std::vector<T>& getVector() {
            std::unique_lock<std::mutex> lock1(mMutexBuffer);
            return mVector;
        }

        /*std::vector<T>& getVector() const {
            std::unique_lock<std::mutex> lock1(mMutexBuffer);
            return mVector;
        }*/

        void clear() {
            std::unique_lock<std::mutex> lock1(mMutexBuffer);
            mVector.clear();
        }

        void push_back(T el) {
            std::unique_lock<std::mutex> lock1(mMutexBuffer);
            mVector.push_back(el);
        }

        void reserve(size_t sz) {
            std::unique_lock<std::mutex> lock1(mMutexBuffer);
            mVector.reserve(sz);
        }

        void resize(size_t sz) {
            std::unique_lock<std::mutex> lock1(mMutexBuffer);
            mVector.resize(sz);
        }

        size_t size() {
            std::unique_lock<std::mutex> lock1(mMutexBuffer);
            return mVector.size();
        }
    protected:
        std::vector<T> mVector;
        std::mutex mMutexBuffer;
    };

    template<typename T>
    class MySharedPtr {
    public:
        MySharedPtr() : mPtr() {}
        explicit MySharedPtr(std::shared_ptr<T> ptr) : mPtr(std::move(ptr)) {}
        explicit MySharedPtr(T* ptr) : mPtr(std::shared_ptr<T>(ptr)) {}

        std::shared_ptr<T>& get() {
            std::unique_lock<std::mutex> lock(mMtxPtr);
            return mPtr;
        }

        std::shared_ptr<T>& get() const {
            return mPtr;
        }

        T* getPtr() {
            std::unique_lock<std::mutex> lock(mMtxPtr);
            return mPtr.get();
        }

        std::shared_ptr<T>& operator->() {
            std::unique_lock<std::mutex> lock(mMtxPtr);
            return mPtr;
        }
    protected:
        std::shared_ptr<T> mPtr;
        std::mutex mMtxPtr;
    };

    class MyDepthMap {
    public:

        bool checkBounds(float x, float y) const;

        void populate(const std::vector<cv::KeyPoint>& vKPts,
                const std::vector<cv::Point3f>& pts3d, const std::vector<int>& matches);

        float getDepth(float x, float y) const;

        float getDepthLinInterp(float x, float y) const;

        static float pointDistance(float x, float y) {
            return std::sqrt(x*x + y*y);
        }

        struct PointComparison {
            bool operator() (const std::pair<float, float> &first, const std::pair<float, float> &second) const {
                return pointDistance(first.first, first.second) < pointDistance(second.first, second.second);
            }
        };

    private:
        int mnPts;
        std::vector<float> vX;
        std::vector<float> vY;
        std::map<std::pair<float, float>, float, PointComparison> mDepthMap;
    };

    class PoseDepthInfo {
    public:
        PoseDepthInfo();
        PoseDepthInfo(const EvFramePtr& pFrame0, const EvFramePtr& pFrame1);
        explicit PoseDepthInfo(const std::shared_ptr<PoseDepthInfo>& pPoseInfo);

        void reset();

        void updateLastPose(const EvFramePtr& pFrame0, const EvFramePtr& pFrame1);
        void updateLastPose(ORB_SLAM3::Frame* pFr0, ORB_SLAM3::Frame* pFr1, float depth);
        void updateLastPose(ORB_SLAM3::KeyFrame* pKF0, ORB_SLAM3::KeyFrame* pKF1);
        void updateLastPose(double ts0, const cv::Mat& Tc0w, double ts1, const cv::Mat& Tc1w, float depth);

        void getLastInfo(double& ts, cv::Mat& Tc1w, float& lastDepth);
        void getIniInfo(double& ts, cv::Mat& Tc0w);
        void getDPose(double& ts0, double& dts, cv::Mat& Tc1c0, float& lastDepth);
        cv::Mat getDPose();

        void setLastDepth(float d);
        double getLastDepth();

        bool isInitialized() const { return mbIsInitialized; }

    private:
        cv::Mat mTc1w, mTc0w, mTc1c0;

        double mLastTimeStamp, mIniTimeStamp, mDTs;

        float mLastDepth;

        std::mutex mMtxUpdate;

        bool mbIsInitialized;
    };

    typedef std::shared_ptr<PoseDepthInfo> PoseDepthPtr;

    class FrameInfo {
    public:
        FrameInfo() = default;

        bool checkDataIntegrity() const;

        void reset();
        void pushState(ORB_SLAM3::Frame* pCurFr, bool isLost);
        void pushState(ulong frId, const cv::Mat& Tcr, ORB_SLAM3::KeyFrame* pRefKF, double ts, bool isLost);
        void pushLastState(bool isLost);

        void getAllState(std::list<cv::Mat>& lRelFramePoses, std::list<ORB_SLAM3::KeyFrame*>& lpRefKFs,
                         std::list<double>& lTs, std::list<bool>& lbLost) const;
        std::list<bool>& getIsLostList();
        cv::Mat getLastRelPose();

        void setAllState(const std::list<cv::Mat>& lRelFramePoses, const std::list<ORB_SLAM3::KeyFrame*>& lpRefKFs,
                         const std::list<double>& lTs, const std::list<bool>& lbLost);
        void setIsLostList(const std::list<bool>& lbLost);
        //void setIsLostRange(ulong iniFrId, ulong lastFrId);
        void setIsLostRange(ulong iniKFrId, ulong lastKFrId, bool isLost, bool resetAll = false);
        void setIsLostRange(double iniFrTs, double lastFrTs, bool isLost, bool resetAll = false);

    private:
        std::list<cv::Mat> mlRelativeFramePoses;
        std::list<ORB_SLAM3::KeyFrame*> mlpReferences;
        std::list<double> mlFrameTimes;
        std::list<bool> mlbLost;

        std::list<ulong> msFrameIds;
    };

    class EvKfInfo {
    public:

        void addKeyPoints(const std::vector<cv::KeyPoint>& kpts);
        //void addMatches();
        //void addPose();
        //void add3dPts();
        void addConnection(const cv::Mat& R, const cv::Mat& t, const std::vector<cv::Point3f>& pts3d,
                           const std::vector<int>& mch, const std::vector<bool>& vbInliers, const std::vector<float>& mchErr);

        void addPose(const cv::Mat& Tcw);
        cv::Mat getLastPose();

        unsigned getPointTracks(std::vector<cv::Mat>& kpts);

        std::vector<std::vector<cv::KeyPoint>> getKeyPoints() { return mvKpts; }

        void clear();

    private:
        std::vector<std::vector<cv::KeyPoint>> mvKpts;
        std::vector<std::vector<int>> mvMatches;
        std::vector<std::vector<float>> mvMchErr;

        std::vector<cv::Mat> mvTrans;
        std::vector<cv::Mat> mvRot;
        std::vector<std::vector<cv::Point3f>> mvPts3d;
        std::vector<std::vector<bool>> mvReconstInliers;

        std::vector<cv::Mat> mvTcw;
    };

} //EORB_SLAM


#endif //ORB_SLAM3_MYDATATYPES_H
