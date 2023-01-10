//
// Created by root on 12/24/20.
//

#ifndef ORB_SLAM3_EVENTLOADER_H
#define ORB_SLAM3_EVENTLOADER_H

#include "EventData.h"
#include "DataStore.h"
#include "MyCalibrator.h"


namespace EORB_SLAM {

    class EventDataStore : public BaseLoader {
    public:
        EventDataStore() : mLastEvTs(0.0) {}
        explicit EventDataStore(const std::string &filePath, double tsFactor);
        ~EventDataStore() override;

        unsigned long getEventChunk(unsigned long chunkSize, std::vector<EventData> &evBuffer);
        unsigned long getEventChunk(double tsEnd, std::vector<EventData> &evBuffer);
        // WARNING!! These methods only support Pinhole cameras
        unsigned long getEventChunkRectified(unsigned long chunkSize, std::vector<EventData> &evBuffer,
                const cv::Mat& K, const cv::Mat& distCoefs, const cv::Mat& rectMat,
                const cv::Scalar& imageSize, bool checkInImage = true);
        unsigned long getEventChunkRectified(double tsEnd, std::vector<EventData> &evBuffer,
                const cv::Mat& K, const cv::Mat& distCoefs, const cv::Mat& rectMat,
                const cv::Scalar& imageSize, bool checkInImage = true);
        // Use these methods whenever possible since they use camModel-free rectification
        unsigned long getEventChunkRectified(unsigned long chunkSize, std::vector<EventData> &evBuffer,
                                             const MyCalibPtr& pCalib, bool checkInImage = true);
        unsigned long getEventChunkRectified(double tsEnd, std::vector<EventData> &evBuffer,
                                             const MyCalibPtr& pCalib, bool checkInImage = true);
        //TODO: Develop methods to process event statistics too.
        //unsigned long getEventChunk(unsigned long chunkSize, std::vector<EventData> &evBuffer, EventStat& evStat);

        void reset() override;

    protected:
        boost::any parseLine(const std::string &evStr) override;
        EventData parseLine(const std::string &evStr, const cv::Mat& K, const cv::Mat& distCoefs, const cv::Mat& rectMat);
        EventData parseLine(const std::string &evStr, const MyCalibPtr& pCalib);

        //void parseEventData(float tsEnd, std::vector<EventData> &evBuffer);

    private:

        double mLastEvTs;
    };

    class EvEthzLoader : public EurocLoader {
    public:
        explicit EvEthzLoader(const std::string &fSettings);
        ~EvEthzLoader() override;

        unsigned long getNextEvents(unsigned long chunkSize, std::vector<EventData>& evs,
                bool undistPoints = false, bool checkInImage = true);
        unsigned long getNextEvents(double tsEnd, std::vector<EventData> &evs,
                bool undistPoints = false, bool checkInImage = true);

        std::shared_ptr<EvParams> getEventParams() const { return mpEventParams; }

        std::string getTrajFileName() override;

        std::string printLoaderStat() const override;

        void resetCurrSequence() override;

        static bool EoSEQ(const EORB_SLAM::SensorConfigPtr& dsConf, unsigned int nImages, int ni);
        bool EoSEQ(unsigned int nImages, int ni);

    protected:
        bool resolveDsFormat(const cv::FileStorage &fsSettings) override;
        void resolveDsPaths(const cv::FileNode &dsPathNode) override;
        bool parseParameters(const cv::FileStorage& fsSettings) override;

        //void loadData();
        void loadSequence(const std::string &dsRoot, const std::string &seqPath, size_t idx) override;
        //bool updateLoadStat();
        bool checkLoadStat() override;

        std::string mPathEvents;
        std::vector<std::unique_ptr<EventDataStore>> mvpEvDs;

        std::shared_ptr<EvParams> mpEventParams;

    private:
        //bool parseSettings(const std::string &settingsFile);

    };

} //EORB_SLAM


#endif //ORB_SLAM3_EVENTLOADER_H
