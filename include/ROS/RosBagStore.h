//
// Created by root on 12/22/20.
//

#ifndef ORB_SLAM3_ROSBAGSTORE_H
#define ORB_SLAM3_ROSBAGSTORE_H

#include "DataStore.h"

namespace EORB_SLAM {

    struct RosbagTopics {
        string camInfo;
        string image;
        string event;
        string imu;
        string gtPose;
        string rImage;
        string rEvent;

        void toStrArr(vector<string>& topics);
        void print() const;
    };

    class RosBagStore : public EurocLoader {
    public:
        explicit RosBagStore(const string &fSettings);
        //~RosBagStore();

        int getBagFiles(string& dataFile, string& gtFile);
        int getTopics(vector<string>& topics);
        void getTopics(RosbagTopics& topics);
        void printLoaderStat() override;

    protected:
        bool resolveDsFormat(const cv::FileStorage &fsSettings) override;
        void resolveDsPaths(const cv::FileNode &dsPathNode) override;

        //void loadData();
        void loadSequence(const string &dsRoot, const string &seqPath, size_t idx) override;
        bool checkLoadStat() override;

    private:
        //bool parseSettings(const string &settingsFile);
        RosbagTopics mTopics;
        vector<vector<string>> mvSeqBagPaths;
    };

} //EORB_SLAM

#endif //ORB_SLAM3_ROSBAGSTORE_H
