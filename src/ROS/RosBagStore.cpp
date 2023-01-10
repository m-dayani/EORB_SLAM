//
// Created by root on 12/22/20.
//

#include "RosBagStore.h"

using namespace boost::filesystem;

namespace EORB_SLAM {

    void RosbagTopics::toStrArr(vector<string>& topics) {

        topics.push_back(camInfo);
        topics.push_back(image);
        topics.push_back(event);
        topics.push_back(imu);
        topics.push_back(gtPose);
    }

    void RosbagTopics::print() const {

        cout << "CamInfo (left): " << camInfo << endl;
        cout << "Images (left): " << image << endl;
        cout << "Events (left): " << event << endl;
        cout << "Imu Data: " << imu << endl;
        cout << "Ground-truth Pose: " << gtPose << endl;
    }

    /* ================================ RosBagStore ================================= */

    RosBagStore::RosBagStore(const string &fSettings) {

        //check settings file exists and is valid
        path pathSettings = path(fSettings);
        try {
            if (exists(pathSettings) && is_regular_file(pathSettings) &&
                (BaseLoader::checkExtension(pathSettings, ".yaml") ||
                 BaseLoader::checkExtension(pathSettings, ".yml"))) {

                mPathSettings = fSettings;
                //parse ds settings
                bool res = this->parseSettings(fSettings);
                if (res) {
                    //load data
                    this->loadData();
                }
            }
            else {
                cerr << "Ds root path or settings are not valid.\n";
            }
            // Update variables
            if (!this->updateLoadStat()) {
                cerr << "Load state: BAD_DATA, check dataset settings.\n";
            }
        }
        catch (const boost::filesystem::filesystem_error& ex)
        {
            cerr << ex.what() << '\n';
        }
    }

    bool RosBagStore::resolveDsFormat(const cv::FileStorage &fsSettings) {

        string dsFormat = MyYamlParser::parseString(fsSettings, "DS.format", BaseLoader::mapDsFormats(BaseLoader::EV_MVSEC));
        mDsFormat = BaseLoader::mapDsFormats(dsFormat);
        return (mDsFormat == BaseLoader::EV_MVSEC);
    }

    void RosBagStore::resolveDsPaths(const cv::FileNode &dsPathNode) {

        // We have rosbag topics instead of Ds Paths
        mTopics.camInfo = MyYamlParser::parseString(dsPathNode, "imageFile", string("/davis/left/camera_info"));
        mTopics.image = MyYamlParser::parseString(dsPathNode, "imageBase", string("/davis/left/image_raw"));
        mTopics.event = MyYamlParser::parseString(dsPathNode, "events", string("/davis/left/events"));
        mTopics.imu = MyYamlParser::parseString(dsPathNode, "imu", string("/davis/left/imu"));
        mTopics.gtPose = MyYamlParser::parseString(dsPathNode, "gt", string("/davis/left/pose"));
        //this->mvSeqBagPaths.resize(5);
    }

    // Deal with timestamp units from the beginning
    void RosBagStore::loadSequence(const string &dsRoot, const string &sqPath, const size_t idx) {

        if (this->mvSeqBagPaths.empty()) {
            this->mvSeqBagPaths.resize(mSeqCount);
        }
        string seqPath = dsRoot + '/' + sqPath + '/';
        string dataPath = seqPath + sqPath + "_data.bag";
        string gtPath = seqPath + sqPath + "_gt.bag";

        if (BaseLoader::checkFile(dataPath) && BaseLoader::checkFile(gtPath)) {

            this->mvSeqBagPaths[idx].resize(2);
            this->mvSeqBagPaths[idx][0] = dataPath;
            this->mvSeqBagPaths[idx][1] = gtPath;
        }
    }

    //RosBagStore::~RosBagStore() {}

    bool RosBagStore::checkLoadStat() {


        if (this->mvSeqBagPaths.empty()) {
            return false;
        }
        else {
            for (size_t i = 0; i < this->mvSeqBagPaths.size(); i++) {
                if (this->mvSeqBagPaths[i].empty())
                    return false;
            }
        }
        mSeqCount = this->mvSeqBagPaths.size();
        mLoadStat = BaseLoader::GOOD;
        return true;
    }

    void RosBagStore::printLoaderStat() {
        EurocLoader::printLoaderStat();
        mTopics.print();

        for (size_t i = 0; i < mvSeqBagPaths.size(); i++) {

            cout << "Sequence: #" << i << endl;
            cout << "Data rosbag: " << mvSeqBagPaths[i][0] << endl;
            cout << "Gt rosbag: " << mvSeqBagPaths[i][1] << endl;
        }
    }

    int RosBagStore::getBagFiles(string& dataFile, string& gtFile) {
        if (!checkSequence(mSeqIdx))
            return 0;
        dataFile = this->mvSeqBagPaths[mSeqIdx][0];
        gtFile = this->mvSeqBagPaths[mSeqIdx][1];

        return 2;
    }

    int RosBagStore::getTopics(vector<string>& topics) {

        mTopics.toStrArr(topics);
        if (topics.empty() || topics[0].empty()) {
            return 0;
        }
        else {
            return topics.size();
        }
    }

    void RosBagStore::getTopics(RosbagTopics &topics) {

        topics = mTopics;
    }

} //EORB_SLAM
