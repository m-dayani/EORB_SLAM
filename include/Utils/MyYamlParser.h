//
// Created by root on 11/27/20.
//

#ifndef EORB_SLAM_MYYAMLPARSER_H
#define EORB_SLAM_MYYAMLPARSER_H

#include <vector>
#include <string>
#include <opencv2/core/core.hpp>

//using namespace std;

namespace EORB_SLAM {

    class MyYamlParser {
    public:
        static std::string parseString(const cv::FileStorage &fs,
                const std::string &strArg, const std::string &defVal = std::string());
        static std::string parseString(const cv::FileNode &n,
                const std::string &strArg, const std::string &defVal = std::string());

        static int parseInt(const cv::FileStorage &fs, const std::string &strArg, const int &defVal = 0);

        static unsigned int parseStringSequence(const cv::FileStorage &fs,
                const std::string &strArg, std::vector<std::string> &outSeq);
    };

}// EORB_SLAM
#endif //EORB_SLAM_MYYAMLPARSER_H
