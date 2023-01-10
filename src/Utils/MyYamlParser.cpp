//
// Created by root on 11/27/20.
//

#include "MyYamlParser.h"

namespace EORB_SLAM {

    std::string MyYamlParser::parseString(const cv::FileStorage &fs,
            const std::string &strArg, const std::string &defVal) {

        cv::FileNode node = fs[strArg];
        if (node.isString())
            return node.string();
        return defVal;
    }

    std::string MyYamlParser::parseString(const cv::FileNode &n,
            const std::string &strArg, const std::string &defVal) {

        cv::FileNode node = n[strArg];
        if (node.isString())
            return node.string();
        return defVal;
    }

    int MyYamlParser::parseInt(const cv::FileStorage &fs, const std::string &strArg, const int &defVal) {

        cv::FileNode node = fs[strArg];
        if (node.isInt())
            return (int) node;
        return defVal;
    }

    unsigned int MyYamlParser::parseStringSequence(const cv::FileStorage &fs,
            const std::string &strArg, std::vector<std::string> &outSeq) {

        unsigned int cnt = 0;
        cv::FileNode n = fs[strArg]; // Read string sequence - Get node
        if (n.type() != cv::FileNode::SEQ) {
            return cnt;
        }
        cv::FileNodeIterator it = n.begin(), it_end = n.end(); // Go through the node
        for (; it != it_end; ++it) {
            outSeq.push_back((std::string) *it);
            cnt++;
        }
        return cnt;
    }

}// EORB_SLAM

