//
// Created by root on 1/1/21.
//

#ifndef ORB_SLAM3_VISUALIZATION_H
#define ORB_SLAM3_VISUALIZATION_H

#include <iostream>
#include <chrono>
#include <utility>
#include <vector>
#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
//#include <opencv2/viz.hpp>
//#include <opencv2/video/tracking.hpp>

#include "Atlas.h"
#include "Converter.h"
#include "MyDataTypes.h"


namespace EORB_SLAM {

    //typedef std::multimap<float, std::pair<std::pair<double, std::string>, std::pair<cv::Mat&, cv::Mat&>>, std::greater<float>> MciInfo;
    typedef std::multimap<float, PoseImagePtr, std::greater<float>> MciInfo;
    typedef std::shared_ptr<ORB_SLAM3::Atlas> AtlasPtr;
    typedef std::vector<ORB_SLAM3::KeyFrame*> VecPtrKF;

    typedef SharedQueue<ImageTsPtr> ImageQueue;
    typedef std::shared_ptr<ImageQueue> ImageQueuePtr;


    class SimpleImageDisplay {
    public:
        enum DisplayMode {
            NONE,
            SYNCH_FEED,
            ASYNCH_FEED
        };

        explicit SimpleImageDisplay(const float fps, std::string winName = "Window") :
                mbStop(false), mWindowName(std::move(winName)), mLastTs(-1.0), mRefMode(SYNCH_FEED) {

            if (fps > 0) {
                mT = static_cast<int>(1000.f / fps);
            }
            else {
                mT = static_cast<int>(1000.f / 30.f);
            }
            mqpImages = std::make_shared<ImageQueue>();
            mpLastImage = make_shared<ImageTs>(0.0, cv::Mat(), "dummy-path.png");
        }

        SimpleImageDisplay(const float fps, const DisplayMode dispMode, const std::string& winName = "Window") :
                SimpleImageDisplay(fps, winName)
        {
            mRefMode = dispMode;
        }


        void run();
        void requestStop() { mbStop = true; }
        bool isStopped() const { return mbStop; }

        void addImage(const ImageTsPtr& image) { mqpImages->push(image); }
        ImageTsPtr frontImage();
        void popImage() { mqpImages->pop(); }

        ImageQueuePtr mqpImages;
        //float mfps;
        int mT;
        bool mbStop;
        std::string mWindowName;
        double mLastTs;

        DisplayMode mRefMode;

        ImageTsPtr mpLastImage;
    };

    class Visualization {
    public:

        // Print

        static void printPose(const cv::Mat& pose);
        static void printPose(const g2o::SE3Quat& pose);
        static void printPose(const g2o::Sim3& pose);
        static void printPose(const Eigen::MatrixXd& pose);

        static void printPoseQuat(const cv::Mat& pose);
        static void printPoseQuat(const g2o::SE3Quat& pose);
        static void printPoseQuat(const g2o::Sim3& pose);

        // Draw

        static void myDrawMatches(const cv::Mat &im1, const std::vector<cv::KeyPoint> &vkp1, const cv::Mat &im2,
                                  const std::vector<cv::KeyPoint> &vkp2, const std::vector<int> &matches,
                                  cv::Mat &newIm);

        static void drawMatchesInPlace(const cv::Mat& image, const std::vector<cv::KeyPoint> &vkp1,
                const std::vector<cv::KeyPoint> &vkp2, const std::vector<int>& vMatches12, cv::Mat& newImg);

        static void visualizePoseAndMap(const std::vector<cv::Affine3f>& vPoses,
                const std::vector<cv::Point3f>& vP3D, const cv::Mat& K);

        static void visualizePoseAndMap(const std::shared_ptr<ORB_SLAM3::Atlas>& pEvAtlas, const cv::Mat& K);

        static void showImage(const std::string &winName, const cv::Mat &img, int delay = 0);

        // Save

        static void saveImage(const std::string& fileName, const cv::Mat& img);

        static void saveReconstImages(const MciInfo& rImages, const string& rootPath);

        static void saveActivePose(const AtlasPtr& pEvAtlas, const std::string& poseFile);
        static void saveAllPose(const AtlasPtr& pEvAtlas, const std::string& poseFile);
        static void saveAllPoseRefs(const VecPtrKF& vpRefKFs, const std::string& poseFile,
                                    double tsc = 1.0, int ts_prec=9);
        static void saveAllPoseLV(const VecPtrKF& vpLocalKFs, const std::string& poseFile);

        static void saveMapPoints(const std::vector<ORB_SLAM3::MapPoint*>& vpMPs, const std::string& mapFile);

        static void saveActiveMap(const AtlasPtr& pEvAtlas, const std::string& mapFile);
        static void saveAtlas(const AtlasPtr& pEvAtlas, const std::string& mapFile);

        static void savePoseMap(const VecPtrKF& vpMapKFs, const std::string& poseFile, const std::string& mapFile,
                                double tsc = 1.0, int ts_prec=9);

        static void saveAllFramePoses(ORB_SLAM3::Atlas* pAtlas, const FrameInfo& frameInfo, const std::string& poseFile,
                                      bool selBestMap = false, bool isInertial = false, double tsc = 1.0, int ts_prec = 9, int p_prec = 9);
        static void saveAllFramePoses(const std::vector<ORB_SLAM3::KeyFrame*>& vpKFs, const FrameInfo& frameInfo,
                                      const std::string& poseFile, ORB_SLAM3::Map* pBiggerMap = nullptr, bool isInertial = false,
                                      double tsc = 1.0, int ts_prec = 9, int p_prec = 9);
        static void saveAllKeyFramePoses(ORB_SLAM3::Atlas* pAtlas, const std::string& poseFile, bool selBestMap = false,
                                         bool isInertial = false, double tsc = 1.0, int ts_prec = 9, int p_prec = 9);
        static void saveAllKeyFramePoses(const std::vector<ORB_SLAM3::KeyFrame*>& vpKFs, const std::string& poseFile,
                                         bool isInertial = false, double tsc = 1.0, int ts_prec = 9, int p_prec = 9);
    };

} // EORB_SLAM


#endif //ORB_SLAM3_VISUALIZATION_H
