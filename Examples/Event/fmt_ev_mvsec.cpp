/**
 * General test module supports all kinds of sensor configurations
 * Dataset must be in MVSEC ROS-Bag format
 * Changes times from ns to s automatically
 *
 * Author: M. Dayani
*/

#include <iostream>
#include <chrono>

#include <opencv2/core/core.hpp>

#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PoseStamped.h>
#include <dvs_msgs/EventArray.h>

#include "System.h"
#include "RosBagStore.h"

using namespace std;
//using namespace cv;
//using namespace EORB_SLAM;

//static const std::string OPENCV_WINDOW = "Image window";

void print_help() {
    cerr << endl << "Usage: ./fmt_ev_mvsec path_to_settings.yaml" << endl;
}

int main(int argc, char* argv[]) {

    if(argc < 2) {
        print_help();
        return 1;
    }

    const string settingsFile = argv[1];

    // Load all sequences:
    EORB_SLAM::RosBagStore eurocLoader(settingsFile);

    const unsigned int num_seq = eurocLoader.getNumTargetSequences();
    EORB_SLAM::RosbagTopics topics;
    eurocLoader.getTopics(topics);
    vector<string> strTopics;
    topics.toStrArr(strTopics);
    string file_name = eurocLoader.getTrajFileName();
    //unsigned int tot_images = eurocLoader.getNumTotalImages();
    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    //vTimesTrack.resize(tot_images);

    //cout << "Num. active sequences = " << num_seq << endl;
    //cout << "Out traj. file name: " << file_name << endl;
    eurocLoader.printLoaderStat();
    cout << endl << "-------" << endl;
    cout.precision(17);

    const EORB_SLAM::SensorConfigPtr dsConf = eurocLoader.getConfigStat();
    EORB_SLAM::EvParamsPtr evParams = std::make_shared<EORB_SLAM::EvParams>();
    evParams->parseParams(settingsFile);
    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(eurocLoader.getPathOrbVoc(), dsConf, eurocLoader.getCamParams(), eurocLoader.getORBParams(),
                           eurocLoader.getIMUParams(), eurocLoader.getViewerParams(), evParams);

    int seq;
    for (seq = 0; seq<num_seq; seq++) {

        string mvsecData, mvsecGt;
        int res = eurocLoader.getBagFiles(mvsecData, mvsecGt);
        if (!res) {
            cerr << "Cannot get bag files.\n";
            continue;
        }
        rosbag::Bag bag, gtBag;
        bag.open(mvsecData, rosbag::bagmode::Read);
        rosbag::View view(bag, rosbag::TopicQuery(strTopics));

        ros::Time bag_begin_time = view.getBeginTime();
        ros::Time bag_end_time = view.getEndTime();

        cout << "Beginning Sequence: #" << seq << endl;
        std::cout << "ROS bag time: " << (bag_end_time - bag_begin_time).toSec() << "(s)" << std::endl;

        cv::Mat im;
        vector<ORB_SLAM3::IMU::Point> vImuMeas;
        vector<EORB_SLAM::EventData> vEvLittleBuff;
        vector<EORB_SLAM::EventData> vEvBuffer;
        //unsigned int nImages = eurocLoader.getNumImages();
        int proccIm = 0;
        int imCnt = 0, camInfoCnt = 0;
        // Load all messages into our stereo dataset
        foreach(rosbag::MessageInstance const m, view) {

            //Load data
            bool newImage = false;
            double tframe = 0.0;
            cv_bridge::CvImagePtr cv_ptr;

            if (m.getTopic() == topics.image || ("/" + m.getTopic() == topics.image)) {

                sensor_msgs::Image::ConstPtr l_img = m.instantiate<sensor_msgs::Image>();
                try {
                    cv_ptr = cv_bridge::toCvCopy(l_img, sensor_msgs::image_encodings::MONO8);
                }
                catch (cv_bridge::Exception &e) {
                    ROS_ERROR("cv_bridge exception: %s", e.what());
                    break;
                }
                if (l_img != NULL) {
                    tframe = l_img->header.stamp.toSec();
                    im = cv_ptr->image;
                    newImage = true;
                    imCnt++;
                }
                else {
                    cerr << endl << "Failed to load image." << endl;
                    break;
                }
            }

            if (m.getTopic() == topics.event || ("/" + m.getTopic() == topics.event)) {
                dvs_msgs::EventArray::ConstPtr evData = m.instantiate<dvs_msgs::EventArray>();
                if (evData != NULL) {
                    vEvLittleBuff.clear();
                    for (int i = 0; i < evData->events.size(); i++) {
                        dvs_msgs::Event currEv = evData->events[i];
                        vEvLittleBuff.push_back(EORB_SLAM::EventData(currEv.ts.toSec(),
                                currEv.x, currEv.y, currEv.polarity));
                    }
                }
            }
            if (m.getTopic() == topics.imu || ("/" + m.getTopic() == topics.imu)) {
                sensor_msgs::Imu::ConstPtr imuData = m.instantiate<sensor_msgs::Imu>();
                if (imuData != NULL) {
                    vImuMeas.push_back(ORB_SLAM3::IMU::Point(imuData->linear_acceleration.x, imuData->linear_acceleration.y,
                                                             imuData->linear_acceleration.z, imuData->angular_velocity.x,
                                                             imuData->angular_velocity.y, imuData->angular_velocity.z,
                                                             imuData->header.stamp.toSec()));
                }
            }

            if (!newImage) continue;

#ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
            std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

            // Process Data
            // cout << "tframe = " << tframe << endl;
            switch (dsConf->getConfig()) {
                case EORB_SLAM::MySensorConfig::MONOCULAR:
                    SLAM.TrackMonocular(im,tframe);
                    break;
                case EORB_SLAM::MySensorConfig::IMU_MONOCULAR:
                    vImuMeas.clear();
                    eurocLoader.getNextImu(tframe, vImuMeas);
                    SLAM.TrackMonocular(im,tframe,vImuMeas);
                    break;
                default: //Idle
                    cout << "Idle mode...\n";
                    break;
            }
            vImuMeas.clear();

#ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
            std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

            double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

            vTimesTrack.push_back(ttrack);

            // Wait to load the next frame
//            double T=0;
//            if(ni<nImages-1)
//                T = eurocLoader.getImageTime(ni+1)-tframe;
//            else if(ni>0)
//                T = tframe-eurocLoader.getImageTime(ni-1);
//
//            if(ttrack<T)
//                usleep((T-ttrack)*1e6); // 1e6
        }

        eurocLoader.incSequence();
        if(seq < num_seq - 1)
        {
            cout << "Changing the dataset" << endl;

            SLAM.ChangeDataset();
        }
        bag.close();

        //gtBag.open(mvsecGt, rosbag::BagMode::Read);
        //rosbag::View gtView(gtBag, rosbag::TopicQuery(topics));
        /*foreach(rosbag::MessageInstance const m, gtView) {

                cout << "something" << endl;
                if (m.getTopic() == gtPoseTopic || ("/" + m.getTopic() == gtPoseTopic)) {
                    auto gtData = m.instantiate<geometry_msgs::PoseStamped>();
                    if (gtData != NULL)
                        cout << gtData << endl;
                }
                break;
            }*/
        //gtBag.close();
    }

    return 0;
}
