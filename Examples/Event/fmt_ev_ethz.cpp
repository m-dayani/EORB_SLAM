/**
 * General test module supports sensor configurations: Monocular Image, Event and IMU
 * Dataset must be in EuRoC format
 * Changes times from ns to s automatically
 *
 * Author: M. Dayani
 */

#include "compiler_options.h"

#include <iostream>
#include <chrono>
#include <string>

#include <opencv2/core/core.hpp>

#include <glog/logging.h>

#include "System.h"
#include "EventLoader.h"

using namespace std;

#define EV_CNT_BREAK 20
#define DEF_MAX_JAMMED_CNT 2000

//unsigned jammedCnt = 0;

void print_help() {
    cerr << endl << "Usage: ./fmt_euroc path_to_settings.yaml" << endl;
}

void savePoseFileHeader(const std::string& fileName, const std::string& msg) {

    ofstream poseFile;
    poseFile.open(fileName, std::ios_base::app);
    if (poseFile.is_open()) {
        poseFile << msg;
    }
    poseFile.close();
}

int main(int argc, char **argv)
{
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    //google::FlushLogFiles(google::INFO);

    if(argc < 2) {
        print_help();
        return 1;
    }

    const string settingsFile = argv[1];

    google::ParseCommandLineFlags(&argc, &argv, false);

    // Load all sequences:
    EORB_SLAM::EvEthzLoader evEthzLoader(settingsFile);
    if (!evEthzLoader.isGood()) {
        LOG(FATAL) << "Cannot Load data.\n";
    }

    const unsigned int num_seq = evEthzLoader.getNumTargetSequences();
    const unsigned int num_iter = evEthzLoader.getMaxNumIter();

    const string res_folder = evEthzLoader.getDatasetName();
    const string res_path = "results/" + res_folder;

    //unsigned long defChunkSize = 10000;

    DLOG(INFO) << "Num. active sequences = " << num_seq << endl;
    DLOG(INFO) << evEthzLoader.printLoaderStat() + "\n-------\n";

    cout.precision(17);

    const EORB_SLAM::SensorConfigPtr dsConf = evEthzLoader.getConfigStat();
    const EORB_SLAM::EvParamsPtr evParams = evEthzLoader.getEventParams();
    const bool needUndistEvs = !evParams->isRectified;
    // Since we rectify events before feeding the algorithm, set rect. evs:
    if (needUndistEvs) {
        evParams->isRectified = needUndistEvs;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(evEthzLoader.getPathOrbVoc(), dsConf, evEthzLoader.getCamParams(),
            evEthzLoader.getPairCamCalibrator(), evEthzLoader.getORBParams(), evEthzLoader.getIMUParams(),
            evEthzLoader.getViewerParams(), evParams);

    //unsigned evCnt = 0;
    int seq, iter;
    for (seq = 0; seq<num_seq; seq++)
    {
        cout << ">> Iterate Sequence: " << evEthzLoader.getSequenceName() << ", " << num_iter << " time(s)" << endl;

        string file_name = evEthzLoader.getTrajFileName();
        DLOG(INFO) << "Out traj. file name: " << file_name << endl;
        unsigned int nImages = evEthzLoader.getNumImages();

        for (iter = 0; iter < num_iter; iter++) {

            cout << "-- Iter #" << iter << endl;

            EORB_SLAM::MySmartTimer mainTrackingTimer(string("Main Tracking Thread, Iter#")+to_string(iter));

            // Main loop
            cv::Mat im;
            vector<ORB_SLAM3::IMU::Point> vImuMeas;
            vector<EORB_SLAM::EventData> vEvBuff;
            //int proccIm = 0;

            for (int ni = 0; !evEthzLoader.EoSEQ(nImages, ni); ni++) {

                //Load data
                // Read image from file
                double tframe = 0.0;
                if (dsConf->isImage()) {

                    evEthzLoader.getImage(ni, im, tframe);
                    if (im.empty()) {
                        LOG(ERROR) << "Failed to load image at: " << evEthzLoader.getImageFileName(ni) << endl;
                        break;
                    }
                }
                // Read events
                if (dsConf->isEvent()) {

                    vEvBuff.clear();
                    if (dsConf->isImage()) { // If image timestamp is retrieved

                        evEthzLoader.getNextEvents(tframe, vEvBuff, needUndistEvs);
                    }
                    else {
                        evEthzLoader.getNextEvents(evParams->getL2ChunkSize(), vEvBuff, needUndistEvs);

                        if (vEvBuff.empty()) {
                            LOG(ERROR) << "Failed to load event chunk\n";
                            break;
                        }
                        tframe = vEvBuff.back().ts;
                    }
                    DVLOG(1) << "** main: size ev. buff. -> " << vEvBuff.size() << ", last ts: "
                                        << ((!vEvBuff.empty()) ? to_string(vEvBuff.back().ts) : "??") << endl;
                    //cout << std::fixed << std::setprecision(9) << vEvBuff[0].ts << endl;
                }
                // Read Imu data
                if (dsConf->isInertial()) {

                    vImuMeas.clear();
                    evEthzLoader.getNextImu(tframe, vImuMeas);
                    if (vImuMeas.empty()) {
                        LOG(ERROR) << "Failed to load imu data to image timestamp: " << tframe << endl;
                        //break; // IMU is not the pipeline arbiter!
                    }
                }

                mainTrackingTimer.tic();

                // Pass the image to the SLAM system
                // cout << "tframe = " << tframe << endl;
                switch (dsConf->getConfig()) {
                    case EORB_SLAM::MySensorConfig::MONOCULAR:
                        SLAM.TrackMonocular(im, tframe);
                        break;
                    case EORB_SLAM::MySensorConfig::IMU_MONOCULAR:
                        SLAM.TrackMonocular(im, tframe, vImuMeas);
                        break;
                    case EORB_SLAM::MySensorConfig::EVENT_ONLY:
                        SLAM.TrackEvent(vEvBuff);
                        break;
                    case EORB_SLAM::MySensorConfig::EVENT_IMU:
                        SLAM.TrackEvent(vEvBuff, vImuMeas);
                        break;
                    case EORB_SLAM::MySensorConfig::EVENT_MONO:
                        SLAM.TrackEvMono(vEvBuff, im, tframe);
                        break;
                    case EORB_SLAM::MySensorConfig::EVENT_IMU_MONO:
                        SLAM.TrackEvMono(vEvBuff, im, tframe, vImuMeas);
                        break;
                    default: //Idle
                        break;
                }

                mainTrackingTimer.toc();
                mainTrackingTimer.push();
                double ttrack = mainTrackingTimer.getLastDtime();

                // Wait to load the next frame
                double T = 0;
                if (dsConf->isImage()) {
                    if (ni < nImages - 1)
                        T = evEthzLoader.getImageTime(ni + 1) - tframe;
                    else if (ni > 0)
                        T = tframe - evEthzLoader.getImageTime(ni - 1);
                }
                else if (dsConf->isEvent()) { // Event Only (no tframe!)
                    T = vEvBuff.back().ts - tframe;
                }

                if (ttrack < T)
                    std::this_thread::sleep_for(std::chrono::microseconds(static_cast<long>((T - ttrack) * 1e6)));

                //bool debug_break = false;
                //if (debug_break) break;
            }

            // give time to finish jobs
            for (ulong i = 0; i < 1000; i++)
                std::this_thread::sleep_for(std::chrono::microseconds(3000));

            // Print tracking time stats
            size_t nTotTTrack = mainTrackingTimer.numDtimes();
            double avgTTrack = mainTrackingTimer.getAverageTime();
            std::string mainTrackingTimeStr = mainTrackingTimer.getCommentedTimeStat();

            cout << "\tNum. All Tracked Frames: " << nTotTTrack << "\n";
            cout << "\tAverage Tracking Time: " << avgTTrack << "\n";
            LOG(INFO) << mainTrackingTimeStr;

            // Save camera trajectory for different schemes
            if (dsConf->isImage()) { // Im, Im-IMU, Ev-Im, Ev-Im-IMU
                ostringstream kf_file, f_file;
                kf_file << res_path << "/kf_" << file_name << "_" << iter << "_" << avgTTrack << ".txt";
                f_file << res_path << "/f_" << file_name << "_" << iter << "_" << avgTTrack << ".txt";

                savePoseFileHeader(f_file.str(), mainTrackingTimeStr);
                savePoseFileHeader(kf_file.str(), mainTrackingTimeStr);

                SLAM.SaveTrajectoryEuRoC(f_file.str(), 1.0, 9);
                SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file.str(), 1.0, 9);
            }
            else if (dsConf->isEvent()) { // Ev, Ev-IMU
                ostringstream ef_file, ekf_file;
                ekf_file << res_path << "/ekf_" << file_name << "_" << iter << "_" << avgTTrack << ".txt";
                ef_file << res_path << "/ef_" << file_name << "_" << iter << "_" << avgTTrack << ".txt";

                savePoseFileHeader(ef_file.str(), mainTrackingTimeStr);
                savePoseFileHeader(ekf_file.str(), mainTrackingTimeStr);

                SLAM.SaveTrajectoryEvent(ef_file.str());
                SLAM.SaveKeyFrameTrajectoryEvent(ekf_file.str());
            }
            // Fuse Event and ORB maps if accessible and save the result

#ifdef SAVE_ALL_IM_MPS
            SLAM.SaveMap("../data/orb_map.txt");
#endif

            if (iter < num_iter - 1) {
                LOG(INFO) << "Preparing next iteration" << endl;
                SLAM.PrepareNextIeration();
            }

            evEthzLoader.resetCurrSequence();
        }

        evEthzLoader.incSequence();

        if(seq < num_seq - 1)
        {
            LOG(INFO) << "Changing the dataset" << endl;
            SLAM.ChangeDataset();
        }
    }

    // Stop all threads
    SLAM.Shutdown();

    return 0;
}

