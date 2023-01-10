/**
 * General test module supports sensor configurations: Monocular Image & Inertial
 * Dataset must be in EuRoC format
 * Changes times from ns to s automatically
 *
 * Author: M. Dayani
 */

#include "compiler_options.h"

#include <iostream>
#include <chrono>

#include <opencv2/core/core.hpp>

#include <glog/logging.h>

#include "System.h"
#include "DataStore.h"

using namespace std;


void print_help() {
    cerr << endl << "Usage: ./fmt_euroc path_to_settings.yaml" << endl;
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

    // Load all sequences:
    EORB_SLAM::EurocLoader eurocLoader(settingsFile);
    if (!eurocLoader.isGood()) {
        LOG(FATAL) << "Cannot Load data.\n";
    }

    const unsigned int num_seq = eurocLoader.getNumTargetSequences();
    const unsigned int num_iter = eurocLoader.getMaxNumIter();

    DLOG(INFO) << "Num. active sequences = " << num_seq << endl;
    DLOG(INFO) << eurocLoader.printLoaderStat() + "\n-------\n";

    cout.precision(17);

    const EORB_SLAM::SensorConfigPtr dsConf = eurocLoader.getConfigStat();
    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(eurocLoader.getPathOrbVoc(), dsConf, eurocLoader.getCamParams(), eurocLoader.getPairCamCalibrator(),
                           eurocLoader.getORBParams(), eurocLoader.getIMUParams(), eurocLoader.getViewerParams());

    int seq, iter;
    for (seq = 0; seq < num_seq; seq++)
    {
        cout << ">> Iterate Sequence: " << eurocLoader.getSequenceName() << ", " << num_iter << " time(s)" <<endl;

        unsigned int nImages = eurocLoader.getNumImages();
        string file_name = eurocLoader.getTrajFileName();
        DLOG(INFO) << "Out traj. file name: " << file_name << endl;

        for (iter = 0; iter < num_iter; iter++) {

            cout << "-- Iter #" << iter << endl;

            // Vector for tracking time statistics
            vector<float> vTimesTrack;
            vTimesTrack.reserve(nImages);

            // Main loop
            cv::Mat im;
            vector<ORB_SLAM3::IMU::Point> vImuMeas;
            int proccIm = 0;

            for (int ni = 0; ni < nImages; ni++, proccIm++) {
                // Read image from file
                double tframe = 0.0;

                if (dsConf->isImage()) {

                    eurocLoader.getImage(ni, im, tframe);
                    if (im.empty()) {
                        LOG(ERROR) << endl << "Failed to load image at: " << eurocLoader.getImageFileName(ni) << endl;
                        break;
                    }
                }
                // Read Imu data
                if (dsConf->isInertial() && dsConf->isImage()) {

                    vImuMeas.clear();
                    eurocLoader.getNextImu(tframe, vImuMeas);
                    if (vImuMeas.empty()) {
                        LOG(ERROR) << "Failed to load imu data to image timestamp: " << tframe << endl;
                        //break; // IMU is not the pipeline arbiter!
                    }
                }

#ifdef COMPILEDWITHC11
                std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
                std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

                // Pass the image to the SLAM system
                // cout << "tframe = " << tframe << endl;
                switch (dsConf->getConfig()) {
                    case EORB_SLAM::MySensorConfig::MONOCULAR:
                        SLAM.TrackMonocular(im, tframe);
                        break;
                    case EORB_SLAM::MySensorConfig::IMU_MONOCULAR:
                        SLAM.TrackMonocular(im, tframe, vImuMeas);
                        break;
                    default: //Idle
                        break;
                }

#ifdef COMPILEDWITHC11
                std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
                std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

                double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

                vTimesTrack.push_back(ttrack);

                // Wait to load the next frame
                double T = 0;
                if (ni < nImages - 1)
                    T = eurocLoader.getImageTime(ni + 1) - tframe;
                else if (ni > 0)
                    T = tframe - eurocLoader.getImageTime(ni - 1);

                if (ttrack < T)
                    std::this_thread::sleep_for(
                            std::chrono::microseconds(static_cast<long>((T - ttrack) * 1e6))); // 1e6
            }

            // Print tracking time stats
            sort(vTimesTrack.begin(), vTimesTrack.end());
            double sumTTrack = 0;
            int nTotTTrack = 0;
            for (const float ttrack : vTimesTrack) {
                sumTTrack += ttrack;
                nTotTTrack++;
            }

            double avgTTrack = sumTTrack / nTotTTrack;

            cout << "\tNum. All Tracked Frames: " << nTotTTrack << "\n";
            cout << "\tAverage Tracking Time: " << avgTTrack << "\n";
            LOG(INFO) << "** Median Tracking Time: " << vTimesTrack[nTotTTrack/2] << " **\n";
            LOG(INFO) << "** (min, max) Tracking Time: (" << vTimesTrack[0] << ", " << vTimesTrack.back() << ") **\n";

            // Save camera trajectory
            const string kf_file =  "results/euroc/kf_" + file_name + "_" + to_string(avgTTrack)  + ".txt";
            const string f_file =  "results/euroc/f_" + file_name + "_" + to_string(avgTTrack)  + ".txt";
            SLAM.SaveTrajectoryEuRoC(f_file);
            SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file);

            if (iter < num_iter - 1) {
                LOG(INFO) << "Preparing next iteration" << endl;
                SLAM.PrepareNextIeration();
            }

            eurocLoader.resetCurrSequence();
        }

        eurocLoader.incSequence();

        if(seq < num_seq - 1)
        {
            LOG(INFO) << "-- Changing the dataset ..." << endl;
            SLAM.ChangeDataset();
        }
    }

    // Stop all threads
    SLAM.Shutdown();

    return 0;
}

