/**
 * General test module supports sensor configurations: Monocular Image & Inertial
 * Dataset must be in EuRoC format
 * Changes times from ns to s automatically
 *
 * Author: M. Dayani
 */

//#include "compiler_options.h"

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

    // Load all sequences:
    EORB_SLAM::EurocLoader eurocLoader(settingsFile);
    if (!eurocLoader.isGood()) {
        LOG(FATAL) << "Cannot Load data.\n";
    }

    const unsigned int num_seq = eurocLoader.getNumTargetSequences();
    const unsigned int num_iter = eurocLoader.getMaxNumIter();

    const string res_folder = eurocLoader.getDatasetName();
    const string res_path = "results/" + res_folder;

    DLOG(INFO) << "Num. active sequences = " << num_seq << endl;
    DLOG(INFO) << eurocLoader.printLoaderStat() + "\n-------\n";

    cout.precision(17);

    const EORB_SLAM::SensorConfigPtr dsConf = eurocLoader.getConfigStat();
    const EORB_SLAM::PairCalibPtr pCalibPiar = eurocLoader.getPairCamCalibrator();
    const EORB_SLAM::CamParamsPtr pCamParams = eurocLoader.getCamParams();

//    cv::Mat M1l,M2l,M1r,M2r;
//    if (dsConf->isStereo() && pCamParams->mLinkedCam) {
//
//        const EORB_SLAM::MyCamParams* pCamParams2 = pCamParams->mLinkedCam;
//        cv::Size imSize(pCamParams->mImWidth, pCamParams->mImHeight);
//
//        cv::initUndistortRectifyMap(pCamParams->mK,pCamParams->mDistCoef,pCamParams->mR,
//                pCamParams->mP.rowRange(0,3).colRange(0,3),imSize,CV_32F,M1l,M2l);
//        cv::initUndistortRectifyMap(pCamParams2->mK,pCamParams2->mDistCoef,pCamParams2->mR,
//                pCamParams2->mP.rowRange(0,3).colRange(0,3),imSize,CV_32F,M1r,M2r);
//    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(eurocLoader.getPathOrbVoc(), dsConf, pCamParams, pCalibPiar,
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

            EORB_SLAM::MySmartTimer mainTrackingTimer(string("Main Tracking Thread, Iter#")+to_string(iter));

            // Main loop
            cv::Mat imLeft, imRight, imLeftRect, imRightRect;
            vector<ORB_SLAM3::IMU::Point> vImuMeas;
            //int proccIm = 0;

            for (int ni = 0; ni < nImages; ni++) {
                // Read image from file
                double tframe = 0.0, tframeRight = 0.0;

                if (dsConf->isImage()) {

                    eurocLoader.getImage(ni, imLeft, tframe);
                    if (imLeft.empty()) {
                        LOG(ERROR) << endl << "Failed to load image at: " << eurocLoader.getImageFileName(ni) << endl;
                        break;
                    }
                    if (dsConf->isStereo()) {

                        eurocLoader.getImage(ni, imRight, tframeRight, true);
                        if (imLeft.empty()) {
                            LOG(ERROR) << endl << "Failed to load right image at: "
                                       << eurocLoader.getImageFileName(ni, true, true) << endl;
                            break;
                        }

                        // images must be undistorted in stereo case only!
                        //pCalibPiar.first->undistImageMaps(imLeft, imLeftRect);
                        //pCalibPiar.second->undistImageMaps(imRight, imRightRect);
                        //cv::remap(imLeft,imLeftRect,M1l,M2l,cv::INTER_LINEAR);
                        //cv::remap(imRight,imRightRect,M1r,M2r,cv::INTER_LINEAR);
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

                mainTrackingTimer.tic();

                // Pass the image to the SLAM system
                // cout << "tframe = " << tframe << endl;
                switch (dsConf->getConfig()) {
                    case EORB_SLAM::MySensorConfig::MONOCULAR:
                        SLAM.TrackMonocular(imLeft, tframe);
                        break;
                    case EORB_SLAM::MySensorConfig::IMU_MONOCULAR:
                        SLAM.TrackMonocular(imLeft, tframe, vImuMeas);
                        break;
                    case EORB_SLAM::MySensorConfig::STEREO:
                        SLAM.TrackStereo(imLeftRect, imRightRect, tframe);
                        break;
                    case EORB_SLAM::MySensorConfig::IMU_STEREO:
                        SLAM.TrackStereo(imLeftRect, imRightRect, tframe, vImuMeas);
                        break;
                    default: //Idle
                        break;
                }

                mainTrackingTimer.toc();
                mainTrackingTimer.push();
                double ttrack = mainTrackingTimer.getLastDtime();

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
            size_t nTotTTrack = mainTrackingTimer.numDtimes();
            double avgTTrack = mainTrackingTimer.getAverageTime();
            std::string mainTrackingTimeStr = mainTrackingTimer.getCommentedTimeStat();

            cout << "\tNum. All Tracked Frames: " << nTotTTrack << "\n";
            cout << "\tAverage Tracking Time: " << avgTTrack << "\n";
            LOG(INFO) << mainTrackingTimeStr;

            // Save camera trajectory
            ostringstream kf_file, f_file;
            kf_file << res_path << "/kf_" << file_name << "_" << iter << "_" << avgTTrack << ".txt";
            f_file << res_path << "/f_" << file_name << "_" << iter << "_" << avgTTrack << ".txt";

            savePoseFileHeader(f_file.str(), mainTrackingTimeStr);
            savePoseFileHeader(kf_file.str(), mainTrackingTimeStr);

            SLAM.SaveTrajectoryEuRoC(f_file.str());
            SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file.str());

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

