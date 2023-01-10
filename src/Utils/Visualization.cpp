//
// Created by root on 1/1/21.
//

#include "Visualization.h"

#include "KeyFrame.h"
#include "MapPoint.h"
#include "Map.h"
//#include "Atlas.h"

using namespace cv;
using namespace std;
using namespace ORB_SLAM3;


namespace EORB_SLAM {

    void SimpleImageDisplay::run()  {

        DLOG(INFO) << "SimpleImageDisplay::run: Starting image display: " << mWindowName << endl;
        cv::namedWindow(mWindowName, cv::WINDOW_AUTOSIZE);

        //cv::startWindowThread();

        while (!mbStop) {

            if (!mqpImages->isEmpty()) {

                ImageTsPtr pImage = mqpImages->front();

                if (pImage->mImage.empty()) {
                    mqpImages->pop();
                    DLOG(WARNING) << "SimpleImageDisplay::run: Empty image, abort!\n";
                    break;
                }

                double imTs = pImage->ts;

                cv::Mat imageToShow = pImage->mImage.clone();

                cv::cvtColor(imageToShow, imageToShow, cv::COLOR_GRAY2RGB);

                cv::putText(imageToShow, std::to_string(imTs), cv::Point2f(10,10),
                            cv::FONT_HERSHEY_COMPLEX_SMALL,((float)imageToShow.cols*1.5f)/720.f,
                            cv::Scalar(0, 180, 0), 1);

                cv::imshow(mWindowName, imageToShow);

                // waitKey is different according to display mode
                if (mRefMode == ASYNCH_FEED) {
                    double duration = imTs - mLastTs;
                    if (mLastTs < 0) {
                        duration = imTs - 0;
                    }
                    mLastTs = imTs;
                    int duration_ms = static_cast<int>(duration * 1000);
                    if (duration_ms > 0) {
                        cv::waitKey(duration_ms);
                    } else {
                        cv::waitKey(1);
                    }
                }
                else if (mRefMode == SYNCH_FEED) {
                    cv::waitKey(mT);
                }
                else {
                    // Default, do nothing (waitKey is called from other processes)
                    DLOG_EVERY_N(INFO, 10000) <<
                        "SimpleImageDisplay::run: No waitKey is called in Display mode NONE!\n";
                    //cv::waitKey(mT);
                }

                mqpImages->pop();
            }
        }

        DLOG(INFO) << "SimpleImageDisplay::run: Exiting main loop, Bye!\n";
        cv::destroyWindow(mWindowName);
        cv::waitKey(1);
    }

    ImageTsPtr SimpleImageDisplay::frontImage() {

//        if (!mqpImages->isEmpty()) {
//            mpLastImage = mqpImages->front();
//            mpLastImage->mImage = mqpImages->front()->mImage.clone();
//        }
//        return mpLastImage;
        if (mqpImages->isEmpty()) {
            return mpLastImage; //make_shared<ImageTs>(0.0, cv::Mat(), "");
        }
        else {
            return mqpImages->front();
        }
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    void Visualization::printPose(const cv::Mat &pose) {

        cout << ORB_SLAM3::Converter::toString(pose);
    }

    void Visualization::printPose(const g2o::SE3Quat& pose) {

        cout << ORB_SLAM3::Converter::toString(pose);
    }

    void Visualization::printPose(const g2o::Sim3& pose) {

        cout << ORB_SLAM3::Converter::toString(pose);
    }

    void Visualization::printPose(const Eigen::MatrixXd& pose) {

        cout << ORB_SLAM3::Converter::toString(pose);
    }

    void Visualization::printPoseQuat(const cv::Mat& pose) {

        cout << ORB_SLAM3::Converter::toStringQuat(pose);
    }

    void Visualization::printPoseQuat(const g2o::SE3Quat& pose) {

        cout << ORB_SLAM3::Converter::toStringQuat(pose);
    }

    void Visualization::printPoseQuat(const g2o::Sim3& pose) {

        cout << ORB_SLAM3::Converter::toStringQuat(pose);
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    void Visualization::myDrawMatches(const cv::Mat &im1, const std::vector<cv::KeyPoint> &vkp1,
            const cv::Mat &im2, const std::vector<cv::KeyPoint> &vkp2, const std::vector<int> &matches, cv::Mat &newIm) {

        int totRows = im1.rows + im2.rows;
        int totCols = im1.cols + im2.cols;
        int maxRows = fmax(im1.rows, im2.rows);
        newIm = cv::Mat::zeros(maxRows, totCols, CV_8UC1);
        im1.copyTo(newIm.colRange(0, im1.cols).rowRange(0, im1.rows));
        im2.copyTo(newIm.colRange(im1.cols, totCols).rowRange(0, im2.rows));

        cv::cvtColor(newIm, newIm, CV_GRAY2BGR);

        for (size_t i = 0; i < vkp1.size(); i++) {
            cv::KeyPoint kp1 = vkp1[i];
            int match12 = matches[i];

            if (match12 < 0 || match12 >= vkp2.size())
                continue;

            cv::KeyPoint kp2 = vkp2[match12];

            cv::line(newIm, cv::Point(kp1.pt.x,kp1.pt.y),
                     cv::Point(kp2.pt.x+im1.cols,kp2.pt.y), cv::Scalar(0,255,0));
        }
    }

    void Visualization::drawMatchesInPlace(const cv::Mat& image, const std::vector<cv::KeyPoint> &vkp1,
                                           const std::vector<cv::KeyPoint> &vkp2, const std::vector<int>& vMatches12,
                                           cv::Mat& newImg) {
        newImg = image.clone();
        cv::cvtColor(newImg, newImg, CV_GRAY2BGR);

        assert(vMatches12.size() == vkp1.size());

        for (size_t i = 0; i < vkp1.size(); i++) {
            cv::KeyPoint kp1 = vkp1[i];
            int match12 = vMatches12[i];

            if (match12 < 0 || match12 >= vkp2.size())
                continue;

            cv::KeyPoint kp2 = vkp2[match12];

            cv::circle(newImg, kp1.pt, 2, Scalar(0, 0, 255), FILLED, LINE_8);
            cv::circle(newImg, kp1.pt, 2, Scalar(255, 0, 0), FILLED, LINE_8);
        }
    }

    void Visualization::visualizePoseAndMap(const std::vector<cv::Affine3f> &vPoses,
            const std::vector<cv::Point3f> &vP3D, const cv::Mat& K) {

        Matx33d KK = Matx33d( K.at<float>(0,0), 0, K.at<float>(0,2),
                             0, K.at<float>(1,1), K.at<float>(1, 2),
                             0, 0,  1);

//        viz::Viz3d window("Coordinate Frame");
//        window.setWindowSize(Size(500,500));
//        window.setWindowPosition(Point(150,150));
//        window.setBackgroundColor(); // black by default
//
//        viz::WCloud cloud_widget(vP3D, viz::Color::green());
//        window.showWidget("point_cloud", cloud_widget);
//
//        window.showWidget("cameras_frames_and_lines", viz::WTrajectory(vPoses,
//                viz::WTrajectory::BOTH, 0.1, viz::Color::green()));
//        window.showWidget("cameras_frustums", viz::WTrajectoryFrustums(vPoses, KK, 0.1, viz::Color::yellow()));
//        window.setViewerPose(vPoses[0]);
//
//        window.spin();
    }

    void Visualization::visualizePoseAndMap(const AtlasPtr& pEvAtlas, const cv::Mat& K) {

        // Prepare Poses
        vector<ORB_SLAM3::KeyFrame*> vKFs = pEvAtlas->GetAllKeyFrames();
        int nKFs = vKFs.size();
        vector<cv::Affine3f> vPoses(nKFs);

        for (int i = 0; i < nKFs; i++) {

            cv::Mat currPose = vKFs[i]->GetPose();
            cv::Mat R = currPose.rowRange(0,3).colRange(0,3);
            cv::Mat t = currPose.rowRange(0,3).col(3);
            vPoses[i] = cv::Affine3f(R, t);
        }

        // Prepare Map Points
        vector<ORB_SLAM3::MapPoint*> vP3D = pEvAtlas->GetAllMapPoints();
        int nMPs = vP3D.size();
        vector<cv::Point3f> vMPs(nMPs);

        for (int i = 0; i < nMPs; i++) {

            ORB_SLAM3::MapPoint* mp = vP3D[i];
            if (!mp || mp->isBad())
                continue;

            vMPs[i] = Point3f(mp->GetWorldPos());
        }

        Visualization::visualizePoseAndMap(vPoses, vMPs, K);
    }

    void Visualization::showImage(const std::string &winName, const cv::Mat &img, int delay) {

        cv::imshow(winName, img);
        cv::waitKey(delay);
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    void Visualization::saveImage(const std::string &fileName, const cv::Mat &img) {

        cv::imwrite(fileName, img);
    }

    void Visualization::saveReconstImages(const MciInfo& rImages, const string& rootPath) {

        int cnt = 0;
        for (const auto& rIm : rImages) {

            cv::Mat im = rIm.second->mImage.clone();
            if (!im.empty()) {

                double imTs = rIm.second->ts;
                cv::cvtColor(im, im, CV_GRAY2BGR);
                std::ostringstream imText;
                imText << rIm.second->uri << ", ";
                if (cnt == 0) {
                    imText << "Best, ";
                }
                imText << "ts: " << imTs << ", STD: " << rIm.first;
                cv::putText(im, imText.str(), Point2f(10,10),cv::FONT_HERSHEY_COMPLEX_SMALL,
                            ((float)im.cols*1.5f)/720.f, cv::Scalar(0, 180, 0), 1);
                Visualization::saveImage(rootPath+"/l1_reconst_im_"+std::to_string(cnt)+".png", im);
                cnt++;
            }
        }
    }

    void Visualization::saveActivePose(const AtlasPtr& pEvAtlas, const std::string& poseName) {

        ofstream poseFile(poseName);
        poseFile << fixed << setprecision(9);

        for (ORB_SLAM3::KeyFrame* kf : pEvAtlas->GetCurrentMap()->GetAllKeyFrames()) {

            if (kf->isBad())
                continue;

            cv::Mat currPose = kf->GetPose();
            vector<float> q = ORB_SLAM3::Converter::toQuaternion(currPose.rowRange(0,3).colRange(0,3));
            poseFile << kf->mTimeStamp << ' ' << currPose.at<float>(0,3) << ' ' <<
                     currPose.at<float>(1,3) << ' ' << currPose.at<float>(2,3) << ' ' <<
                     q[0] << ' ' << q[1] << ' ' << q[2] << ' ' << q[3] << endl;
        }

        poseFile.close();
    }

    void Visualization::saveAllPose(const AtlasPtr& pEvAtlas, const std::string& poseName) {

        ofstream poseFile(poseName);
        poseFile << fixed << setprecision(9);

        // Gather all KFs in a sorted structure
        multimap<double, ORB_SLAM3::KeyFrame*> allKFs;
        for (ORB_SLAM3::Map* pMap : pEvAtlas->GetAllMaps()) {
            for (ORB_SLAM3::KeyFrame *kf : pMap->GetAllKeyFrames()) {

                allKFs.insert(make_pair(kf->mTimeStamp, kf));
            }
        }
        // Add all poses
        for (const auto& kfPair : allKFs) {

            ORB_SLAM3::KeyFrame* kf = kfPair.second;
            if (kf->isBad())
                continue;

            cv::Mat currPose = kf->GetPose();
            vector<float> q = ORB_SLAM3::Converter::toQuaternion(currPose.rowRange(0, 3).colRange(0, 3));
            poseFile << kf->mTimeStamp << ' ' << currPose.at<float>(0, 3) << ' ' <<
                     currPose.at<float>(1, 3) << ' ' << currPose.at<float>(2, 3) << ' ' <<
                     q[0] << ' ' << q[1] << ' ' << q[2] << ' ' << q[3] << endl;
        }

        poseFile.close();
    }

    void Visualization::saveAllPoseRefs(const std::vector<ORB_SLAM3::KeyFrame*>& vpLocalKFs,
                                        const std::string& poseFileName, const double tsc, const int ts_prec)
    {

        ofstream poseFile(poseFileName, std::ios_base::app);
        poseFile << fixed;

        // Save all poses
        for (const auto& pRefKF : vpLocalKFs) {

            if (pRefKF->isBad())
                continue;

            // Add reference key frame
            cv::Mat Rwc = pRefKF->GetRotation().t();
            cv::Mat twc = pRefKF->GetCameraCenter();

            poseFile << std::setprecision(ts_prec) << tsc * pRefKF->mTimeStamp << ' '
                     << ORB_SLAM3::Converter::toStringQuatRaw(twc, Rwc) << endl;

            // Add children
            for (const auto& pChildKF : pRefKF->GetChilds()) {

                if (pChildKF->isBad())
                    continue;

                Rwc = pChildKF->GetRotation().t();
                twc = pChildKF->GetCameraCenter();

                poseFile << std::setprecision(ts_prec) << tsc * pChildKF->mTimeStamp << ' '
                         << ORB_SLAM3::Converter::toStringQuatRaw(twc, Rwc) << endl;
            }
        }

        poseFile.close();
    }

    void Visualization::saveAllPoseLV(const std::vector<ORB_SLAM3::KeyFrame*>& vpLocalKFs,
            const std::string& poseFileName) {

        // "ev_asynch_tracker_pose_chain.txt"
        ofstream poseFile(poseFileName, std::ios_base::app);
        //poseFile.precision(9);

        // Save all poses
        for (const auto& kf : vpLocalKFs) {

            if (kf->isBad())
                continue;

            cv::Mat currPose = kf->GetPose();
            vector<float> q = ORB_SLAM3::Converter::toQuaternion(currPose.rowRange(0, 3).colRange(0, 3));
            poseFile << std::fixed << std::setprecision(9) << kf->mTimeStamp << ' ' << currPose.at<float>(0, 3) << ' ' <<
                     currPose.at<float>(1, 3) << ' ' << currPose.at<float>(2, 3) << ' ' <<
                     q[0] << ' ' << q[1] << ' ' << q[2] << ' ' << q[3] << endl;
        }

        poseFile.close();
    }

    void Visualization::saveMapPoints(const std::vector<ORB_SLAM3::MapPoint*>& vpMPs, const std::string& mapPath) {

        ofstream mapFile(mapPath);

        for (ORB_SLAM3::MapPoint* pMP : vpMPs) {

            if (!pMP || pMP->isBad())
                continue;

            cv::Mat currPos = pMP->GetWorldPos();
            mapFile << currPos.at<float>(0,0) << ' ' << currPos.at<float>(1,0) << ' '
                    << currPos.at<float>(2,0) << endl;
        }

        mapFile.close();
    }

    void Visualization::saveActiveMap(const AtlasPtr& pEvAtlas, const std::string& mapPath) {

        ofstream mapFile(mapPath);

        for (ORB_SLAM3::MapPoint* pMP : pEvAtlas->GetCurrentMap()->GetAllMapPoints()) {

            if (!pMP || pMP->isBad())
                continue;

            cv::Mat currPos = pMP->GetWorldPos();
            mapFile << currPos.at<float>(0,0) << ' ' << currPos.at<float>(1,0) << ' '
                    << currPos.at<float>(2,0) << endl;
        }

        mapFile.close();
    }

    void Visualization::saveAtlas(const AtlasPtr& pEvAtlas, const std::string& mapFileName) {

        // "ev_asynch_tracker_map.txt"
        ofstream mapFile(mapFileName);

        for (ORB_SLAM3::Map* pMap : pEvAtlas->GetAllMaps()) {
            for (ORB_SLAM3::MapPoint *pMP : pMap->GetAllMapPoints()) {

                if (!pMP || pMP->isBad())
                    continue;

                cv::Mat currPos = pMP->GetWorldPos();
                mapFile << currPos.at<float>(0, 0) << ' ' << currPos.at<float>(1, 0) << ' '
                        << currPos.at<float>(2, 0) << endl;
            }
        }

        mapFile.close();
    }

    void Visualization::savePoseMap(const VecPtrKF &vpMapKFs, const std::string &pfName, const std::string &mfName,
                                    const double tsc, const int ts_prec) {

        // Normally the first KF should be the reference KF
        set<ORB_SLAM3::MapPoint*> spAllMPs;

        ofstream poseFile(pfName);
        poseFile << fixed;
        ofstream mapFile(mfName);

        // Save all poses as Tcw
        for (const auto& pKF : vpMapKFs) {

            if (pKF->isBad())
                continue;

            cv::Mat Rwc = pKF->GetRotation().t();
            cv::Mat twc = pKF->GetCameraCenter();

            poseFile << std::setprecision(ts_prec) << tsc * pKF->mTimeStamp << ' '
                     << ORB_SLAM3::Converter::toStringQuatRaw(twc, Rwc) << endl;

            for (ORB_SLAM3::MapPoint* pMP : pKF->GetMapPoints()) {

                if (pMP && !pMP->isBad() && !spAllMPs.count(pMP)) {

                    spAllMPs.insert(pMP);
                }
            }
        }

        for (ORB_SLAM3::MapPoint* pMP : spAllMPs) {

            cv::Mat p3d = pMP->GetWorldPos();
            mapFile << p3d.at<float>(0, 0) << " " << p3d.at<float>(1, 0) << " " << p3d.at<float>(2, 0) << endl;
        }

        poseFile.close();
        mapFile.close();
    }

    void Visualization::saveAllFramePoses(ORB_SLAM3::Atlas *pAtlas, const FrameInfo &frameInfo, const std::string &poseFile,
                                          const bool selBestMap, const bool isInertial, const double tsc,
                                          const int ts_prec, const int p_prec) {

        // Retrieve all key frames to find initial pose
        vector<KeyFrame *> vpKFs;
        Map *pBiggerMap = nullptr;

        if (selBestMap) {
            vector<Map *> vpMaps = pAtlas->GetAllMaps();
            int numMaxKFs = 0;

            for (Map *pMap :vpMaps) {
                if (pMap->GetAllKeyFrames().size() > numMaxKFs) {
                    numMaxKFs = pMap->GetAllKeyFrames().size();
                    pBiggerMap = pMap;
                }
            }

            if (numMaxKFs <= 0) {
                LOG(WARNING) << "System::SaveTrajectoryEuRoC: Cannot retrieve map.\n";
                return;
            }

            vpKFs = pBiggerMap->GetAllKeyFrames();
        }
        else {
            vpKFs = pAtlas->GetAllKeyFrames();
        }

        // sort according to ids
        sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

        saveAllFramePoses(vpKFs, frameInfo, poseFile, pBiggerMap, isInertial, tsc, ts_prec, p_prec);
    }

    void Visualization::saveAllFramePoses(const std::vector<ORB_SLAM3::KeyFrame*>& vpKFs, const FrameInfo& frameInfo,
                                          const std::string& poseFile, ORB_SLAM3::Map* pBiggerMap, const bool isInertial,
                                          const double tsc, const int ts_prec, const int p_prec) {

        // Transform all keyframes so that the first keyframe is at the origin.
        // After a loop closure the first keyframe might not be at the origin.
        cv::Mat Twb; // Can be world to cam0 or world to b depending on IMU or not.
        if (isInertial)
            Twb = vpKFs[0]->GetImuPose();
        else
            Twb = vpKFs[0]->GetPoseInverse();

        if (Twb.empty()) {
            LOG(WARNING) << "Visualization::saveAllFramePoses: Empty Twb\n";
            return;
        }

        ofstream f;
        f.open(poseFile.c_str(), std::ios_base::app);
        f << fixed;

        // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
        // We need to get first the keyframe pose and then concatenate the relative transformation.
        // Frames not localized (tracking failure) are not saved.
        list<cv::Mat> mlRelativeFramePoses;
        list<KeyFrame*> mlpReferences;
        list<double> mlFrameTimes;
        list<bool> mlbLost;
        frameInfo.getAllState(mlRelativeFramePoses, mlpReferences, mlFrameTimes, mlbLost);

        // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
        // which is true when tracking failed (lbL).
        auto lRit = mlpReferences.begin();
        auto lT = mlFrameTimes.begin();
        auto lbL = mlbLost.begin();

        for(auto lit=mlRelativeFramePoses.begin(), lend=mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++)
        {
            if(*lbL)
                continue;

            KeyFrame* pKF = *lRit;
            if (!pKF)
                continue;

            cv::Mat Trw = cv::Mat::eye(4,4,CV_32FC1);

            // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
            while(pKF && pKF->isBad())
            {
                Trw = Trw*pKF->mTcp;
                pKF = pKF->GetParent();
            }

            if(!pKF || pKF->GetPose().empty() || (pBiggerMap && pKF->GetMap() != pBiggerMap))
                continue;

            Trw = Trw*pKF->GetPose()*Twb; // Tcp*Tpw*Twb0=Tcb0 where b0 is the new world reference

            if (isInertial)
            {
                cv::Mat Tbw = pKF->mImuCalib.Tbc*(*lit)*Trw;
                cv::Mat Rwb = Tbw.rowRange(0,3).colRange(0,3).t();
                cv::Mat twb = -Rwb*Tbw.rowRange(0,3).col(3);

                f << setprecision(ts_prec) << tsc * (*lT) << " " << Converter::toStringQuatRaw(twb, Rwb, p_prec) << endl;
            }
            else
            {
                cv::Mat Tcw = (*lit)*Trw;
                cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
                cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

                f << setprecision(ts_prec) << tsc * (*lT) << " " << Converter::toStringQuatRaw(twc, Rwc, p_prec) << endl;
            }
        }

        f.close();
    }

    void Visualization::saveAllKeyFramePoses(ORB_SLAM3::Atlas *pAtlas, const std::string &poseFile, const bool selBestMap,
                                             const bool isInertial, const double tsc, const int ts_prec, const int p_prec) {

        vector<KeyFrame*> vpKFs;
        Map *pBiggerMap = nullptr;

        if (selBestMap) {
            vector<Map *> vpMaps = pAtlas->GetAllMaps();
            int numMaxKFs = 0;

            for (Map *pMap :vpMaps) {
                if (pMap->GetAllKeyFrames().size() > numMaxKFs) {
                    numMaxKFs = pMap->GetAllKeyFrames().size();
                    pBiggerMap = pMap;
                }
            }

            if (numMaxKFs <= 0) {
                LOG(WARNING) << "System::SaveKeyFrameTrajectoryEuRoC: Cannot retrieve map.\n";
                return;
            }

            vpKFs = pBiggerMap->GetAllKeyFrames();
        }
        else {
            vpKFs = pAtlas->GetAllKeyFrames();
        }

        sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

        saveAllKeyFramePoses(vpKFs, poseFile, isInertial, tsc, ts_prec, p_prec);
    }

    void Visualization::saveAllKeyFramePoses(const std::vector<ORB_SLAM3::KeyFrame *> &vpKFs, const std::string &poseFile,
                                             bool isInertial, double tsc, int ts_prec, int p_prec) {

        // Transform all keyframes so that the first keyframe is at the origin.
        // After a loop closure the first keyframe might not be at the origin.
        ofstream f;
        // Open file for append (to protect headers)
        f.open(poseFile.c_str(), std::ios_base::app);
        f << fixed;

        for(auto pKF : vpKFs)
        {
            if(pKF->isBad())
                continue;

            if (isInertial)
            {
                cv::Mat Rwb = pKF->GetImuRotation();
                cv::Mat twb = pKF->GetImuPosition();

                f << setprecision(ts_prec) << tsc*pKF->mTimeStamp  << " " << Converter::toStringQuatRaw(twb, Rwb, p_prec) << endl;
            }
            else
            {
                cv::Mat Rwc = pKF->GetRotation().t();
                cv::Mat twc = pKF->GetCameraCenter();

                f << setprecision(ts_prec) << tsc*pKF->mTimeStamp << " " << Converter::toStringQuatRaw(twc, Rwc, p_prec) << endl;
            }
        }

        f.close();
    }

} // EORB_SLAM
