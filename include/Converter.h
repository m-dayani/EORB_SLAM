/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef CONVERTER_H
#define CONVERTER_H

#include<opencv2/core/core.hpp>

#include<Eigen/Dense>
#include"Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include"Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

namespace ORB_SLAM3
{

class Converter
{
public:
    static std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors);

    static g2o::SE3Quat toSE3Quat(const cv::Mat &CvT);
    static g2o::SE3Quat toSE3Quat(const g2o::Sim3 &gSim3);

    static cv::Mat toCvMat(const g2o::SE3Quat &SE3);
    static cv::Mat toCvMat(const g2o::Sim3 &Sim3);
    static cv::Mat toCvMat(const Eigen::Matrix<double,4,4> &m);
    static cv::Mat toCvMat(const Eigen::Matrix3d &m);
    static cv::Mat toCvMat(const Eigen::Matrix<double,3,1> &m);
    static cv::Mat toCvMat(const Eigen::MatrixXd &m);

    static cv::Mat toCvSE3(const Eigen::Matrix<double,3,3> &R, const Eigen::Matrix<double,3,1> &t);
    static cv::Mat tocvSkewMatrix(const cv::Mat &v);

    static Eigen::Matrix<double,3,1> toVector3d(const cv::Mat &cvVector);
    static Eigen::Matrix<double,3,1> toVector3d(const cv::Point3f &cvPoint);
    static Eigen::Matrix<double,3,3> toMatrix3d(const cv::Mat &cvMat3);
    static Eigen::Matrix<double,4,4> toMatrix4d(const cv::Mat &cvMat4);
    static std::vector<float> toQuaternion(const cv::Mat &M);

    static bool isRotationMatrix(const cv::Mat &R);
    static std::vector<float> toEuler(const cv::Mat &R);

    /* -------------------------------------------------------------------------------------------------------------- */

    // Lie Algebra Functions
    static cv::Mat ExpSO3(const float &x, const float &y, const float &z);
    static cv::Mat ExpSO3(const cv::Mat &v);
    static cv::Mat LogSO3(const cv::Mat &R);
    static cv::Mat RightJacobianSO3(const float &x, const float &y, const float &z);
    static cv::Mat RightJacobianSO3(const cv::Mat &v);
    static cv::Mat InverseRightJacobianSO3(const float &x, const float &y, const float &z);
    static cv::Mat InverseRightJacobianSO3(const cv::Mat &v);
    static cv::Mat Skew(const cv::Mat &v);
    static cv::Mat NormalizeRotation(const cv::Mat &R);

    static Eigen::Matrix3d ExpSO3(const double &x, const double &y, const double &z);
    static Eigen::Matrix3d ExpSO3(const Eigen::Vector3d &w);
    static Eigen::Vector3d LogSO3(const Eigen::Matrix3d &R);
    static Eigen::Matrix3d InverseRightJacobianSO3(const Eigen::Vector3d &v);
    static Eigen::Matrix3d RightJacobianSO3(const Eigen::Vector3d &v);
    static Eigen::Matrix3d RightJacobianSO3(const double x, const double y, const double z);
    static Eigen::Matrix3d Skew(const Eigen::Vector3d &w);
    static Eigen::Matrix3d InverseRightJacobianSO3(const double x, const double y, const double z);
    static Eigen::Matrix3d NormalizeRotation(const Eigen::Matrix3d &R);

    static Eigen::Matrix<double, 6, 1> SE3_to_se3(const Eigen::Matrix3d& R, const Eigen::Vector3d& t, const double dt);
    static Eigen::Matrix<double, 6, 1> SE3_to_se3(const Eigen::Matrix4d& T, const double dt);
    static Eigen::Matrix4d se3_to_SE3(const Eigen::Matrix<double, 6, 1>& speed, const double dt);

    static cv::Mat SE3_to_se3(const cv::Mat& Tcr, const double dt);
    static cv::Mat se3_to_SE3(const cv::Mat& speed, const double dt);

    /* -------------------------------------------------------------------------------------------------------------- */

    static cv::Mat toCvSE3(const cv::Mat& R, const cv::Mat& t);
    static cv::Mat toCvSE3Inv(const cv::Mat& CvT);
    static void scaleSE3(cv::Mat& Tsc, const cv::Mat& Tscaler);
    static cv::Mat getTc1c0(const cv::Mat& Tc0w, const cv::Mat& Tc1w);
    static cv::Mat getCurrTcw(const cv::Mat& Tc0w, const cv::Mat& Tcc0);

    static g2o::SE3Quat interpTcw(const g2o::SE3Quat& Tcw0, const g2o::SE3Quat& Tcw1, double dTs0, double dTs);

    /* -------------------------------------------------------------------------------------------------------------- */

    static std::string toString(const cv::Mat& pose);
    static std::string toString(const g2o::SE3Quat& pose);
    static std::string toString(const g2o::Sim3& pose);
    static std::string toString(const Eigen::MatrixXd& pose);

    static std::string toStringQuatRaw(const cv::Mat& pose, const int prec=9, const std::string& delim=" ");
    static std::string toStringQuatRaw(const cv::Mat& t, const cv::Mat& R, const int prec=9, const std::string& delim=" ");
    static std::string toStringQuat(const cv::Mat& pose);
    static std::string toStringQuat(const g2o::SE3Quat& pose);
    static std::string toStringQuat(const g2o::Sim3& pose);
};

}// namespace ORB_SLAM

#endif // CONVERTER_H
