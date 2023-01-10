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

#include "Converter.h"

#include <iomanip>

using namespace std;

namespace ORB_SLAM3
{

const float eps = 1e-4;

std::vector<cv::Mat> Converter::toDescriptorVector(const cv::Mat &Descriptors)
{
    std::vector<cv::Mat> vDesc;
    vDesc.reserve(Descriptors.rows);
    for (int j=0;j<Descriptors.rows;j++)
        vDesc.push_back(Descriptors.row(j));

    return vDesc;
}

g2o::SE3Quat Converter::toSE3Quat(const cv::Mat &cvT)
{
    Eigen::Matrix<double,3,3> R;
    R << cvT.at<float>(0,0), cvT.at<float>(0,1), cvT.at<float>(0,2),
         cvT.at<float>(1,0), cvT.at<float>(1,1), cvT.at<float>(1,2),
         cvT.at<float>(2,0), cvT.at<float>(2,1), cvT.at<float>(2,2);

    Eigen::Matrix<double,3,1> t(cvT.at<float>(0,3), cvT.at<float>(1,3), cvT.at<float>(2,3));

    return g2o::SE3Quat(R,t);
}

cv::Mat Converter::toCvMat(const g2o::SE3Quat &SE3)
{
    Eigen::Matrix<double,4,4> eigMat = SE3.to_homogeneous_matrix();
    return toCvMat(eigMat);
}

cv::Mat Converter::toCvMat(const g2o::Sim3 &Sim3)
{
    Eigen::Matrix3d eigR = Sim3.rotation().toRotationMatrix();
    Eigen::Vector3d eigt = Sim3.translation();
    double s = Sim3.scale();
    return toCvSE3(s*eigR,eigt);
}

cv::Mat Converter::toCvMat(const Eigen::Matrix<double,4,4> &m)
{
    cv::Mat cvMat(4,4,CV_32F);
    for(int i=0;i<4;i++)
        for(int j=0; j<4; j++)
            cvMat.at<float>(i,j)=m(i,j);

    return cvMat.clone();
}

cv::Mat Converter::toCvMat(const Eigen::Matrix3d &m)
{
    cv::Mat cvMat(3,3,CV_32F);
    for(int i=0;i<3;i++)
        for(int j=0; j<3; j++)
            cvMat.at<float>(i,j)=m(i,j);

    return cvMat.clone();
}

cv::Mat Converter::toCvMat(const Eigen::MatrixXd &m)
{
    cv::Mat cvMat(m.rows(),m.cols(),CV_32F);
    for(int i=0;i<m.rows();i++)
        for(int j=0; j<m.cols(); j++)
            cvMat.at<float>(i,j)=m(i,j);

    return cvMat.clone();
}

cv::Mat Converter::toCvMat(const Eigen::Matrix<double,3,1> &m)
{
    cv::Mat cvMat(3,1,CV_32F);
    for(int i=0;i<3;i++)
            cvMat.at<float>(i)=m(i);

    return cvMat.clone();
}

cv::Mat Converter::toCvSE3(const Eigen::Matrix<double,3,3> &R, const Eigen::Matrix<double,3,1> &t)
{
    cv::Mat cvMat = cv::Mat::eye(4,4,CV_32F);
    for(int i=0;i<3;i++)
    {
        for(int j=0;j<3;j++)
        {
            cvMat.at<float>(i,j)=R(i,j);
        }
    }
    for(int i=0;i<3;i++)
    {
        cvMat.at<float>(i,3)=t(i);
    }

    return cvMat.clone();
}

Eigen::Matrix<double,3,1> Converter::toVector3d(const cv::Mat &cvVector)
{
    Eigen::Matrix<double,3,1> v;
    v << cvVector.at<float>(0), cvVector.at<float>(1), cvVector.at<float>(2);

    return v;
}

Eigen::Matrix<double,3,1> Converter::toVector3d(const cv::Point3f &cvPoint)
{
    Eigen::Matrix<double,3,1> v;
    v << cvPoint.x, cvPoint.y, cvPoint.z;

    return v;
}

Eigen::Matrix<double,3,3> Converter::toMatrix3d(const cv::Mat &cvMat3)
{
    Eigen::Matrix<double,3,3> M;

    M << cvMat3.at<float>(0,0), cvMat3.at<float>(0,1), cvMat3.at<float>(0,2),
         cvMat3.at<float>(1,0), cvMat3.at<float>(1,1), cvMat3.at<float>(1,2),
         cvMat3.at<float>(2,0), cvMat3.at<float>(2,1), cvMat3.at<float>(2,2);

    return M;
}

Eigen::Matrix<double,4,4> Converter::toMatrix4d(const cv::Mat &cvMat4)
{
    Eigen::Matrix<double,4,4> M;

    M << cvMat4.at<float>(0,0), cvMat4.at<float>(0,1), cvMat4.at<float>(0,2), cvMat4.at<float>(0,3),
         cvMat4.at<float>(1,0), cvMat4.at<float>(1,1), cvMat4.at<float>(1,2), cvMat4.at<float>(1,3),
         cvMat4.at<float>(2,0), cvMat4.at<float>(2,1), cvMat4.at<float>(2,2), cvMat4.at<float>(2,3),
         cvMat4.at<float>(3,0), cvMat4.at<float>(3,1), cvMat4.at<float>(3,2), cvMat4.at<float>(3,3);
    return M;
}


std::vector<float> Converter::toQuaternion(const cv::Mat &M)
{
    Eigen::Matrix<double,3,3> eigMat = toMatrix3d(M);
    Eigen::Quaterniond q(eigMat);

    std::vector<float> v(4);
    v[0] = q.x();
    v[1] = q.y();
    v[2] = q.z();
    v[3] = q.w();

    return v;
}

cv::Mat Converter::tocvSkewMatrix(const cv::Mat &v)
{
    return (cv::Mat_<float>(3,3) <<             0, -v.at<float>(2), v.at<float>(1),
            v.at<float>(2),               0,-v.at<float>(0),
            -v.at<float>(1),  v.at<float>(0),              0);
}

bool Converter::isRotationMatrix(const cv::Mat &R)
{
    cv::Mat Rt;
    cv::transpose(R, Rt);
    cv::Mat shouldBeIdentity = Rt * R;
    cv::Mat I = cv::Mat::eye(3,3, shouldBeIdentity.type());

    return  cv::norm(I, shouldBeIdentity) < 1e-6;

}

std::vector<float> Converter::toEuler(const cv::Mat &R)
{
    assert(isRotationMatrix(R));
    float sy = sqrt(R.at<float>(0,0) * R.at<float>(0,0) +  R.at<float>(1,0) * R.at<float>(1,0) );

    bool singular = sy < 1e-6; // If

    float x, y, z;
    if (!singular)
    {
        x = atan2(R.at<float>(2,1) , R.at<float>(2,2));
        y = atan2(-R.at<float>(2,0), sy);
        z = atan2(R.at<float>(1,0), R.at<float>(0,0));
    }
    else
    {
        x = atan2(-R.at<float>(1,2), R.at<float>(1,1));
        y = atan2(-R.at<float>(2,0), sy);
        z = 0;
    }

    std::vector<float> v_euler(3);
    v_euler[0] = x;
    v_euler[1] = y;
    v_euler[2] = z;

    return v_euler;
}

/* ------------------------------------------------------------------------------------------------------------------ */

cv::Mat Converter::NormalizeRotation(const cv::Mat &R)
{
    cv::Mat U,w,Vt;
    cv::SVDecomp(R,w,U,Vt,cv::SVD::FULL_UV);
    // assert(cv::determinant(U*Vt)>0);
    return U*Vt;
}

cv::Mat Converter::Skew(const cv::Mat &v)
{
    const float x = v.at<float>(0);
    const float y = v.at<float>(1);
    const float z = v.at<float>(2);
    return (cv::Mat_<float>(3,3) << 0, -z, y,
            z, 0, -x,
            -y,  x, 0);
}

cv::Mat Converter::ExpSO3(const float &x, const float &y, const float &z)
{
    cv::Mat I = cv::Mat::eye(3,3,CV_32F);
    const float d2 = x*x+y*y+z*z;
    const float d = sqrt(d2);
    cv::Mat W = (cv::Mat_<float>(3,3) << 0, -z, y,
            z, 0, -x,
            -y,  x, 0);
    if(d<eps)
        return (I + W + 0.5f*W*W);
    else
        return (I + W*sin(d)/d + W*W*(1.0f-cos(d))/d2);
}

cv::Mat Converter::ExpSO3(const cv::Mat &v)
{
    return ExpSO3(v.at<float>(0),v.at<float>(1),v.at<float>(2));
}

cv::Mat Converter::LogSO3(const cv::Mat &R)
{
    const float tr = R.at<float>(0,0)+R.at<float>(1,1)+R.at<float>(2,2);
    cv::Mat w = (cv::Mat_<float>(3,1) <<(R.at<float>(2,1)-R.at<float>(1,2))/2,
            (R.at<float>(0,2)-R.at<float>(2,0))/2,
            (R.at<float>(1,0)-R.at<float>(0,1))/2);
    const float costheta = (tr-1.0f)*0.5f;
    if(costheta>1 || costheta<-1)
        return w;
    const float theta = acos(costheta);
    const float s = sin(theta);
    if(fabs(s)<eps)
        return w;
    else
        return theta*w/s;
}

cv::Mat Converter::RightJacobianSO3(const float &x, const float &y, const float &z)
{
    cv::Mat I = cv::Mat::eye(3,3,CV_32F);
    const float d2 = x*x+y*y+z*z;
    const float d = sqrt(d2);
    cv::Mat W = (cv::Mat_<float>(3,3) << 0, -z, y,
            z, 0, -x,
            -y,  x, 0);
    if(d<eps)
    {
        return cv::Mat::eye(3,3,CV_32F);
    }
    else
    {
        return I - W*(1.0f-cos(d))/d2 + W*W*(d-sin(d))/(d2*d);
    }
}

cv::Mat Converter::RightJacobianSO3(const cv::Mat &v)
{
    return RightJacobianSO3(v.at<float>(0),v.at<float>(1),v.at<float>(2));
}

cv::Mat Converter::InverseRightJacobianSO3(const float &x, const float &y, const float &z)
{
    cv::Mat I = cv::Mat::eye(3,3,CV_32F);
    const float d2 = x*x+y*y+z*z;
    const float d = sqrt(d2);
    cv::Mat W = (cv::Mat_<float>(3,3) << 0, -z, y,
            z, 0, -x,
            -y,  x, 0);
    if(d<eps)
    {
        return cv::Mat::eye(3,3,CV_32F);
    }
    else
    {
        return I + W/2 + W*W*(1.0f/d2 - (1.0f+cos(d))/(2.0f*d*sin(d)));
    }
}

cv::Mat Converter::InverseRightJacobianSO3(const cv::Mat &v)
{
    return InverseRightJacobianSO3(v.at<float>(0),v.at<float>(1),v.at<float>(2));
}

// SO3 FUNCTIONS
Eigen::Matrix3d Converter::ExpSO3(const Eigen::Vector3d &w)
{
    return Converter::ExpSO3(w[0],w[1],w[2]);
}

Eigen::Matrix3d Converter::ExpSO3(const double &x, const double &y, const double &z)
{
Eigen::Matrix<double,3,3> I = Eigen::MatrixXd::Identity(3,3);
const double d2 = x*x+y*y+z*z;
const double d = sqrt(d2);
Eigen::Matrix<double,3,3> W;
W(0,0) = 0;
W(0,1) = -z;
W(0,2) = y;
W(1,0) = z;
W(1,1) = 0;
W(1,2) = -x;
W(2,0) = -y;
W(2,1) = x;
W(2,2) = 0;

if(d<eps)
    return (I + W + 0.5*W*W);
else
    return (I + W*sin(d)/d + W*W*(1.0-cos(d))/d2);
}

Eigen::Vector3d Converter::LogSO3(const Eigen::Matrix3d &R)
{
    const double tr = R(0,0)+R(1,1)+R(2,2);
    Eigen::Vector3d w;
    w << (R(2,1)-R(1,2))/2, (R(0,2)-R(2,0))/2, (R(1,0)-R(0,1))/2;
    const double costheta = (tr-1.0)*0.5f;
    if(costheta>1 || costheta<-1)
        return w;
    const double theta = acos(costheta);
    const double s = sin(theta);
    if(fabs(s)<1e-5)
        return w;
    else
        return theta*w/s;
}

Eigen::Matrix3d Converter::InverseRightJacobianSO3(const Eigen::Vector3d &v)
{
    return InverseRightJacobianSO3(v[0],v[1],v[2]);
}

Eigen::Matrix3d Converter::InverseRightJacobianSO3(const double x, const double y, const double z)
{
    const double d2 = x*x+y*y+z*z;
    const double d = sqrt(d2);

    Eigen::Matrix3d W;
    W << 0.0, -z, y,z, 0.0, -x,-y,  x, 0.0;
    if(d<1e-5)
        return Eigen::Matrix3d::Identity();
    else
        return Eigen::Matrix3d::Identity() + W/2 + W*W*(1.0/d2 - (1.0+cos(d))/(2.0*d*sin(d)));
}

Eigen::Matrix3d Converter::RightJacobianSO3(const Eigen::Vector3d &v)
{
    return RightJacobianSO3(v[0],v[1],v[2]);
}

Eigen::Matrix3d Converter::RightJacobianSO3(const double x, const double y, const double z)
{
    const double d2 = x*x+y*y+z*z;
    const double d = sqrt(d2);

    Eigen::Matrix3d W;
    W << 0.0, -z, y,z, 0.0, -x,-y,  x, 0.0;
    if(d<1e-5)
    {
        return Eigen::Matrix3d::Identity();
    }
    else
    {
        return Eigen::Matrix3d::Identity() - W*(1.0-cos(d))/d2 + W*W*(d-sin(d))/(d2*d);
    }
}

Eigen::Matrix3d Converter::Skew(const Eigen::Vector3d &w)
{
    Eigen::Matrix3d W;
    W << 0.0, -w[2], w[1],w[2], 0.0, -w[0],-w[1],  w[0], 0.0;
    return W;
}

Eigen::Matrix3d Converter::NormalizeRotation(const Eigen::Matrix3d &R)
{
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R,Eigen::ComputeFullU | Eigen::ComputeFullV);
    return svd.matrixU()*svd.matrixV();
}

Eigen::Matrix<double, 6, 1>
Converter::SE3_to_se3(const Eigen::Matrix3d &R, const Eigen::Vector3d &t, const double dt) {

    double invDt = 1.0 / dt;
    Eigen::Vector3d omega = LogSO3(R) * invDt;
    Eigen::Vector3d v = t * invDt;
    Eigen::Matrix<double, 6, 1> speed;

    speed << omega[0], omega[1], omega[2], v[0], v[1], v[2];
    return speed;
}

Eigen::Matrix<double, 6, 1> Converter::SE3_to_se3(const Eigen::Matrix4d &T, const double dt) {

    return SE3_to_se3(T.block<3,3>(0,0), T.block<3,1>(0,3), dt);
}

Eigen::Matrix4d Converter::se3_to_SE3(const Eigen::Matrix<double, 6, 1> &speed, const double dt) {

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3,3>(0,0) = ExpSO3(speed.block<3,1>(0,0) * dt);
    T.block<3,1>(0,3) = speed.block<3,1>(3,0) * dt;
    return T;
}

cv::Mat Converter::SE3_to_se3(const cv::Mat &Tcr, const double dt) {

    double invDt = 1.0/dt;
    cv::Mat speed = cv::Mat::zeros(6,1,CV_32FC1);
    speed.rowRange(0,3).col(0) = LogSO3(Tcr.rowRange(0,3).colRange(0,3) * invDt);
    speed.rowRange(3,6).col(0) = Tcr.rowRange(0,3).col(3) * invDt;
    return speed.clone();
}

cv::Mat Converter::se3_to_SE3(const cv::Mat &speed, const double dt) {

    cv::Mat T = cv::Mat::eye(4,4,CV_32FC1);
    T.rowRange(0,3).colRange(0,3) = ExpSO3(speed.rowRange(0,3).col(0) * dt);
    T.rowRange(0,3).col(3) = speed.rowRange(3,6).col(0) * dt;
    return T.clone();
}

/* ------------------------------------------------------------------------------------------------------------------ */

cv::Mat Converter::toCvSE3(const cv::Mat &R, const cv::Mat &t) {

    cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);

    R.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
    t.copyTo(Tcw.rowRange(0,3).col(3));

    return Tcw.clone();
}

cv::Mat Converter::toCvSE3Inv(const cv::Mat &CvT) {

    cv::Mat Tinv = cv::Mat::eye(4,4,CV_32F);

    cv::Mat Rt = CvT.rowRange(0,3).colRange(0,3).t();
    Rt.copyTo(Tinv.rowRange(0, 3).colRange(0, 3));

    cv::Mat tinv = - Rt * CvT.rowRange(0,3).col(3);
    tinv.copyTo(Tinv.rowRange(0,3).col(3));

    return Tinv.clone();
}

void Converter::scaleSE3(cv::Mat &Tsc, const cv::Mat &Tscaler) {

    float sc1 = cv::norm(Tscaler.rowRange(0,3).col(3));
    cv::Mat tsc = Tsc.rowRange(0,3).col(3).clone();
    float sc2 = cv::norm(tsc);

    tsc = (sc1/sc2) * tsc;

    tsc.copyTo(Tsc.rowRange(0,3).col(3));
}

// Use this carefully (care to what is relative to what)!
cv::Mat Converter::getTc1c0(const cv::Mat& Tc0w, const cv::Mat& Tc1w) {

    return Tc1w * toCvSE3Inv(Tc0w);
}

cv::Mat Converter::getCurrTcw(const cv::Mat &Tc0w, const cv::Mat &Tcc0) {

    return Tcc0 * Tc0w;
}

// TODO: This seem to be mathematically wrong
g2o::SE3Quat Converter::interpTcw(const g2o::SE3Quat &Tcw0, const g2o::SE3Quat &Tcw1, const double dTs0, const double dTs) {

    g2o::SE3Quat Tcwi;
    const double s = dTs0 / dTs;

    Tcwi.setRotation(Tcw0.rotation().slerp(s, Tcw1.rotation()));
    Tcwi.setTranslation(Tcw0.translation() * (1-s) + Tcw1.translation() * s);

    return Tcwi;
}

/* ================================================================================================================== */
// Pose to string

std::string Converter::toString(const cv::Mat& pose) {

    std::ostringstream oss;

    oss << "[";
    for (int i = 0; i < pose.rows; i++) {
        for (int j = 0; j < pose.cols; j++) {
            oss << pose.at<float>(i,j) << ((j == pose.cols-1) ? "" : " ");
        }
        oss << ((i == pose.rows-1) ? "]\n" : "\n");
    }
    return oss.str();
}

std::string Converter::toString(const g2o::SE3Quat& pose) {

    return toString(toCvMat(pose));
}

std::string Converter::toString(const g2o::Sim3& pose) {

    std::ostringstream oss;
    oss << toString(g2o::SE3Quat(pose.rotation(), pose.translation()));
    oss << ", s = " << pose.scale() << endl;
    return oss.str();
}

std::string Converter::toString(const Eigen::MatrixXd& pose) {

    return toString(toCvMat(pose));
}

std::string Converter::toStringQuatRaw(const cv::Mat &pose, const int prec, const std::string& delim) {

    stringstream oss;

    vector<float> Qcur = ORB_SLAM3::Converter::toQuaternion(pose.rowRange(0,3).colRange(0,3));
    cv::Mat trans = pose.rowRange(0,3).col(3);

    oss << setprecision(prec) << trans.at<float>(0,0) << delim << trans.at<float>(1,0) << delim <<
        trans.at<float>(2,0) << delim;

    for (size_t i = 0; i < Qcur.size(); i++) {
        oss << Qcur[i] << ((i == Qcur.size()-1) ? "" : delim);
    }

    return oss.str();
}

std::string Converter::toStringQuatRaw(const cv::Mat &t, const cv::Mat& R, const int prec, const std::string& delim) {

    stringstream oss;

    vector<float> Qcur = ORB_SLAM3::Converter::toQuaternion(R);

    oss << setprecision(prec) << t.at<float>(0,0) << delim << t.at<float>(1,0) << delim <<
        t.at<float>(2,0) << delim;

    for (size_t i = 0; i < Qcur.size(); i++) {
        oss << Qcur[i] << ((i == Qcur.size()-1) ? "" : delim);
    }

    return oss.str();
}

std::string Converter::toStringQuat(const cv::Mat &pose) {

    stringstream oss;

    vector<float> Qcur = ORB_SLAM3::Converter::toQuaternion(pose.rowRange(0,3).colRange(0,3));
    cv::Mat trans = pose.rowRange(0,3).col(3);

    oss << "[" << trans.at<float>(0,0) << ", " << trans.at<float>(1,0) << ", " <<
                  trans.at<float>(2,0) << ", ";

    for (size_t i = 0; i < Qcur.size(); i++) {
        oss << Qcur[i] << ((i == Qcur.size()-1) ? "" : ", ");
    }
    oss << "]\n";

    return oss.str();
}

std::string Converter::toStringQuat(const g2o::SE3Quat& pose) {

    ostringstream oss;
    oss << "[" << pose.translation().x() << ", " << pose.translation().y() << ", " << pose.translation().z() << ", ";
    oss << pose.rotation().x() << ", " << pose.rotation().y() << ", " <<
           pose.rotation().z() << ", " << pose.rotation().w() << "]\n";
    return oss.str();
}

std::string Converter::toStringQuat(const g2o::Sim3& pose) {

    ostringstream oss;
    oss << toStringQuat(g2o::SE3Quat(pose.rotation(), pose.translation()));
    oss << ", scale = " << pose.scale() << endl;
    return oss.str();
}

} //namespace ORB_SLAM
