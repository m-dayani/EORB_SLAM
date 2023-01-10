//
// Created by root on 9/1/21.
//

#include "MyOptimTypes.h"

#include <utility>

using namespace ORB_SLAM3;

namespace EORB_SLAM {

    void EdgeEventContrast::linearizeOplus() {
        g2o::VertexSE3Expmap * vi = static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);
        //TODO: ???? Do we need minus sign????
        _jacobianOplusXi = -EORB_SLAM::EvImConverter::ev2mci_gg_f_jac(mEvents, mpCamera, vi, mMedianDepth,
                                                                      mImSize.width, mImSize.height, mImSigma);
    }

    bool EdgeEventContrast::read(std::istream& is)
    {
        is >> _measurement >> information()(0,0);
        return true;
    }

    bool EdgeEventContrast::write(std::ostream& os) const
    {
        os  << _measurement << " " << information()(0,0);
        return os.good();
    }

    /* ============================================================================================================== */

    /*void EdgeSamePose::computeError()
    {
        const g2o::VertexSE3Expmap* VPi = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        const g2o::VertexSE3Expmap* VPj = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);

        Eigen::Matrix3d Riw = VPi->estimate().rotation().toRotationMatrix();
        Eigen::Vector3d tiw = VPi->estimate().translation();

        Eigen::Matrix3d Rjw = VPj->estimate().rotation().toRotationMatrix();
        Eigen::Vector3d tjw = VPj->estimate().translation();

        double s = 1.0;
        const VertexScale* VS = static_cast<const VertexScale*>(_vertices[2]);

        if (isScaled && VS) {
            s = VS->estimate();
        }

        // TODO: Check These Tcw vs. Twc???
        Eigen::Vector3d er = LogSO3(Riw*Rjw.transpose()); // * I(3) -> implicit observation
        Eigen::Vector3d et = Riw * (-s * Rjw.transpose() * tjw) + tiw; // -Zeros(3) -> obs

        _error << er, et;
    }

    void EdgeSamePose::linearizeOplus() {

        const g2o::VertexSE3Expmap* VPi = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        const g2o::VertexSE3Expmap* VPj = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);

        Eigen::Matrix3d Riw = VPi->estimate().rotation().toRotationMatrix();
        Eigen::Vector3d tiw = VPi->estimate().translation();

        Eigen::Matrix3d Rjw = VPj->estimate().rotation().toRotationMatrix();
        Eigen::Vector3d tjw = VPj->estimate().translation();

        double s = 1.0;
        const VertexScale* VS = static_cast<const VertexScale*>(_vertices[2]);

        if (isScaled && VS) {
            s = VS->estimate();
        }

        Eigen::Vector3d er = LogSO3(Riw*Rjw.transpose());
        const Eigen::Matrix3d invJr = InverseRightJacobianSO3(er);

        // Pose 1
        _jacobianOplus[0].setZero();
        // rotation
        _jacobianOplus[0].block<3,3>(0,0) = invJr * Rjw * Rjw.transpose();
        _jacobianOplus[0].block<3,3>(3,0) = s * Skew(Riw * Rjw.transpose() * tjw);
        // translation
        _jacobianOplus[0].block<3,3>(3,3) = Riw;

        // Pose 2
        _jacobianOplus[1].setZero();
        // rotation
        _jacobianOplus[1].block<3,3>(0,6) = -invJr * Rjw;
        _jacobianOplus[1].block<3,3>(3,6) = -s * Riw * Rjw.transpose() * Skew(tjw);
        // translation
        _jacobianOplus[1].block<3,3>(3,9) = -s * Riw;

        // Scale
        _jacobianOplus[2].setZero();
        if (isScaled) {
            _jacobianOplus[2].block<3,1>(3,12) = -s * Riw * Rjw.transpose() * tjw;
        }
    }*/

    // ** Big Note! To avoid misinterpretation use VertexPose (Twb) which we know how it's updated

    /*EdgePosePrior::EdgePosePrior(Frame *pFrame, const bool scaled) : isScaled(scaled) {

        Rcw = Converter::toMatrix3d(pFrame->mTcw.rowRange(0,3).colRange(0,3));
        Rwc = Rcw.transpose();
        tcw = Converter::toVector3d(pFrame->mTcw.rowRange(0,3).col(3));
        twc = -Rwc * tcw;
    }

    EdgePosePrior::EdgePosePrior(KeyFrame *pKF, const bool scaled) : isScaled(scaled) {

        Rcw = Converter::toMatrix3d(pKF->GetRotation());
        Rwc = Rcw.transpose();
        tcw = Converter::toVector3d(pKF->GetTranslation());
        twc = -Rwc * tcw;
    }

    void EdgePosePrior::computeError() {

        const g2o::VertexSE3Expmap* VP = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        Eigen::Matrix3d Rcw_est = VP->estimate().rotation().toRotationMatrix();
        Eigen::Matrix3d Rwc_est = Rcw_est.transpose();
        Eigen::Vector3d tcw_est = VP->estimate().translation();
        Eigen::Vector3d twc_est = - Rwc_est * tcw_est;

        const VertexScale* VS = static_cast<const VertexScale*>(_vertices[1]);
        double s = 1.0;
        if (isScaled && VS) {
            s = VS->estimate();
        }

        const Eigen::Vector3d er = LogSO3(Rwc.transpose()*Rwc_est);
        const Eigen::Vector3d et = Rwc.transpose()*(s*twc_est-twc);

        _error << er, et;
    }

    void EdgePosePrior::linearizeOplus() {

        const g2o::VertexSE3Expmap* VP = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        Eigen::Matrix3d Rcw_est = VP->estimate().rotation().toRotationMatrix();
        Eigen::Matrix3d Rwc_est = Rcw_est.transpose();
        Eigen::Vector3d tcw_est = VP->estimate().translation();
        Eigen::Vector3d twc_est = - Rwc_est * tcw_est;

        const VertexScale* VS = static_cast<const VertexScale*>(_vertices[1]);
        double s = VS->estimate();

        const Eigen::Vector3d er = LogSO3(Rwc.transpose()*Rwc_est);

        _jacobianOplusXi.setZero();
        _jacobianOplusXi.block<3,3>(0,0) = InverseRightJacobianSO3(er);
        _jacobianOplusXi.block<3,3>(3,3) = s * Rwc.transpose() * Rwc_est;

        _jacobianOplusXj.setZero();
        _jacobianOplusXj.block<3,1>(3,6) = s * Rwc.transpose() * twc_est;
    }*/

    EdgeDPosePrior::EdgeDPosePrior(ORB_SLAM3::Frame *pPrevFr, ORB_SLAM3::Frame *pCurrFr, const bool scaled) :
        isScaled(scaled) {

        cv::Mat Tc0w = pPrevFr->mTcw;
        cv::Mat Twc1 = pCurrFr->mTcw.inv(cv::DECOMP_SVD);

        cv::Mat Tc0c1 = Tc0w * Twc1;
        Rb0b1 = Converter::toMatrix3d(Tc0c1.rowRange(0, 3).colRange(0, 3));
        tb0b1 = Converter::toVector3d(Tc0c1.rowRange(0, 3).col(3));
    }

    EdgeDPosePrior::EdgeDPosePrior(ORB_SLAM3::KeyFrame *pPrevKF, ORB_SLAM3::KeyFrame *pCurrKF, const bool scaled) :
        isScaled(scaled) {

        cv::Mat Tc0w = pPrevKF->GetPose();
        cv::Mat Twc1 = pCurrKF->GetPoseInverse();

        cv::Mat Tc0c1 = Tc0w * Twc1;
        Rb0b1 = Converter::toMatrix3d(Tc0c1.rowRange(0, 3).colRange(0, 3));
        tb0b1 = Converter::toVector3d(Tc0c1.rowRange(0, 3).col(3));
    }

    void EdgeDPosePrior::computeError() {

        VertexPose* VP1 = static_cast<VertexPose*>(_vertices[0]);
        Eigen::Matrix3d Rwb0 = VP1->estimate().Rwb;
        Eigen::Vector3d twb0 = VP1->estimate().twb;

        VertexPose* VP2 = static_cast<VertexPose*>(_vertices[1]);
        Eigen::Matrix3d Rwb1 = VP2->estimate().Rwb;
        Eigen::Vector3d twb1 = VP2->estimate().twb;

        VertexScale* VS = static_cast<VertexScale*>(_vertices[2]);
        double s = 1.0;
        if (isScaled) {
            s = VS->estimate();
        }

        // For rotation, using Rwc is not very important -> consider update: Exp(phi) Rcw
        Eigen::Vector3d er = Converter::LogSO3(Rb0b1.transpose() * Rwb0.transpose() * Rwb1);
        // Writes these in terms of twc to avoid update problems (Ri dpi + pi)
        Eigen::Vector3d et = s * Rwb0.transpose() * (twb1 - twb0) - tb0b1;

        _error << er, et;
    }

    void EdgeDPosePrior::linearizeOplus() {

        VertexPose* VPi = static_cast<VertexPose*>(_vertices[0]);
        VertexPose* VPj = static_cast<VertexPose*>(_vertices[1]);

        Eigen::Matrix3d Rwi = VPi->estimate().Rwb;
        Eigen::Vector3d twi = VPi->estimate().twb;

        Eigen::Matrix3d Rwj = VPj->estimate().Rwb;
        Eigen::Vector3d twj = VPj->estimate().twb;

        double s = 1.0;
        const VertexScale* VS = static_cast<const VertexScale*>(_vertices[2]);
        if (isScaled && VS) {
            s = VS->estimate();
        }

        Eigen::Vector3d er = Converter::LogSO3(Rb0b1.transpose() * Rwi.transpose() * Rwj);
        const Eigen::Matrix3d invJr = Converter::InverseRightJacobianSO3(er);

        // Pose 1
        _jacobianOplus[0].setZero();
        // rotation
        _jacobianOplus[0].block<3,3>(0,0) = -invJr * Rwj.transpose() * Rwi;
        _jacobianOplus[0].block<3,3>(3,0) = s * Converter::Skew(Rwi.transpose() * (twj - twi));
        // translation
        _jacobianOplus[0].block<3,3>(3,3) = -s * Eigen::Matrix3d::Identity();

        // Pose 2
        _jacobianOplus[1].setZero();
        // rotation
        _jacobianOplus[1].block<3,3>(0,6) = invJr;
        //_jacobianOplus[1].block<3,3>(3,6) = -s * Riw * Rjw.transpose() * Skew(tjw);
        // translation
        _jacobianOplus[1].block<3,3>(3,9) = s * Rwi.transpose() * Rwj;

        // Scale
        _jacobianOplus[2].setZero();
        if (isScaled) {
            _jacobianOplus[2].block<3,1>(3,12) = s * Rwi.transpose() * (twj - twi);
        }
    }

    EdgePoseMedDepth::EdgePoseMedDepth(const cv::Point2d &pt2d, ORB_SLAM3::GeometricCamera *pCamera, int _cam_idx)
            : cam_idx(_cam_idx) {

        Pt2d = Converter::toVector3d(pCamera->unproject(pt2d));
    }

    EdgePoseMedDepth::EdgePoseMedDepth(Eigen::Vector3d _Pt2d, int _cam_idx) : Pt2d(std::move(_Pt2d)), cam_idx(_cam_idx) {}

    void EdgePoseMedDepth::computeError() {

        VertexPose* VP = static_cast<VertexPose*>(_vertices[0]);
        VertexScale* VZ = static_cast<VertexScale*>(_vertices[1]);

        Eigen::Vector2d obs(_measurement);

        _error << obs - VP->estimate().Project(VZ->estimate() * Pt2d, cam_idx);
    }

    void EdgePoseMedDepth::linearizeOplus() {

        const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[0]);
        VertexScale* VZ = static_cast<VertexScale*>(_vertices[1]);
        double d = VZ->estimate();

        const Eigen::Matrix3d &Rcw = VPose->estimate().Rcw[cam_idx];
        const Eigen::Vector3d &tcw = VPose->estimate().tcw[cam_idx];
        const Eigen::Vector3d Xc = Rcw * (d * Pt2d) + tcw;
        const Eigen::Vector3d Xb = VPose->estimate().Rbc[cam_idx]*Xc+VPose->estimate().tbc[cam_idx];
        const Eigen::Matrix3d &Rcb = VPose->estimate().Rcb[cam_idx];

        const Eigen::Matrix<double,2,3> proj_jac = VPose->estimate().pCamera[cam_idx]->projectJac(Xc);

        Eigen::Matrix<double,3,6> SE3deriv;
        double x = Xb(0);
        double y = Xb(1);
        double z = Xb(2);

        SE3deriv << 0.0, z,   -y, 1.0, 0.0, 0.0,
                -z , 0.0, x, 0.0, 1.0, 0.0,
                y ,  -x , 0.0, 0.0, 0.0, 1.0;

        // Pose Jacobian is the same
        _jacobianOplusXi = proj_jac * Rcb * SE3deriv; // TODO optimize this product

        // But the scale Jac. is different -> 2x1
        _jacobianOplusXj = -proj_jac * Rcw * (d * Pt2d);
    }
} // EORB_SLAM