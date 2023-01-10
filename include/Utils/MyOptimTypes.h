//
// Created by root on 9/1/21.
//

#ifndef ORB_SLAM3_MYOPTIMTYPES_H
#define ORB_SLAM3_MYOPTIMTYPES_H

#include "OptimizableTypes.h"
#include "G2oTypes.h"


namespace EORB_SLAM {

    /* -------------------------------------------------------------------------------------------------------------- */

    class  EdgeEventContrast: public  g2o::BaseUnaryEdge<1, double, g2o::VertexSE3Expmap>{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        EdgeEventContrast() : mMedianDepth(1.f), mpCamera(nullptr), mImSigma(1.f) {}
        EdgeEventContrast(std::vector<EORB_SLAM::EventData> vEvData, const float medDepth,
                          ORB_SLAM3::GeometricCamera* pCamera, const cv::Size& imSize, const float imSigma):
                mEvents(std::move(vEvData)), mMedianDepth(medDepth), mpCamera(pCamera), mImSize(imSize), mImSigma(imSigma) {}

        bool read(std::istream& is) override;

        bool write(std::ostream& os) const override;

        void computeError() override  {
            const g2o::VertexSE3Expmap* v1 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);

            // Reconstruct Image
            cv::Mat mcImage = EORB_SLAM::EvImConverter::ev2mci_gg_f(mEvents, mpCamera, ORB_SLAM3::Converter::toCvMat(v1->estimate()),
                                                                    mMedianDepth, mImSize.width, mImSize.height, mImSigma, false, false);

            // Calculate image variance
            double imSTD = EORB_SLAM::EvImConverter::measureImageFocus(mcImage);
            // Minus sign to find minimum (not maximum)
            _error[0] = -(imSTD*imSTD);
        }

        void linearizeOplus() override;

        //Eigen::Vector3d Xw;
        std::vector<EORB_SLAM::EventData> mEvents;
        float mMedianDepth;
        ORB_SLAM3::GeometricCamera* mpCamera;
        cv::Size mImSize;
        float mImSigma;
    };

    /* ============================================================================================================== */

    class VertexMedDepth : public g2o::BaseVertex<1,ORB_SLAM3::InvDepthPoint>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        VertexMedDepth()= default;
        VertexMedDepth(double mDepth, double u, double v, ORB_SLAM3::KeyFrame* pHostKF){
            setEstimate(ORB_SLAM3::InvDepthPoint(mDepth, u, v, pHostKF));
        }

        bool read(std::istream& is) override{return false;}
        bool write(std::ostream& os) const override{return false;}

        void setToOriginImpl() override {
        }

        void oplusImpl(const double* update_) override{
            _estimate.Update(update_);
            updateCache();
        }
    };

    // theta, tx, ty
    class VertexSE2 : public g2o::BaseVertex<3, Eigen::Vector3d> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        VertexSE2() = default;

        bool read(std::istream& /*is*/) override
        {
            cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
            return false;
        }

        bool write(std::ostream& /*os*/) const override
        {
            cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
            return false;
        }

        void setToOriginImpl() override
        {
            cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
        }

        void oplusImpl(const double* update) override
        {
            Eigen::Vector3d::ConstMapType v(update);
            _estimate += v;
        }
    };

    // theta, tx, ty, sc
    class VertexSim2 : public g2o::BaseVertex<4, Eigen::Vector4d> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        VertexSim2() = default;

        bool read(std::istream& /*is*/) override
        {
            cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
            return false;
        }

        bool write(std::ostream& /*os*/) const override
        {
            cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
            return false;
        }

        void setToOriginImpl() override
        {
            cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
        }

        void oplusImpl(const double* update) override
        {
            Eigen::Vector4d::ConstMapType v(update);
            _estimate += v;
        }
    };

    class EdgeSE2PointXY : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexSE2>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        EdgeSE2PointXY(const float dataX, const float dataY)
        {
            _data << dataX, dataY;
        }

        bool read(std::istream& /*is*/) override
        {
            cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
            return false;
        }
        bool write(std::ostream& /*os*/) const override
        {
            cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
            return false;
        }

        void computeError() override
        {
            const VertexSE2* params = static_cast<const VertexSE2*>(vertex(0));

            const double& theta = params->estimate()(0);
            const double& tx = params->estimate()(1);
            const double& ty = params->estimate()(2);

            double xp = cos(theta) * _data[0] - sin(theta) * _data[1] + tx;
            double yp = sin(theta) * _data[0] + cos(theta) * _data[1] + ty;

            _error(0) = xp - measurement()(0);
            _error(1) = yp - measurement()(1);
        }
    protected:
        Eigen::Vector2d _data;
    };

    class EdgeSim2PointXY : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexSim2>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        EdgeSim2PointXY(const float dataX, const float dataY)
        {
            _data << dataX, dataY;
        }

        bool read(std::istream& /*is*/) override
        {
            cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
            return false;
        }
        bool write(std::ostream& /*os*/) const override
        {
            cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
            return false;
        }

        void computeError() override
        {
            const VertexSim2* params = static_cast<const VertexSim2*>(vertex(0));

            const double& theta = params->estimate()(0);
            const double& tx = params->estimate()(1);
            const double& ty = params->estimate()(2);
            const double& sc = params->estimate()(3);

            double xp = sc * (cos(theta) * _data[0] - sin(theta) * _data[1]) + tx;
            double yp = sc * (sin(theta) * _data[0] + cos(theta) * _data[1]) + ty;

            _error(0) = xp - measurement()(0);
            _error(1) = yp - measurement()(1);
        }
    protected:
        Eigen::Vector2d _data;
    };

    // 2 SE3 vertex must remain the same or scaled versions of each other
    /*class EdgeSamePose : public g2o::BaseMultiEdge<6, ORB_SLAM3::Vector6d> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

//        explicit EdgeSamePose(const Eigen::Matrix4d &Tc1c2){
//            //Tij = deltaT;
//            Rij = Tc1c2.block<3,3>(0,0);
//            tij = Tc1c2.block<3,1>(0,3);
//        }
//
//        explicit EdgeSamePose(const g2o::SE3Quat &Tc1c2){
//            //Tij = deltaT;
//            Rij = Tc1c2.rotation().toRotationMatrix();
//            tij = Tc1c2.translation();
//        }
        explicit EdgeSamePose(const bool scaled = false) { isScaled = scaled; }

        bool read(std::istream& is) override {return false;}
        bool write(std::ostream& os) const override {return false;}

        void computeError() override;

        void linearizeOplus() override; // numerical implementation

        //Eigen::Matrix4d Tij;
        //Eigen::Matrix3d Rij;
        //Eigen::Vector3d tij;
        bool isScaled;
    };*/

    // ** Note: These edges work with Tcw (not Twc)
    // Based on EdgePriorPoseImu, non-IMU -> All use (u+du) replacement technique
    // Can use other edges instead -> use a DPose edge with one vertex fixed and Tij = I(4)
    /*class EdgePosePrior : public g2o::BaseBinaryEdge<6, ORB_SLAM3::Vector6d, g2o::VertexSE3Expmap, ORB_SLAM3::VertexScale>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        explicit EdgePosePrior(ORB_SLAM3::Frame* pFrame, bool scaled = false);
        explicit EdgePosePrior(ORB_SLAM3::KeyFrame* pKF, bool scaled = false);

        bool read(std::istream& is) override{return false;}
        bool write(std::ostream& os) const override{return false;}

        void computeError() override;
        void linearizeOplus() override;

        Eigen::Matrix<double,6,6> GetHessian() {
            linearizeOplus();
            Eigen::Matrix<double,6,6> J;
            J.block<6,6>(0,0) = _jacobianOplusXi;
            return J.transpose()*information()*J;
        }

        Eigen::Matrix3d Rcw, Rwc;
        Eigen::Vector3d tcw, twc;

        bool isScaled;
    };*/

    // ** Note: Use g2o's default EdgeSE3 for Tcw case, this is Twb case
    // Based on EdgeInertialGS, Error between dPose(vp1, vp2) (scaled) and measurement.
    // Can implement SamePose with this if the obs = I
    class EdgeDPosePrior : public g2o::BaseMultiEdge<6, ORB_SLAM3::Vector6d>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        // EdgeInertialGS(IMU::Preintegrated* pInt);
        EdgeDPosePrior(ORB_SLAM3::Frame* pPrevFr, ORB_SLAM3::Frame* pCurrFr, bool scaled = false);
        EdgeDPosePrior(ORB_SLAM3::KeyFrame* pPrevKF, ORB_SLAM3::KeyFrame* pCurrKF, bool scaled = false);

        bool read(std::istream& is) override{return false;}
        bool write(std::ostream& os) const override{return false;}

        void computeError() override;
        void linearizeOplus() override;

        Eigen::Matrix3d Rb0b1;
        Eigen::Vector3d tb0b1;

        bool isScaled;
    };

    // Based on G2oTypes' EdgeMono
    class EdgePoseMedDepth : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, ORB_SLAM3::VertexPose, ORB_SLAM3::VertexScale> {
    public:
        EdgePoseMedDepth(const cv::Point2d& pt2d, ORB_SLAM3::GeometricCamera* pCamera, int _cam_idx = 0);
        EdgePoseMedDepth(Eigen::Vector3d  _Pt2d, int _cam_idx = 0);

        bool read(std::istream& is) override{return false;}
        bool write(std::ostream& os) const override{return false;}

        void computeError() override;

        void linearizeOplus() override;

        bool isDepthPositive()
        {
            //const g2o::VertexSBAPointXYZ* VPoint = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
            const ORB_SLAM3::VertexPose* VPose = static_cast<const ORB_SLAM3::VertexPose*>(_vertices[0]);
            ORB_SLAM3::VertexScale* VZ = static_cast<ORB_SLAM3::VertexScale*>(_vertices[1]);
            return VPose->estimate().isDepthPositive(VZ->estimate() * Pt2d, cam_idx);
        }

        /*Eigen::Matrix<double,2,9> GetJacobian(){
            linearizeOplus();
            Eigen::Matrix<double,2,9> J;
            J.block<2,3>(0,0) = _jacobianOplusXi;
            J.block<2,6>(0,3) = _jacobianOplusXj;
            return J;
        }

        Eigen::Matrix<double,9,9> GetHessian(){
            linearizeOplus();
            Eigen::Matrix<double,2,9> J;
            J.block<2,3>(0,0) = _jacobianOplusXi;
            J.block<2,6>(0,3) = _jacobianOplusXj;
            return J.transpose()*information()*J;
        }*/

        // Normalized reference 2d point in homo-coords
        Eigen::Vector3d Pt2d;
        //cv::Point2f pt0;
        //ORB_SLAM3::GeometricCamera* pCamera;

        const int cam_idx;
    };

} // EORB_SLAM


#endif //ORB_SLAM3_MYOPTIMTYPES_H
