//
// Created by root on 1/13/21.
//

#ifndef ORB_SLAM3_EVOPTIMIZER_H
#define ORB_SLAM3_EVOPTIMIZER_H

#include <utility>
#include <vector>

#include <ceres/ceres.h>
#include <glog/logging.h>

#include <opencv2/core.hpp>

#include "EventData.h"
#include "EventConversion.h"
#include "MyCalibrator.h"

#include "GeometricCamera.h"
#include "Optimizer.h"

namespace ORB_SLAM3 {

    class EdgeMono;
    class EdgeSE3ProjectXYZ;
    class EdgeSE3ProjectXYZOnlyPose;
    class EdgeMonoOnlyPose;
}

namespace EORB_SLAM {

#define DEF_SIGMA_EF_RANGE 5.f

    class Rosenbrock : public ceres::FirstOrderFunction {
    public:
        virtual bool Evaluate(const double* parameters,
                              double* cost,
                              double* gradient) const {
            const double x = parameters[0];
            const double y = parameters[1];

            cost[0] = (1.0 - x) * (1.0 - x) + 100.0 * (y - x * x) * (y - x * x);
            if (gradient != nullptr) {
                gradient[0] = -2.0 * (1.0 - x) - 200.0 * (y - x * x) * 2.0 * x;
                gradient[1] = 200.0 * (y - x * x);
            }
            return true;
        }

        virtual int NumParameters() const { return 2; }
    };

    /*struct EvBA_ReprojectionError {
        EvBA_ReprojectionError(double observed_x, double observed_y)
                : observed_x(observed_x), observed_y(observed_y) {}

        template <typename T>
        bool operator()(const T* const camera, const T* const point, T* residuals) const {

            // camera[0,1,2] are the angle-axis rotation.
            T p[3];
            ceres::AngleAxisRotatePoint(camera, point, p);
            // camera[3,4,5] are the translation.
            p[0] += camera[3]; p[1] += camera[4]; p[2] += camera[5];

            // Compute the center of distortion. The sign change comes from
            // the camera model that Noah Snavely's Bundler assumes, whereby
            // the camera coordinate system has a negative z axis.
            T xp = p[0] / p[2];
            T yp = p[1] / p[2];

            // Compute final projected point position.
            const T& focal = camera[6];
            T predicted_x = focal * xp;
            T predicted_y = focal * yp;

            // The error is the difference between the predicted and observed position.
            residuals[0] = predicted_x - T(observed_x);
            residuals[1] = predicted_y - T(observed_y);
            return true;
        }

        // Factory to hide the construction of the CostFunction object from
        // the client code.
        static ceres::CostFunction* Create(const double observed_x,
                                           const double observed_y) {
            return (new ceres::AutoDiffCostFunction<EvBA_ReprojectionError, 2, 9, 3>(
                    new EvBA_ReprojectionError(observed_x, observed_y)));
        }

        double observed_x;
        double observed_y;
    };*/

    class EvFocus_MS_RT2D : public ceres::FirstOrderFunction {
    public:
        explicit EvFocus_MS_RT2D(std::vector<EventData>  vEvObs, std::shared_ptr<EvParams> pEvParams,
                ORB_SLAM3::GeometricCamera* pCamera) :
            mvEvObs(std::move(vEvObs)), mpCamera(pCamera), mpEvParams(std::move(pEvParams))
        {}

        virtual bool Evaluate(const double* parameters, double* cost, double* gradient) const;

        virtual int NumParameters() const { return 3; }

    private:
        std::vector<EventData> mvEvObs;
        ORB_SLAM3::GeometricCamera* mpCamera;
        std::shared_ptr<EvParams> mpEvParams;
    };

    class EvOptimizer : public ORB_SLAM3::Optimizer {
    public:
        static void optimizeFocusPlan3dof();

        static void optimizeFocus_MS_RT2D(const std::vector<EventData>&  vEvObs, std::shared_ptr<EvParams> pEvParams,
                                          ORB_SLAM3::GeometricCamera* pCamera, double& omega0, double& vx0, double& vy0);

        /* ---------------------------------------------------------------------------------------------------------- */

        void static BundleAdjustment(const std::vector<ORB_SLAM3::KeyFrame*> &vpKF, const std::vector<ORB_SLAM3::MapPoint*> &vpMP,
                                     int nIterations = 5, bool *pbStopFlag=NULL, const unsigned long nLoopKF=0,
                                     const bool bRobust = true);
        void static GlobalBundleAdjustemnt(ORB_SLAM3::Map* pMap, int nIterations=5, bool *pbStopFlag=NULL,
                                           const unsigned long nLoopKF=0, const bool bRobust = true);

        void static FullInertialBA(ORB_SLAM3::Map *pMap, int its, const bool bFixLocal=false, const unsigned long nLoopKF=0,
                                   bool *pbStopFlag=NULL, bool bInit=false, float priorG = 1e2, float priorA=1e6,
                                   Eigen::VectorXd *vSingVal = NULL, bool *bHess=NULL);

        void static LocalBundleAdjustment(ORB_SLAM3::KeyFrame* pKF, bool *pbStopFlag, ORB_SLAM3::Map *pMap, int& num_fixedKF, int& num_OptKF,
                                          int& num_MPs, int& num_edges);

        int static PoseOptimization(ORB_SLAM3::Frame* pFrame);

        int static PoseInertialOptimizationLastKeyFrame(ORB_SLAM3::Frame* pFrame, bool bRecInit = false);
        int static PoseInertialOptimizationLastFrame(ORB_SLAM3::Frame *pFrame, bool bRecInit = false);

        void static LocalInertialBA(ORB_SLAM3::KeyFrame* pKF, bool *pbStopFlag, ORB_SLAM3::Map *pMap, int& num_fixedKF, int& num_OptKF,
                int& num_MPs, int& num_edges, bool bLarge = false, bool bRecInit = false);

        /* ---------------------------------------------------------------------------------------------------------- */

        uint static setEventMapVxAndEdges(g2o::SparseOptimizer& optimizer, const std::vector<ORB_SLAM3::KeyFrame*>& vpOrbKFs,
                                          const int maxId, const double thHuber, const bool bRobust, const bool inertial);

        uint static setEventMapVxAndEdges(g2o::SparseOptimizer& optimizer, const std::set<ORB_SLAM3::MapPoint*>& spEvMpts,
                                          vector<ORB_SLAM3::EdgeSE3ProjectXYZ*>& vpEvEdgesMono,
                                          std::vector<ORB_SLAM3::KeyFrame*>& vpKFsMono, std::vector<ORB_SLAM3::MapPoint*>& vpMptsMono,
                                          int& maxId, const double thHuber, const bool bRobust);

        uint static setEventMapVxAndEdges(g2o::SparseOptimizer& optimizer, const std::set<ORB_SLAM3::MapPoint*>& spEvMpts,
                                          vector<ORB_SLAM3::EdgeMono*>& vpEvEdgesMono,
                                          std::vector<ORB_SLAM3::KeyFrame*>& vpKFsMono, std::vector<ORB_SLAM3::MapPoint*>& vpMptsMono,
                                          int& maxId, const double thHuber, const bool bRobust);

        int static setEventMapVxAndEdges(g2o::SparseOptimizer& optimizer, ORB_SLAM3::Frame* pEvFrame,
                                         vector<ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose*>& vpEdgesMonoEv,
                                         std::vector<std::size_t>& vnIndexEdgeMonoEv,
                                         int poseId, const double thHuber, const bool bRobust = true);

        int static setEventMapVxAndEdges(g2o::SparseOptimizer& optimizer, ORB_SLAM3::Frame* pEvFrame,
                                         vector<ORB_SLAM3::EdgeMonoOnlyPose*>& vpEdgesMonoEv,
                                         std::vector<std::size_t>& vnIndexEdgeMonoEv,
                                         int poseId, const double thHuber, const bool bRobust = true);

        void static retrieveAllMapPoints(const std::vector<ORB_SLAM3::KeyFrame*>& vpOrbKFs, std::set<ORB_SLAM3::MapPoint*>& spEvMpts);

        void static retrieveAllMapPoints(const std::list<ORB_SLAM3::KeyFrame*>& vpOrbKFs, std::set<ORB_SLAM3::MapPoint*>& spEvMpts);

    };

} // EORB_SLAM

#endif //ORB_SLAM3_EVOPTIMIZER_H
