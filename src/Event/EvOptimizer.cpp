//
// Created by root on 1/13/21.
//

#include "EvOptimizer.h"

#include <utility>
#include <complex>
#include<mutex>

#include <Eigen/StdVector>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include "Thirdparty/g2o/g2o/core/sparse_block_matrix.h"
#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_gauss_newton.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "G2oTypes.h"
#include "Converter.h"

#include "OptimizableTypes.h"


using namespace std;
using namespace cv;
using namespace ORB_SLAM3;


namespace EORB_SLAM {

    // Warp coordinates respect to the center of sensor
    void warp_RT2D(const float& x, const float& y, const double& tk, const double& omega,
            const double& vx, const double& vy, float& xp, float& yp) {

        double theta_k = omega * tk;
        xp = x * cos(theta_k) - y * sin(theta_k) + vx * tk;
        yp = x * sin(theta_k) + y * cos(theta_k) + vy * tk;
    }

    bool EvFocus_MS_RT2D::Evaluate(const double *parameters, double *cost, double *gradient) const {

        int imWidth = mpEvParams->imWidth;
        int imHeight = mpEvParams->imHeight;

        float imSigma = mpEvParams->l1ImSigma;
        float sig2 = powf(imSigma, 2);
        int lenHalfWin = static_cast<int>(ceilf(imSigma*DEF_SIGMA_EF_RANGE));

        cv::Mat Ix = cv::Mat::zeros(imHeight, imWidth, CV_32FC1);
        cv::Mat Ix_omega = cv::Mat::zeros(imHeight, imWidth, CV_32FC1);
        cv::Mat Ix_vx = cv::Mat::zeros(imHeight, imWidth, CV_32FC1);
        cv::Mat Ix_vy = cv::Mat::zeros(imHeight, imWidth, CV_32FC1);

        double mu_Ix = 0.0;
        float maxVal = -1000000.0;
        float minVal = 0.0;
        float bk = 1.f;
        double t0 = mvEvObs[0].ts;

        const double omega = parameters[0];
        const double vx = parameters[1];
        const double vy = parameters[2];

        // Calculate Event Images
        for (const EventData& ev : mvEvObs) {

            float ex = ev.x;
            float ey = ev.y;
            double tk = ev.ts - t0;

            // Compute every thing in Sensor Space
            cv::Point3f cvP3d = mpCamera->unproject(cv::Point2f(ex, ey));

            // Calculate Warped coordinates
            float xp, yp;
            warp_RT2D(cvP3d.x, cvP3d.y, tk, omega, vx, vy, xp, yp);

            // Convert xp, yp back to image sensor to find image indices
            cv::Point2f uv = mpCamera->project(cv::Point3f(xp, yp, cvP3d.z));

            int xIdx, yIdx;
            float xRes, yRes;
            breakFloatCoords(uv.x, uv.y, xIdx, yIdx, xRes, yRes);

            for (int i = -lenHalfWin; i <= lenHalfWin; i++)
            {
                for (int j = -lenHalfWin; j <= lenHalfWin; j++)
                {
                    // Assign values in Image Space
                    int xnIdx = xIdx + i;
                    int ynIdx = yIdx + j;

                    if (!MyCalibrator::isInImage(xnIdx, ynIdx, imWidth, imHeight)) {
                        continue;
                    }

                    cv::Point3f cvXn3d = mpCamera->unproject(cv::Point2f(xnIdx, ynIdx));

                    float val = exp_XY2f(cvXn3d.x - xp, cvXn3d.y - yp, sig2);
                    //float polSign = resolvePolarity(pol, ev.p);

                    // Regular Image
                    Ix.at<float>(ynIdx, xnIdx) += bk * val;
                    mu_Ix += bk * val;

                    if (gradient != nullptr) {

                        float theta_k = tk * omega;
                        // Image Omega
                        float vv_omeg = bk * tk * ((cvXn3d.y-yp) * (cvP3d.x*cos(theta_k)-cvP3d.y*sin(theta_k)) -
                                (cvXn3d.x-xp) * (cvP3d.x*sin(theta_k)+cvP3d.y*cos(theta_k))) * val / sig2;
                        Ix_omega.at<float>(ynIdx, xnIdx) += vv_omeg;

                        // Image Vx
                        float vv_vx = bk * tk * (cvXn3d.x-xp) * val / sig2;
                        Ix_vx.at<float>(ynIdx, xnIdx) += vv_vx;

                        // Image Vy
                        float vv_vy = bk * tk * (cvXn3d.y-yp) * val / sig2;
                        Ix_vy.at<float>(ynIdx, xnIdx) += vv_vy;
                    }
                }
            }
        }

        double MS_val = 0.0;
        double gradOmega = 0.0;
        double gradVx = 0.0;
        double gradVy = 0.0;

        for (int i = 0; i < imHeight; i++) {
            for (int j = 0; j < imWidth; j++) {

                float Ix_val = Ix.at<float>(i,j);
                MS_val += powf(Ix_val,2);

                if (gradient != nullptr) {

                    gradOmega += Ix_omega.at<float>(i,j) * Ix_val;
                    gradVx += Ix_vx.at<float>(i,j) * Ix_val;
                    gradVy += Ix_vy.at<float>(i,j) * Ix_val;
                }
            }
        }

        cost[0] = -1.0 * MS_val / ((double)(imWidth*imHeight));

        if (gradient != nullptr) {

            gradient[0] = -2.0 * gradOmega / ((double)(imWidth*imHeight));
            gradient[1] = -2.0 * gradVx / ((double)(imWidth*imHeight));
            gradient[2] = -2.0 * gradVy / ((double)(imWidth*imHeight));
        }
        return true;
    }

    void EvOptimizer::optimizeFocusPlan3dof() {

        google::InitGoogleLogging("abdul");
        double parameters[2] = {-1.2, 1.0};
        ceres::GradientProblemSolver::Options options;
        options.minimizer_progress_to_stdout = true;
        ceres::GradientProblemSolver::Summary summary;
        ceres::GradientProblem problem(new Rosenbrock());
        ceres::Solve(options, problem, parameters, &summary);
        std::cout << summary.FullReport() << "\n";
        std::cout << "Initial x: " << -1.2 << " y: " << 1.0 << "\n";
        std::cout << "Final   x: " << parameters[0] << " y: " << parameters[1]
                  << "\n";

    }

    void EvOptimizer::optimizeFocus_MS_RT2D(const std::vector<EventData>& vEvObs, std::shared_ptr<EvParams> pEvParams,
            ORB_SLAM3::GeometricCamera *pCamera, double &omega0, double &vx0, double &vy0) {

        //google::InitGoogleLogging("optimizeFocus_MS_RT2D");

        double parameters[3] = {omega0, vx0, vy0};

        ceres::GradientProblemSolver::Options options;
        options.minimizer_progress_to_stdout = true;

        ceres::GradientProblemSolver::Summary summary;

        ceres::GradientProblem problem(new EvFocus_MS_RT2D(vEvObs, std::move(pEvParams), pCamera));

        ceres::Solve(options, problem, parameters, &summary);

        std::cout << summary.FullReport() << "\n";
        std::cout << "Final   x: " << parameters[0] << " y: " << parameters[1] << " vy: " << parameters[2] << "\n";

        omega0 = parameters[0];
        vx0 = parameters[1];
        vy0 = parameters[2];
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    void EvOptimizer::GlobalBundleAdjustemnt(Map* pMap, int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
    {
        vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
        vector<MapPoint*> vpMP = pMap->GetAllMapPoints();
        EvOptimizer::BundleAdjustment(vpKFs,vpMP,nIterations,pbStopFlag, nLoopKF, bRobust);
    }


    void EvOptimizer::BundleAdjustment(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP,
                                       int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
    {
        if (vpKFs.empty() || vpMP.empty()) {
            LOG(WARNING) << "Optimizer::BundleAdjustment: Called with no constraints: nKFs: "
                         << vpKFs.size() << ", nMPs: " << vpMP.size() << endl;
            return;
        }

        vector<bool> vbNotIncludedMP;
        vbNotIncludedMP.resize(vpMP.size());

        Map* pMap = vpKFs[0]->GetMap();

        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

        g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        if(pbStopFlag)
            optimizer.setForceStopFlag(pbStopFlag);

        long unsigned int maxKFid = 0;

        const int nExpectedSize = (vpKFs.size())*vpMP.size();

        vector<ORB_SLAM3::EdgeSE3ProjectXYZ*> vpEdgesMono;
        vpEdgesMono.reserve(nExpectedSize);

        vector<ORB_SLAM3::EdgeSE3ProjectXYZToBody*> vpEdgesBody;
        vpEdgesBody.reserve(nExpectedSize);

        vector<KeyFrame*> vpEdgeKFMono;
        vpEdgeKFMono.reserve(nExpectedSize);

        vector<KeyFrame*> vpEdgeKFBody;
        vpEdgeKFBody.reserve(nExpectedSize);

        vector<MapPoint*> vpMapPointEdgeMono;
        vpMapPointEdgeMono.reserve(nExpectedSize);

        vector<MapPoint*> vpMapPointEdgeBody;
        vpMapPointEdgeBody.reserve(nExpectedSize);

        vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
        vpEdgesStereo.reserve(nExpectedSize);

        vector<KeyFrame*> vpEdgeKFStereo;
        vpEdgeKFStereo.reserve(nExpectedSize);

        vector<MapPoint*> vpMapPointEdgeStereo;
        vpMapPointEdgeStereo.reserve(nExpectedSize);


        // Set KeyFrame vertices
        unsigned nVertexKF = 0;
        for(auto pKF : vpKFs) {

            if(pKF->isBad())
                continue;
            g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
            vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));
            vSE3->setId(pKF->mnId);
            vSE3->setFixed(pKF->mnId==pMap->GetInitKFid());
            optimizer.addVertex(vSE3);
            if(pKF->mnId>maxKFid)
                maxKFid=pKF->mnId;
            //cout << "KF id: " << pKF->mnId << endl;
            nVertexKF++;
        }

        if (nVertexKF <= 0) {
            LOG(WARNING) << "Optimizer::BundleAdjustment: No valid KF vertex can be added, abort...\n";
            return;
        }

        const float thHuber2D = sqrt(5.99f);
        const float thHuber3D = sqrt(7.815f);

        // Set MapPoint vertices
        //cout << "start inserting MPs" << endl;

        unsigned nVertexMP = 0;
        ulong maxVertexId = maxKFid;

        for(size_t i=0; i<vpMP.size(); i++)
        {
            MapPoint* pMP = vpMP[i];
            if(pMP->isBad())
                continue;
            g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
            vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
            const int id = pMP->mnId+maxKFid+1;
            vPoint->setId(id);
            vPoint->setMarginalized(true);
            optimizer.addVertex(vPoint);

            const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();

            int nEdges = 0;
            //SET EDGES
            for(map<KeyFrame*,tuple<int,int>>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)
            {
                KeyFrame* pKF = mit->first;
                if(pKF->isBad() || pKF->mnId>maxKFid)
                    continue;
                if(optimizer.vertex(id) == NULL || optimizer.vertex(pKF->mnId) == NULL)
                    continue;
                nEdges++;

                const int leftIndex = get<0>(mit->second);

                if(leftIndex != -1 && pKF->mvuRight[get<0>(mit->second)]<0)
                {
                    const cv::KeyPoint &kpUn = pKF->getUndistKPtMono(leftIndex);

                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    ORB_SLAM3::EdgeSE3ProjectXYZ* e = new ORB_SLAM3::EdgeSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKF->getKPtInvLevelSigma2(leftIndex);
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    if(bRobust)
                    {
                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuber2D);
                    }

                    e->pCamera = pKF->mpCamera;

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKF);
                    vpMapPointEdgeMono.push_back(pMP);
                }
                else if(leftIndex != -1 && pKF->mvuRight[leftIndex] >= 0) //Stereo observation
                {
                    // TODO: Check original
                    const cv::KeyPoint &kpUn = pKF->getUndistKPtMono(leftIndex);

                    Eigen::Matrix<double,3,1> obs;
                    const float kp_ur = pKF->mvuRight[get<0>(mit->second)];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKF->getORBInvLevelSigma2(kpUn.octave);
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                    e->setInformation(Info);

                    if(bRobust)
                    {
                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuber3D);
                    }

                    e->fx = pKF->fx;
                    e->fy = pKF->fy;
                    e->cx = pKF->cx;
                    e->cy = pKF->cy;
                    e->bf = pKF->mbf;

                    optimizer.addEdge(e);

                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKF);
                    vpMapPointEdgeStereo.push_back(pMP);
                }

                if(pKF->mpCamera2){
                    int rightIndex = get<1>(mit->second);

                    if(rightIndex != -1 && rightIndex < pKF->numAllKPtsRight()){
                        rightIndex -= pKF->numAllKPtsLeft();

                        Eigen::Matrix<double,2,1> obs;
                        cv::KeyPoint kp = pKF->getKPtRight(rightIndex);
                        obs << kp.pt.x, kp.pt.y;

                        ORB_SLAM3::EdgeSE3ProjectXYZToBody *e = new ORB_SLAM3::EdgeSE3ProjectXYZToBody();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                        e->setMeasurement(obs);
                        const float &invSigma2 = pKF->getORBInvLevelSigma2(kp.octave);
                        e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuber2D);

                        e->mTrl = Converter::toSE3Quat(pKF->mTrl);

                        e->pCamera = pKF->mpCamera2;

                        optimizer.addEdge(e);
                        vpEdgesBody.push_back(e);
                        vpEdgeKFBody.push_back(pKF);
                        vpMapPointEdgeBody.push_back(pMP);
                    }
                }
            }


            if(nEdges==0)
            {
                optimizer.removeVertex(vPoint);
                vbNotIncludedMP[i]=true;
            }
            else
            {
                vbNotIncludedMP[i]=false;
                nVertexMP++;
            }

            if (id > maxVertexId) {
                maxVertexId = id;
            }
        }

        // Set Event MapPoint vertices and edges
        set<MapPoint*> spEvMpts;
        retrieveAllMapPoints(vpKFs, spEvMpts);

        int maxEvMpId = maxVertexId;
        vector<EdgeSE3ProjectXYZ*> vpEvEdgesMono;
        vpEvEdgesMono.reserve(nExpectedSize);
        vector<KeyFrame*> vpEvEdgesKFMono;
        vpEvEdgesMono.reserve(nExpectedSize);
        vector<MapPoint*> vpEvEdgesMpMono;
        vpEvEdgesMpMono.reserve(nExpectedSize);

        const uint nVertexEvMP = setEventMapVxAndEdges(optimizer, spEvMpts, vpEvEdgesMono, vpEdgeKFMono,
                                                       vpEvEdgesMpMono, maxEvMpId, thHuber2D, bRobust);

        if (nVertexMP+nVertexEvMP <= 0) {
            LOG(WARNING) << "Optimizer::BundleAdjustment: No valid MP vertex can be added, abort...\n";
            return;
        }

        VLOG(40) << "Ev-Im-GBA: nVertexOrbKF: " << nVertexKF << ", nVertexOrbMP: "
                           << nVertexMP << ", nVertexEvMP: " << nVertexEvMP
                           << ", nEdges: " << optimizer.edges().size() << endl;

        //cout << "end inserting MPs" << endl;
        // Optimize!
        optimizer.setVerbose(false);
        optimizer.initializeOptimization();
        optimizer.optimize(nIterations);
        Verbose::PrintMess("BA: End of the optimization", Verbose::VERBOSITY_NORMAL);

        // Recover optimized data

        //Keyframes
        for (auto pKF : vpKFs)
        {
            if(pKF->isBad())
                continue;
            g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));

            g2o::SE3Quat SE3quat = vSE3->estimate();
            if(nLoopKF==pMap->GetOriginKF()->mnId)
            {
                cv::Mat Tcw = Converter::toCvMat(SE3quat);
                pKF->SetPose(Tcw);

                // Event Pose
                KeyFrame* pEvKF = pKF->mpSynchEvKF;
                if (pEvKF) {
                    pEvKF->SetPose(Tcw);
                }
            }
            else
            {
                /*if(!vSE3->fixed())
                {
                    //cout << "KF " << pKF->mnId << ": " << endl;
                    pKF->mHessianPose = cv::Mat(6, 6, CV_64F);
                    pKF->mbHasHessian = true;
                    for(int r=0; r<6; ++r)
                    {
                        for(int c=0; c<6; ++c)
                        {
                            //cout  << vSE3->hessian(r, c) << ", ";
                            pKF->mHessianPose.at<double>(r, c) = vSE3->hessian(r, c);
                        }
                        //cout << endl;
                    }
                }*/


                pKF->mTcwGBA.create(4,4,CV_32F);
                Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
                pKF->mnBAGlobalForKF = nLoopKF;

                cv::Mat mTwc = pKF->GetPoseInverse();
                cv::Mat mTcGBA_c = pKF->mTcwGBA * mTwc;
                cv::Vec3d vector_dist =  mTcGBA_c.rowRange(0, 3).col(3);
                double dist = cv::norm(vector_dist);
                if(dist > 1)
                {
                    int numMonoBadPoints = 0, numMonoOptPoints = 0;
                    int numStereoBadPoints = 0, numStereoOptPoints = 0;
                    vector<MapPoint*> vpMonoMPsOpt, vpStereoMPsOpt;

                    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
                    {
                        ORB_SLAM3::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
                        MapPoint* pMP = vpMapPointEdgeMono[i];
                        KeyFrame* pKFedge = vpEdgeKFMono[i];

                        if(pKF != pKFedge)
                        {
                            continue;
                        }

                        if(pMP->isBad())
                            continue;

                        if(e->chi2()>5.991 || !e->isDepthPositive())
                        {
                            numMonoBadPoints++;

                        }
                        else
                        {
                            numMonoOptPoints++;
                            vpMonoMPsOpt.push_back(pMP);
                        }

                    }

                    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
                    {
                        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
                        MapPoint* pMP = vpMapPointEdgeStereo[i];
                        KeyFrame* pKFedge = vpEdgeKFMono[i];

                        if(pKF != pKFedge)
                        {
                            continue;
                        }

                        if(pMP->isBad())
                            continue;

                        if(e->chi2()>7.815 || !e->isDepthPositive())
                        {
                            numStereoBadPoints++;
                        }
                        else
                        {
                            numStereoOptPoints++;
                            vpStereoMPsOpt.push_back(pMP);
                        }
                    }
                    Verbose::PrintMess("GBA: KF " + to_string(pKF->mnId) + " had been moved " + to_string(dist) + " meters", Verbose::VERBOSITY_DEBUG);
                    Verbose::PrintMess("--Number of observations: " + to_string(numMonoOptPoints) + " in mono and " + to_string(numStereoOptPoints) + " in stereo", Verbose::VERBOSITY_DEBUG);
                    Verbose::PrintMess("--Number of discarded observations: " + to_string(numMonoBadPoints) + " in mono and " + to_string(numStereoBadPoints) + " in stereo", Verbose::VERBOSITY_DEBUG);
                }
            }
        }
        Verbose::PrintMess("BA: KFs updated", Verbose::VERBOSITY_DEBUG);

        //Points
        for(size_t i=0; i<vpMP.size(); i++)
        {
            if(vbNotIncludedMP[i])
                continue;

            MapPoint* pMP = vpMP[i];

            if(pMP->isBad())
                continue;
            g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));

            if(nLoopKF==pMap->GetOriginKF()->mnId)
            {
                pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
                pMP->UpdateNormalAndDepth();
            }
            else
            {
                pMP->mPosGBA.create(3,1,CV_32F);
                Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
                pMP->mnBAGlobalForKF = nLoopKF;
            }
        }

        // Event Points
        for (auto* pEvMpt : spEvMpts) {

            if (!pEvMpt || pEvMpt->isBad())
                continue;

            g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pEvMpt->mnId+maxVertexId+1));

            if(nLoopKF==pMap->GetOriginKF()->mnId)
            {
                pEvMpt->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
                pEvMpt->UpdateNormalAndDepth();
            }
        }
    }

    int EvOptimizer::PoseOptimization(Frame *pFrame)
    {
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

        linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

        g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        int nInitialCorrespondences=0;
        int nInitialCorrespondencesEv=0;

        // Set Frame vertex
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        vSE3->setId(0);
        vSE3->setFixed(false);
        optimizer.addVertex(vSE3);

        // Set MapPoint vertices
        const int N = pFrame->numAllKPts();

        vector<ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono, vpEdgesMonoEv;
        vector<ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody *> vpEdgesMono_FHR;
        vector<size_t> vnIndexEdgeMono, vnIndexEdgeRight, vnIndexEdgeMonoEv;
        vpEdgesMono.reserve(N);
        vpEdgesMono_FHR.reserve(N);
        vnIndexEdgeMono.reserve(N);
        vnIndexEdgeRight.reserve(N);

        vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;
        vector<size_t> vnIndexEdgeStereo;
        vpEdgesStereo.reserve(N);
        vnIndexEdgeStereo.reserve(N);

        const float deltaMono = sqrtf(5.991);
        const float deltaStereo = sqrtf(7.815);

        FramePtr pEvFrame = pFrame->mpEvFrame.lock();

        {
            unique_lock<mutex> lock(MapPoint::mGlobalMutex);

            for(int i=0; i<N; i++)
            {
                MapPoint* pMP = pFrame->getMapPoint(i);
                if(pMP)
                {
                    //Conventional SLAM
                    if(!pFrame->mpCamera2){
                        // Monocular observation
                        if(pFrame->mvuRight[i]<0)
                        {
                            nInitialCorrespondences++;
                            pFrame->setMPOutlier(i, false);

                            Eigen::Matrix<double,2,1> obs;
                            const cv::KeyPoint &kpUn = pFrame->getUndistKPtMono(i);
                            obs << kpUn.pt.x, kpUn.pt.y;

                            ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose* e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose();

                            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                            e->setMeasurement(obs);
                            const float invSigma2 = pFrame->getKPtInvLevelSigma2(i);//getORBInvLevelSigma2(kpUn.octave);
                            e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                            e->setRobustKernel(rk);
                            rk->setDelta(deltaMono);

                            e->pCamera = pFrame->mpCamera;
                            cv::Mat Xw = pMP->GetWorldPos();
                            e->Xw[0] = Xw.at<float>(0);
                            e->Xw[1] = Xw.at<float>(1);
                            e->Xw[2] = Xw.at<float>(2);

                            optimizer.addEdge(e);

                            vpEdgesMono.push_back(e);
                            vnIndexEdgeMono.push_back(i);
                        }
                        else  // Stereo observation
                        {
                            nInitialCorrespondences++;
                            pFrame->setMPOutlier(i, false);

                            //SET EDGE
                            Eigen::Matrix<double,3,1> obs;
                            const cv::KeyPoint &kpUn = pFrame->getUndistKPtMono(i);
                            const float &kp_ur = pFrame->mvuRight[i];
                            obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                            g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                            e->setMeasurement(obs);
                            const float invSigma2 = pFrame->getKPtInvLevelSigma2(i);//getORBInvLevelSigma2(kpUn.octave);
                            Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                            e->setInformation(Info);

                            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                            e->setRobustKernel(rk);
                            rk->setDelta(deltaStereo);

                            e->fx = pFrame->fx;
                            e->fy = pFrame->fy;
                            e->cx = pFrame->cx;
                            e->cy = pFrame->cy;
                            e->bf = pFrame->mbf;
                            cv::Mat Xw = pMP->GetWorldPos();
                            e->Xw[0] = Xw.at<float>(0);
                            e->Xw[1] = Xw.at<float>(1);
                            e->Xw[2] = Xw.at<float>(2);

                            optimizer.addEdge(e);

                            vpEdgesStereo.push_back(e);
                            vnIndexEdgeStereo.push_back(i);
                        }
                    }
                        //SLAM with respect a rigid body
                    else{
                        nInitialCorrespondences++;

                        cv::KeyPoint kpUn;

                        if (i < pFrame->numKPtsLeft()) {    //Left camera observation
                            kpUn = pFrame->getDistKPtMono(i);

                            pFrame->setMPOutlier(i, false);

                            Eigen::Matrix<double, 2, 1> obs;
                            obs << kpUn.pt.x, kpUn.pt.y;

                            ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose *e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose();

                            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                            e->setMeasurement(obs);
                            const float invSigma2 = pFrame->getKPtInvLevelSigma2(i);//getORBInvLevelSigma2(kpUn.octave);
                            e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                            g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                            e->setRobustKernel(rk);
                            rk->setDelta(deltaMono);

                            e->pCamera = pFrame->mpCamera;
                            cv::Mat Xw = pMP->GetWorldPos();
                            e->Xw[0] = Xw.at<float>(0);
                            e->Xw[1] = Xw.at<float>(1);
                            e->Xw[2] = Xw.at<float>(2);

                            optimizer.addEdge(e);

                            vpEdgesMono.push_back(e);
                            vnIndexEdgeMono.push_back(i);
                        }
                        else {   //Right camera observation
                            //continue;
                            kpUn = pFrame->getKPtRight(i - pFrame->numKPtsLeft());

                            Eigen::Matrix<double, 2, 1> obs;
                            obs << kpUn.pt.x, kpUn.pt.y;

                            pFrame->setMPOutlier(i, false);

                            ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody *e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody();

                            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                            e->setMeasurement(obs);
                            const float invSigma2 = pFrame->getORBInvLevelSigma2(kpUn.octave);
                            e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                            g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                            e->setRobustKernel(rk);
                            rk->setDelta(deltaMono);

                            e->pCamera = pFrame->mpCamera2;
                            cv::Mat Xw = pMP->GetWorldPos();
                            e->Xw[0] = Xw.at<float>(0);
                            e->Xw[1] = Xw.at<float>(1);
                            e->Xw[2] = Xw.at<float>(2);

                            e->mTrl = Converter::toSE3Quat(pFrame->mTrl);

                            optimizer.addEdge(e);

                            vpEdgesMono_FHR.push_back(e);
                            vnIndexEdgeRight.push_back(i);
                        }
                    }
                }
            }

            // Set Event Edges
            if (pEvFrame) {
                nInitialCorrespondencesEv = setEventMapVxAndEdges(optimizer, pEvFrame.get(), vpEdgesMonoEv, vnIndexEdgeMonoEv, 0, deltaMono);
            }
        }

        //cout << "PO: vnIndexEdgeMono.size() = " << vnIndexEdgeMono.size() << "   vnIndexEdgeRight.size() = " << vnIndexEdgeRight.size() << endl;
        if(nInitialCorrespondences<3)
            return 0;

        // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
        // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
        const float chi2Mono[4]={5.991,5.991,5.991,5.991};
        const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
        const int its[4]={10,10,10,10};

        int nBad=0;
        for(size_t it=0; it<4; it++)
        {

            vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
            optimizer.initializeOptimization(0);
            optimizer.optimize(its[it]);

            nBad=0;
            for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
            {
                ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

                const size_t idx = vnIndexEdgeMono[i];

                if(pFrame->getMPOutlier(idx))
                {
                    e->computeError();
                }

                const float chi2 = e->chi2();

                if(chi2>chi2Mono[it])
                {
                    pFrame->setMPOutlier(idx, true);
                    e->setLevel(1);
                    nBad++;
                }
                else
                {
                    pFrame->setMPOutlier(idx, false);
                    e->setLevel(0);
                }

                if(it==2)
                    e->setRobustKernel(0);
            }

            for(size_t i=0, iend=vpEdgesMono_FHR.size(); i<iend; i++)
            {
                ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody* e = vpEdgesMono_FHR[i];

                const size_t idx = vnIndexEdgeRight[i];

                if(pFrame->getMPOutlier(idx))
                {
                    e->computeError();
                }

                const float chi2 = e->chi2();

                if(chi2>chi2Mono[it])
                {
                    pFrame->setMPOutlier(idx, true);
                    e->setLevel(1);
                    nBad++;
                }
                else
                {
                    pFrame->setMPOutlier(idx, false);
                    e->setLevel(0);
                }

                if(it==2)
                    e->setRobustKernel(0);
            }

            for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
            {
                g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = vpEdgesStereo[i];

                const size_t idx = vnIndexEdgeStereo[i];

                if(pFrame->getMPOutlier(idx))
                {
                    e->computeError();
                }

                const float chi2 = e->chi2();

                if(chi2>chi2Stereo[it])
                {
                    pFrame->setMPOutlier(idx, true);
                    e->setLevel(1);
                    nBad++;
                }
                else
                {
                    e->setLevel(0);
                    pFrame->setMPOutlier(idx, false);
                }

                if(it==2)
                    e->setRobustKernel(0);
            }

            for(size_t i=0, iend=vpEdgesMonoEv.size(); i<iend; i++)
            {
                ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMonoEv[i];

                const size_t idx = vnIndexEdgeMonoEv[i];

                if(pEvFrame->getMPOutlier(idx))
                {
                    e->computeError();
                }

                const float chi2 = e->chi2();

                if(chi2>chi2Mono[it])
                {
                    pEvFrame->setMPOutlier(idx, true);
                    e->setLevel(1);
                    nBad++;
                }
                else
                {
                    pEvFrame->setMPOutlier(idx, false);
                    e->setLevel(0);
                }

                if(it==2)
                    e->setRobustKernel(0);
            }

            if(optimizer.edges().size()<10)
                break;
        }

        // Recover optimized pose and return number of inliers
        g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
        g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
        cv::Mat pose = Converter::toCvMat(SE3quat_recov);
        pFrame->SetPose(pose);
        if (pEvFrame) {
            pEvFrame->SetPose(pose);
        }

        //cout << "[PoseOptimization]: initial correspondences-> " << nInitialCorrespondences << " --- outliers-> " << nBad << endl;

        return nInitialCorrespondences+nInitialCorrespondencesEv-nBad;
    }

    void EvOptimizer::LocalBundleAdjustment(KeyFrame *pKF, bool* pbStopFlag, Map* pMap, int& num_fixedKF, int& num_OptKF,
                                            int& num_MPs, int& num_edges)
    {
        // Local KeyFrames: First Breath Search from Current Keyframe
        list<KeyFrame*> lLocalKeyFrames;

        lLocalKeyFrames.push_back(pKF);
        pKF->mnBALocalForKF = pKF->mnId;
        Map* pCurrentMap = pKF->GetMap();

        const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
        for(int i=0, iend=vNeighKFs.size(); i<iend; i++)
        {
            KeyFrame* pKFi = vNeighKFs[i];
            pKFi->mnBALocalForKF = pKF->mnId;
            if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                lLocalKeyFrames.push_back(pKFi);
        }

        // Local MapPoints seen in Local KeyFrames
        num_fixedKF = 0;
        list<MapPoint*> lLocalMapPoints;
        set<MapPoint*> sNumObsMP;
        for(auto pKFi : lLocalKeyFrames)
        {
            if(pKFi->mnId==pMap->GetInitKFid())
            {
                num_fixedKF = 1;
            }
            vector<MapPoint*> vpMPs = pKFi->GetMapPointMatches();
            for(auto pMP : vpMPs)
            {
                if(pMP)
                    if(!pMP->isBad() && pMP->GetMap() == pCurrentMap)
                    {
                        if(pMP->mnBALocalForKF!=pKF->mnId)
                        {
                            lLocalMapPoints.push_back(pMP);
                            pMP->mnBALocalForKF=pKF->mnId;
                        }
                    }
            }
        }
        num_MPs = lLocalMapPoints.size();

        // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
        list<KeyFrame*> lFixedCameras;
        for(auto & lLocalMapPoint : lLocalMapPoints)
        {
            map<KeyFrame*,tuple<int,int>> observations = lLocalMapPoint->GetObservations();
            for(auto & observation : observations)
            {
                KeyFrame* pKFi = observation.first;

                if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId )
                {
                    pKFi->mnBAFixedForKF=pKF->mnId;
                    if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                        lFixedCameras.push_back(pKFi);
                }
            }
        }
        num_fixedKF = lFixedCameras.size() + num_fixedKF;
        if(num_fixedKF < 2)
        {
            list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin();
            int lowerId = pKF->mnId;
            KeyFrame* pLowerKf;
            int secondLowerId = pKF->mnId;
            KeyFrame* pSecondLowerKF;

            for(; lit != lLocalKeyFrames.end(); lit++)
            {
                KeyFrame* pKFi = *lit;
                if(pKFi == pKF || pKFi->mnId == pMap->GetInitKFid())
                {
                    continue;
                }

                if(pKFi->mnId < lowerId)
                {
                    lowerId = pKFi->mnId;
                    pLowerKf = pKFi;
                }
                else if(pKFi->mnId < secondLowerId)
                {
                    secondLowerId = pKFi->mnId;
                    pSecondLowerKF = pKFi;
                }
            }
            lFixedCameras.push_back(pLowerKf);
            lLocalKeyFrames.remove(pLowerKf);
            num_fixedKF++;
            if(num_fixedKF < 2)
            {
                lFixedCameras.push_back(pSecondLowerKF);
                lLocalKeyFrames.remove(pSecondLowerKF);
                num_fixedKF++;
            }
        }

        if(num_fixedKF == 0)
        {
            Verbose::PrintMess("LM-LBA: There are 0 fixed KF in the optimizations, LBA aborted", Verbose::VERBOSITY_QUIET);
            return;
        }

        if (lLocalKeyFrames.empty() || (lLocalKeyFrames.size() == 1 && (*(lLocalKeyFrames.begin()))->mnId == pMap->GetInitKFid())) {

            LOG(WARNING) << "LM-LBA: There are 0 local KFs in the optimization but SBA is not supported, LBA aborted!\n";
            return;
        }

        // Setup optimizer
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

        g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        if (pMap->IsInertial())
            solver->setUserLambdaInit(100.0);

        optimizer.setAlgorithm(solver);
        optimizer.setVerbose(false);

        if(pbStopFlag)
            optimizer.setForceStopFlag(pbStopFlag);

        unsigned long maxKFid = 0;

        const size_t nAllKFs = lLocalKeyFrames.size()+lFixedCameras.size();

        if (nAllKFs <= 0) {
            LOG(WARNING) << "LM-LBA: No KeyFrame constraints available, abort optimization...\n";
            return;
        }

        // Set Local KeyFrame vertices
        for(auto pKFi : lLocalKeyFrames)
        {
            if (!pKFi) // insurance!
                continue;
            g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
            vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
            vSE3->setId(pKFi->mnId);
            vSE3->setFixed(pKFi->mnId==pMap->GetInitKFid());
            optimizer.addVertex(vSE3);
            if(pKFi->mnId>maxKFid)
                maxKFid=pKFi->mnId;
        }
        num_OptKF = lLocalKeyFrames.size();

        // Set Fixed KeyFrame vertices
        for(auto pKFi : lFixedCameras)
        {
            if (!pKFi)
                continue;
            g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
            vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
            vSE3->setId(pKFi->mnId);
            vSE3->setFixed(true);
            optimizer.addVertex(vSE3);
            if(pKFi->mnId>maxKFid)
                maxKFid=pKFi->mnId;
        }

        // Set MapPoint vertices
        const int nExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size())*lLocalMapPoints.size();

        vector<ORB_SLAM3::EdgeSE3ProjectXYZ*> vpEdgesMono;
        vpEdgesMono.reserve(nExpectedSize);

        vector<ORB_SLAM3::EdgeSE3ProjectXYZToBody*> vpEdgesBody;
        vpEdgesBody.reserve(nExpectedSize);

        vector<KeyFrame*> vpEdgeKFMono;
        vpEdgeKFMono.reserve(nExpectedSize);

        vector<KeyFrame*> vpEdgeKFBody;
        vpEdgeKFBody.reserve(nExpectedSize);

        vector<MapPoint*> vpMapPointEdgeMono;
        vpMapPointEdgeMono.reserve(nExpectedSize);

        vector<MapPoint*> vpMapPointEdgeBody;
        vpMapPointEdgeBody.reserve(nExpectedSize);

        vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
        vpEdgesStereo.reserve(nExpectedSize);

        vector<KeyFrame*> vpEdgeKFStereo;
        vpEdgeKFStereo.reserve(nExpectedSize);

        vector<MapPoint*> vpMapPointEdgeStereo;
        vpMapPointEdgeStereo.reserve(nExpectedSize);

        const float thHuberMono = sqrt(5.991);
        const float thHuberStereo = sqrt(7.815);

        int nPoints = 0;
        int maxVertexId = maxKFid;

        int nKFs = lLocalKeyFrames.size()+lFixedCameras.size(), nEdges = 0;

        for(auto pMP : lLocalMapPoints)
        {
            g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
            vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
            int id = pMP->mnId+maxKFid+1;
            vPoint->setId(id);
            vPoint->setMarginalized(true);
            optimizer.addVertex(vPoint);
            nPoints++;

            const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();

            //Set edges
            for(const auto & observation : observations)
            {
                KeyFrame* pKFi = observation.first;

                if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                {
                    const int leftIndex = get<0>(observation.second);

                    // Monocular observation
                    if(leftIndex != -1 && pKFi->mvuRight[get<0>(observation.second)]<0)
                    {
                        const cv::KeyPoint &kpUn = pKFi->getUndistKPtMono(leftIndex);
                        Eigen::Matrix<double,2,1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y;

                        ORB_SLAM3::EdgeSE3ProjectXYZ* e = new ORB_SLAM3::EdgeSE3ProjectXYZ();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);
                        const float &invSigma2 = pKFi->getKPtInvLevelSigma2(leftIndex);//[kpUn.octave];
                        e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        e->pCamera = pKFi->mpCamera;

                        optimizer.addEdge(e);
                        vpEdgesMono.push_back(e);
                        vpEdgeKFMono.push_back(pKFi);
                        vpMapPointEdgeMono.push_back(pMP);

                        nEdges++;
                    }
                    else if(leftIndex != -1 && pKFi->mvuRight[get<0>(observation.second)]>=0)// Stereo observation
                    {
                        const cv::KeyPoint &kpUn = pKFi->getUndistKPtMono(leftIndex);
                        Eigen::Matrix<double,3,1> obs;
                        const float kp_ur = pKFi->mvuRight[get<0>(observation.second)];
                        obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                        g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);
                        const float &invSigma2 = pKFi->getKPtInvLevelSigma2(leftIndex);//[kpUn.octave];
                        Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                        e->setInformation(Info);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberStereo);

                        e->fx = pKFi->fx;
                        e->fy = pKFi->fy;
                        e->cx = pKFi->cx;
                        e->cy = pKFi->cy;
                        e->bf = pKFi->mbf;

                        optimizer.addEdge(e);
                        vpEdgesStereo.push_back(e);
                        vpEdgeKFStereo.push_back(pKFi);
                        vpMapPointEdgeStereo.push_back(pMP);

                        nEdges++;
                    }

                    if(pKFi->mpCamera2){
                        int rightIndex = get<1>(observation.second);

                        if(rightIndex != -1 ){
                            rightIndex -= pKFi->numAllKPtsLeft();

                            Eigen::Matrix<double,2,1> obs;
                            cv::KeyPoint kp = pKFi->getKPtRight(rightIndex);
                            obs << kp.pt.x, kp.pt.y;

                            ORB_SLAM3::EdgeSE3ProjectXYZToBody *e = new ORB_SLAM3::EdgeSE3ProjectXYZToBody();

                            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                            e->setMeasurement(obs);
                            const float &invSigma2 = pKFi->getORBInvLevelSigma2(kp.octave);
                            e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                            e->setRobustKernel(rk);
                            rk->setDelta(thHuberMono);

                            e->mTrl = Converter::toSE3Quat(pKFi->mTrl);

                            e->pCamera = pKFi->mpCamera2;

                            optimizer.addEdge(e);
                            vpEdgesBody.push_back(e);
                            vpEdgeKFBody.push_back(pKFi);
                            vpMapPointEdgeBody.push_back(pMP);

                            nEdges++;
                        }
                    }
                }
            }

            if (id > maxVertexId) {
                maxVertexId = id;
            }
        }
        num_edges = nEdges;

        // Set Event Map Edges
        set<MapPoint*> spAllEvMpts;
        retrieveAllMapPoints(lLocalKeyFrames, spAllEvMpts);
        retrieveAllMapPoints(lFixedCameras, spAllEvMpts);

        int maxEvVxId = maxVertexId;
        vector<EdgeSE3ProjectXYZ*> vpEvEdgesMono;
        vpEvEdgesMono.reserve(nExpectedSize);
        vector<KeyFrame*> vpEvEdgesKFMono;
        vpEvEdgesMono.reserve(nExpectedSize);
        vector<MapPoint*> vpEvEdgesMpMono;
        vpEvEdgesMpMono.reserve(nExpectedSize);
        uint nEvPoints = setEventMapVxAndEdges(optimizer, spAllEvMpts, vpEvEdgesMono, vpEvEdgesKFMono,
                                               vpEvEdgesMpMono, maxEvVxId, thHuberMono, true);

        if (nPoints+nEvPoints <= 0) {
            LOG(WARNING) << "LM-LBA: No Map Point constraints available, abort optimization...\n";
            return;
        }

        if (nEdges <= 0) {
            LOG(WARNING) << "LM-LBA: Num. KF consts: " << nAllKFs << ", Num MP consts: " << nPoints << ", but no edges available!\n";
        }

        DLOG(INFO) << "LM-LBA: nFixedKFs: " << lFixedCameras.size() << ", nNonFixedKFs: " << lLocalKeyFrames.size()
                   << ", nMPs: " << nPoints << endl;

        // Make sure we have vertices to optimize!
        bool allVerticesMarginalized = true;
        for (auto v : optimizer.indexMapping()) {
            if (! v->marginalized()){
                allVerticesMarginalized = false;
                break;
            }
        }
        if( allVerticesMarginalized ) {
            DLOG(WARNING) << "LM-LBA: All vertices are set to marginalized, abort optimization at " << pKF->mTimeStamp << endl;
            //return;
        }

        if(pbStopFlag)
            if(*pbStopFlag)
                return;

        optimizer.initializeOptimization();

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        optimizer.optimize(5);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        bool bDoMore= true;

        if(pbStopFlag)
            if(*pbStopFlag)
                bDoMore = false;

        if(bDoMore)
        {

            // Check inlier observations
            int nMonoBadObs = 0;
            for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
            {
                ORB_SLAM3::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
                MapPoint* pMP = vpMapPointEdgeMono[i];

                if(pMP->isBad())
                    continue;

                if(e->chi2()>5.991 || !e->isDepthPositive())
                {
                    nMonoBadObs++;
                }
            }

            int nBodyBadObs = 0;
            for(size_t i=0, iend=vpEdgesBody.size(); i<iend;i++)
            {
                ORB_SLAM3::EdgeSE3ProjectXYZToBody* e = vpEdgesBody[i];
                MapPoint* pMP = vpMapPointEdgeBody[i];

                if(pMP->isBad())
                    continue;

                if(e->chi2()>5.991 || !e->isDepthPositive())
                {
                    nBodyBadObs++;
                }
            }

            int nStereoBadObs = 0;
            for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
            {
                g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
                MapPoint* pMP = vpMapPointEdgeStereo[i];

                if(pMP->isBad())
                    continue;

                if(e->chi2()>7.815 || !e->isDepthPositive())
                {
                    nStereoBadObs++;
                }
            }

            int nEvMonoBadObs = 0;
            for(size_t i=0, iend=vpEvEdgesMono.size(); i<iend;i++)
            {
                ORB_SLAM3::EdgeSE3ProjectXYZ* e = vpEvEdgesMono[i];
                MapPoint* pMP = vpEvEdgesMpMono[i];

                if(pMP->isBad())
                    continue;

                if(e->chi2()>5.991 || !e->isDepthPositive())
                {
                    nEvMonoBadObs++;
                }
            }

            // Optimize again
            optimizer.initializeOptimization(0);
            optimizer.optimize(10);

        }

        vector<pair<KeyFrame*,MapPoint*> > vToErase;
        vToErase.reserve(vpEdgesMono.size()+vpEdgesBody.size()+vpEdgesStereo.size());

        // Check inlier observations
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
        {
            ORB_SLAM3::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
            MapPoint* pMP = vpMapPointEdgeMono[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>5.991 || !e->isDepthPositive())
            {
                KeyFrame* pKFi = vpEdgeKFMono[i];
                vToErase.push_back(make_pair(pKFi,pMP));
            }
        }

        for(size_t i=0, iend=vpEdgesBody.size(); i<iend;i++)
        {
            ORB_SLAM3::EdgeSE3ProjectXYZToBody* e = vpEdgesBody[i];
            MapPoint* pMP = vpMapPointEdgeBody[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>5.991 || !e->isDepthPositive())
            {
                KeyFrame* pKFi = vpEdgeKFBody[i];
                vToErase.push_back(make_pair(pKFi,pMP));
            }
        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
        {
            g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
            MapPoint* pMP = vpMapPointEdgeStereo[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>7.815 || !e->isDepthPositive())
            {
                KeyFrame* pKFi = vpEdgeKFStereo[i];
                vToErase.push_back(make_pair(pKFi,pMP));
            }
        }

        for(size_t i=0, iend=vpEvEdgesMono.size(); i<iend;i++)
        {
            ORB_SLAM3::EdgeSE3ProjectXYZ* e = vpEvEdgesMono[i];
            MapPoint* pMP = vpEvEdgesMpMono[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>5.991 || !e->isDepthPositive())
            {
                KeyFrame* pKFi = vpEvEdgesKFMono[i];
                vToErase.push_back(make_pair(pKFi,pMP));
            }
        }

        // Get Map Mutex
        unique_lock<mutex> lock(pMap->mMutexMapUpdate);

        if(!vToErase.empty())
        {
            for(size_t i=0;i<vToErase.size();i++)
            {
                KeyFrame* pKFi = vToErase[i].first;
                MapPoint* pMPi = vToErase[i].second;
                pKFi->EraseMapPointMatch(pMPi);
                pMPi->EraseObservation(pKFi);
            }
        }

        // Recover optimized data
        //Keyframes
        bool bShowStats = false;
        for(auto pKFi : lLocalKeyFrames)
        {
            g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKFi->mnId));
            g2o::SE3Quat SE3quat = vSE3->estimate();
            cv::Mat Tcw = Converter::toCvMat(SE3quat);
            pKFi->SetPose(Tcw);

            // Events
            KeyFrame* pEvKFi = pKFi->mpSynchEvKF;
            if (pEvKFi) {
                pEvKFi->SetPose(Tcw);
            }
        }

        //Points
        for(auto pMP : lLocalMapPoints)
        {
            g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
        }

        // Event Map Points
        for(auto pMP : spAllEvMpts)
        {
            g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxVertexId+1));
            if (vPoint) {
                pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
                pMP->UpdateNormalAndDepth();
            }
        }

        // TODO Check this changeindex
        pMap->IncreaseChangeIndex();
    }


    void EvOptimizer::FullInertialBA(Map *pMap, int its, const bool bFixLocal, const long unsigned int nLoopId, bool *pbStopFlag,
                                     bool bInit, float priorG, float priorA, Eigen::VectorXd *vSingVal, bool *bHess)
    {
        if (!pMap) {
            return;
        }

        long unsigned int maxKFid = pMap->GetMaxKFid();
        const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
        const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

        // Setup optimizer
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolverX::LinearSolverType * linearSolver;

        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

        g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        solver->setUserLambdaInit(1e-5);
        optimizer.setAlgorithm(solver);
        optimizer.setVerbose(false);

        if(pbStopFlag)
            optimizer.setForceStopFlag(pbStopFlag);

        int nNonFixed = 0;

        // Set KeyFrame vertices
        KeyFrame* pIncKF;
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            KeyFrame* pKFi = vpKFs[i];
            if(!pKFi || pKFi->mnId>maxKFid)
                continue;
            VertexPose * VP = new VertexPose(pKFi);
            VP->setId(pKFi->mnId);
            pIncKF=pKFi;
            bool bFixed = false;
            if(bFixLocal)
            {
                bFixed = (pKFi->mnBALocalForKF>=(maxKFid-1)) || (pKFi->mnBAFixedForKF>=(maxKFid-1));
                if(!bFixed)
                    nNonFixed++;
                VP->setFixed(bFixed);
            }
            optimizer.addVertex(VP);

            if(pKFi->bImu)
            {
                VertexVelocity* VV = new VertexVelocity(pKFi);
                VV->setId(maxKFid+3*(pKFi->mnId)+1);
                VV->setFixed(bFixed);
                optimizer.addVertex(VV);
                if (!bInit)
                {
                    VertexGyroBias* VG = new VertexGyroBias(pKFi);
                    VG->setId(maxKFid+3*(pKFi->mnId)+2);
                    VG->setFixed(bFixed);
                    optimizer.addVertex(VG);
                    VertexAccBias* VA = new VertexAccBias(pKFi);
                    VA->setId(maxKFid+3*(pKFi->mnId)+3);
                    VA->setFixed(bFixed);
                    optimizer.addVertex(VA);
                }
            }
        }

        if (bInit)
        {
            VertexGyroBias* VG = new VertexGyroBias(pIncKF);
            VG->setId(4*maxKFid+2);
            VG->setFixed(false);
            optimizer.addVertex(VG);
            VertexAccBias* VA = new VertexAccBias(pIncKF);
            VA->setId(4*maxKFid+3);
            VA->setFixed(false);
            optimizer.addVertex(VA);
        }

        if(bFixLocal)
        {
            if(nNonFixed<3)
                return;
        }

        // IMU links
        for(size_t i=0;i<vpKFs.size();i++)
        {
            KeyFrame* pKFi = vpKFs[i];
            if (!pKFi)
                continue;
            KeyFrame* pPreKF = pKFi->mPrevKF;

            if(!pPreKF || pMap != pPreKF->GetMap())
            {
                Verbose::PrintMess("NOT INERTIAL LINK TO PREVIOUS FRAME!", Verbose::VERBOSITY_NORMAL);
                continue;
            }

            if(pPreKF && pKFi->mnId<=maxKFid)
            {
                if(pKFi->isBad() || pPreKF->mnId>maxKFid)
                    continue;
                if(pKFi->bImu && pPreKF->bImu)
                {
                    pKFi->mpImuPreintegrated->SetNewBias(pPreKF->GetImuBias());
                    g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pPreKF->mnId);
                    g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid+3*(pPreKF->mnId)+1);

                    g2o::HyperGraph::Vertex* VG1;
                    g2o::HyperGraph::Vertex* VA1;
                    g2o::HyperGraph::Vertex* VG2;
                    g2o::HyperGraph::Vertex* VA2;
                    if (!bInit)
                    {
                        VG1 = optimizer.vertex(maxKFid+3*(pPreKF->mnId)+2);
                        VA1 = optimizer.vertex(maxKFid+3*(pPreKF->mnId)+3);
                        VG2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+2);
                        VA2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+3);
                    }
                    else
                    {
                        VG1 = optimizer.vertex(4*maxKFid+2);
                        VA1 = optimizer.vertex(4*maxKFid+3);
                    }

                    g2o::HyperGraph::Vertex* VP2 =  optimizer.vertex(pKFi->mnId);
                    g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+1);

                    if (!bInit)
                    {
                        if(!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2)
                        {
                            cout << "Error" << VP1 << ", "<< VV1 << ", "<< VG1 << ", "<< VA1 << ", " << VP2 << ", " << VV2 <<  ", "<< VG2 << ", "<< VA2 <<endl;

                            continue;
                        }
                    }
                    else
                    {
                        if(!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2)
                        {
                            cout << "Error" << VP1 << ", "<< VV1 << ", "<< VG1 << ", "<< VA1 << ", " << VP2 << ", " << VV2 <<endl;

                            continue;
                        }
                    }

                    EdgeInertial* ei = new EdgeInertial(pKFi->mpImuPreintegrated);
                    ei->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
                    ei->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
                    ei->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG1));
                    ei->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA1));
                    ei->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
                    ei->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));

                    g2o::RobustKernelHuber* rki = new g2o::RobustKernelHuber;
                    ei->setRobustKernel(rki);
                    rki->setDelta(sqrt(16.92));

                    optimizer.addEdge(ei);

                    if (!bInit)
                    {
                        EdgeGyroRW* egr= new EdgeGyroRW();
                        egr->setVertex(0,VG1);
                        egr->setVertex(1,VG2);
                        cv::Mat cvInfoG = pKFi->mpImuPreintegrated->C.rowRange(9,12).colRange(9,12).inv(cv::DECOMP_SVD);
                        Eigen::Matrix3d InfoG;
                        for(int r=0;r<3;r++)
                            for(int c=0;c<3;c++)
                                InfoG(r,c)=cvInfoG.at<float>(r,c);
                        egr->setInformation(InfoG);
                        egr->computeError();
                        optimizer.addEdge(egr);

                        EdgeAccRW* ear = new EdgeAccRW();
                        ear->setVertex(0,VA1);
                        ear->setVertex(1,VA2);
                        cv::Mat cvInfoA = pKFi->mpImuPreintegrated->C.rowRange(12,15).colRange(12,15).inv(cv::DECOMP_SVD);
                        Eigen::Matrix3d InfoA;
                        for(int r=0;r<3;r++)
                            for(int c=0;c<3;c++)
                                InfoA(r,c)=cvInfoA.at<float>(r,c);
                        ear->setInformation(InfoA);
                        ear->computeError();
                        optimizer.addEdge(ear);
                    }
                }
                else
                {
                    cout << pKFi->mnId << " or " << pKFi->mPrevKF->mnId << " no imu" << endl;
                }
            }
        }

        if (bInit)
        {
            g2o::HyperGraph::Vertex* VG = optimizer.vertex(4*maxKFid+2);
            g2o::HyperGraph::Vertex* VA = optimizer.vertex(4*maxKFid+3);

            // Add prior to comon biases
            EdgePriorAcc* epa = new EdgePriorAcc(cv::Mat::zeros(3,1,CV_32F));
            epa->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
            double infoPriorA = priorA; //
            epa->setInformation(infoPriorA*Eigen::Matrix3d::Identity());
            optimizer.addEdge(epa);

            EdgePriorGyro* epg = new EdgePriorGyro(cv::Mat::zeros(3,1,CV_32F));
            epg->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
            double infoPriorG = priorG; //
            epg->setInformation(infoPriorG*Eigen::Matrix3d::Identity());
            optimizer.addEdge(epg);
        }

        const float thHuberMono = sqrt(5.991);
        const float thHuberStereo = sqrt(7.815);

        const unsigned long iniMPid = maxKFid*5;

        vector<bool> vbNotIncludedMP(vpMPs.size(),false);

        ulong maxVertexId = iniMPid;

        for(size_t i=0; i<vpMPs.size(); i++)
        {
            MapPoint* pMP = vpMPs[i];
            g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
            vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
            unsigned long id = pMP->mnId+iniMPid+1;
            vPoint->setId(id);
            vPoint->setMarginalized(true);
            optimizer.addVertex(vPoint);

            const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();


            bool bAllFixed = true;

            //Set edges
            for(map<KeyFrame*,tuple<int,int>>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
            {
                KeyFrame* pKFi = mit->first;

                if(pKFi->mnId>maxKFid)
                    continue;

                if(!pKFi->isBad())
                {
                    const int leftIndex = get<0>(mit->second);
                    cv::KeyPoint kpUn;

                    if(leftIndex != -1 && pKFi->mvuRight[get<0>(mit->second)]<0) // Monocular observation
                    {
                        kpUn = pKFi->getUndistKPtMono(leftIndex);
                        Eigen::Matrix<double,2,1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y;

                        EdgeMono* e = new EdgeMono(0);

                        g2o::OptimizableGraph::Vertex* VP = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId));
                        if(bAllFixed)
                            if(!VP->fixed())
                                bAllFixed=false;

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        e->setVertex(1, VP);
                        e->setMeasurement(obs);
                        const float invSigma2 = pKFi->getKPtInvLevelSigma2(leftIndex);

                        e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        optimizer.addEdge(e);
                    }
                    else if(leftIndex != -1 && pKFi->mvuRight[leftIndex] >= 0) // stereo observation
                    {
                        kpUn = pKFi->getUndistKPtMono(leftIndex);
                        const float kp_ur = pKFi->mvuRight[leftIndex];
                        Eigen::Matrix<double,3,1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                        EdgeStereo* e = new EdgeStereo(0);

                        g2o::OptimizableGraph::Vertex* VP = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId));
                        if(bAllFixed)
                            if(!VP->fixed())
                                bAllFixed=false;

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        e->setVertex(1, VP);
                        e->setMeasurement(obs);
                        const float invSigma2 = pKFi->getORBInvLevelSigma2(kpUn.octave);

                        e->setInformation(Eigen::Matrix3d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberStereo);

                        optimizer.addEdge(e);
                    }

                    if(pKFi->mpCamera2){ // Monocular right observation
                        int rightIndex = get<1>(mit->second);

                        if(rightIndex != -1 && rightIndex < pKFi->numAllKPtsRight()){
                            rightIndex -= pKFi->numAllKPtsLeft();

                            Eigen::Matrix<double,2,1> obs;
                            kpUn = pKFi->getKPtRight(rightIndex);
                            obs << kpUn.pt.x, kpUn.pt.y;

                            EdgeMono *e = new EdgeMono(1);

                            g2o::OptimizableGraph::Vertex* VP = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId));
                            if(bAllFixed)
                                if(!VP->fixed())
                                    bAllFixed=false;

                            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                            e->setVertex(1, VP);
                            e->setMeasurement(obs);
                            const float invSigma2 = pKFi->getORBInvLevelSigma2(kpUn.octave);
                            e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                            e->setRobustKernel(rk);
                            rk->setDelta(thHuberMono);

                            optimizer.addEdge(e);
                        }
                    }
                }
            }

            if(bAllFixed)
            {
                optimizer.removeVertex(vPoint);
                vbNotIncludedMP[i]=true;
            }

            if (id > maxVertexId) {
                maxVertexId = id;
            }
        }

        // Event Map Point Edges
        set<MapPoint*> spEvMpts;
        retrieveAllMapPoints(vpKFs, spEvMpts);
        int nExpectedSizeEv = spEvMpts.size();
        int maxEvMpId = maxVertexId;
        vector<EdgeMono*> vpEvEdgesMono;
        vpEvEdgesMono.reserve(nExpectedSizeEv);
        vector<KeyFrame*> vpEvEdgesKFMono;
        vpEvEdgesMono.reserve(nExpectedSizeEv);
        vector<MapPoint*> vpEvEdgesMpMono;
        vpEvEdgesMpMono.reserve(nExpectedSizeEv);

        const uint nVertexEvMP = setEventMapVxAndEdges(optimizer, spEvMpts, vpEvEdgesMono, vpEvEdgesKFMono,
                                                       vpEvEdgesMpMono, maxEvMpId, thHuberMono, true);

        if(pbStopFlag)
            if(*pbStopFlag)
                return;


        optimizer.initializeOptimization();
        optimizer.optimize(its);


        // Recover optimized data
        //Keyframes
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            KeyFrame* pKFi = vpKFs[i];
            if(!pKFi || pKFi->mnId>maxKFid)
                continue;

            VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(pKFi->mnId));
            KeyFrame* pEvKFi = pKFi->mpSynchEvKF;

            if(nLoopId==0)
            {
                cv::Mat Tcw = Converter::toCvSE3(VP->estimate().Rcw[0], VP->estimate().tcw[0]);
                pKFi->SetPose(Tcw);

                if (pEvKFi) {
                    pEvKFi->SetPose(Tcw);
                }
            }
            else
            {
                pKFi->mTcwGBA = cv::Mat::eye(4,4,CV_32F);
                Converter::toCvMat(VP->estimate().Rcw[0]).copyTo(pKFi->mTcwGBA.rowRange(0,3).colRange(0,3));
                Converter::toCvMat(VP->estimate().tcw[0]).copyTo(pKFi->mTcwGBA.rowRange(0,3).col(3));
                pKFi->mnBAGlobalForKF = nLoopId;

            }
            if(pKFi->bImu)
            {
                VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+1));
                if(nLoopId==0)
                {
                    pKFi->SetVelocity(Converter::toCvMat(VV->estimate()));
                }
                else
                {
                    pKFi->mVwbGBA = Converter::toCvMat(VV->estimate());
                }

                VertexGyroBias* VG;
                VertexAccBias* VA;
                if (!bInit)
                {
                    VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+2));
                    VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+3));
                }
                else
                {
                    VG = static_cast<VertexGyroBias*>(optimizer.vertex(4*maxKFid+2));
                    VA = static_cast<VertexAccBias*>(optimizer.vertex(4*maxKFid+3));
                }

                Vector6d vb;
                vb << VG->estimate(), VA->estimate();
                IMU::Bias b (vb[3],vb[4],vb[5],vb[0],vb[1],vb[2]);
                if(nLoopId==0)
                {
                    pKFi->SetNewBias(b);
                }
                else
                {
                    pKFi->mBiasGBA = b;
                }
            }
        }

        //Points
        for(size_t i=0; i<vpMPs.size(); i++)
        {
            if(vbNotIncludedMP[i])
                continue;

            MapPoint* pMP = vpMPs[i];
            g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+iniMPid+1));

            if(nLoopId==0)
            {
                pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
                pMP->UpdateNormalAndDepth();
            }
            else
            {
                pMP->mPosGBA.create(3,1,CV_32F);
                Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
                pMP->mnBAGlobalForKF = nLoopId;
            }

        }

        for (auto* pEvMP : spEvMpts) {

            g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pEvMP->mnId+maxVertexId+1));

            if(nLoopId==0)
            {
                pEvMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
                pEvMP->UpdateNormalAndDepth();
            }
        }

        pMap->IncreaseChangeIndex();
    }

    void EvOptimizer::LocalInertialBA(KeyFrame *pKF, bool *pbStopFlag, Map *pMap, int& num_fixedKF, int& num_OptKF,
                                      int& num_MPs, int& num_edges, bool bLarge, bool bRecInit)
    {
        Map* pCurrentMap = pKF->GetMap();

        int maxOpt=10;
        int opt_it=10;
        if(bLarge)
        {
            maxOpt=25;
            opt_it=4;
        }
        const int Nd = std::min((int)pCurrentMap->KeyFramesInMap()-2,maxOpt);
        const unsigned long maxKFid = pKF->mnId;

        vector<KeyFrame*> vpOptimizableKFs;
        const vector<KeyFrame*> vpNeighsKFs = pKF->GetVectorCovisibleKeyFrames();
        list<KeyFrame*> lpOptVisKFs;

        vpOptimizableKFs.reserve(Nd);
        vpOptimizableKFs.push_back(pKF);
        pKF->mnBALocalForKF = pKF->mnId;
        for(int i=1; i<Nd; i++)
        {
            if(vpOptimizableKFs.back()->mPrevKF)
            {
                vpOptimizableKFs.push_back(vpOptimizableKFs.back()->mPrevKF);
                vpOptimizableKFs.back()->mnBALocalForKF = pKF->mnId;
            }
            else
                break;
        }

        int N = vpOptimizableKFs.size();

        // Optimizable points seen by temporal optimizable keyframes
        list<MapPoint*> lLocalMapPoints;
        for(int i=0; i<N; i++)
        {
            vector<MapPoint*> vpMPs = vpOptimizableKFs[i]->GetMapPointMatches();
            for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
            {
                MapPoint* pMP = *vit;
                if(pMP)
                    if(!pMP->isBad())
                        if(pMP->mnBALocalForKF!=pKF->mnId)
                        {
                            lLocalMapPoints.push_back(pMP);
                            pMP->mnBALocalForKF=pKF->mnId;
                        }
            }
        }

        // Fixed Keyframe: First frame previous KF to optimization window)
        list<KeyFrame*> lFixedKeyFrames;
        if(vpOptimizableKFs.back()->mPrevKF)
        {
            lFixedKeyFrames.push_back(vpOptimizableKFs.back()->mPrevKF);
            vpOptimizableKFs.back()->mPrevKF->mnBAFixedForKF=pKF->mnId;
        }
        else
        {
            vpOptimizableKFs.back()->mnBALocalForKF=0;
            vpOptimizableKFs.back()->mnBAFixedForKF=pKF->mnId;
            lFixedKeyFrames.push_back(vpOptimizableKFs.back());
            vpOptimizableKFs.pop_back();
        }

        // Optimizable visual KFs
        const int maxCovKF = 0;
        for(int i=0, iend=vpNeighsKFs.size(); i<iend; i++)
        {
            if(lpOptVisKFs.size() >= maxCovKF)
                break;

            KeyFrame* pKFi = vpNeighsKFs[i];
            if(pKFi->mnBALocalForKF == pKF->mnId || pKFi->mnBAFixedForKF == pKF->mnId)
                continue;
            pKFi->mnBALocalForKF = pKF->mnId;
            if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
            {
                lpOptVisKFs.push_back(pKFi);

                vector<MapPoint*> vpMPs = pKFi->GetMapPointMatches();
                for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
                {
                    MapPoint* pMP = *vit;
                    if(pMP)
                        if(!pMP->isBad())
                            if(pMP->mnBALocalForKF!=pKF->mnId)
                            {
                                lLocalMapPoints.push_back(pMP);
                                pMP->mnBALocalForKF=pKF->mnId;
                            }
                }
            }
        }

        // Fixed KFs which are not covisible optimizable
        const int maxFixKF = 200;

        for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
        {
            map<KeyFrame*,tuple<int,int>> observations = (*lit)->GetObservations();
            for(map<KeyFrame*,tuple<int,int>>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
            {
                KeyFrame* pKFi = mit->first;

                if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)
                {
                    pKFi->mnBAFixedForKF=pKF->mnId;
                    if(!pKFi->isBad())
                    {
                        lFixedKeyFrames.push_back(pKFi);
                        break;
                    }
                }
            }
            if(lFixedKeyFrames.size()>=maxFixKF)
                break;
        }

        bool bNonFixed = (lFixedKeyFrames.size() == 0);

        // Setup optimizer
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolverX::LinearSolverType * linearSolver;
        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

        g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

        if(bLarge)
        {
            g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
            solver->setUserLambdaInit(1e-2); // to avoid iterating for finding optimal lambda
            optimizer.setAlgorithm(solver);
        }
        else
        {
            g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
            solver->setUserLambdaInit(1e0);
            optimizer.setAlgorithm(solver);
        }


        // Set Local temporal KeyFrame vertices
        N=vpOptimizableKFs.size();
        num_fixedKF = 0;
        num_OptKF = 0;
        num_MPs = 0;
        num_edges = 0;
        for(int i=0; i<N; i++)
        {
            KeyFrame* pKFi = vpOptimizableKFs[i];

            VertexPose * VP = new VertexPose(pKFi);
            VP->setId(pKFi->mnId);
            VP->setFixed(false);
            optimizer.addVertex(VP);

            if(pKFi->bImu)
            {
                VertexVelocity* VV = new VertexVelocity(pKFi);
                VV->setId(maxKFid+3*(pKFi->mnId)+1);
                VV->setFixed(false);
                optimizer.addVertex(VV);
                VertexGyroBias* VG = new VertexGyroBias(pKFi);
                VG->setId(maxKFid+3*(pKFi->mnId)+2);
                VG->setFixed(false);
                optimizer.addVertex(VG);
                VertexAccBias* VA = new VertexAccBias(pKFi);
                VA->setId(maxKFid+3*(pKFi->mnId)+3);
                VA->setFixed(false);
                optimizer.addVertex(VA);
            }
            num_OptKF++;
        }

        // Set Local visual KeyFrame vertices
        for(list<KeyFrame*>::iterator it=lpOptVisKFs.begin(), itEnd = lpOptVisKFs.end(); it!=itEnd; it++)
        {
            KeyFrame* pKFi = *it;
            VertexPose * VP = new VertexPose(pKFi);
            VP->setId(pKFi->mnId);
            VP->setFixed(false);
            optimizer.addVertex(VP);

            num_OptKF++;
        }

        // Set Fixed KeyFrame vertices
        for(list<KeyFrame*>::iterator lit=lFixedKeyFrames.begin(), lend=lFixedKeyFrames.end(); lit!=lend; lit++)
        {
            KeyFrame* pKFi = *lit;
            VertexPose * VP = new VertexPose(pKFi);
            VP->setId(pKFi->mnId);
            VP->setFixed(true);
            optimizer.addVertex(VP);

            if(pKFi->bImu) // This should be done only for keyframe just before temporal window
            {
                VertexVelocity* VV = new VertexVelocity(pKFi);
                VV->setId(maxKFid+3*(pKFi->mnId)+1);
                VV->setFixed(true);
                optimizer.addVertex(VV);
                VertexGyroBias* VG = new VertexGyroBias(pKFi);
                VG->setId(maxKFid+3*(pKFi->mnId)+2);
                VG->setFixed(true);
                optimizer.addVertex(VG);
                VertexAccBias* VA = new VertexAccBias(pKFi);
                VA->setId(maxKFid+3*(pKFi->mnId)+3);
                VA->setFixed(true);
                optimizer.addVertex(VA);
            }
            num_fixedKF++;
        }

        // Create intertial constraints
        vector<EdgeInertial*> vei(N,(EdgeInertial*)NULL);
        vector<EdgeGyroRW*> vegr(N,(EdgeGyroRW*)NULL);
        vector<EdgeAccRW*> vear(N,(EdgeAccRW*)NULL);

        for(int i=0;i<N;i++)
        {
            KeyFrame* pKFi = vpOptimizableKFs[i];

            if(!pKFi->mPrevKF)
            {
                cout << "NOT INERTIAL LINK TO PREVIOUS FRAME!!!!" << endl;
                continue;
            }
            if(pKFi->bImu && pKFi->mPrevKF->bImu && pKFi->mpImuPreintegrated)
            {
                pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
                g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
                g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+1);
                g2o::HyperGraph::Vertex* VG1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+2);
                g2o::HyperGraph::Vertex* VA1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+3);
                g2o::HyperGraph::Vertex* VP2 =  optimizer.vertex(pKFi->mnId);
                g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+1);
                g2o::HyperGraph::Vertex* VG2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+2);
                g2o::HyperGraph::Vertex* VA2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+3);

                if(!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2)
                {
                    cerr << "Error " << VP1 << ", "<< VV1 << ", "<< VG1 << ", "<< VA1 << ", " << VP2 << ", " << VV2 <<  ", "<< VG2 << ", "<< VA2 <<endl;
                    continue;
                }

                vei[i] = new EdgeInertial(pKFi->mpImuPreintegrated);

                vei[i]->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
                vei[i]->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
                vei[i]->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG1));
                vei[i]->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA1));
                vei[i]->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
                vei[i]->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));

                if(i==N-1 || bRecInit)
                {
                    // All inertial residuals are included without robust cost function, but not that one linking the
                    // last optimizable keyframe inside of the local window and the first fixed keyframe out. The
                    // information matrix for this measurement is also downweighted. This is done to avoid accumulating
                    // error due to fixing variables.
                    g2o::RobustKernelHuber* rki = new g2o::RobustKernelHuber;
                    vei[i]->setRobustKernel(rki);
                    if(i==N-1)
                        vei[i]->setInformation(vei[i]->information()*1e-2);
                    rki->setDelta(sqrt(16.92));
                }
                optimizer.addEdge(vei[i]);

                vegr[i] = new EdgeGyroRW();
                vegr[i]->setVertex(0,VG1);
                vegr[i]->setVertex(1,VG2);
                cv::Mat cvInfoG = pKFi->mpImuPreintegrated->C.rowRange(9,12).colRange(9,12).inv(cv::DECOMP_SVD);
                Eigen::Matrix3d InfoG;

                for(int r=0;r<3;r++)
                    for(int c=0;c<3;c++)
                        InfoG(r,c)=cvInfoG.at<float>(r,c);
                vegr[i]->setInformation(InfoG);
                optimizer.addEdge(vegr[i]);
                num_edges++;

                vear[i] = new EdgeAccRW();
                vear[i]->setVertex(0,VA1);
                vear[i]->setVertex(1,VA2);
                cv::Mat cvInfoA = pKFi->mpImuPreintegrated->C.rowRange(12,15).colRange(12,15).inv(cv::DECOMP_SVD);
                Eigen::Matrix3d InfoA;
                for(int r=0;r<3;r++)
                    for(int c=0;c<3;c++)
                        InfoA(r,c)=cvInfoA.at<float>(r,c);
                vear[i]->setInformation(InfoA);

                optimizer.addEdge(vear[i]);
                num_edges++;
            }
            else
                cout << "ERROR building inertial edge" << endl;
        }

        // Set MapPoint vertices
        const int nExpectedSize = (N+lFixedKeyFrames.size())*lLocalMapPoints.size();

        // Mono
        vector<EdgeMono*> vpEdgesMono;
        vpEdgesMono.reserve(nExpectedSize);

        vector<KeyFrame*> vpEdgeKFMono;
        vpEdgeKFMono.reserve(nExpectedSize);

        vector<MapPoint*> vpMapPointEdgeMono;
        vpMapPointEdgeMono.reserve(nExpectedSize);

        // Stereo
        vector<EdgeStereo*> vpEdgesStereo;
        vpEdgesStereo.reserve(nExpectedSize);

        vector<KeyFrame*> vpEdgeKFStereo;
        vpEdgeKFStereo.reserve(nExpectedSize);

        vector<MapPoint*> vpMapPointEdgeStereo;
        vpMapPointEdgeStereo.reserve(nExpectedSize);



        const float thHuberMono = sqrt(5.991);
        const float chi2Mono2 = 5.991;
        const float thHuberStereo = sqrt(7.815);
        const float chi2Stereo2 = 7.815;

        const unsigned long iniMPid = maxKFid*5;
        ulong maxVertexId = iniMPid;

        map<int,int> mVisEdges;
        for(int i=0;i<N;i++)
        {
            KeyFrame* pKFi = vpOptimizableKFs[i];
            mVisEdges[pKFi->mnId] = 0;
        }
        for(list<KeyFrame*>::iterator lit=lFixedKeyFrames.begin(), lend=lFixedKeyFrames.end(); lit!=lend; lit++)
        {
            mVisEdges[(*lit)->mnId] = 0;
        }

        num_MPs = lLocalMapPoints.size();
        for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
        {
            MapPoint* pMP = *lit;
            g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
            vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));

            unsigned long id = pMP->mnId+iniMPid+1;
            vPoint->setId(id);
            vPoint->setMarginalized(true);
            optimizer.addVertex(vPoint);
            const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();

            // Create visual constraints
            for(map<KeyFrame*,tuple<int,int>>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
            {
                KeyFrame* pKFi = mit->first;

                if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)
                    continue;

                if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                {
                    const int leftIndex = get<0>(mit->second);

                    cv::KeyPoint kpUn;

                    // Monocular left observation
                    if(leftIndex != -1 && pKFi->mvuRight[leftIndex]<0)
                    {
                        mVisEdges[pKFi->mnId]++;

                        kpUn = pKFi->getUndistKPtMono(leftIndex);
                        Eigen::Matrix<double,2,1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y;

                        EdgeMono* e = new EdgeMono(0);

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);

                        // Add here uncerteinty
                        const float unc2 = pKFi->mpCamera->uncertainty2(obs);

                        const float &invSigma2 = pKFi->getKPtInvLevelSigma2(leftIndex)/unc2;//mvInvLevelSigma2[kpUn.octave]/unc2;
                        e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        optimizer.addEdge(e);
                        vpEdgesMono.push_back(e);
                        vpEdgeKFMono.push_back(pKFi);
                        vpMapPointEdgeMono.push_back(pMP);

                        num_edges++;
                    }
                        // Stereo-observation
                    else if(leftIndex != -1)// Stereo observation
                    {
                        kpUn = pKFi->getUndistKPtMono(leftIndex);
                        mVisEdges[pKFi->mnId]++;

                        const float kp_ur = pKFi->mvuRight[leftIndex];
                        Eigen::Matrix<double,3,1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                        EdgeStereo* e = new EdgeStereo(0);

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);

                        // Add here uncerteinty
                        const float unc2 = pKFi->mpCamera->uncertainty2(obs.head(2));

                        const float &invSigma2 = pKFi->getKPtInvLevelSigma2(leftIndex)/unc2;//mvInvLevelSigma2[kpUn.octave]/unc2;
                        e->setInformation(Eigen::Matrix3d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberStereo);

                        optimizer.addEdge(e);
                        vpEdgesStereo.push_back(e);
                        vpEdgeKFStereo.push_back(pKFi);
                        vpMapPointEdgeStereo.push_back(pMP);

                        num_edges++;
                    }

                    // Monocular right observation
                    if(pKFi->mpCamera2){
                        int rightIndex = get<1>(mit->second);

                        if(rightIndex != -1 ){
                            rightIndex -= pKFi->numAllKPtsLeft();
                            mVisEdges[pKFi->mnId]++;

                            Eigen::Matrix<double,2,1> obs;
                            cv::KeyPoint kp = pKFi->getKPtRight(rightIndex);
                            obs << kp.pt.x, kp.pt.y;

                            EdgeMono* e = new EdgeMono(1);

                            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                            e->setMeasurement(obs);

                            // Add here uncerteinty
                            const float unc2 = pKFi->mpCamera->uncertainty2(obs);

                            const float &invSigma2 = pKFi->getORBInvLevelSigma2(kpUn.octave)/unc2;//mvInvLevelSigma2[kpUn.octave]/unc2;
                            e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                            e->setRobustKernel(rk);
                            rk->setDelta(thHuberMono);

                            optimizer.addEdge(e);
                            vpEdgesMono.push_back(e);
                            vpEdgeKFMono.push_back(pKFi);
                            vpMapPointEdgeMono.push_back(pMP);

                            num_edges++;
                        }
                    }
                }
            }

            if (id > maxVertexId) {
                maxVertexId = id;
            }
        }

        //cout << "Total map points: " << lLocalMapPoints.size() << endl;
        for(map<int,int>::iterator mit=mVisEdges.begin(), mend=mVisEdges.end(); mit!=mend; mit++)
        {
            assert(mit->second>=3);
        }


        // Event Map Point Edges
        set<MapPoint*> spEvMpts;
        retrieveAllMapPoints(vpOptimizableKFs, spEvMpts);
        retrieveAllMapPoints(lpOptVisKFs, spEvMpts);
        retrieveAllMapPoints(lFixedKeyFrames, spEvMpts);

        int nExpectedSizeEv = (N+lFixedKeyFrames.size())*spEvMpts.size();
        int maxEvMpId = maxVertexId;
        vector<EdgeMono*> vpEvEdgesMono;
        vpEvEdgesMono.reserve(nExpectedSizeEv);
        vector<KeyFrame*> vpEvEdgesKFMono;
        vpEvEdgesMono.reserve(nExpectedSizeEv);
        vector<MapPoint*> vpEvEdgesMpMono;
        vpEvEdgesMpMono.reserve(nExpectedSizeEv);

        const uint nVertexEvMP = setEventMapVxAndEdges(optimizer, spEvMpts, vpEvEdgesMono, vpEvEdgesKFMono,
                                                       vpEvEdgesMpMono, maxEvMpId, thHuberMono, true);


        optimizer.initializeOptimization();
        optimizer.computeActiveErrors();

        float err = optimizer.activeRobustChi2();
        optimizer.optimize(opt_it); // Originally to 2
        float err_end = optimizer.activeRobustChi2();
        if(pbStopFlag)
            optimizer.setForceStopFlag(pbStopFlag);


        vector<pair<KeyFrame*,MapPoint*> > vToErase;
        vToErase.reserve(vpEdgesMono.size()+vpEdgesStereo.size());

        // Check inlier observations
        // Mono
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
        {
            EdgeMono* e = vpEdgesMono[i];
            MapPoint* pMP = vpMapPointEdgeMono[i];
            bool bClose = pMP->mTrackDepth<10.f;

            if(pMP->isBad())
                continue;

            if((e->chi2()>chi2Mono2 && !bClose) || (e->chi2()>1.5f*chi2Mono2 && bClose) || !e->isDepthPositive())
            {
                KeyFrame* pKFi = vpEdgeKFMono[i];
                vToErase.push_back(make_pair(pKFi,pMP));
            }
        }

        // Stereo
        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
        {
            EdgeStereo* e = vpEdgesStereo[i];
            MapPoint* pMP = vpMapPointEdgeStereo[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>chi2Stereo2)
            {
                KeyFrame* pKFi = vpEdgeKFStereo[i];
                vToErase.push_back(make_pair(pKFi,pMP));
            }
        }

        // Event Mono
        for(size_t i=0, iend=vpEvEdgesMono.size(); i<iend;i++)
        {
            EdgeMono* e = vpEvEdgesMono[i];
            MapPoint* pMP = vpEvEdgesMpMono[i];
            bool bClose = pMP->mTrackDepth<10.f;

            if(pMP->isBad())
                continue;

            if((e->chi2()>chi2Mono2 && !bClose) || (e->chi2()>1.5f*chi2Mono2 && bClose) || !e->isDepthPositive())
            {
                KeyFrame* pKFi = vpEvEdgesKFMono[i];
                vToErase.push_back(make_pair(pKFi,pMP));
            }
        }

        // Get Map Mutex and erase outliers
        unique_lock<mutex> lock(pMap->mMutexMapUpdate);

        if((2*err < err_end || isnan(err) || isnan(err_end)) && !bLarge)
        {
            cout << "FAIL LOCAL-INERTIAL BA!!!!" << endl;
            return;
        }


        if(!vToErase.empty())
        {
            for(size_t i=0;i<vToErase.size();i++)
            {
                KeyFrame* pKFi = vToErase[i].first;
                MapPoint* pMPi = vToErase[i].second;
                pKFi->EraseMapPointMatch(pMPi);
                pMPi->EraseObservation(pKFi);
            }
        }


        // Display main statistcis of optimization
        Verbose::PrintMess("LIBA KFs: " + to_string(N), Verbose::VERBOSITY_DEBUG);
        Verbose::PrintMess("LIBA bNonFixed?: " + to_string(bNonFixed), Verbose::VERBOSITY_DEBUG);
        Verbose::PrintMess("LIBA KFs visual outliers: " + to_string(vToErase.size()), Verbose::VERBOSITY_DEBUG);

        for(list<KeyFrame*>::iterator lit=lFixedKeyFrames.begin(), lend=lFixedKeyFrames.end(); lit!=lend; lit++)
            (*lit)->mnBAFixedForKF = 0;

        // Recover optimized data
        // Local temporal Keyframes
        N=vpOptimizableKFs.size();
        for(int i=0; i<N; i++)
        {
            KeyFrame* pKFi = vpOptimizableKFs[i];

            VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(pKFi->mnId));
            cv::Mat Tcw = Converter::toCvSE3(VP->estimate().Rcw[0], VP->estimate().tcw[0]);
            pKFi->SetPose(Tcw);
            pKFi->mnBALocalForKF=0;

            if(pKFi->bImu)
            {
                VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+1));
                pKFi->SetVelocity(Converter::toCvMat(VV->estimate()));
                VertexGyroBias* VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+2));
                VertexAccBias* VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+3));
                Vector6d b;
                b << VG->estimate(), VA->estimate();
                pKFi->SetNewBias(IMU::Bias(b[3],b[4],b[5],b[0],b[1],b[2]));
            }

            if (pKFi->mpSynchEvKF) {
                pKFi->mpSynchEvKF->SetPose(Tcw);
            }
        }

        // Local visual KeyFrame
        for(list<KeyFrame*>::iterator it=lpOptVisKFs.begin(), itEnd = lpOptVisKFs.end(); it!=itEnd; it++)
        {
            KeyFrame* pKFi = *it;
            VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(pKFi->mnId));
            cv::Mat Tcw = Converter::toCvSE3(VP->estimate().Rcw[0], VP->estimate().tcw[0]);
            pKFi->SetPose(Tcw);
            pKFi->mnBALocalForKF=0;

            if (pKFi->mpSynchEvKF) {
                pKFi->mpSynchEvKF->SetPose(Tcw);
            }
        }

        //Points
        for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
        {
            MapPoint* pMP = *lit;
            g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+iniMPid+1));
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
        }

        for(auto* pEvMp : spEvMpts)
        {
            g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pEvMp->mnId+maxVertexId+1));
            pEvMp->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pEvMp->UpdateNormalAndDepth();
        }

        pMap->IncreaseChangeIndex();
    }

    int EvOptimizer::PoseInertialOptimizationLastKeyFrame(Frame *pFrame, bool bRecInit)
    {
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolverX::LinearSolverType * linearSolver;

        linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

        g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

        g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
        optimizer.setVerbose(false);
        optimizer.setAlgorithm(solver);

        int nInitialMonoCorrespondences=0;
        int nInitialStereoCorrespondences=0;
        int nInitialCorrespondences=0;

        // Set Frame vertex
        VertexPose* VP = new VertexPose(pFrame);
        VP->setId(0);
        VP->setFixed(false);
        optimizer.addVertex(VP);
        VertexVelocity* VV = new VertexVelocity(pFrame);
        VV->setId(1);
        VV->setFixed(false);
        optimizer.addVertex(VV);
        VertexGyroBias* VG = new VertexGyroBias(pFrame);
        VG->setId(2);
        VG->setFixed(false);
        optimizer.addVertex(VG);
        VertexAccBias* VA = new VertexAccBias(pFrame);
        VA->setId(3);
        VA->setFixed(false);
        optimizer.addVertex(VA);

        // Set MapPoint vertices
        const int N = pFrame->numAllKPts();
        const int Nleft = pFrame->numKPtsLeft();
        const bool bRight = (Nleft!=-1);

        vector<EdgeMonoOnlyPose*> vpEdgesMono;
        vector<EdgeStereoOnlyPose*> vpEdgesStereo;
        vector<size_t> vnIndexEdgeMono;
        vector<size_t> vnIndexEdgeStereo;
        vpEdgesMono.reserve(N);
        vpEdgesStereo.reserve(N);
        vnIndexEdgeMono.reserve(N);
        vnIndexEdgeStereo.reserve(N);

        const float thHuberMono = sqrt(5.991);
        const float thHuberStereo = sqrt(7.815);

        int nInitialCorrespondencesEv=0;
        vector<ORB_SLAM3::EdgeMonoOnlyPose*> vpEdgesMonoEv;
        vector<size_t> vnIndexEdgeMonoEv;
        FramePtr pEvFrame = pFrame->mpEvFrame.lock();

        {
            unique_lock<mutex> lock(MapPoint::mGlobalMutex);

            for(int i=0; i<N; i++)
            {
                MapPoint* pMP = pFrame->getMapPoint(i);
                if(pMP)
                {
                    cv::KeyPoint kpUn;

                    // Left monocular observation
                    if((!bRight && pFrame->mvuRight[i]<0) || i < Nleft)
                    {
                        float fInvLevelSigma2;
                        if(i < Nleft) {// pair left-right
                            kpUn = pFrame->getDistKPtMono(i); // Check original -> Done: This is stereo case!
                            fInvLevelSigma2 = pFrame->getORBInvLevelSigma2(kpUn.octave);
                        }
                        else {
                            kpUn = pFrame->getUndistKPtMono(i);
                            fInvLevelSigma2 = pFrame->getKPtInvLevelSigma2(i);
                        }

                        nInitialMonoCorrespondences++;
                        pFrame->setMPOutlier(i, false);

                        Eigen::Matrix<double,2,1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y;

                        EdgeMonoOnlyPose* e = new EdgeMonoOnlyPose(pMP->GetWorldPos(),0);

                        e->setVertex(0,VP);
                        e->setMeasurement(obs);

                        // Add here uncerteinty
                        const float unc2 = pFrame->mpCamera->uncertainty2(obs);

                        const float invSigma2 = fInvLevelSigma2/unc2;
                        e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        optimizer.addEdge(e);

                        vpEdgesMono.push_back(e);
                        vnIndexEdgeMono.push_back(i);
                    }
                        // Stereo observation
                    else if(!bRight)
                    {
                        nInitialStereoCorrespondences++;
                        pFrame->setMPOutlier(i, false);

                        kpUn = pFrame->getUndistKPtMono(i);
                        const float kp_ur = pFrame->mvuRight[i];
                        Eigen::Matrix<double,3,1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                        EdgeStereoOnlyPose* e = new EdgeStereoOnlyPose(pMP->GetWorldPos());

                        e->setVertex(0, VP);
                        e->setMeasurement(obs);

                        // Add here uncerteinty
                        const float unc2 = pFrame->mpCamera->uncertainty2(obs.head(2));

                        const float &invSigma2 = pFrame->getKPtInvLevelSigma2(i)/unc2;
                        e->setInformation(Eigen::Matrix3d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberStereo);

                        optimizer.addEdge(e);

                        vpEdgesStereo.push_back(e);
                        vnIndexEdgeStereo.push_back(i);
                    }

                    // Right monocular observation
                    if(bRight && i >= Nleft)
                    {
                        nInitialMonoCorrespondences++;
                        pFrame->setMPOutlier(i, false);

                        kpUn = pFrame->getKPtRight(i - Nleft);
                        Eigen::Matrix<double,2,1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y;

                        EdgeMonoOnlyPose* e = new EdgeMonoOnlyPose(pMP->GetWorldPos(),1);

                        e->setVertex(0,VP);
                        e->setMeasurement(obs);

                        // Add here uncerteinty
                        const float unc2 = pFrame->mpCamera->uncertainty2(obs);

                        const float invSigma2 = pFrame->getORBInvLevelSigma2(kpUn.octave)/unc2;
                        e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        optimizer.addEdge(e);

                        vpEdgesMono.push_back(e);
                        vnIndexEdgeMono.push_back(i);
                    }
                }
            }

            // Set Event Edges
            if (pEvFrame) {
                nInitialCorrespondencesEv = setEventMapVxAndEdges(optimizer, pEvFrame.get(), vpEdgesMonoEv, vnIndexEdgeMonoEv, 0, thHuberMono);
            }
        }

        nInitialCorrespondences = nInitialCorrespondencesEv + nInitialMonoCorrespondences + nInitialStereoCorrespondences;

        KeyFrame* pKF = pFrame->mpLastKeyFrame;
        VertexPose* VPk = new VertexPose(pKF);
        VPk->setId(4);
        VPk->setFixed(true);
        optimizer.addVertex(VPk);
        VertexVelocity* VVk = new VertexVelocity(pKF);
        VVk->setId(5);
        VVk->setFixed(true);
        optimizer.addVertex(VVk);
        VertexGyroBias* VGk = new VertexGyroBias(pKF);
        VGk->setId(6);
        VGk->setFixed(true);
        optimizer.addVertex(VGk);
        VertexAccBias* VAk = new VertexAccBias(pKF);
        VAk->setId(7);
        VAk->setFixed(true);
        optimizer.addVertex(VAk);

        EdgeInertial* ei = new EdgeInertial(pFrame->mpImuPreintegrated);

        ei->setVertex(0, VPk);
        ei->setVertex(1, VVk);
        ei->setVertex(2, VGk);
        ei->setVertex(3, VAk);
        ei->setVertex(4, VP);
        ei->setVertex(5, VV);
        optimizer.addEdge(ei);

        EdgeGyroRW* egr = new EdgeGyroRW();
        egr->setVertex(0,VGk);
        egr->setVertex(1,VG);
        cv::Mat cvInfoG = pFrame->mpImuPreintegrated->C.rowRange(9,12).colRange(9,12).inv(cv::DECOMP_SVD);
        Eigen::Matrix3d InfoG;
        for(int r=0;r<3;r++)
            for(int c=0;c<3;c++)
                InfoG(r,c)=cvInfoG.at<float>(r,c);
        egr->setInformation(InfoG);
        optimizer.addEdge(egr);

        EdgeAccRW* ear = new EdgeAccRW();
        ear->setVertex(0,VAk);
        ear->setVertex(1,VA);
        cv::Mat cvInfoA = pFrame->mpImuPreintegrated->C.rowRange(12,15).colRange(12,15).inv(cv::DECOMP_SVD);
        Eigen::Matrix3d InfoA;
        for(int r=0;r<3;r++)
            for(int c=0;c<3;c++)
                InfoA(r,c)=cvInfoA.at<float>(r,c);
        ear->setInformation(InfoA);
        optimizer.addEdge(ear);

        // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
        // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
        float chi2Mono[4]={12,7.5,5.991,5.991};
        float chi2Stereo[4]={15.6,9.8,7.815,7.815};

        int its[4]={10,10,10,10};

        int nBad=0;
        int nBadMono = 0;
        int nBadMonoEv = 0;
        int nBadStereo = 0;
        int nInliersMono = 0;
        int nInliersMonoEv = 0;
        int nInliersStereo = 0;
        int nInliers=0;
        bool bOut = false;
        for(size_t it=0; it<4; it++)
        {
            optimizer.initializeOptimization(0);
            optimizer.optimize(its[it]);

            nBad=0;
            nBadMono = 0;
            nBadMonoEv = 0;
            nBadStereo = 0;
            nInliers=0;
            nInliersMono=0;
            nInliersMonoEv = 0;
            nInliersStereo=0;
            float chi2close = 1.5*chi2Mono[it];

            // For monocular observations
            for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
            {
                EdgeMonoOnlyPose* e = vpEdgesMono[i];

                const size_t idx = vnIndexEdgeMono[i];

                if(pFrame->getMPOutlier(idx))
                {
                    e->computeError();
                }

                const float chi2 = e->chi2();
                bool bClose = pFrame->getMapPoint(idx)->mTrackDepth<10.f;

                if((chi2>chi2Mono[it]&&!bClose)||(bClose && chi2>chi2close)||!e->isDepthPositive())
                {
                    pFrame->setMPOutlier(idx, true);
                    e->setLevel(1);
                    nBadMono++;
                }
                else
                {
                    pFrame->setMPOutlier(idx, false);
                    e->setLevel(0);
                    nInliersMono++;
                }

                if (it==2)
                    e->setRobustKernel(0);
            }

            // For stereo observations
            for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
            {
                EdgeStereoOnlyPose* e = vpEdgesStereo[i];

                const size_t idx = vnIndexEdgeStereo[i];

                if(pFrame->getMPOutlier(idx))
                {
                    e->computeError();
                }

                const float chi2 = e->chi2();

                if(chi2>chi2Stereo[it])
                {
                    pFrame->setMPOutlier(idx, true);
                    e->setLevel(1); // not included in next optimization
                    nBadStereo++;
                }
                else
                {
                    pFrame->setMPOutlier(idx, false);
                    e->setLevel(0);
                    nInliersStereo++;
                }

                if(it==2)
                    e->setRobustKernel(0);
            }

            // Mono Event
            for(size_t i=0, iend=vpEdgesMonoEv.size(); i<iend; i++)
            {
                EdgeMonoOnlyPose* e = vpEdgesMonoEv[i];

                const size_t idx = vnIndexEdgeMonoEv[i];

                if(pEvFrame->getMPOutlier(idx))
                {
                    e->computeError();
                }

                const float chi2 = e->chi2();
                bool bClose = pEvFrame->getMapPoint(idx)->mTrackDepth<10.f;

                if((chi2>chi2Mono[it]&&!bClose)||(bClose && chi2>chi2close)||!e->isDepthPositive())
                {
                    pEvFrame->setMPOutlier(idx, true);
                    e->setLevel(1);
                    nBadMonoEv++;
                }
                else
                {
                    pEvFrame->setMPOutlier(idx, false);
                    e->setLevel(0);
                    nInliersMonoEv++;
                }

                if (it==2)
                    e->setRobustKernel(0);
            }

            nInliers = nInliersMono + nInliersStereo + nInliersMonoEv;
            nBad = nBadMono + nBadStereo + nBadMonoEv;

            if(optimizer.edges().size()<10)
            {
                cout << "PIOLKF: NOT ENOUGH EDGES" << endl;
                break;
            }

        }

        // If not too much tracks, recover not too bad points
        if ((nInliers<30) && !bRecInit)
        {
            nBad=0;
            const float chi2MonoOut = 18.f;
            const float chi2StereoOut = 24.f;
            EdgeMonoOnlyPose* e1, *e3;
            EdgeStereoOnlyPose* e2;
            for(size_t i=0, iend=vnIndexEdgeMono.size(); i<iend; i++)
            {
                const size_t idx = vnIndexEdgeMono[i];
                e1 = vpEdgesMono[i];
                e1->computeError();
                if (e1->chi2()<chi2MonoOut)
                    pFrame->setMPOutlier(idx, false);
                else
                    nBad++;
            }
            for(size_t i=0, iend=vnIndexEdgeMonoEv.size(); i<iend; i++)
            {
                const size_t idx = vnIndexEdgeMonoEv[i];
                e3 = vpEdgesMonoEv[i];
                e3->computeError();
                if (e3->chi2()<chi2MonoOut)
                    pEvFrame->setMPOutlier(idx, false);
                else
                    nBad++;
            }
            for(size_t i=0, iend=vnIndexEdgeStereo.size(); i<iend; i++)
            {
                const size_t idx = vnIndexEdgeStereo[i];
                e2 = vpEdgesStereo[i];
                e2->computeError();
                if (e2->chi2()<chi2StereoOut)
                    pFrame->setMPOutlier(idx, false);
                else
                    nBad++;
            }
        }

        // Recover optimized pose, velocity and biases
        cv::Mat Rwb = Converter::toCvMat(VP->estimate().Rwb);
        cv::Mat twb = Converter::toCvMat(VP->estimate().twb);
        cv::Mat Vel = Converter::toCvMat(VV->estimate());
        pFrame->SetImuPoseVelocity(Rwb, twb, Vel);
        Vector6d b;
        b << VG->estimate(), VA->estimate();
        IMU::Bias imuBias = IMU::Bias(b[3],b[4],b[5],b[0],b[1],b[2]);
        pFrame->mImuBias = imuBias;

        // Recover Hessian, marginalize keyFframe states and generate new prior for frame
        Eigen::Matrix<double,15,15> H;
        H.setZero();

        H.block<9,9>(0,0)+= ei->GetHessian2();
        H.block<3,3>(9,9) += egr->GetHessian2();
        H.block<3,3>(12,12) += ear->GetHessian2();

        int tot_in = 0, tot_out = 0;
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            EdgeMonoOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if(!pFrame->getMPOutlier(idx))
            {
                H.block<6,6>(0,0) += e->GetHessian();
                tot_in++;
            }
            else
                tot_out++;
        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
        {
            EdgeStereoOnlyPose* e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if(!pFrame->getMPOutlier(idx))
            {
                H.block<6,6>(0,0) += e->GetHessian();
                tot_in++;
            }
            else
                tot_out++;
        }

        for(size_t i=0, iend=vpEdgesMonoEv.size(); i<iend; i++)
        {
            EdgeMonoOnlyPose* e = vpEdgesMonoEv[i];

            const size_t idx = vnIndexEdgeMonoEv[i];

            if(!pEvFrame->getMPOutlier(idx))
            {
                H.block<6,6>(0,0) += e->GetHessian();
                tot_in++;
            }
            else
                tot_out++;
        }

        pFrame->mpcpi = new ConstraintPoseImu(VP->estimate().Rwb,VP->estimate().twb,VV->estimate(),VG->estimate(),VA->estimate(),H);

        if (pEvFrame) {
            pEvFrame->SetImuPoseVelocity(Rwb, twb, Vel);
            pEvFrame->mImuBias = imuBias;

            pEvFrame->mpcpi = new ConstraintPoseImu(VP->estimate().Rwb,VP->estimate().twb,VV->estimate(),VG->estimate(),VA->estimate(),H);
        }

        return nInitialCorrespondences-nBad;
    }

    int EvOptimizer::PoseInertialOptimizationLastFrame(Frame *pFrame, bool bRecInit)
    {
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolverX::LinearSolverType * linearSolver;

        linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

        g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

        g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
        optimizer.setAlgorithm(solver);
        optimizer.setVerbose(false);

        int nInitialMonoCorrespondences=0;
        int nInitialStereoCorrespondences=0;
        int nInitialCorrespondences=0;

        // Set Current Frame vertex
        VertexPose* VP = new VertexPose(pFrame);
        VP->setId(0);
        VP->setFixed(false);
        optimizer.addVertex(VP);
        VertexVelocity* VV = new VertexVelocity(pFrame);
        VV->setId(1);
        VV->setFixed(false);
        optimizer.addVertex(VV);
        VertexGyroBias* VG = new VertexGyroBias(pFrame);
        VG->setId(2);
        VG->setFixed(false);
        optimizer.addVertex(VG);
        VertexAccBias* VA = new VertexAccBias(pFrame);
        VA->setId(3);
        VA->setFixed(false);
        optimizer.addVertex(VA);

        // Set MapPoint vertices
        const int N = pFrame->numAllKPts();
        const int Nleft = pFrame->numKPtsLeft();
        const bool bRight = (Nleft!=-1);

        vector<EdgeMonoOnlyPose*> vpEdgesMono;
        vector<EdgeStereoOnlyPose*> vpEdgesStereo;
        vector<size_t> vnIndexEdgeMono;
        vector<size_t> vnIndexEdgeStereo;
        vpEdgesMono.reserve(N);
        vpEdgesStereo.reserve(N);
        vnIndexEdgeMono.reserve(N);
        vnIndexEdgeStereo.reserve(N);

        const float thHuberMono = sqrt(5.991);
        const float thHuberStereo = sqrt(7.815);

        int nInitialCorrespondencesEv=0;
        vector<ORB_SLAM3::EdgeMonoOnlyPose*> vpEdgesMonoEv;
        vector<size_t> vnIndexEdgeMonoEv;
        FramePtr pEvFrame = pFrame->mpEvFrame.lock();

        {
            unique_lock<mutex> lock(MapPoint::mGlobalMutex);

            for(int i=0; i<N; i++)
            {
                MapPoint* pMP = pFrame->getMapPoint(i);
                if(pMP)
                {
                    cv::KeyPoint kpUn;
                    // Left monocular observation
                    if((!bRight && pFrame->mvuRight[i]<0) || i < Nleft)
                    {
                        float fInvLevelSigma2;
                        if(i < Nleft) {// pair left-right
                            kpUn = pFrame->getDistKPtMono(i); // Check original -> Done!
                            fInvLevelSigma2 = pFrame->getORBInvLevelSigma2(kpUn.octave);
                        }
                        else {
                            kpUn = pFrame->getUndistKPtMono(i);
                            fInvLevelSigma2 = pFrame->getKPtInvLevelSigma2(i);
                        }

                        nInitialMonoCorrespondences++;
                        pFrame->setMPOutlier(i, false);

                        Eigen::Matrix<double,2,1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y;

                        EdgeMonoOnlyPose* e = new EdgeMonoOnlyPose(pMP->GetWorldPos(),0);

                        e->setVertex(0,VP);
                        e->setMeasurement(obs);

                        // Add here uncerteinty
                        const float unc2 = pFrame->mpCamera->uncertainty2(obs);

                        const float invSigma2 = fInvLevelSigma2/unc2;
                        e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        optimizer.addEdge(e);

                        vpEdgesMono.push_back(e);
                        vnIndexEdgeMono.push_back(i);
                    }
                        // Stereo observation
                    else if(!bRight)
                    {
                        nInitialStereoCorrespondences++;
                        pFrame->setMPOutlier(i, false);

                        kpUn = pFrame->getUndistKPtMono(i);
                        const float kp_ur = pFrame->mvuRight[i];
                        Eigen::Matrix<double,3,1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                        EdgeStereoOnlyPose* e = new EdgeStereoOnlyPose(pMP->GetWorldPos());

                        e->setVertex(0, VP);
                        e->setMeasurement(obs);

                        // Add here uncerteinty
                        const float unc2 = pFrame->mpCamera->uncertainty2(obs.head(2));

                        const float &invSigma2 = pFrame->getKPtInvLevelSigma2(i)/unc2;
                        e->setInformation(Eigen::Matrix3d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberStereo);

                        optimizer.addEdge(e);

                        vpEdgesStereo.push_back(e);
                        vnIndexEdgeStereo.push_back(i);
                    }

                    // Right monocular observation
                    if(bRight && i >= Nleft)
                    {
                        nInitialMonoCorrespondences++;
                        pFrame->setMPOutlier(i, false);

                        kpUn = pFrame->getKPtRight(i - Nleft);
                        Eigen::Matrix<double,2,1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y;

                        EdgeMonoOnlyPose* e = new EdgeMonoOnlyPose(pMP->GetWorldPos(),1);

                        e->setVertex(0,VP);
                        e->setMeasurement(obs);

                        // Add here uncerteinty
                        const float unc2 = pFrame->mpCamera->uncertainty2(obs);

                        const float invSigma2 = pFrame->getORBInvLevelSigma2(kpUn.octave)/unc2;
                        e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        optimizer.addEdge(e);

                        vpEdgesMono.push_back(e);
                        vnIndexEdgeMono.push_back(i);
                    }
                }
            }

            // Set Event Edges
            if (pEvFrame) {
                nInitialCorrespondencesEv = setEventMapVxAndEdges(optimizer, pEvFrame.get(), vpEdgesMonoEv, vnIndexEdgeMonoEv, 0, thHuberMono);
            }
        }

        nInitialCorrespondences = nInitialMonoCorrespondences + nInitialStereoCorrespondences + nInitialCorrespondencesEv;

        // Set Previous Frame Vertex
        Frame* pFp = pFrame->mpPrevFrame;

        VertexPose* VPk = new VertexPose(pFp);
        VPk->setId(4);
        VPk->setFixed(false);
        optimizer.addVertex(VPk);
        VertexVelocity* VVk = new VertexVelocity(pFp);
        VVk->setId(5);
        VVk->setFixed(false);
        optimizer.addVertex(VVk);
        VertexGyroBias* VGk = new VertexGyroBias(pFp);
        VGk->setId(6);
        VGk->setFixed(false);
        optimizer.addVertex(VGk);
        VertexAccBias* VAk = new VertexAccBias(pFp);
        VAk->setId(7);
        VAk->setFixed(false);
        optimizer.addVertex(VAk);

        EdgeInertial* ei = new EdgeInertial(pFrame->mpImuPreintegratedFrame);

        ei->setVertex(0, VPk);
        ei->setVertex(1, VVk);
        ei->setVertex(2, VGk);
        ei->setVertex(3, VAk);
        ei->setVertex(4, VP);
        ei->setVertex(5, VV);
        optimizer.addEdge(ei);

        EdgeGyroRW* egr = new EdgeGyroRW();
        egr->setVertex(0,VGk);
        egr->setVertex(1,VG);
        cv::Mat cvInfoG = pFrame->mpImuPreintegratedFrame->C.rowRange(9,12).colRange(9,12).inv(cv::DECOMP_SVD);
        Eigen::Matrix3d InfoG;
        for(int r=0;r<3;r++)
            for(int c=0;c<3;c++)
                InfoG(r,c)=cvInfoG.at<float>(r,c);
        egr->setInformation(InfoG);
        optimizer.addEdge(egr);

        EdgeAccRW* ear = new EdgeAccRW();
        ear->setVertex(0,VAk);
        ear->setVertex(1,VA);
        cv::Mat cvInfoA = pFrame->mpImuPreintegratedFrame->C.rowRange(12,15).colRange(12,15).inv(cv::DECOMP_SVD);
        Eigen::Matrix3d InfoA;
        for(int r=0;r<3;r++)
            for(int c=0;c<3;c++)
                InfoA(r,c)=cvInfoA.at<float>(r,c);
        ear->setInformation(InfoA);
        optimizer.addEdge(ear);

        if (!pFp->mpcpi)
            Verbose::PrintMess("pFp->mpcpi does not exist!!!\nPrevious Frame " + to_string(pFp->mnId), Verbose::VERBOSITY_NORMAL);

        EdgePriorPoseImu* ep = new EdgePriorPoseImu(pFp->mpcpi);

        ep->setVertex(0,VPk);
        ep->setVertex(1,VVk);
        ep->setVertex(2,VGk);
        ep->setVertex(3,VAk);
        g2o::RobustKernelHuber* rkp = new g2o::RobustKernelHuber;
        ep->setRobustKernel(rkp);
        rkp->setDelta(5);
        optimizer.addEdge(ep);

        // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
        // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.

        const float chi2Mono[4]={5.991,5.991,5.991,5.991};
        const float chi2Stereo[4]={15.6f,9.8f,7.815f,7.815f};
        const int its[4]={10,10,10,10};

        int nBad=0;
        int nBadMono = 0;
        int nBadMonoEv = 0;
        int nBadStereo = 0;
        int nInliersMono = 0;
        int nInliersMonoEv = 0;
        int nInliersStereo = 0;
        int nInliers=0;
        for(size_t it=0; it<4; it++)
        {
            optimizer.initializeOptimization(0);
            optimizer.optimize(its[it]);

            nBad=0;
            nBadMono = 0;
            nBadMonoEv = 0;
            nBadStereo = 0;
            nInliers=0;
            nInliersMono=0;
            nInliersMonoEv=0;
            nInliersStereo=0;
            float chi2close = 1.5*chi2Mono[it];

            for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
            {
                EdgeMonoOnlyPose* e = vpEdgesMono[i];

                const size_t idx = vnIndexEdgeMono[i];
                bool bClose = pFrame->getMapPoint(idx)->mTrackDepth<10.f;

                if(pFrame->getMPOutlier(idx))
                {
                    e->computeError();
                }

                const float chi2 = e->chi2();

                if((chi2>chi2Mono[it]&&!bClose)||(bClose && chi2>chi2close)||!e->isDepthPositive())
                {
                    pFrame->setMPOutlier(idx, true);
                    e->setLevel(1);
                    nBadMono++;
                }
                else
                {
                    pFrame->setMPOutlier(idx, false);
                    e->setLevel(0);
                    nInliersMono++;
                }

                if (it==2)
                    e->setRobustKernel(0);

            }

            for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
            {
                EdgeStereoOnlyPose* e = vpEdgesStereo[i];

                const size_t idx = vnIndexEdgeStereo[i];

                if(pFrame->getMPOutlier(idx))
                {
                    e->computeError();
                }

                const float chi2 = e->chi2();

                if(chi2>chi2Stereo[it])
                {
                    pFrame->setMPOutlier(idx, true);
                    e->setLevel(1);
                    nBadStereo++;
                }
                else
                {
                    pFrame->setMPOutlier(idx, false);
                    e->setLevel(0);
                    nInliersStereo++;
                }

                if(it==2)
                    e->setRobustKernel(0);
            }

            for(size_t i=0, iend=vpEdgesMonoEv.size(); i<iend; i++)
            {
                EdgeMonoOnlyPose* e = vpEdgesMonoEv[i];

                const size_t idx = vnIndexEdgeMonoEv[i];
                bool bClose = pEvFrame->getMapPoint(idx)->mTrackDepth<10.f;

                if(pEvFrame->getMPOutlier(idx))
                {
                    e->computeError();
                }

                const float chi2 = e->chi2();

                if((chi2>chi2Mono[it]&&!bClose)||(bClose && chi2>chi2close)||!e->isDepthPositive())
                {
                    pEvFrame->setMPOutlier(idx, true);
                    e->setLevel(1);
                    nBadMonoEv++;
                }
                else
                {
                    pEvFrame->setMPOutlier(idx, false);
                    e->setLevel(0);
                    nInliersMonoEv++;
                }

                if (it==2)
                    e->setRobustKernel(0);

            }

            nInliers = nInliersMono + nInliersStereo + nInliersMonoEv;
            nBad = nBadMono + nBadStereo + nBadMonoEv;

            if(optimizer.edges().size()<10)
            {
                cout << "PIOLF: NOT ENOUGH EDGES" << endl;
                break;
            }
        }


        if ((nInliers<30) && !bRecInit)
        {
            nBad=0;
            const float chi2MonoOut = 18.f;
            const float chi2StereoOut = 24.f;
            EdgeMonoOnlyPose* e1, *e3;
            EdgeStereoOnlyPose* e2;
            for(size_t i=0, iend=vnIndexEdgeMono.size(); i<iend; i++)
            {
                const size_t idx = vnIndexEdgeMono[i];
                e1 = vpEdgesMono[i];
                e1->computeError();
                if (e1->chi2()<chi2MonoOut)
                    pFrame->setMPOutlier(idx, false);
                else
                    nBad++;

            }
            for(size_t i=0, iend=vnIndexEdgeStereo.size(); i<iend; i++)
            {
                const size_t idx = vnIndexEdgeStereo[i];
                e2 = vpEdgesStereo[i];
                e2->computeError();
                if (e2->chi2()<chi2StereoOut)
                    pFrame->setMPOutlier(idx, false);
                else
                    nBad++;
            }
            for(size_t i=0, iend=vnIndexEdgeMonoEv.size(); i<iend; i++)
            {
                const size_t idx = vnIndexEdgeMonoEv[i];
                e3 = vpEdgesMonoEv[i];
                e3->computeError();
                if (e3->chi2()<chi2MonoOut)
                    pEvFrame->setMPOutlier(idx, false);
                else
                    nBad++;

            }
        }

        nInliers = nInliersMono + nInliersStereo + nInliersMonoEv;


        // Recover optimized pose, velocity and biases
        cv::Mat Rwb = Converter::toCvMat(VP->estimate().Rwb);
        cv::Mat twb = Converter::toCvMat(VP->estimate().twb);
        cv::Mat Vel = Converter::toCvMat(VV->estimate());
        pFrame->SetImuPoseVelocity(Rwb, twb, Vel);
        Vector6d b;
        b << VG->estimate(), VA->estimate();
        IMU::Bias imuBias = IMU::Bias(b[3],b[4],b[5],b[0],b[1],b[2]);
        pFrame->mImuBias = imuBias;

        // Recover Hessian, marginalize previous frame states and generate new prior for frame
        Eigen::Matrix<double,30,30> H;
        H.setZero();

        H.block<24,24>(0,0)+= ei->GetHessian();

        Eigen::Matrix<double,6,6> Hgr = egr->GetHessian();
        H.block<3,3>(9,9) += Hgr.block<3,3>(0,0);
        H.block<3,3>(9,24) += Hgr.block<3,3>(0,3);
        H.block<3,3>(24,9) += Hgr.block<3,3>(3,0);
        H.block<3,3>(24,24) += Hgr.block<3,3>(3,3);

        Eigen::Matrix<double,6,6> Har = ear->GetHessian();
        H.block<3,3>(12,12) += Har.block<3,3>(0,0);
        H.block<3,3>(12,27) += Har.block<3,3>(0,3);
        H.block<3,3>(27,12) += Har.block<3,3>(3,0);
        H.block<3,3>(27,27) += Har.block<3,3>(3,3);

        H.block<15,15>(0,0) += ep->GetHessian();

        int tot_in = 0, tot_out = 0;
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            EdgeMonoOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if(!pFrame->getMPOutlier(idx))
            {
                H.block<6,6>(15,15) += e->GetHessian();
                tot_in++;
            }
            else
                tot_out++;
        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
        {
            EdgeStereoOnlyPose* e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if(!pFrame->getMPOutlier(idx))
            {
                H.block<6,6>(15,15) += e->GetHessian();
                tot_in++;
            }
            else
                tot_out++;
        }

        for(size_t i=0, iend=vpEdgesMonoEv.size(); i<iend; i++)
        {
            EdgeMonoOnlyPose* e = vpEdgesMonoEv[i];

            const size_t idx = vnIndexEdgeMonoEv[i];

            if(!pEvFrame->getMPOutlier(idx))
            {
                H.block<6,6>(15,15) += e->GetHessian();
                tot_in++;
            }
            else
                tot_out++;
        }

        H = Marginalize(H,0,14);

        pFrame->mpcpi = new ConstraintPoseImu(VP->estimate().Rwb,VP->estimate().twb,VV->estimate(),VG->estimate(),VA->estimate(),H.block<15,15>(15,15));
        delete pFp->mpcpi;
        pFp->mpcpi = NULL;

        if (pEvFrame) {
            pEvFrame->SetImuPoseVelocity(Rwb, twb, Vel);
            pEvFrame->mImuBias = imuBias;

            pEvFrame->mpcpi = new ConstraintPoseImu(VP->estimate().Rwb,VP->estimate().twb,VV->estimate(),VG->estimate(),VA->estimate(),H.block<15,15>(15,15));
        }

        return nInitialCorrespondences-nBad;
    }

    // InertialOptimization methods are safe without override (because don't use map points)

    // TODO: Take care about some loop-closing optimization methods (Especially mergeInertial)

    /* -------------------------------------------------------------------------------------------------------------- */

    uint EvOptimizer::setEventMapVxAndEdges(g2o::SparseOptimizer &optimizer, const std::vector<ORB_SLAM3::KeyFrame *> &vpOrbKFs,
                                            const int maxId, const double thHuber, const bool bRobust, const bool inertial) {

        unsigned nEvVertexMP = 0;
        for (auto pOrbKF : vpOrbKFs) {

            if (!pOrbKF)
                continue;

            auto pEvKF = pOrbKF->mpSynchEvKF;
            if (!pEvKF)
                continue;


        }

        return nEvVertexMP;
    }

    uint EvOptimizer::setEventMapVxAndEdges(g2o::SparseOptimizer &optimizer, const std::set<ORB_SLAM3::MapPoint *> &spEvMpts,
                                            vector<ORB_SLAM3::EdgeSE3ProjectXYZ*>& vpEvEdgesMono,
                                            std::vector<ORB_SLAM3::KeyFrame*>& vpKFsMono, std::vector<ORB_SLAM3::MapPoint*>& vpMptsMono,
                                            int& maxId, const double thHuber, const bool bRobust) {

        uint nPoints = 0, nEdges = 0;
        uint maxLocalId = maxId;

        for (auto pMP : spEvMpts) {

            if (!pMP || pMP->isBad())
                continue;

            nEdges = 0;
            const int p3dId = static_cast<int>(pMP->mnId)+maxId+1;

            g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
            vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
            vPoint->setId(p3dId);
            vPoint->setMarginalized(true);
            optimizer.addVertex(vPoint);

            const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();

            //Set edges
            for(const auto & observation : observations)
            {
                KeyFrame* pEvKF = observation.first;

                if (!pEvKF)
                    continue;

                KeyFrame* pOrbKF = pEvKF->mpSynchOrbKF;

                if (!pOrbKF || pOrbKF->isBad())
                    continue;

                const int leftIndex = get<0>(observation.second);

                // Monocular observation
                if(leftIndex != -1 && pEvKF->mvuRight[get<0>(observation.second)]<0)
                {
                    const int kfId = static_cast<int>(pOrbKF->mnId);
                    auto* pVKF = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(kfId));

                    if (!pVKF)
                        continue;

                    const cv::KeyPoint &kpUn = pEvKF->getUndistKPtMono(leftIndex);
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    const float &invSigma2 = pEvKF->getKPtInvLevelSigma2(leftIndex);//[kpUn.octave];

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    rk->setDelta(thHuber);

                    ORB_SLAM3::EdgeSE3ProjectXYZ* e = new ORB_SLAM3::EdgeSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(p3dId)));
                    e->setVertex(1, pVKF);
                    e->setMeasurement(obs);
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);
                    e->pCamera = pEvKF->mpCamera;

                    if (bRobust) {
                        e->setRobustKernel(rk);
                    }
                    optimizer.addEdge(e);

                    vpEvEdgesMono.push_back(e);
                    vpKFsMono.push_back(pEvKF);
                    vpMptsMono.push_back(pMP);

                    nEdges++;
                }
            }

            if(nEdges==0) {
                optimizer.removeVertex(vPoint);
                //vbValidPts3d[i] = false;
            }
            else {
                //vbNotIncludedMP[i]=false;
                nPoints++;
            }

            if (p3dId > maxLocalId) {
                maxLocalId = p3dId;
            }
        }

        maxId = maxLocalId;
        return nPoints;
    }

    uint EvOptimizer::setEventMapVxAndEdges(g2o::SparseOptimizer &optimizer, const std::set<ORB_SLAM3::MapPoint *> &spEvMpts,
                                            vector<ORB_SLAM3::EdgeMono*>& vpEvEdgesMono,
                                            std::vector<ORB_SLAM3::KeyFrame*>& vpKFsMono, std::vector<ORB_SLAM3::MapPoint*>& vpMptsMono,
                                            int& maxId, const double thHuber, const bool bRobust) {

        uint nPoints = 0, nEdges = 0;
        uint maxLocalId = maxId;

        for (auto pMP : spEvMpts) {

            if (!pMP || pMP->isBad())
                continue;

            nEdges = 0;
            const int p3dId = static_cast<int>(pMP->mnId)+maxId+1;

            g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
            vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
            vPoint->setId(p3dId);
            vPoint->setMarginalized(true);
            optimizer.addVertex(vPoint);

            const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();

            //Set edges
            for(const auto & observation : observations)
            {
                KeyFrame* pEvKF = observation.first;

                if (!pEvKF)
                    continue;

                KeyFrame* pOrbKF = pEvKF->mpSynchOrbKF;

                if (!pOrbKF || pOrbKF->isBad())
                    continue;

                const int leftIndex = get<0>(observation.second);

                // Monocular observation
                if(leftIndex != -1 && pEvKF->mvuRight[get<0>(observation.second)]<0)
                {
                    const int kfId = static_cast<int>(pOrbKF->mnId);
                    auto* pVKF = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(kfId));

                    if (!pVKF)
                        continue;

                    const cv::KeyPoint &kpUn = pEvKF->getUndistKPtMono(leftIndex);
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    const float &invSigma2 = pEvKF->getKPtInvLevelSigma2(leftIndex);//[kpUn.octave];

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    rk->setDelta(thHuber);

                    EdgeMono* e = new EdgeMono(0);

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(p3dId)));
                    e->setVertex(1, pVKF);
                    e->setMeasurement(obs);
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    if (bRobust) {
                        e->setRobustKernel(rk);
                    }
                    optimizer.addEdge(e);

                    vpEvEdgesMono.push_back(e);
                    vpKFsMono.push_back(pEvKF);
                    vpMptsMono.push_back(pMP);

                    nEdges++;
                }
            }

            if(nEdges==0) {
                optimizer.removeVertex(vPoint);
                //vbValidPts3d[i] = false;
            }
            else {
                //vbNotIncludedMP[i]=false;
                nPoints++;
            }

            if (p3dId > maxLocalId) {
                maxLocalId = p3dId;
            }
        }

        maxId = maxLocalId;
        return nPoints;
    }

    int EvOptimizer::setEventMapVxAndEdges(g2o::SparseOptimizer &optimizer, ORB_SLAM3::Frame *pEvFrame,
            vector<ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose *> &vpEdgesMonoEv,
            std::vector<std::size_t> &vnIndexEdgeMonoEv, int poseId, const double thHuber, const bool bRobust) {

        const int nEvPts = pEvFrame->numAllKPts();

        int nInitialCorrespondencesEv = 0;

        vpEdgesMonoEv.reserve(nEvPts);
        vnIndexEdgeMonoEv.reserve(nEvPts);

        for(int i=0; i<nEvPts; i++)
        {
            MapPoint* pMP = pEvFrame->getMapPoint(i);
            if(pMP)
            {
                // Only Monocular SLAM supported
                //Conventional SLAM
                if(!pEvFrame->mpCamera2){
                    // Monocular observation
                    if(pEvFrame->mvuRight[i]<0)
                    {
                        nInitialCorrespondencesEv++;
                        pEvFrame->setMPOutlier(i, false);

                        Eigen::Matrix<double,2,1> obs;
                        const cv::KeyPoint &kpUn = pEvFrame->getUndistKPtMono(i);
                        obs << kpUn.pt.x, kpUn.pt.y;

                        ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose* e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(poseId)));
                        e->setMeasurement(obs);
                        const float invSigma2 = pEvFrame->getKPtInvLevelSigma2(i);
                        e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuber);

                        e->pCamera = pEvFrame->mpCamera;
                        cv::Mat Xw = pMP->GetWorldPos();
                        e->Xw[0] = Xw.at<float>(0);
                        e->Xw[1] = Xw.at<float>(1);
                        e->Xw[2] = Xw.at<float>(2);

                        optimizer.addEdge(e);

                        vpEdgesMonoEv.push_back(e);
                        vnIndexEdgeMonoEv.push_back(i);
                    }
                }
            }
        }

        return nInitialCorrespondencesEv;
    }

    int EvOptimizer::setEventMapVxAndEdges(g2o::SparseOptimizer &optimizer, ORB_SLAM3::Frame *pEvFrame,
                                           vector<ORB_SLAM3::EdgeMonoOnlyPose *> &vpEdgesMonoEv,
                                           std::vector<std::size_t> &vnIndexEdgeMonoEv, int poseId, const double thHuber, const bool bRobust) {

        const int nEvPts = pEvFrame->numAllKPts();

        int nInitialCorrespondencesEv = 0;

        vpEdgesMonoEv.reserve(nEvPts);
        vnIndexEdgeMonoEv.reserve(nEvPts);

        for(int i=0; i<nEvPts; i++)
        {
            MapPoint* pMP = pEvFrame->getMapPoint(i);
            if(pMP)
            {
                // Only Monocular SLAM supported
                //Conventional SLAM
                if(!pEvFrame->mpCamera2){
                    // Monocular observation
                    if(pEvFrame->mvuRight[i]<0)
                    {
                        nInitialCorrespondencesEv++;
                        pEvFrame->setMPOutlier(i, false);

                        Eigen::Matrix<double,2,1> obs;
                        const cv::KeyPoint &kpUn = pEvFrame->getUndistKPtMono(i);
                        obs << kpUn.pt.x, kpUn.pt.y;

                        EdgeMonoOnlyPose* e = new EdgeMonoOnlyPose(pMP->GetWorldPos(),0);

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(poseId)));
                        e->setMeasurement(obs);
                        const float invSigma2 = pEvFrame->getKPtInvLevelSigma2(i);
                        e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuber);

                        optimizer.addEdge(e);

                        vpEdgesMonoEv.push_back(e);
                        vnIndexEdgeMonoEv.push_back(i);
                    }
                }
            }
        }

        return nInitialCorrespondencesEv;
    }

    void EvOptimizer::retrieveAllMapPoints(const std::vector<ORB_SLAM3::KeyFrame *> &vpOrbKFs, std::set<ORB_SLAM3::MapPoint *> &spEvMpts) {

        for (auto pOrbKF : vpOrbKFs) {

            if (!pOrbKF)
                continue;

            auto pEvKF = pOrbKF->mpSynchEvKF;
            if (!pEvKF)
                continue;

            vector<MapPoint*> vpMpts = pEvKF->GetMapPointMatches();

            for (auto pMpt : vpMpts) {

                if (pMpt && !pMpt->isBad()) {

                    spEvMpts.insert(pMpt);
                }
            }

            if (!pEvKF->mpSynchOrbKF) {

                LOG_EVERY_N(WARNING, 100) << "EvOptimizer::retrieveAllMapPoints: KF #" << pEvKF->mnId << " has no ORB connection!\n";
                pEvKF->mpSynchOrbKF = pOrbKF;
            }
        }
    }

    void EvOptimizer::retrieveAllMapPoints(const std::list<ORB_SLAM3::KeyFrame *> &vpOrbKFs, std::set<ORB_SLAM3::MapPoint *> &spEvMpts) {

        for (auto pOrbKF : vpOrbKFs) {

            if (!pOrbKF)
                continue;

            auto pEvKF = pOrbKF->mpSynchEvKF;
            if (!pEvKF)
                continue;

            vector<MapPoint*> vpMpts = pEvKF->GetMapPointMatches();

            for (auto pMpt : vpMpts) {

                if (pMpt && !pMpt->isBad()) {

                    spEvMpts.insert(pMpt);
                }
            }

            if (!pEvKF->mpSynchOrbKF) {

                LOG_EVERY_N(WARNING, 100) << "EvOptimizer::retrieveAllMapPoints: KF #" << pEvKF->mnId << " has no ORB connection!\n";
                pEvKF->mpSynchOrbKF = pOrbKF;
            }
        }
    }

} // EORB_SLAM


