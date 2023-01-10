//
// Created by root on 3/1/21.
//

#include "MyOptimizer.h"

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
#include "Converter.h"

#include "MyOptimTypes.h"
#include "Visualization.h"


using namespace ORB_SLAM3;


namespace EORB_SLAM {

    bool x3d_comprator(const cv::Mat& first, const cv::Mat& second) {

        return cv::norm(first) < cv::norm(second);
    }

    struct X3dCmp {
        bool operator()(const cv::Mat& first, const cv::Mat& second) {

            return cv::norm(first) < cv::norm(second);
        }
    };

    double MyOptimizer::OptimInitV(std::vector<EORB_SLAM::EvFramePtr>& vpEvFrames, std::vector<cv::Point3f> &vPts3d,
                                 std::vector<int>& vbValidPts3d, const int nIterations, const bool bRobust,
                                 const bool addPrior) {

        const int numFrames = vpEvFrames.size();
        //assert(!kfIni.mTcw.empty() && !kfCur.mTcw.empty() && !vPts3d.empty() && kfCur.numAllKPts() == vPts3d.size());
        if (vpEvFrames.empty() || numFrames < 2 || vPts3d.size() != vbValidPts3d.size()) {
            DLOG(ERROR) << "MyOptimizer::OptimInitV: Wrong Input: num frames = " << numFrames
                        << ", numPts3d = " << vPts3d.size() << ", numValidPts = " << vbValidPts3d.size() << endl;
            return MAXFLOAT;
        }

        // Better to work with maxId instead of numFrames because these can be different
        const int maxFrId = static_cast<int>(std::max(vpEvFrames.back()->mnId, vpEvFrames.size()));

        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

        g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        const float deltaMono = sqrt(5.991);
        const float thHuber2D = sqrt(5.99f);
        const float thHuber3D = sqrt(7.815f);

        // Set Frame vertices
        // Assume Init. Frame is the first in vector
        int vcnt = 0;
        const int vfactorPose = vcnt;
        addPoseVertices(optimizer, vpEvFrames, vcnt);

        // Prior Edges (and Vertices)
        const int vfactorPrior = vcnt;
        if (addPrior) {
            addPriorEdges(optimizer, vpEvFrames, vfactorPose, vcnt, thHuber2D, bRobust);
        }

        // Set MapPoint vertices and edges
        int maxId = vcnt * maxFrId;
        addMapPointEdges(optimizer, vpEvFrames, vPts3d, vbValidPts3d, vfactorPose, maxId, thHuber2D, bRobust);

        // Optimize
        optimizer.setVerbose(false);
        optimizer.initializeOptimization();
        optimizer.optimize(nIterations);
        Verbose::PrintMess("BA: End of the optimization", Verbose::VERBOSITY_NORMAL);

        // Recover optimized poses
        recoverPose(optimizer, vpEvFrames, vfactorPose);

        // Also update map point matches
        recoverMapPoints(optimizer, vPts3d, vbValidPts3d, maxId);

        return optimizer.chi2();
    }

    // TODO: What about Prior IMU Bias??
    double MyOptimizer::OptimInitVI(vector<EORB_SLAM::EvFramePtr> &vpEvFrames, vector<cv::Point3f> &vPts3d,
                                  vector<int>& vbValidPts3d, double& medDepth, InitInfoImuPtr& pInfoImu,
                                  const int nIterations, const bool bRobust,
                                  const bool addPrior, const bool addInertial, const bool addGDirScale,
                                  const bool singleBias) {

        if (vpEvFrames.empty() || vpEvFrames.size() < 2) {
            DLOG(ERROR) << "MyOptimizer::OptimInitV: Not enough Event Frames: " << vpEvFrames.size() << endl;
            return MAXFLOAT;
        }

        const int maxFrId = static_cast<int>(std::max(vpEvFrames.back()->mnId, vpEvFrames.size()));

        g2o::SparseOptimizer optimizer;
        g2o::BlockSolverX::LinearSolverType * linearSolver;

        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

        g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        solver->setUserLambdaInit(1e-5);
        optimizer.setAlgorithm(solver);

        const float deltaMono = sqrt(5.991);
        const float thHuber2D = sqrt(5.99f);
        const float thHuber3D = sqrt(7.815f);

        int vcnt = 0;

        // Add Vertices

        // #1 Pose Vertices & Velocity
        int vfactorPose = vcnt;
        addPoseVertices(optimizer, vpEvFrames, vcnt, addInertial);

        // #2 IMU Related Vertices
        int vfactorBias = -1, vfactorGS = -1;
        if (addInertial) {

            vfactorBias = vcnt;
            addBiasIMU(optimizer, vpEvFrames, pInfoImu, vcnt, singleBias);

            if (addGDirScale) {
                vfactorGS = vcnt;
                addScaleGDirIMU(optimizer, pInfoImu, maxFrId, vcnt);
            }
        }

        // #5 Other Edges (Prior & ...)
        int vfactorPrior = -1;
        if (addPrior) {

            if (addInertial) {
                addPriorImuEdges(optimizer, vpEvFrames, vfactorPose, vfactorBias, thHuber3D, bRobust, singleBias);
            }
            else {
                vfactorPrior = vcnt;
                addPriorEdges(optimizer, vpEvFrames, vfactorPose, vcnt, thHuber3D, bRobust);
            }
        }

        int maxId = vcnt*maxFrId;

        // #3 Map Point Vertices and Edges
        if (vPts3d.empty()) {
            addMedianDepthEdges(optimizer, vpEvFrames, medDepth, vfactorPose, maxId, thHuber2D, bRobust);
        }
        else {
            addMapPointEdges(optimizer, vpEvFrames, vPts3d, vbValidPts3d, vfactorPose, maxId, thHuber2D, bRobust, addInertial);
        }

        // #4 Edge IMU
        if (addInertial) {
            addImuEdges(optimizer, vpEvFrames, pInfoImu, vfactorPose, vfactorBias, vfactorGS, thHuber2D, bRobust, singleBias);
        }

        // Optimize
        optimizer.setVerbose(false);
        optimizer.initializeOptimization();
        optimizer.optimize(nIterations);
        Verbose::PrintMess("BA: End of the optimization", Verbose::VERBOSITY_NORMAL);

        // Recover optimized pose
        if (addInertial) {
            recoverPoseIMU(optimizer, vpEvFrames, pInfoImu, vfactorPose, vfactorBias, vfactorGS, singleBias);
        }
        else {
            recoverPose(optimizer, vpEvFrames, vfactorPose);
        }

        // Also update map point matches
        if (vPts3d.empty()) {
            // Recover median depth
            VertexScale* VMZ = static_cast<VertexScale*>(optimizer.vertex(maxId+1));
            medDepth = VMZ->estimate();
        }
        else {
            recoverMapPoints(optimizer, vPts3d, vbValidPts3d, maxId);
        }

        return optimizer.chi2();
    }

    double MyOptimizer::optimizeSim3L1(std::vector<EvFramePtr> &vpKfs, double &sc0, const float th2,
                                     const int nIter, const bool bFixScale) {

        const int numFrames = vpKfs.size();
        if (numFrames < 2) {
            return MAXFLOAT;
        }

        g2o::SparseOptimizer optimizer;
        g2o::BlockSolverX::LinearSolverType * linearSolver;

        linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

        g2o::BlockSolverX* solver_ptr = new g2o::BlockSolverX(linearSolver);

        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        const float deltaHuber = sqrt(th2);

        // Add Pose Vertices
        // Set Sim3 vertex
        for (size_t i = 0; i < numFrames; i++) {

            bool bFixed = i == 0;

            EORB_SLAM::EvFramePtr currFrame = vpKfs[i];
            g2o::SE3Quat g2oTcw = Converter::toSE3Quat(currFrame->mTcw);

            ORB_SLAM3::VertexSim3Expmap *vSim3 = new ORB_SLAM3::VertexSim3Expmap();
            vSim3->setEstimate(g2o::Sim3(g2oTcw.rotation(), g2oTcw.translation(), sc0));
            vSim3->setId(i);
            vSim3->setFixed(bFixed);
            vSim3->pCamera1 = currFrame->mpCamera;
            //vSim3->pCamera2 = currFrame.mpCamera;
            vSim3->_fix_scale = bFixScale;

            optimizer.addVertex(vSim3);
        }

        // Set Map Point Vertices (Point directions from first camera) & Edges
        EORB_SLAM::EvFramePtr refFrame = vpKfs[0];
        //vector<cv::KeyPoint> vRefPts2d = refFrame.getAllUndistKPtsMono();
        //assert(refFrame.numAllKPts() == vMatchesCnt.size());

        int mpVxId = numFrames;
        int nEdges = 0;
        for (size_t i = 0; i < refFrame->numAllKPts(); i++, mpVxId++) {

            // Set Map Point Vertices
            cv::KeyPoint kpt = refFrame->getUndistKPtMono(i);

            cv::Mat unProjMat = refFrame->mpCamera->unprojectMat(kpt.pt);

            g2o::VertexSBAPointXYZ* vMapPt = new g2o::VertexSBAPointXYZ();
            vMapPt->setEstimate(Converter::toVector3d(unProjMat));
            vMapPt->setId(mpVxId);
            vMapPt->setFixed(true);
            vMapPt->setMarginalized(true);

            optimizer.addVertex(vMapPt);

            nEdges = 0;

            // Set Edges
            for (size_t j = 0; j < numFrames; j++) {

                EvFramePtr pFri = vpKfs[j];

                int matches12 = i;

                if (j > 0) {
                    matches12 = pFri->getMatches(i);
                    if (matches12 < 0)
                        continue;
                }

                nEdges++;

                Eigen::Matrix<double, 2, 1> obs1;
                const cv::KeyPoint &kpUn1 = pFri->getUndistKPtMono(matches12);
                obs1 << kpUn1.pt.x, kpUn1.pt.y;

                const float invSigmaSquare1 = pFri->getKPtInvLevelSigma2(matches12);

                ORB_SLAM3::EdgeSim3ProjectXYZ *e12 = new ORB_SLAM3::EdgeSim3ProjectXYZ();

                e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(mpVxId)));
                e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(j)));
                e12->setMeasurement(obs1);
                e12->setInformation(Eigen::Matrix2d::Identity() * invSigmaSquare1);

                g2o::RobustKernelHuber *rk1 = new g2o::RobustKernelHuber;
                e12->setRobustKernel(rk1);
                rk1->setDelta(deltaHuber);

                optimizer.addEdge(e12);
            }
            if (nEdges == 0) {
                optimizer.removeVertex(vMapPt);
            }
        }

        // Optimize!
        optimizer.initializeOptimization();
        optimizer.optimize(nIter);

        double scales = 0.0;

        // Recover Poses and average Scale
        for (size_t i = 0; i < numFrames; i++) {

            EvFramePtr pFri = vpKfs[i];

            g2o::VertexSim3Expmap* vSim3_recov = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(i));
            g2o::Sim3 sim3Tcw = vSim3_recov->estimate();

            pFri->SetPose(Converter::toCvSE3(sim3Tcw.rotation().toRotationMatrix(), sim3Tcw.translation()));
            scales += sim3Tcw.scale();
        }

        sc0 = scales / static_cast<double>(numFrames);

        return optimizer.chi2();
    }

    // 2D stuff
    void MyOptimizer::optimize2D(const vector<EORB_SLAM::EvFrame>& vEvFrames, cv::Mat& paramsTrans2d,
                               const int maxIterations, const bool verbose) {

        int nFrames = vEvFrames.size();
        int nParams = paramsTrans2d.rows;
        assert(nFrames >= 2 && nParams >= 3);

        // some handy typedefs
        typedef g2o::BlockSolver< g2o::BlockSolverTraits<Eigen::Dynamic, Eigen::Dynamic> >  MyBlockSolver;
        typedef g2o::LinearSolverDense<MyBlockSolver::PoseMatrixType> MyLinearSolver;

        // setup the solver
        g2o::SparseOptimizer optimizer;
        optimizer.setVerbose(false);

        MyLinearSolver* pLinearSolver = new MyLinearSolver();
        MyBlockSolver* pbSolver = new MyBlockSolver(pLinearSolver);
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(pbSolver);

        optimizer.setAlgorithm(solver);

        // build the optimization problem given the points
        //vector<cv::KeyPoint> dataKPts = vEvFrames[0].getAllUndistKPtsMono();
        EORB_SLAM::EvFrame refFrame = vEvFrames[0];
        int nKPts = refFrame.numAllKPts();
        vector<VertexSE2*> vSE2Trans(nFrames-1);
        vector<VertexSim2*> vSim2Trans(nFrames-1);

        cv::Mat K = vEvFrames[0].mK.clone();

        for (int i = 1, vId = 0; i < vEvFrames.size(); ++i, ++vId) {

            EORB_SLAM::EvFrame currFrame = vEvFrames[i];
            vector<int> vMatches12 = currFrame.getMatches();
            //vector<cv::KeyPoint> currKPts = currFrame.getAllUndistKPtsMono();

            // 1. add the SE2 vertex for each frame
            if (nParams == 3) {

                VertexSE2 *vSE2 = new VertexSE2();
                vSE2->setId(vId);
                vSE2->setEstimate(Converter::toVector3d(paramsTrans2d));
                optimizer.addVertex(vSE2);
                vSE2Trans[vId] = vSE2;
            }
            else if (nParams == 4){

                VertexSim2 *vSim2 = new VertexSim2();
                vSim2->setId(vId);
                Eigen::Vector4d paramsSim2;
                paramsSim2 << paramsTrans2d.at<float>(0, 0), paramsTrans2d.at<float>(1, 0),
                        paramsTrans2d.at<float>(2, 0), paramsTrans2d.at<float>(3, 0);
                vSim2->setEstimate(paramsSim2);
                optimizer.addVertex(vSim2);
                vSim2Trans[vId] = vSim2;
            }

            for (int j = 0; j < nKPts; j++) {

                if (vMatches12[j] < 0)
                    continue;

                cv::KeyPoint currData = refFrame.getUndistKPtMono(j);
                float xd = (currData.pt.x - K.at<float>(0,2))/K.at<float>(0,0);
                float yd = (currData.pt.y - K.at<float>(1,2))/K.at<float>(1,1);

                cv::KeyPoint currObs = currFrame.getUndistKPtMono(j);
                float xob = (currObs.pt.x - K.at<float>(0,2))/K.at<float>(0,0);
                float yob = (currObs.pt.y - K.at<float>(1,2))/K.at<float>(1,1);
                Eigen::Vector2d obs;
                obs << xob, yob;

                // 2. add edges between SE2 transformation, data and observations
                if (nParams == 3) {

                    EdgeSE2PointXY *e = new EdgeSE2PointXY(xd, yd);
                    e->setInformation(Eigen::Matrix2d::Identity() * currFrame.getMedianPxDisp());
                    e->setVertex(0, vSE2Trans[vId]);
                    e->setMeasurement(obs);
                    optimizer.addEdge(e);
                }
                else if (nParams == 4) {

                    EdgeSim2PointXY *e = new EdgeSim2PointXY(xd, yd);
                    e->setInformation(Eigen::Matrix2d::Identity() * currFrame.getMedianPxDisp());
                    e->setVertex(0, vSim2Trans[vId]);
                    e->setMeasurement(obs);
                    optimizer.addEdge(e);
                }
            }
        }

        // perform the optimization
        optimizer.initializeOptimization();
        optimizer.setVerbose(verbose);
        optimizer.optimize(maxIterations);

        if (nParams == 3) {
            paramsTrans2d.at<float>(0, 0) = vSE2Trans.back()->estimate()(0);
            paramsTrans2d.at<float>(1, 0) = vSE2Trans.back()->estimate()(1);
            paramsTrans2d.at<float>(2, 0) = vSE2Trans.back()->estimate()(2);
        }
        else if (nParams == 4) {
            paramsTrans2d.at<float>(0, 0) = vSim2Trans.back()->estimate()(0);
            paramsTrans2d.at<float>(1, 0) = vSim2Trans.back()->estimate()(1);
            paramsTrans2d.at<float>(2, 0) = vSim2Trans.back()->estimate()(2);
            paramsTrans2d.at<float>(3, 0) = vSim2Trans.back()->estimate()(3);
        }
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    int MyOptimizer::PoseOptimization(Frame *pFrame, EvFrame& currEvFrame)
    {
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

        linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

        g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        int nInitialCorrespondences=0;

        // Set Frame vertex
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        vSE3->setId(0);
        vSE3->setFixed(false);
        optimizer.addVertex(vSE3);

        // Set MapPoint vertices
        const int N = pFrame->numAllKPts();

        vector<ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
        vector<ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody *> vpEdgesMono_FHR;
        vector<size_t> vnIndexEdgeMono, vnIndexEdgeRight;
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

        map<cv::Mat, float, X3dCmp> mOrbInvLevelSigma2;
        float maxDepth = -1, minDepth = 1e6, maxDepthILS2 = -1, minDepthILS2 = 1e6;
        float maxInvLevelSigma2 = -1, minInvLevelSigma2 = 1e6;

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

                            if (Xw.at<float>(2) > maxDepth) {
                                maxDepth = Xw.at<float>(2);
                                maxDepthILS2 = invSigma2;
                            }
                            if (Xw.at<float>(2) < minDepth) {
                                minDepth = Xw.at<float>(2);
                                minDepthILS2 = invSigma2;
                            }
                            if (maxInvLevelSigma2 < invSigma2) {
                                maxInvLevelSigma2 = invSigma2;
                            }
                            if (minInvLevelSigma2 > invSigma2) {
                                minInvLevelSigma2 = invSigma2;
                            }
                            mOrbInvLevelSigma2.insert(make_pair(Xw, invSigma2));
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
        }

        // Add Event Edges
        int nEvPts = currEvFrame.numAllKPts();

        int nEvInitialCorrespondences = 0;
        vector<ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose*> vpEvEdgesMono;
        vector<size_t> vnIndexEvEdgeMono;
        vpEvEdgesMono.reserve(nEvPts);
        vnIndexEvEdgeMono.reserve(nEvPts);

        {
            unique_lock<mutex> lock(MapPoint::mGlobalMutex);

            for(int i=0; i<nEvPts; i++)
            {
                MapPoint* pMP = currEvFrame.getMapPoint(i);
                if(pMP)
                {
                    //Conventional SLAM
                    if(!currEvFrame.mpCamera2){
                        // Monocular observation
                        if(currEvFrame.mvuRight[i]<0)
                        {
                            nEvInitialCorrespondences++;
                            currEvFrame.setMPOutlier(i, false);

                            Eigen::Matrix<double,2,1> obs;
                            const cv::KeyPoint &kpUn = currEvFrame.getUndistKPtMono(i);
                            obs << kpUn.pt.x, kpUn.pt.y;

                            ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose* e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose();

                            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                            e->setMeasurement(obs);
                            const float invSigma2 = currEvFrame.getKPtInvLevelSigma2(i);//getORBInvLevelSigma2(kpUn.octave);
                            e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                            e->setRobustKernel(rk);
                            rk->setDelta(deltaMono);

                            e->pCamera = currEvFrame.mpCamera;
                            cv::Mat Xw = pMP->GetWorldPos();
                            e->Xw[0] = Xw.at<float>(0);
                            e->Xw[1] = Xw.at<float>(1);
                            e->Xw[2] = Xw.at<float>(2);

                            optimizer.addEdge(e);

                            vpEvEdgesMono.push_back(e);
                            vnIndexEvEdgeMono.push_back(i);
                        }
                    }
                }
            }
        }

#ifdef SAVE_EV_IM_SYNCH_MPS
        // Some debuging output:
        DLOG(INFO) << "EvImPoseOptim: maxDepth: " << maxDepth << ", minDepth: " << minDepth
                   << ", maxILS2: " << maxInvLevelSigma2 << ", minILS2: " << minInvLevelSigma2
                   << ", maxDILS2: " << maxDepthILS2 << ", minDILS2: " << minDepthILS2 << endl;
        Visualization::saveMapPoints(pFrame->getAllMapPointsMono(),
                "../data/ev_im_pose_optim_orb_mps_"+to_string(pFrame->mTimeStamp)+".txt");
        Visualization::saveMapPoints(currEvFrame.getAllMapPointsMono(),
                "../data/ev_im_pose_optim_ev_mps_"+to_string(pFrame->mTimeStamp)+".txt");
#endif

        //cout << "PO: vnIndexEdgeMono.size() = " << vnIndexEdgeMono.size() << "   vnIndexEdgeRight.size() = " << vnIndexEdgeRight.size() << endl;
        if(nInitialCorrespondences + nEvInitialCorrespondences < 3)
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

            for(size_t i=0, iend=vpEvEdgesMono.size(); i<iend; i++)
            {
                ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose* e = vpEvEdgesMono[i];

                const size_t idx = vnIndexEvEdgeMono[i];

                if(currEvFrame.getMPOutlier(idx))
                {
                    e->computeError();
                }

                const float chi2 = e->chi2();

                if(chi2>chi2Mono[it])
                {
                    currEvFrame.setMPOutlier(idx, true);
                    e->setLevel(1);
                    nBad++;
                }
                else
                {
                    currEvFrame.setMPOutlier(idx, false);
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

            if(optimizer.edges().size()<10)
                break;
        }

        // Recover optimized pose and return number of inliers
        g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
        g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
        cv::Mat pose = Converter::toCvMat(SE3quat_recov);
        pFrame->SetPose(pose);

        //cout << "[PoseOptimization]: initial correspondences-> " << nInitialCorrespondences << " --- outliers-> " << nBad << endl;
        currEvFrame.SetPose(pose);

        return nInitialCorrespondences+nEvInitialCorrespondences-nBad;
    }

    void MyOptimizer::StructureBA(std::vector<ORB_SLAM3::KeyFrame*>& vpKFs, int nIterations, bool structOnly,
                                  bool *pbStopFlag, unsigned long nLoopKF, bool bRobust) {

        if (vpKFs.size() < 2) {
            DLOG(WARNING) << "MyOptimizer::SBA: Not enough key frames, abort optimization...\n";
            return;
        }

        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

        g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);
        optimizer.setVerbose(false);

        if(pbStopFlag)
            optimizer.setForceStopFlag(pbStopFlag);

        int currId = 0;

        map<KeyFrame*, int> mpConstKF;

        // Set KeyFrame vertices

        unsigned nVertexKF = 0;
        for(auto pKF : vpKFs) {

            if(!pKF || pKF->isBad())
                continue;

            g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
            vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));
            vSE3->setId(currId);

            vSE3->setFixed(!(currId > 0 && !structOnly));
            optimizer.addVertex(vSE3);

            mpConstKF.insert(make_pair(pKF,currId));

            currId++;
            nVertexKF++;
        }

        if (nVertexKF <= 0) {
            LOG(WARNING) << "Optimizer::SBA: No valid KF vertex can be added, abort...\n";
            return;
        }

        const float thHuber2D = sqrt(5.99f);
        const float thHuber3D = sqrt(7.815f);

        // Set MapPoint vertices
        //cout << "start inserting MPs" << endl;
        map<MapPoint*, int> mConstMP;

        unsigned nVertexMP = 0;
        // The first KF is always considered the ref. KF
        for(MapPoint* pMP : vpKFs[0]->GetMapPointMatches())
        {
            if(!pMP || pMP->isBad())
                continue;

            g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
            vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
            const int id = currId;
            vPoint->setId(id);
            vPoint->setMarginalized(true);
            optimizer.addVertex(vPoint);

            const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();

            int nEdges = 0;
            //SET EDGES
            for(map<KeyFrame*,tuple<int,int>>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)
            {
                KeyFrame* pKF = mit->first;
                if(pKF->isBad())
                    continue;
                if(optimizer.vertex(id) == nullptr || !mpConstKF.count(pKF))
                    continue;

                auto constKF = mpConstKF.find(pKF);
                const int kfId = constKF->second;
                if (optimizer.vertex(kfId) == nullptr)
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
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(kfId)));
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
                }
            }

            if(nEdges==0)
            {
                optimizer.removeVertex(vPoint);
                //vbNotIncludedMP[i]=true;
            }
            else
            {
                nVertexMP++;
                currId++;
                mConstMP.insert(make_pair(pMP, id));
            }
        }

        if (nVertexMP <= 0) {
            LOG(WARNING) << "Optimizer::SBA: No valid MP vertex can be added, abort...\n";
            return;
        }

        if (structOnly) {
            DLOG(WARNING) << "Optimizer::SBA: Structure only BA currently not supported\n";
            return;
//            DLOG(INFO) << "Optimizer::SBA: Performing structure-only BA:" << endl;
//            g2o::StructureOnlySolver<3> structure_only_ba;
//            g2o::OptimizableGraph::VertexContainer points;
//            for (g2o::OptimizableGraph::VertexIDMap::const_iterator it = optimizer.vertices().begin(); it != optimizer.vertices().end(); ++it) {
//                g2o::OptimizableGraph::Vertex* v = static_cast<g2o::OptimizableGraph::Vertex*>(it->second);
//                if (v->dimension() == 3)
//                    points.push_back(v);
//            }
//
//            structure_only_ba.calc(points, 10);
        }

        //cout << "end inserting MPs" << endl;
        // Optimize!
        optimizer.setVerbose(false);
        optimizer.initializeOptimization();
        optimizer.optimize(nIterations);
        Verbose::PrintMess("Optimizer::SBA: End of the optimization", Verbose::VERBOSITY_NORMAL);

        // Recover optimized data

        //Points
        for(auto& constMP : mConstMP)
        {
            MapPoint* pMP = constMP.first;
            const int mpId = constMP.second;

            if(pMP->isBad())
                continue;

            g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(mpId));

            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
        }
    }

    void MyOptimizer::GlobalBundleAdjustment(Map* pMap, const std::vector<ORB_SLAM3::KeyFrame*> &vpEvRefKF,
                                             int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
    {
        vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
        vector<MapPoint*> vpMP = pMap->GetAllMapPoints();
        BundleAdjustment(vpKFs, vpMP, vpEvRefKF, nIterations, pbStopFlag, nLoopKF, bRobust);
    }

    void MyOptimizer::BundleAdjustment(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP,
                                       const std::vector<ORB_SLAM3::KeyFrame*> &vpEvKFs, int nIterations,
                                       bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
    {
        if (vpKFs.empty() || vpMP.empty()) {
            LOG(WARNING) << "MyOptimizer::BundleAdjustment: Called with no constraints: nKFs: "
                         << vpKFs.size() << ", nMPs: " << vpMP.size() << endl;
            return;
        }
        if (vpKFs.size() > 2) {
            LOG(WARNING) << "MyOptimizer::BundleAdjustment: Only supports initialization optim., "
                         << vpKFs.size() << " > 2 key frames given\n";
            Optimizer::BundleAdjustment(vpKFs, vpMP, nIterations, pbStopFlag, nLoopKF, bRobust);
            return;
        }
        // Maybe even check if ORB & Ev KFs timestamps comply
        assert(!vpEvKFs.empty());

        vector<bool> vbNotIncludedMP;
        vbNotIncludedMP.resize(vpMP.size());

        Map* pMap = vpKFs[0]->GetMap();

        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

        g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);
        optimizer.setVerbose(false);

        if(pbStopFlag)
            optimizer.setForceStopFlag(pbStopFlag);

        long unsigned int maxKFid = 0;
        long unsigned int refKfId, currKfId;

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
            const bool isFixed = pKF->mnId==pMap->GetInitKFid();
            vSE3->setFixed(isFixed);
            optimizer.addVertex(vSE3);
            if(pKF->mnId>maxKFid)
                maxKFid=pKF->mnId;
            //cout << "KF id: " << pKF->mnId << endl;
            nVertexKF++;

            if (isFixed) {
                refKfId = pKF->mnId;
            }
            else {
                currKfId = pKF->mnId;
            }
        }

        if (nVertexKF <= 0) {
            LOG(WARNING) << "MyOptimizer::BundleAdjustment: No valid KF vertex can be added, abort...\n";
            return;
        }

        const float thHuber2D = sqrt(5.99f);
        const float thHuber3D = sqrt(7.815f);

        // Set MapPoint vertices
        //cout << "start inserting MPs" << endl;

        long unsigned int globalId = maxKFid;

        map<KeyFrame*, map<cv::Mat, float, X3dCmp>> mmOrbInvLevelSigma2;
        float maxDepth = -1, minDepth = 1e6, maxDepthILS2 = -1, minDepthILS2 = 1e6;
        float maxInvLevelSigma2 = -1, minInvLevelSigma2 = 1e6;

        unsigned nVertexMP = 0;
        for(size_t i=0; i<vpMP.size(); i++)
        {
            MapPoint* pMP = vpMP[i];
            if(pMP->isBad())
                continue;
            g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
            cv::Mat Xw = pMP->GetWorldPos();
            vPoint->setEstimate(Converter::toVector3d(Xw));
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

                    if (Xw.at<float>(2) > maxDepth) {
                        maxDepth = Xw.at<float>(2);
                        maxDepthILS2 = invSigma2;
                    }
                    if (Xw.at<float>(2) < minDepth) {
                        minDepth = Xw.at<float>(2);
                        minDepthILS2 = invSigma2;
                    }
                    if (maxInvLevelSigma2 < invSigma2) {
                        maxInvLevelSigma2 = invSigma2;
                    }
                    if (minInvLevelSigma2 > invSigma2) {
                        minInvLevelSigma2 = invSigma2;
                    }

                    if (!mmOrbInvLevelSigma2.count(pKF)) {
                        mmOrbInvLevelSigma2.insert(make_pair(pKF, map<cv::Mat, float, X3dCmp>()));
                        mmOrbInvLevelSigma2[pKF].insert(make_pair(Xw, invSigma2));
                    }
                    else {
                        mmOrbInvLevelSigma2[pKF].insert(make_pair(Xw, invSigma2));
                    }
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

                if (id > globalId) {
                    globalId = id;
                }
            }
        }

        // Add Event Synch. Constraints
        unsigned long currEvId = globalId+1;
        unsigned nEvVertexMP = 0;
        // We assume first member of vdfj is ref. event KF
        KeyFrame* pRefEvKF = vpEvKFs[0];
        KeyFrame* pCurEvKF = vpEvKFs[1];
        vector<MapPoint*> vpEvMPs = pCurEvKF->GetMapPointMatches();
        int nEvMPs = vpEvMPs.size();

//        const int nExpectedSizeEv = vpEvKFs.size()*nEvMPs;
//
//        vector<bool> vbNotIncludedEvMP;
//        vbNotIncludedEvMP.resize(nEvMPs);
//        vector<ORB_SLAM3::EdgeSE3ProjectXYZ*> vpEvEdgesMono;
//        vpEvEdgesMono.reserve(nExpectedSizeEv);
//        vector<KeyFrame*> vpEvEdgeKFMono;
//        vpEvEdgeKFMono.reserve(nExpectedSizeEv);
//        vector<MapPoint*> vpEvMapPointEdgeMono;
//        vpEvMapPointEdgeMono.reserve(nExpectedSizeEv);

        map<unsigned long, MapPoint*> mEvVertexMPs;

        for(size_t i=0; i<vpEvMPs.size(); i++)
        {
            MapPoint* pMP = vpEvMPs[i];
            if(!pMP || pMP->isBad())
                continue;
            g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
            vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
            //const int id = pMP->mnId+maxKFid+1;
            vPoint->setId(static_cast<int>(currEvId));
            vPoint->setMarginalized(true);
            optimizer.addVertex(vPoint);

            const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();

            int nEdges = 0;
            //SET EDGES
            for(map<KeyFrame*,tuple<int,int>>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)
            {
                KeyFrame* pKF = mit->first;
                if(pKF->isBad() || (pKF != pRefEvKF && pKF != pCurEvKF))
                    continue;

                int currKfIdORB = static_cast<int>(refKfId);
                if (pKF == pCurEvKF) {
                    currKfIdORB = static_cast<int>(currKfId);
                }
                if(optimizer.vertex(static_cast<int>(currEvId)) == nullptr || optimizer.vertex(currKfIdORB) == nullptr)
                    continue;
                nEdges++;

                const int leftIndex = get<0>(mit->second);

                if(leftIndex != -1 && pKF->mvuRight[get<0>(mit->second)]<0)
                {
                    const cv::KeyPoint &kpUn = pKF->getUndistKPtMono(leftIndex);

                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    ORB_SLAM3::EdgeSE3ProjectXYZ* e = new ORB_SLAM3::EdgeSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(static_cast<int>(currEvId))));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(currKfIdORB)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKF->getKPtInvLevelSigma2(leftIndex);
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    if(bRobust)
                    {
                        auto* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuber2D);
                    }

                    e->pCamera = pKF->mpCamera;

                    optimizer.addEdge(e);

                    //vpEvEdgesMono.push_back(e);
                    //vpEvEdgeKFMono.push_back(pKF);
                    //vpEvMapPointEdgeMono.push_back(pMP);
                }
            }

            if(nEdges==0)
            {
                optimizer.removeVertex(vPoint);
                //vbNotIncludedEvMP[i]=true;
            }
            else
            {
                //vbNotIncludedEvMP[i]=false;
                nEvVertexMP++;
                mEvVertexMPs.insert(make_pair(currEvId, pMP));
                currEvId++;
            }
        }

        if (nVertexMP + nEvVertexMP <= 0) {
            LOG(WARNING) << "MyOptimizer::BundleAdjustment: No valid MP vertex can be added, abort...\n";
            return;
        }

#ifdef SAVE_EV_IM_SYNCH_MPS
        // Some debuging output:
        DLOG(INFO) << "EvImPoseOptim: maxDepth: " << maxDepth << ", minDepth: " << minDepth
                   << ", maxILS2: " << maxInvLevelSigma2 << ", minILS2: " << minInvLevelSigma2
                   << ", maxDILS2: " << maxDepthILS2 << ", minDILS2: " << minDepthILS2 << endl;
        Visualization::saveMapPoints(vpMP, "../data/ev_im_ba_optim_orb_mps_"+to_string(pRefEvKF->mTimeStamp)+".txt");
        Visualization::saveMapPoints(vpEvMPs, "../data/ev_im_ba_optim_ev_mps_"+to_string(pRefEvKF->mTimeStamp)+".txt");
#endif

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
                pKF->SetPose(Converter::toCvMat(SE3quat));
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
                    Verbose::PrintMess("GBA: KF " + to_string(pKF->mnId) + " had been moved " + to_string(dist) +
                        " meters", Verbose::VERBOSITY_DEBUG);
                    Verbose::PrintMess("--Number of observations: " + to_string(numMonoOptPoints) +
                        " in mono and " + to_string(numStereoOptPoints) + " in stereo", Verbose::VERBOSITY_DEBUG);
                    Verbose::PrintMess("--Number of discarded observations: " + to_string(numMonoBadPoints) +
                        " in mono and " + to_string(numStereoBadPoints) + " in stereo", Verbose::VERBOSITY_DEBUG);
                }
            }

            if (pKF->mnId == refKfId) {
                vpEvKFs[0]->SetPose(pKF->GetPose());
            }
            else if (pKF->mnId == currKfId) {
                vpEvKFs[1]->SetPose(pKF->GetPose());
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

        // Update event map point constraints
        for (auto& constEvMP : mEvVertexMPs) {

            MapPoint* pMP = constEvMP.second;

            if(pMP->isBad())
                continue;
            g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(constEvMP.first));

            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
        }
    }

    void MyOptimizer::getLocalOptKFs(ORB_SLAM3::KeyFrame* pKF, std::vector<ORB_SLAM3::KeyFrame*> &vpOptimizableKFs,
            const int Nd, const bool bMergeMaps) {

        Map* pCurrMap = pKF->GetMap();

        vpOptimizableKFs.push_back(pKF);
        pKF->mnBALocalForKF = pKF->mnId;

        for(int i=1; i<Nd; i++)
        {
            KeyFrame* pPreKF = vpOptimizableKFs.back()->mPrevKF;
            if(pPreKF && (pCurrMap == pPreKF->GetMap() || bMergeMaps))
            {
                vpOptimizableKFs.push_back(vpOptimizableKFs.back()->mPrevKF);
                vpOptimizableKFs.back()->mnBALocalForKF = pKF->mnId;
            }
            else
                break;
        }
    }

    void MyOptimizer::getCovisOptKFs(ORB_SLAM3::KeyFrame* pKF, std::list<ORB_SLAM3::KeyFrame*> &lpOptVisKFs,
            const int maxCovKF, const bool bMergeMaps) {

        const vector<KeyFrame*> vpNeighsKFs = pKF->GetVectorCovisibleKeyFrames();
        Map* pCurrentMap = pKF->GetMap();

        for(auto pKFi : vpNeighsKFs)
        {
            if(lpOptVisKFs.size() >= maxCovKF)
                break;

            if(pKFi->mnBALocalForKF == pKF->mnId || pKFi->mnBAFixedForKF == pKF->mnId)
                continue;
            pKFi->mnBALocalForKF = pKF->mnId;
            if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
            {
                lpOptVisKFs.push_back(pKFi);
            }
        }
    }

    void MyOptimizer::getFixedKFs(ORB_SLAM3::KeyFrame* pKF, std::vector<ORB_SLAM3::KeyFrame*> &vpOptimizableKFs,
                                  const std::list<ORB_SLAM3::MapPoint*>& lLocalMapPoints,
                                  std::list<ORB_SLAM3::KeyFrame*> &lFixedKeyFrames, int maxFixKF, const bool bMergeMaps) {

        KeyFrame* pPreKF = vpOptimizableKFs.back()->mPrevKF;
        Map* pCurrMap = pKF->GetMap();

        // Fixed Keyframe: First frame previous KF to optimization window)
        if(pPreKF && (pPreKF->GetMap() == pCurrMap || bMergeMaps))
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

        // Fixed KFs which are not covisible optimizable
        for(auto & lLocalMapPoint : lLocalMapPoints)
        {
            map<KeyFrame*,tuple<int,int>> observations = lLocalMapPoint->GetObservations();
            for(auto & observation : observations)
            {
                KeyFrame* pKFi = observation.first;

                // TODO: Check this cond.
                bool bMapCond = pCurrMap == pKFi->GetMap() || bMergeMaps;

                if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId && pKFi->mnId < pKF->mnId && bMapCond)
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
    }

    void MyOptimizer::getLocalOptMPts(ORB_SLAM3::KeyFrame* pKF, const std::vector<ORB_SLAM3::KeyFrame*> &vpOptimizableKFs,
                                      const std::list<ORB_SLAM3::KeyFrame*> &lpOptVisKFs,
                                      std::list<ORB_SLAM3::MapPoint*>& lLocalMapPoints, const bool bMergeMaps) {

        Map* pCurrMap = pKF->GetMap();

        for(auto vpOptimizableKF : vpOptimizableKFs)
        {
            vector<MapPoint*> vpMPs = vpOptimizableKF->GetMapPointMatches();
            for(auto pMP : vpMPs)
            {
                if(pMP)
                    if(!pMP->isBad())
                        if(pMP->mnBALocalForKF!=pKF->mnId && (pCurrMap == pMP->GetMap() || bMergeMaps))
                        {
                            lLocalMapPoints.push_back(pMP);
                            pMP->mnBALocalForKF=pKF->mnId;
                        }
            }
        }

        for(auto vpOptimizableKF : lpOptVisKFs)
        {
            vector<MapPoint*> vpMPs = vpOptimizableKF->GetMapPointMatches();
            for(auto pMP : vpMPs)
            {
                if(pMP)
                    if(!pMP->isBad())
                        if(pMP->mnBALocalForKF!=pKF->mnId && (pCurrMap == pMP->GetMap() || bMergeMaps))
                        {
                            lLocalMapPoints.push_back(pMP);
                            pMP->mnBALocalForKF=pKF->mnId;
                        }
            }
        }
    }

    // Filter key frames by id so all sets contain unique ids
    bool MyOptimizer::checkKeyFrameIntegrity(std::vector<ORB_SLAM3::KeyFrame *> &vpOptimizableKFs,
                                             std::list<ORB_SLAM3::KeyFrame *> &lpOptVisKFs,
                                             std::list<ORB_SLAM3::KeyFrame *> &lFixedKeyFrames, ulong& maxId) {

        size_t optSize = vpOptimizableKFs.size(), optVisSize = lpOptVisKFs.size(), fixedSize = lFixedKeyFrames.size();
        map<ulong, KeyFrame*> mAllKFs, mLocalOpt, mVisOpt, mFixed;

        for (auto pKF : vpOptimizableKFs) {

            ulong kfId = pKF->mnId;

            if (mAllKFs.find(kfId) == mAllKFs.end()) {
                mAllKFs.insert(make_pair(kfId, pKF));

                if (mLocalOpt.find(kfId) == mLocalOpt.end()) {
                    mLocalOpt.insert(make_pair(kfId, pKF));
                }
            }
        }

        for (auto pKF : lpOptVisKFs) {

            ulong kfId = pKF->mnId;

            if (mAllKFs.find(kfId) == mAllKFs.end()) {
                mAllKFs.insert(make_pair(kfId, pKF));

                if (mVisOpt.find(kfId) == mVisOpt.end()) {
                    mVisOpt.insert(make_pair(kfId, pKF));
                }
            }
            else {
                LOG(WARNING) << "MyOptimizer::checkKeyFrameIntegrity: Duplicate KF detected in OptVisKFs: " << kfId << endl;
            }
        }

        for (auto pKF : lFixedKeyFrames) {

            ulong kfId = pKF->mnId;

            if (mAllKFs.find(kfId) == mAllKFs.end()) {
                mAllKFs.insert(make_pair(kfId, pKF));

                if (mFixed.find(kfId) == mFixed.end()) {
                    mFixed.insert(make_pair(kfId, pKF));
                }
            }
            else {
                LOG(WARNING) << "MyOptimizer::checkKeyFrameIntegrity: Duplicate KF detected in FixedKFs: " << kfId << endl;
            }
        }

        bool goodSets = true;
        if (mFixed.size() != fixedSize || mVisOpt.size() != optVisSize || mLocalOpt.size() != optSize ||
            mAllKFs.size() != (fixedSize+optSize+optVisSize)) {
            goodSets = false;
        }

        vpOptimizableKFs.clear();
        vpOptimizableKFs.reserve(mLocalOpt.size());
        for (auto ppKF : mLocalOpt) {
            vpOptimizableKFs.push_back(ppKF.second);
        }

        lpOptVisKFs.clear();
        for (auto ppKF : mVisOpt) {
            lpOptVisKFs.push_back(ppKF.second);
        }

        lFixedKeyFrames.clear();
        for (auto ppKF : mFixed) {
            lFixedKeyFrames.push_back(ppKF.second);
        }

        maxId = mAllKFs.rbegin()->first;
        return goodSets;
    }

    void MyOptimizer::LocalInertialBA(KeyFrame *pKF, bool *pbStopFlag, Map *pMap, int& num_fixedKF, int& num_OptKF,
                                      int& num_MPs, int& num_edges, int nMaxOpt, int nIter, bool bLarge, bool bRecInit,
                                      const bool bMergeMaps)
    {
        if (!pKF) {
            return;
        }

        int maxOpt=nMaxOpt;
        int opt_it=nIter;
        if(bLarge)
        {
            maxOpt=25;
            opt_it=4;
        }

        Map* pCurrentMap = pKF->GetMap();

        const int Nd = std::min((int)pCurrentMap->KeyFramesInMap()-2,maxOpt);
        unsigned long maxKFid = pKF->mnId;

        vector<KeyFrame*> vpOptimizableKFs;
        vpOptimizableKFs.reserve(Nd);
        list<KeyFrame*> lpOptVisKFs;
        list<KeyFrame*> lFixedKeyFrames;
        list<MapPoint*> lLocalMapPoints;

        // Optimizable Keyframes
        getLocalOptKFs(pKF, vpOptimizableKFs, Nd, bMergeMaps);
        //int N = vpOptimizableKFs.size();

        // Optimizable visual KFs
        getCovisOptKFs(pKF, lpOptVisKFs, 0, false);

        // Optimizable points seen by temporal optimizable keyframes
        getLocalOptMPts(pKF, vpOptimizableKFs, lpOptVisKFs, lLocalMapPoints, bMergeMaps);

        // Fixed KFs
        getFixedKFs(pKF, vpOptimizableKFs, lLocalMapPoints, lFixedKeyFrames, 200, bMergeMaps);

        checkKeyFrameIntegrity(vpOptimizableKFs, lpOptVisKFs, lFixedKeyFrames, maxKFid);

        bool bNonFixed = lFixedKeyFrames.empty();


        // Setup optimizer
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolverX::LinearSolverType * linearSolver;
        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

        auto * solver_ptr = new g2o::BlockSolverX(linearSolver);

        if(bLarge)
        {
            auto* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
            solver->setUserLambdaInit(1e-2); // to avoid iterating for finding optimal lambda
            optimizer.setAlgorithm(solver);
        }
        else
        {
            auto* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
            solver->setUserLambdaInit(1e0);
            optimizer.setAlgorithm(solver);
        }


        // Set Local temporal KeyFrame vertices
        int N=vpOptimizableKFs.size();
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
        for(auto pKFi : lpOptVisKFs)
        {
            VertexPose * VP = new VertexPose(pKFi);
            VP->setId(pKFi->mnId);
            VP->setFixed(false);
            optimizer.addVertex(VP);

            num_OptKF++;
        }

        // Set Fixed KeyFrame vertices
        for(auto pKFi : lFixedKeyFrames)
        {
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

        // NOTE: if size(lFixedKFs) == 1, means the only fixed connection is the link previous to local window
        // an inertial link must be established in this case to impede growing errors related to IMU biases
        // happens frequently in LK-based feature association
        float infoPreFixedLink = 1e-2;
        if (lFixedKeyFrames.size() == 1) {
            infoPreFixedLink = 1e-3;
        }

        // Create intertial constraints
        vector<EdgeInertial*> vei(N,(EdgeInertial*)nullptr);
        vector<EdgeGyroRW*> vegr(N,(EdgeGyroRW*)nullptr);
        vector<EdgeAccRW*> vear(N,(EdgeAccRW*)nullptr);

        for(int i=0;i<N;i++)
        {
            KeyFrame* pKFi = vpOptimizableKFs[i];

            if(!pKFi->mPrevKF)
            {
                cout << "NO INERTIAL LINK TO PREVIOUS FRAME!!!!" << endl;
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
                        vei[i]->setInformation(vei[i]->information()*infoPreFixedLink);
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


        const float thHuberMono = sqrtf(5.991);
        const float chi2Mono2 = 5.991;
        const float thHuberStereo = sqrtf(7.815);
        const float chi2Stereo2 = 7.815;

        const unsigned long iniMPid = maxKFid*5;

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
        }

        //cout << "Total map points: " << lLocalMapPoints.size() << endl;
        for(map<int,int>::iterator mit=mVisEdges.begin(), mend=mVisEdges.end(); mit!=mend; mit++)
        {
            assert(mit->second>=3);
        }

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
        }

        // Local visual KeyFrame
        for(list<KeyFrame*>::iterator it=lpOptVisKFs.begin(), itEnd = lpOptVisKFs.end(); it!=itEnd; it++)
        {
            KeyFrame* pKFi = *it;
            VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(pKFi->mnId));
            cv::Mat Tcw = Converter::toCvSE3(VP->estimate().Rcw[0], VP->estimate().tcw[0]);
            pKFi->SetPose(Tcw);
            pKFi->mnBALocalForKF=0;
        }

        //Points
        for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
        {
            MapPoint* pMP = *lit;
            g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+iniMPid+1));
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
        }

        pMap->IncreaseChangeIndex();

    }

    void MyOptimizer::LocalBundleAdjustment(KeyFrame *pKF, bool* pbStopFlag, Map* pMap, int& num_fixedKF, int& num_OptKF,
                                            int& num_MPs, int& num_edges)
    {
        // NOTE: pMap must be the current map of pKF
        if (!pKF || !pMap || pKF->GetMap() != pMap) {
            return;
        }

        // Local KeyFrames: First Breath Search from Current Keyframe
        list<KeyFrame*> lLocalKeyFrames;

        lLocalKeyFrames.push_back(pKF);
        pKF->mnBALocalForKF = pKF->mnId;

        const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
        for(int i=0, iend=vNeighKFs.size(); i<iend; i++)
        {
            KeyFrame* pKFi = vNeighKFs[i];
            if (!pKFi)
                continue;

            pKFi->mnBALocalForKF = pKF->mnId;
            if(!pKFi->isBad() && pKFi->GetMap() == pMap && pKFi->mnId < pKF->mnId)
                lLocalKeyFrames.push_back(pKFi);
        }

        // Local MapPoints seen in Local KeyFrames
        num_fixedKF = 0;
        list<MapPoint*> lLocalMapPoints;
        set<MapPoint*> sNumObsMP;
        for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)
        {
            KeyFrame* pKFi = *lit;
            if (!pKFi)
                continue;

            if(pKFi->mnId==pMap->GetInitKFid())
            {
                num_fixedKF = 1;
            }
            vector<MapPoint*> vpMPs = pKFi->GetMapPointMatches();
            for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
            {
                MapPoint* pMP = *vit;
                if(pMP)
                    if(!pMP->isBad() && pMP->GetMap() == pMap)
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
        for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
        {
            map<KeyFrame*,tuple<int,int>> observations = (*lit)->GetObservations();
            for(map<KeyFrame*,tuple<int,int>>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
            {
                KeyFrame* pKFi = mit->first;

                if(pKFi && pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId  && pKFi->mnId < pKF->mnId)
                {
                    pKFi->mnBAFixedForKF=pKF->mnId;
                    if(!pKFi->isBad() && pKFi->GetMap() == pMap)
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
        for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
        {
            KeyFrame* pKFi = *lit;
            if (!pKFi)
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
        for(list<KeyFrame*>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
        {
            KeyFrame* pKFi = *lit;
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

        int nKFs = lLocalKeyFrames.size()+lFixedCameras.size(), nEdges = 0;

        for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
        {
            MapPoint* pMP = *lit;
            if (!pMP)
                continue;
            g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
            vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
            int id = pMP->mnId+maxKFid+1;
            vPoint->setId(id);
            vPoint->setMarginalized(true);
            optimizer.addVertex(vPoint);
            nPoints++;

            const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();

            //Set edges
            for(map<KeyFrame*,tuple<int,int>>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
            {
                KeyFrame* pKFi = mit->first;

                if(pKFi && !pKFi->isBad() && pKFi->GetMap() == pMap)
                {
                    const int leftIndex = get<0>(mit->second);

                    // Monocular observation
                    if(leftIndex != -1 && pKFi->mvuRight[get<0>(mit->second)]<0)
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
                    else if(leftIndex != -1 && pKFi->mvuRight[get<0>(mit->second)]>=0)// Stereo observation
                    {
                        const cv::KeyPoint &kpUn = pKFi->getUndistKPtMono(leftIndex);
                        Eigen::Matrix<double,3,1> obs;
                        const float kp_ur = pKFi->mvuRight[get<0>(mit->second)];
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
                        int rightIndex = get<1>(mit->second);

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
        }
        num_edges = nEdges;

        if (nPoints <= 0) {
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
            LOG(WARNING) << "LM-LBA: All vertices are set to marginalized, abort optimization\n";
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
        for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
        {
            KeyFrame* pKFi = *lit;
            if (!pKF)
                continue;
            g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKFi->mnId));
            g2o::SE3Quat SE3quat = vSE3->estimate();
            pKFi->SetPose(Converter::toCvMat(SE3quat));

        }

        //Points
        for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
        {
            MapPoint* pMP = *lit;
            if (!pMP)
                continue;
            g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
        }

        // TODO Check this changeindex
        pMap->IncreaseChangeIndex();
    }

    void MyOptimizer::LocalBundleAdjustment22(KeyFrame *pKF, bool* pbStopFlag, Map* pMap, int& num_fixedKF, int& num_OptKF,
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
        for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)
        {
            KeyFrame* pKFi = *lit;
            if(pKFi->mnId==pMap->GetInitKFid())
            {
                num_fixedKF = 1;
            }
            vector<MapPoint*> vpMPs = pKFi->GetMapPointMatches();
            for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
            {
                MapPoint* pMP = *vit;
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

        // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
        list<KeyFrame*> lFixedCameras;
        for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
        {
            map<KeyFrame*,tuple<int,int>> observations = (*lit)->GetObservations();
            for(map<KeyFrame*,tuple<int,int>>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
            {
                KeyFrame* pKFi = mit->first;

                if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId )
                {
                    pKFi->mnBAFixedForKF=pKF->mnId;
                    if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                        lFixedCameras.push_back(pKFi);
                }
            }
        }
        num_fixedKF = lFixedCameras.size() + num_fixedKF;


        if(num_fixedKF == 0)
        {
            Verbose::PrintMess("LM-LBA: There are 0 fixed KF in the optimizations, LBA aborted", Verbose::VERBOSITY_NORMAL);
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

        // DEBUG LBA
        //pCurrentMap->msOptKFs.clear();
        //pCurrentMap->msFixedKFs.clear();

        // Set Local KeyFrame vertices
        for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
        {
            KeyFrame* pKFi = *lit;
            g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
            //Sophus::SE3<float> Tcw = pKFi->GetPose();
            vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));//g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
            vSE3->setId(pKFi->mnId);
            vSE3->setFixed(pKFi->mnId==pMap->GetInitKFid());
            optimizer.addVertex(vSE3);
            if(pKFi->mnId>maxKFid)
                maxKFid=pKFi->mnId;
            // DEBUG LBA
            //pCurrentMap->msOptKFs.insert(pKFi->mnId);
        }
        num_OptKF = lLocalKeyFrames.size();

        // Set Fixed KeyFrame vertices
        for(list<KeyFrame*>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
        {
            KeyFrame* pKFi = *lit;
            g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
            //Sophus::SE3<float> Tcw = pKFi->GetPose();
            vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));//(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(),Tcw.translation().cast<double>()));
            vSE3->setId(pKFi->mnId);
            vSE3->setFixed(true);
            optimizer.addVertex(vSE3);
            if(pKFi->mnId>maxKFid)
                maxKFid=pKFi->mnId;
            // DEBUG LBA
            //pCurrentMap->msFixedKFs.insert(pKFi->mnId);
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

        int nEdges = 0;

        for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
        {
            MapPoint* pMP = *lit;
            g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
            vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));//.cast<double>());
            int id = pMP->mnId+maxKFid+1;
            vPoint->setId(id);
            vPoint->setMarginalized(true);
            optimizer.addVertex(vPoint);
            nPoints++;

            const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();

            //Set edges
            for(map<KeyFrame*,tuple<int,int>>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
            {
                KeyFrame* pKFi = mit->first;

                if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                {
                    const int leftIndex = get<0>(mit->second);

                    // Monocular observation
                    if(leftIndex != -1 && pKFi->mvuRight[get<0>(mit->second)]<0)
                    {
                        const cv::KeyPoint &kpUn = pKFi->getUndistKPtMono(leftIndex);
                        Eigen::Matrix<double,2,1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y;

                        ORB_SLAM3::EdgeSE3ProjectXYZ* e = new ORB_SLAM3::EdgeSE3ProjectXYZ();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);
                        const float &invSigma2 = pKFi->getKPtInvLevelSigma2(leftIndex);
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
                    else if(leftIndex != -1 && pKFi->mvuRight[get<0>(mit->second)]>=0)// Stereo observation
                    {
                        const cv::KeyPoint &kpUn = pKFi->getUndistKPtMono(leftIndex);
                        Eigen::Matrix<double,3,1> obs;
                        const float kp_ur = pKFi->mvuRight[get<0>(mit->second)];
                        obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                        g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);
                        const float &invSigma2 = pKFi->getORBInvLevelSigma2(kpUn.octave);
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
                        int rightIndex = get<1>(mit->second);

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

                            //Sophus::SE3f Trl = pKFi-> GetRelativePoseTrl();
                            e->mTrl = Converter::toSE3Quat(pKFi->mTlr);//g2o::SE3Quat(Trl.unit_quaternion().cast<double>(), Trl.translation().cast<double>());

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
        }
        num_edges = nEdges;

        if(pbStopFlag)
            if(*pbStopFlag)
                return;

        optimizer.initializeOptimization();
        optimizer.optimize(10);

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
        for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
        {
            KeyFrame* pKFi = *lit;
            g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKFi->mnId));
            g2o::SE3Quat SE3quat = vSE3->estimate();
            //Sophus::SE3f Tiw(SE3quat.rotation().cast<float>(), SE3quat.translation().cast<float>());
            pKFi->SetPose(Converter::toCvMat(SE3quat));
        }

        //Points
        for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
        {
            MapPoint* pMP = *lit;
            g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate())); //.cast<float>());
            pMP->UpdateNormalAndDepth();
        }

        pMap->IncreaseChangeIndex();
    }


    /* -------------------------------------------------------------------------------------------------------------- */

    int MyOptimizer::findNearestPose(const PoseConstraintList &mPoseList, KeyFrame* pKF,
            PoseConstraintListIter& firstConst, PoseConstraintListIter& secondConst, const float thTs) {

        const double refTs = pKF->mTimeStamp;

        // Find nearest Constraint based on timestamps
        auto firstRef = mPoseList.lower_bound(refTs);

        // If event timestamp is in bound, ...
        if (firstRef != mPoseList.end()) {

            const double distTs = firstRef->first - refTs;

            if (distTs < thTs) {
                // Poses are close, we can use the found vertex
                firstConst = firstRef;
                DLOG_EVERY_N(INFO, 100) << "MyOptimizer::findNearestPose: Found close ts: "
                                           << firstRef->first << " to ref. ts: " << refTs << endl;
                return 1;
            }
            else {
                // Search pose is not close, return the 2 nearest verticies
                if (firstRef == mPoseList.begin()) {
                    firstConst = firstRef;
                    secondConst = std::next(firstRef);
                }
                else {
                    firstConst = std::prev(firstRef);
                    secondConst = firstRef;
                }
                // It is possible that current pose is not near one bound but the other
                if (abs(firstConst->first - refTs) < thTs) {
                    DLOG_EVERY_N(INFO, 100) << "MyOptimizer::findNearestPose: Found close ts: "
                                               << firstConst->first << " to ref. ts: " << refTs << endl;
                    return 1;
                }
                else if (abs(secondConst->first - refTs) < thTs) {
                    firstConst = secondConst;
                    DLOG_EVERY_N(INFO, 100) << "MyOptimizer::findNearestPose: Found close ts: "
                                            << firstConst->first << " to ref. ts: " << refTs << endl;
                    return 1;
                }
                else {
                    DLOG_EVERY_N(INFO, 100) << "MyOptimizer::findNearestPose: Found 2 close ts: "
                                               << firstConst->first << " ~ " << secondConst->first
                                               << " to ref. ts: " << refTs << endl;
                    return 2;
                }
            }
        }
        else {
            DLOG_EVERY_N(INFO, 1000) << "MyOptimizer::findNearestPose: Cannot find any close constraints to ts: "
                                        << refTs << "\n";
            return 0;
        }
    }

    // Key frame in EventVertexPoseList is the one we search for (pEvKF) -> Used for later MP const. addition
    // Key frame in orbConstraints is either a real ORB or an interpolated virtual KF -> Used for search
    // This method adds either an interpolated vertex, or none (exists a near pose or no pose exists)
    int MyOptimizer::addEventVertexPose(g2o::SparseOptimizer& optimizer, EventVertexPoseList& mEvVertex,
            g2o::Sim3& Twowe, ORB_SLAM3::KeyFrame* pEvKF, PoseConstraintList& orbConstraints,
            const unsigned long currId, const float thTs) {

        // TODO: What is matLambda and why don't set all diag. to 1e3
        Eigen::Matrix<double,6,6> matLambda = Eigen::Matrix<double,6,6>::Identity();
        matLambda(0,0) = 1e3;
        matLambda(1,1) = 1e3;
        matLambda(2,2) = 1e3;

        // Find nearest ORB constraint based on timestamps
        PoseConstraintListIter firstORB, lastORB;
        int stat = findNearestPose(orbConstraints, pEvKF, firstORB, lastORB, thTs);

        // If event timestamp is in ORB bound, add event constraint
        if (stat != 0) {

            g2o::SE3Quat Tcw_gi;
            float orbScale = 1.f;
            unsigned long idKF = currId;

            // Resolve SE3 Pose Vertex
            if (stat == 1) {
                // ORB and Event poses are close, we can use ORB vertex
                // In this case we just need to remember ORB KF Id and relative world transformation
                idKF = firstORB->second.second;
                KeyFrame* pOrbKF = firstORB->second.first;
                Tcw_gi = Converter::toSE3Quat(pOrbKF->GetPose());
                orbScale = pOrbKF->ComputeSceneMedianDepth(2);
            }
            else {
                // ORB and Event poses are not close, estimate a virtual vertex
                const double evRefTs = pEvKF->mTimeStamp;
                const double dTs0 = abs(evRefTs - firstORB->first);

                const double firstOrbTs = firstORB->first;
                KeyFrame* firstOrbKF = firstORB->second.first;
                const long firstOrbId = firstORB->second.second;
                g2o::SE3Quat Tcw_g1 = Converter::toSE3Quat(firstOrbKF->GetPose());

                const double lastOrbTs = lastORB->first;
                KeyFrame* lastOrbKF = lastORB->second.first;
                const long lastOrbId = lastORB->second.second;
                g2o::SE3Quat Tcw_g2 = Converter::toSE3Quat(lastOrbKF->GetPose());

                orbScale = firstOrbKF->ComputeSceneMedianDepth(2);

                // Interpolate the middle pose assuming constant speed constraint
                const double dTs = lastOrbTs - firstOrbTs;

                Tcw_gi = Converter::interpTcw(Tcw_g1, Tcw_g2, dTs0, dTs);

                // Add the vertex
                g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
                vSE3->setEstimate(Tcw_gi);
                vSE3->setId(static_cast<int>(idKF));
                vSE3->setFixed(false);
                optimizer.addVertex(vSE3);

                // Set relative constraints to adjacent ORB poses
                g2o::SE3Quat dTw1wi = Tcw_g1.inverse() * Tcw_gi;
                g2o::EdgeSE3* e = new g2o::EdgeSE3();
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(static_cast<int>(idKF))));
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(static_cast<int>(firstOrbId))));
                e->setMeasurement(dTw1wi);

                e->information() = matLambda;
                //e_loop = e;
                optimizer.addEdge(e);

                g2o::SE3Quat dTwiw2 = Tcw_gi.inverse() * Tcw_g2;
                g2o::EdgeSE3* e1 = new g2o::EdgeSE3();
                e1->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(static_cast<int>(lastOrbId))));
                e1->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(static_cast<int>(idKF))));
                e1->setMeasurement(dTwiw2);

                e1->information() = matLambda;
                //e_loop = e;
                optimizer.addEdge(e1);

                auto* pVirKF = new KeyFrame(*(firstOrbKF));
                pVirKF->SetPose(Converter::toCvSE3(Tcw_gi.rotation().toRotationMatrix(), Tcw_gi.translation()));
                //pVirKF->mTimeStamp = pEvKF->mTimeStamp;
                // Also insert the new vertex as a ORB constraint so it can be found later
                orbConstraints.insert(make_pair(evRefTs, make_pair(pVirKF, idKF)));

                DLOG_EVERY_N(INFO, 100) << "MyOptimizer::addEventVertexPose: Added Virtual pose at " << evRefTs
                                           << " with Id: " << idKF << ", interpolated pose:\n" << Converter::toString(Tcw_gi);
            }

            // Save this information for adding map point constraints
            g2o::SE3Quat evPose = Converter::toSE3Quat(pEvKF->GetPose());
            const float evScale = pEvKF->ComputeSceneMedianDepth(2);
            const float relScale = orbScale/(evScale+1e-9);
            // TODO: Check these scale relations!
            g2o::Sim3 Tcwo(Tcw_gi.rotation(), Tcw_gi.translation(), 1.0);
            g2o::Sim3 Tcwe(evPose.rotation(), evPose.translation() * relScale, relScale);
            Twowe = Tcwo.inverse() * Tcwe;

            mEvVertex.insert(make_pair(pEvKF, make_pair(idKF, Twowe)));

            DLOG_EVERY_N(INFO, 100) << "MyOptimizer::addEventVertexPose: Found pose match, orbSc: " << orbScale
                                    << ", evSc: " << evScale << ", relSc: " << relScale
                                    << ", Twowe (e: Search to o: Old):\n" << Converter::toString(Tcw_gi);
        }
        return stat;
    }

    bool kf_ts_comp(KeyFrame* first, KeyFrame* last) {

        return first->mTimeStamp < last->mTimeStamp;
    }

    // Before calling this, merge all internal event rel poses (in localization mode)
    // WARNING!! This will mess up with map & ... containers so call this when everything is over!
    void MyOptimizer::MergeVisualEvent(ORB_SLAM3::Atlas *pAtlasORB, const std::vector<ORB_SLAM3::KeyFrame *> &vpEvRefKF,
            ORB_SLAM3::Atlas *pAtlasMerge, const int nIterations, const bool bRobust) {

        // Build the optimization problem
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

        g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        // Initializing helper containers
        const float thHuber2D = sqrt(5.99f);
        const float thHuber3D = sqrt(7.815f);

        const float thTsSelf = 1e-3;
        const float thTsMut = 0.01;
        double maxTs = -1, minTs = MAXFLOAT;

        unsigned long currId = 0;
        bool fixedRef = true;

        map<KeyFrame*, unsigned long> mPoseConstraints;
        //map<unsigned long, KeyFrame*> mPoseVerticies;
        map<MapPoint*, unsigned long> mMPConstraints;
        //EventVertexPoseList mRelWorldsTrans;
        PoseConstraintList mLastSearchConsts;

        size_t idxEvRefs = 0;
        size_t idxOrbMaps = 0;
        const size_t nEvRefs = vpEvRefKF.size();
        vector<Map*> vpAllOrbMaps = pAtlasORB->GetAllMaps();
        const size_t nOrbMaps = vpAllOrbMaps.size();

        double refTsORB = 0.0;
        double refTsEv = 0.0;

        DLOG(INFO) << "MyOptimizer::MergeVisualEvent: Commencing merge with "
                   << nOrbMaps << " ORB Maps and " << nEvRefs << " Ev Refs.\n";

        // Add Key Frames
        // Traverse through all maps
        while (idxEvRefs < nEvRefs && idxOrbMaps < nOrbMaps) {

            Map* currOrbMap = vpAllOrbMaps[idxOrbMaps];
            if (currOrbMap->GetAllKeyFrames().size() < 3) {
                DLOG(WARNING) << "MyOptimizer::MergeVisualEvent: Low number of KFs: "
                              << currOrbMap->GetAllKeyFrames().size() << " in ORB Map: " << idxOrbMaps << "\n";
                idxOrbMaps++;
                continue;
            }
            KeyFrame* pRefEvKF = vpEvRefKF[idxEvRefs];
            if (!pRefEvKF || pRefEvKF->isBad() || pRefEvKF->GetChilds().empty()) {
                DLOG(WARNING) << "MyOptimizer::MergeVisualEvent: Low number of KFs: "
                              << pRefEvKF->GetChilds().size()+1 << " in Ev Map: " << idxEvRefs << "\n";
                idxEvRefs++;
                continue;
            }

            refTsORB = currOrbMap->GetOriginKF()->mTimeStamp;
            refTsEv = pRefEvKF->mTimeStamp;

            if (refTsORB <= refTsEv) {

                // Tracking starts with ORB maps
                DLOG(INFO) << "MyOptimizer::MergeVisualEvent: Beginning ORB Merge, TsORB: "
                           << refTsORB << ", TsEv: " << refTsEv << ", Lower: ORB\n";

                // Add all ORB key frames first
                // Event ref. timestamps can exactly be the same with lastTs!
                while (idxOrbMaps < nOrbMaps && maxTs <= refTsEv) {

                    currOrbMap = vpAllOrbMaps[idxOrbMaps];

                    vector<KeyFrame*> vpOrbKFs = currOrbMap->GetAllKeyFrames();

                    if (!vpOrbKFs.empty()) {

                        // Sort keyframes based on time stamp
                        //sort(vpOrbKFs.begin(), vpOrbKFs.end(), kf_ts_comp);
                        // ORB-SLAM sorts based on key frames Id:
                        sort(vpOrbKFs.begin(), vpOrbKFs.end(), KeyFrame::lId);

#ifdef SAVE_MAP_BEFORE_MERGE
                        // Save Pose-Map data for offline debugging purposes
                        DLOG(INFO) << "MyOptimizer::MergeVisualEvent: Saving raw ORB map before merge\n";
                        const string posePath = "../data/orb_pose_"+to_string(vpOrbKFs[0]->mTimeStamp)+".txt";
                        const string mapPath = "../data/orb_map_"+to_string(vpOrbKFs[0]->mTimeStamp)+".txt";
                        Visualization::savePoseMap(vpOrbKFs, posePath, mapPath);
#endif

                        DLOG(INFO) << "MyOptimizer::MergeVisualEvent: Stitching ORB Map #" << idxOrbMaps
                                   << ", with " << vpOrbKFs.size() << " KFs, refTs: " << vpOrbKFs[0]->mTimeStamp << endl;

                        stitchPoseConstraints(vpOrbKFs, optimizer, pAtlasMerge, mPoseConstraints, mLastSearchConsts,
                                              currId, minTs, maxTs, thTsMut, fixedRef);
                    }

                    fixedRef = false;
                    idxOrbMaps++;

                    if (idxOrbMaps < nOrbMaps && maxTs <= refTsEv) {

                        const double nextOrbRefTs = vpAllOrbMaps[idxOrbMaps]->GetOriginKF()->mTimeStamp;
                        if (nextOrbRefTs > maxTs) {
                            DLOG(INFO) << "MyOptimizer::MergeVisualEvent: No Ev map between ORB maps, creating new ORB map\n";
                            pAtlasMerge->CreateNewMap();
                            fixedRef = true;
                        }
                    }
                }
            }
            else {
                // Tracking starts with Events, same as above but in reverse order
                DLOG(INFO) << "MyOptimizer::MergeVisualEvent: Beginning Ev Merge, TsORB: "
                           << refTsORB << ", TsEv: " << refTsEv << ", Lower: Ev\n";

                // Add all relative event key frames
                while (idxEvRefs < nEvRefs && maxTs < refTsORB) {

                    // Construct vector of event key frames
                    pRefEvKF = vpEvRefKF[idxEvRefs];
                    assert(pRefEvKF);
                    vector<KeyFrame *> vpEvKFs;
                    size_t nEvKFs = pRefEvKF->GetChilds().size() + 1;
                    vpEvKFs.reserve(nEvKFs);

                    vpEvKFs.push_back(pRefEvKF);
                    for (KeyFrame *pEvKF : pRefEvKF->GetChilds()) {
                        vpEvKFs.push_back(pEvKF);
                    }

#ifdef SAVE_MAP_BEFORE_MERGE
                    DLOG(INFO) << "MyOptimizer::MergeVisualEvent: Saving raw Ev map before merge\n";
                    const string posePath = "../data/ev_pose_"+to_string(vpEvKFs[0]->mTimeStamp)+".txt";
                    const string mapPath = "../data/ev_map_"+to_string(vpEvKFs[0]->mTimeStamp)+".txt";
                    Visualization::savePoseMap(vpEvKFs, posePath, mapPath);
#endif

                    DLOG(INFO) << "MyOptimizer::MergeVisualEvent: Stitching Ev Map #" << idxOrbMaps
                               << ", with " << vpEvKFs.size() << " KFs, refTs: " << vpEvKFs[0]->mTimeStamp << endl;

                    stitchPoseConstraints(vpEvKFs, optimizer, pAtlasMerge, mPoseConstraints, mLastSearchConsts,
                                          currId, minTs, maxTs, thTsMut, fixedRef);
                    fixedRef = false;
                    idxEvRefs++;

                    if (idxEvRefs < nEvRefs && maxTs < refTsORB) {

                        const double nextEvRefTs = vpEvRefKF[idxEvRefs]->mTimeStamp;
                        if (nextEvRefTs > maxTs) {
                            DLOG(INFO) << "MyOptimizer::MergeVisualEvent: No ORB map between Ev maps, creating new Ev map\n";
                            pAtlasMerge->CreateNewMap();
                            fixedRef = true;
                        }
                    }
                }
            }
        }
        // What if Event & ORB vary a lot: One has a lot of tracks and the other not!
        while (idxOrbMaps < nOrbMaps) {


            // Add the rest of ORB Maps
            DLOG(INFO) << "MyOptimizer::MergeVisualEvent: Merging rest of ORB maps...\n";

            Map* currOrbMap = vpAllOrbMaps[idxOrbMaps];

            vector<KeyFrame*> vpOrbKFs = currOrbMap->GetAllKeyFrames();

            if (!vpOrbKFs.empty()) {

                // Sort keyframes based on time stamp
                //sort(vpOrbKFs.begin(), vpOrbKFs.end(), kf_ts_comp);
                sort(vpOrbKFs.begin(), vpOrbKFs.end(), KeyFrame::lId);

#ifdef SAVE_MAP_BEFORE_MERGE
                DLOG(INFO) << "MyOptimizer::MergeVisualEvent: Saving raw ORB map before merge\n";
                const string posePath = "../data/orb_pose_" + to_string(vpOrbKFs[0]->mTimeStamp) + ".txt";
                const string mapPath = "../data/orb_map_" + to_string(vpOrbKFs[0]->mTimeStamp) + ".txt";
                Visualization::savePoseMap(vpOrbKFs, posePath, mapPath);
#endif

                DLOG(INFO) << "MyOptimizer::MergeVisualEvent: Stitching ORB Map #" << idxOrbMaps
                           << ", with " << vpOrbKFs.size() << " KFs, refTs: " << vpOrbKFs[0]->mTimeStamp << endl;

                stitchPoseConstraints(vpOrbKFs, optimizer, pAtlasMerge, mPoseConstraints, mLastSearchConsts,
                                      currId, minTs, maxTs, thTsMut, fixedRef);
            }

            fixedRef = false;
            idxOrbMaps++;

            if (idxOrbMaps < nOrbMaps) {

                const double nextOrbRefTs = vpAllOrbMaps[idxOrbMaps]->GetOriginKF()->mTimeStamp;
                if (nextOrbRefTs > maxTs) {
                    DLOG(INFO) << "MyOptimizer::MergeVisualEvent: No Ev map between ORB maps, creating new ORB map\n";
                    pAtlasMerge->CreateNewMap();
                    fixedRef = true;
                }
            }
        }
        while (idxEvRefs < nEvRefs) {

            // Add the rest of Event Refs
            DLOG(INFO) << "MyOptimizer::MergeVisualEvent: Merging rest of Ev maps...\n";

            // Construct vector of event key frames
            KeyFrame *pRefEvKF = vpEvRefKF[idxEvRefs];
            assert(pRefEvKF);
            vector<KeyFrame *> vpEvKFs;
            size_t nEvKFs = pRefEvKF->GetChilds().size() + 1;
            vpEvKFs.reserve(nEvKFs);

            vpEvKFs.push_back(pRefEvKF);
            for (KeyFrame *pEvKF : pRefEvKF->GetChilds()) {
                vpEvKFs.push_back(pEvKF);
            }

#ifdef SAVE_MAP_BEFORE_MERGE
            DLOG(INFO) << "MyOptimizer::MergeVisualEvent: Saving raw Ev map before merge\n";
            const string posePath = "../data/ev_pose_"+to_string(vpEvKFs[0]->mTimeStamp)+".txt";
            const string mapPath = "../data/ev_map_"+to_string(vpEvKFs[0]->mTimeStamp)+".txt";
            Visualization::savePoseMap(vpEvKFs, posePath, mapPath);
#endif

            DLOG(INFO) << "MyOptimizer::MergeVisualEvent: Stitching Ev Map #" << idxOrbMaps
                       << ", with " << vpEvKFs.size() << " KFs, refTs: " << vpEvKFs[0]->mTimeStamp << endl;

            stitchPoseConstraints(vpEvKFs, optimizer, pAtlasMerge, mPoseConstraints, mLastSearchConsts,
                                  currId, minTs, maxTs, thTsMut, fixedRef);
            fixedRef = false;
            idxEvRefs++;

            if (idxEvRefs < nEvRefs) {

                const double nextEvRefTs = vpEvRefKF[idxEvRefs]->mTimeStamp;
                if (nextEvRefTs > maxTs) {
                    DLOG(INFO) << "MyOptimizer::MergeVisualEvent: No ORB map between Ev maps, creating new Ev map\n";
                    pAtlasMerge->CreateNewMap();
                    fixedRef = true;
                }
            }
        }

        DLOG(INFO) << "MyOptimizer::MergeVisualEvent: " << mPoseConstraints.size()
                   << " pose constraint(s) added, currId: " << currId
                   << ", min Ts: " << minTs << ", max Ts: " << maxTs << "\n";
        DLOG(INFO) << "MyOptimizer::MergeVisualEvent: Adding map point constraints...\n";

        // Add Map Point Constraints
        for(const auto& poseConst : mPoseConstraints)
        {
            KeyFrame* pKFi = poseConst.first;
            const unsigned long kfId = poseConst.second;

            for (MapPoint* pMP : pKFi->GetMapPoints()) {

                if (!pMP || pMP->isBad() || mMPConstraints.count(pMP))
                    continue;

                g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
                vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
                const unsigned long id = currId;
                vPoint->setId(id);
                vPoint->setMarginalized(true);
                optimizer.addVertex(vPoint);

                const map<KeyFrame *, tuple<int, int>> observations = pMP->GetObservations();

                int nEdges = 0;
                //SET EDGES
                for (const auto& mit : observations) {

                    KeyFrame *pKF = mit.first;
                    if (pKF->isBad() || !mPoseConstraints.count(pKF))
                        continue;

                    const unsigned long obsKfId = mPoseConstraints.find(pKF)->second;

                    if (optimizer.vertex(id) == nullptr || optimizer.vertex(obsKfId) == nullptr)
                        continue;
                    nEdges++;

                    const int leftIndex = get<0>(mit.second);

                    if (leftIndex != -1 && pKF->mvuRight[get<0>(mit.second)] < 0) {
                        const cv::KeyPoint &kpUn = pKF->getUndistKPtMono(leftIndex);

                        Eigen::Matrix<double, 2, 1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y;

                        ORB_SLAM3::EdgeSE3ProjectXYZ *e = new ORB_SLAM3::EdgeSE3ProjectXYZ();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(obsKfId)));
                        e->setMeasurement(obs);
                        const float &invSigma2 = pKF->getKPtInvLevelSigma2(leftIndex);
                        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                        if (bRobust) {
                            g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                            e->setRobustKernel(rk);
                            rk->setDelta(thHuber2D);
                        }

                        e->pCamera = pKF->mpCamera;

                        optimizer.addEdge(e);

                        //vpEdgesMono.push_back(e);
                        //vpEdgeKFMono.push_back(pKF);
                        //vpMapPointEdgeMono.push_back(pMP);
                    }
                }

                if (nEdges == 0) {
                    optimizer.removeVertex(vPoint);
                    //vbNotIncludedMP[i] = true;
                } else {
                    //vbNotIncludedMP[i] = false;
                    //if (globalMaxId < id)
                    //    globalMaxId = id;
                    mMPConstraints.insert(make_pair(pMP, currId));
                    currId++;
                }
            }
        }

        // Optimize
        DLOG(INFO) << "MyOptimizer::MergeVisualEvent: Beginning of the optimization...\n";
        bool optimVerbose = false;
        optimizer.setVerbose(optimVerbose);
        optimizer.initializeOptimization();
        optimizer.optimize(nIterations);
        DLOG(INFO) << "MyOptimizer::MergeVisualEvent: End of the optimization...\n";

        // Recover Data
        DLOG(INFO) << "MyOptimizer::MergeVisualEvent: Recovering data...\n";
        // Key frames
        for (auto& poseConst : mPoseConstraints) {

            KeyFrame* pKFi = poseConst.first;
            const unsigned long kfId = poseConst.second;

            if(pKFi->isBad())
                continue;

            g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(kfId));

            g2o::SE3Quat SE3quat = vSE3->estimate();
            pKFi->SetPose(Converter::toCvMat(SE3quat));
        }

        DLOG(INFO) << "MyOptimizer::MergeVisualEvent: " << mPoseConstraints.size() << " poses recovered.\n";

        // Map points
        for(auto& mpConst : mMPConstraints) {

            MapPoint* pMP = mpConst.first;
            const unsigned long mpId = mpConst.second;

            if(pMP->isBad())
                continue;

            g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(mpId));

            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
        }

        DLOG(INFO) << "MyOptimizer::MergeVisualEvent: " << mMPConstraints.size() << " map points recovered.\n";
    }

    // vpKFs: The first key frame is reference KF and all of these KFs span a whole map
    void MyOptimizer::stitchPoseConstraints(const std::vector<KeyFrame*>& vpKFs, g2o::SparseOptimizer& optimizer,
            Atlas* pAtlasMerge, map<KeyFrame*, unsigned long>& mAllVertices, PoseConstraintList& mLastSearchConsts,
            unsigned long& currId, double& minTs, double& maxTs, const float thTs, const bool fixedRef) {

        // TODO: What is matLambda and why don't set all diag. to 1e3
        Eigen::Matrix<double,6,6> matLambda = Eigen::Matrix<double,6,6>::Identity();
        matLambda(0,0) = 1e3;
        matLambda(1,1) = 1e3;
        matLambda(2,2) = 1e3;

        bool refFound = false;
        // Used to remap current graph if we have a connection to older graph
        g2o::SE3Quat iniTw0w1(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
        float iniScale = 1.f;
        // Used to remap map points only once!
        set<MapPoint*> spAllMPs;
        // Used internally to call addEventVertexPose method, for merge optim., it's useless!
        EventVertexPoseList mLocalRWsTrans;
        // This is also constructed locally and finally exchanged with old search consts.
        //PoseConstraintList mNewSearchConsts;

        Map* fuseMap = pAtlasMerge->GetCurrentMap();

        int nAddedKFs = 0, nFound = 0;

        // First key frame in vector is alwayes treated as ref. key frame
        for (size_t i = 0; i < vpKFs.size(); i++) {

            KeyFrame* pKF = vpKFs[i];

            if(pKF->isBad())
                continue;

            // Add Vertex
            g2o::Sim3 Tw0w1;
            // Careful!! This method changes a lot of things (including mRelWorldsTrans, mSearchConsts)
            int stat = addEventVertexPose(optimizer, mLocalRWsTrans, Tw0w1, pKF, mLastSearchConsts, currId, thTs);

            if (stat != 0 && i == 0) {

                refFound = true;
                // Change iniTwowe if this is ref. kf
                iniTw0w1 = g2o::SE3Quat(Tw0w1.rotation(), Tw0w1.translation());
                iniScale = static_cast<float>(Tw0w1.scale());

                DLOG(INFO) << "MyOptimizer::stitchPoseConstraints: Ref. at ts: "
                           << pKF->mTimeStamp << " found with Twr:\n" << Converter::toString(iniTw0w1)
                           << "and relative scale: " << iniScale << endl;
            }
            if (stat == 2) {
                // This means we added a virtual kf, so we need to inc currId
                currId++;
            }

            // Remap Kf pose and all map points if ref. found
            if (refFound) {
                g2o::SE3Quat currTcw1(Converter::toMatrix3d(pKF->GetRotation()),
                                      iniScale * Converter::toVector3d(pKF->GetTranslation()));
                pKF->SetPose(Converter::toCvMat(currTcw1 * iniTw0w1.inverse()));
            }

            // Add key frame to new map
            pKF->mnId = currId;
            pKF->UpdateMap(fuseMap);
            pAtlasMerge->AddKeyFrame(pKF);

            // Also add map points
            for (MapPoint *pMP : pKF->GetMapPoints()) {

                if (pMP && !pMP->isBad() && !spAllMPs.count(pMP)) {

                    if (refFound) {
                        Eigen::Vector3d p3d = iniTw0w1.map(iniScale * Converter::toVector3d(pMP->GetWorldPos()));
                        pMP->SetWorldPos(Converter::toCvMat(p3d));
                    }

                    pMP->UpdateMap(fuseMap);
                    pAtlasMerge->AddMapPoint(pMP);

                    spAllMPs.insert(pMP);
                }
            }

            // Merge vertices always are added
            g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
            vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));
            vSE3->setId(static_cast<int>(currId));
            vSE3->setFixed(i == 0 && fixedRef);
            optimizer.addVertex(vSE3);

            if (stat != 0) {

                // For merge, we also need an additional constraint between event-image graphs
                // This actually implies the 2 poses are the same
                g2o::EdgeSE3* e = new g2o::EdgeSE3();

                const unsigned long oldId = mLocalRWsTrans.find(pKF)->second.first;
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(static_cast<int>(oldId))));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(static_cast<int>(currId))));
                e->setMeasurement(g2o::SE3Quat());

                e->information() = matLambda;
                //e_loop = e;
                optimizer.addEdge(e);

                nFound++;
            }

            // Save supporting info.
            if (pKF->mTimeStamp < minTs)
                minTs = pKF->mTimeStamp;
            if (pKF->mTimeStamp > maxTs)
                maxTs = pKF->mTimeStamp;

            mAllVertices.insert(make_pair(pKF, currId));
            //mNewSearchConsts.insert(make_pair(pKF->mTimeStamp, make_pair(pKF, currId)));
            if (stat == 0) {
                mLastSearchConsts.insert(make_pair(pKF->mTimeStamp, make_pair(pKF, currId)));
            }

            currId++;
            nAddedKFs++;
        }
        // If we assign search KFs globally, we might accidentally connect disconnected graphs
        //mLastSearchConsts = mNewSearchConsts;
        DLOG(INFO) << "MyOptimizer::stitchPoseConstraints: Added " << nAddedKFs << " from " << vpKFs.size() << " KFs\n";
        DLOG(INFO) << "MyOptimizer::stitchPoseConstraints: Found " << nFound << " KFs in graph\n";
        DLOG(INFO) << "MyOptimizer::stitchPoseConstraints: currId: " << currId
                   << ", minTs: " << minTs << ", maxTs: " << maxTs << endl;
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    void MyOptimizer::addPoseVertices(g2o::SparseOptimizer &optimizer, const vector<EORB_SLAM::EvFramePtr> &vpEvFrames,
                                      int &vfactor, const bool addInertial) {

        const int maxFrId = static_cast<int>(std::max(vpEvFrames.back()->mnId, vpEvFrames.size()));

        for(size_t i=0; i < vpEvFrames.size(); i++)
        {
            EvFramePtr pFri = vpEvFrames[i];

            bool bFixed = i == 0;
            int currId = static_cast<int>(pFri->mnId);

            if(addInertial)
            {
                VertexPose* VP = new VertexPose(pFri.get());
                VP->setId(vfactor * maxFrId + currId);
                // The first frame is the reference
                VP->setFixed(bFixed);
                optimizer.addVertex(VP);

                VertexVelocity* VV = new VertexVelocity(pFri.get());
                VV->setId((vfactor+1) * maxFrId + currId);
                VV->setFixed(bFixed);
                optimizer.addVertex(VV);
            }
            else {
                g2o::VertexSE3Expmap* VP = new g2o::VertexSE3Expmap();
                VP->setEstimate(Converter::toSE3Quat(pFri->mTcw));
                VP->setId(vfactor * maxFrId + currId);
                // The first frame is the reference
                VP->setFixed(bFixed);
                optimizer.addVertex(VP);
            }
        }

        if (addInertial) {
            vfactor += 2;
        }
        else {
            vfactor++;
        }
    }

    void MyOptimizer::addBiasIMU(g2o::SparseOptimizer &optimizer, const vector<EORB_SLAM::EvFramePtr> &vpEvFrames,
                                 const InitInfoImuPtr& pInfoImu, int &vfactor, const bool singleBias) {

        const int maxFrId = static_cast<int>(std::max(vpEvFrames.back()->mnId, vpEvFrames.size()));

        IMU::Bias bIniIMU = pInfoImu->getLastImuBias();
        bool fixedBG = pInfoImu->mInfoG <= 0;
        bool fixedBA = pInfoImu->mInfoA <= 0;

        if (singleBias) {

            vpEvFrames.front()->SetNewBias(bIniIMU);

            VertexGyroBias* VG = new VertexGyroBias(vpEvFrames.front().get());
            VG->setId(vfactor * maxFrId + 0);
            VG->setFixed(fixedBG);
            optimizer.addVertex(VG);

            VertexAccBias* VA = new VertexAccBias(vpEvFrames.front().get());
            VA->setId(vfactor * maxFrId + 1);
            VA->setFixed(fixedBA);
            optimizer.addVertex(VA);

            // numFrames must be >= 2 (typically true for optimization!)
            vfactor++;
        }
        else {
            for (size_t i = 0; i < vpEvFrames.size(); i++) {

                EvFramePtr pFri = vpEvFrames[i];
                const int currId = static_cast<int>(pFri->mnId);

                VertexGyroBias* VG = new VertexGyroBias(pFri.get());
                VG->setId(vfactor * maxFrId + currId);
                VG->setFixed(fixedBG);
                optimizer.addVertex(VG);

                VertexAccBias* VA = new VertexAccBias(pFri.get());
                VA->setId((vfactor+1) * maxFrId + currId);
                VA->setFixed(fixedBA);
                optimizer.addVertex(VA);
            }

            vfactor += 2;
        }
    }

    void MyOptimizer::addScaleGDirIMU(g2o::SparseOptimizer &optimizer, const InitInfoImuPtr& pInfoImu,
                                      const int maxFrId, int &vfactor) {

        double scale = pInfoImu->getLastScale();
        bool fixedScale = pInfoImu->mInfoS <= 0;

        cv::Mat cvRwg = pInfoImu->getLastRwg();//.t();
        Eigen::Matrix3d Rwg = Converter::toMatrix3d(cvRwg);
        bool fixedRwg = pInfoImu->mInfoRwg <= 0;

        // Gravity
        VertexGDir* VGDir = new VertexGDir(Rwg);
        VGDir->setId(vfactor * maxFrId + 0);
        VGDir->setFixed(fixedRwg);
        optimizer.addVertex(VGDir);

        // Scale
        VertexScale* VS = new VertexScale(scale);
        VS->setId(vfactor * maxFrId + 1);
        VS->setFixed(fixedScale); // Fixed for stereo case
        optimizer.addVertex(VS);

        // True unless maxFrId < 2
        vfactor++;
    }

    void MyOptimizer::addPriorEdges(g2o::SparseOptimizer &optimizer, const vector<EORB_SLAM::EvFramePtr> &vpEvFrames,
                                    const int vfactorPose, int &vfactor, const double thHuber, const bool bRobust) {

        const int maxFrId = static_cast<int>(std::max(vpEvFrames.back()->mnId, vpEvFrames.size()));

        for (int i = 0; i < vpEvFrames.size(); i++) {

            EvFramePtr pFri = vpEvFrames[i];
            const int currId = static_cast<int>(pFri->mnId);
            const int vposeId = vfactorPose * maxFrId + currId;

            g2o::VertexSE3Expmap *VPi = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(vposeId));

            // TODO: Change this to dpose?? -> hard??
            cv::Mat prTcw = pFri->getPosePrior();
            // Set prior pose if required
            if (!prTcw.empty()) {

                int virId = vfactor * maxFrId + currId;

                // Add a virtual fixed vertex for prior
                g2o::VertexSE3Expmap *VVPi = new g2o::VertexSE3Expmap();
                VVPi->setEstimate(ORB_SLAM3::Converter::toSE3Quat(prTcw));
                VVPi->setId(virId);
                VVPi->setFixed(true);
                optimizer.addVertex(VVPi);

                // Add a DPose edge with measurement equal to identity dpose
                g2o::EdgeSE3 *EPPi = new g2o::EdgeSE3();
                EPPi->setVertex(virId, VPi);
                EPPi->setVertex(vposeId, VVPi);
                EPPi->setMeasurement(g2o::SE3Quat());

                if (bRobust) {
                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    EPPi->setRobustKernel(rk);
                    rk->setDelta(thHuber);
                }

                optimizer.addEdge(EPPi);
            }
        }
        vfactor++;
    }

    void MyOptimizer::addPriorImuEdges(g2o::SparseOptimizer &optimizer, const vector<EORB_SLAM::EvFramePtr> &vpEvFrames,
                                       const int vfactorPose, const int vfactorBias, const double thHuber,
                                       const bool bRobust, const bool singleBias) {

        const int maxFrId = static_cast<int>(std::max(vpEvFrames.back()->mnId, vpEvFrames.size()));

        VertexGyroBias* VGk;
        VertexAccBias* VAk;

        for (int i = 0; i < vpEvFrames.size(); i++) {

            EvFramePtr pFri = vpEvFrames[i];
            const int currId = static_cast<int>(pFri->mnId);

            VertexPose* VPk = static_cast<VertexPose*>(optimizer.vertex(vfactorPose*maxFrId+currId));
            VertexVelocity* VVk = static_cast<VertexVelocity*>(optimizer.vertex((vfactorPose+1)*maxFrId+currId));

            if (singleBias) {
                VGk = static_cast<VertexGyroBias *>(optimizer.vertex(vfactorBias * maxFrId + 0));
                VAk = static_cast<VertexAccBias *>(optimizer.vertex((vfactorBias + 1) * maxFrId + 1));
            }
            else {
                VGk = static_cast<VertexGyroBias *>(optimizer.vertex(vfactorBias * maxFrId + currId));
                VAk = static_cast<VertexAccBias *>(optimizer.vertex((vfactorBias + 1) * maxFrId + currId));
            }

            if (pFri->mpcpi) {

                EdgePriorPoseImu* ep = new EdgePriorPoseImu(pFri->mpcpi);

                ep->setVertex(0,VPk);
                ep->setVertex(1,VVk);
                ep->setVertex(2,VGk);
                ep->setVertex(3,VAk);

                if (bRobust) {
                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    ep->setRobustKernel(rk);
                    rk->setDelta(thHuber);
                }

                optimizer.addEdge(ep);
            }
        }
    }

    void MyOptimizer::addImuEdges(g2o::SparseOptimizer &optimizer, const vector<EORB_SLAM::EvFramePtr> &vpEvFrames,
                                  const InitInfoImuPtr& pInfoImu, const int vfactorPose, const int vfactorBias,
                                  const int vfactorGS, const double thHuber, const bool bRobust, const bool singleBias) {

        //const int numFrames = vpEvFrames.size();
        const int maxFrId = static_cast<int>(std::max(vpEvFrames.back()->mnId, vpEvFrames.size()));

        g2o::HyperGraph::Vertex *VGk, *VAk;
        if (singleBias) {
            VGk = optimizer.vertex(vfactorBias*maxFrId+0);
            VAk = optimizer.vertex(vfactorBias*maxFrId+1);
        }

        for (size_t i = 1; i < vpEvFrames.size(); i++) {

            EvFramePtr pFr2 = vpEvFrames[i];
            EvFramePtr pFr1 = pFr2->getPrevFrame();

            g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(vfactorPose*maxFrId+pFr1->mnId);
            g2o::HyperGraph::Vertex* VV1 = optimizer.vertex((vfactorPose+1)*maxFrId+pFr1->mnId);
            g2o::HyperGraph::Vertex* VP2 =  optimizer.vertex(vfactorPose*maxFrId+pFr2->mnId);
            g2o::HyperGraph::Vertex* VV2 = optimizer.vertex((vfactorPose+1)*maxFrId+pFr2->mnId);

            if(!singleBias) {
                VGk = optimizer.vertex(vfactorBias*maxFrId+pFr1->mnId);
                VAk = optimizer.vertex((vfactorBias+1)*maxFrId+pFr1->mnId);
            }

            if (vfactorGS < 0) {

                // No GravityDir/Scale

                EdgeInertial* ei = new EdgeInertial(pFr2->mpImuPreintegrated);
                ei->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
                ei->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
                ei->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VGk));
                ei->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VAk));
                ei->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
                ei->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));

                // TODO: What about information of these edges??

                if (bRobust) {
                    g2o::RobustKernelHuber *rki = new g2o::RobustKernelHuber;
                    ei->setRobustKernel(rki);
                    rki->setDelta(thHuber);
                }

                optimizer.addEdge(ei);
            }
            else {

                //pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
                g2o::HyperGraph::Vertex* VGDir = optimizer.vertex(vfactorGS*maxFrId+0);
                g2o::HyperGraph::Vertex* VS = optimizer.vertex(vfactorGS*maxFrId+1);

                //if(!VP1 || !VV1 || !VG || !VA || !VP2 || !VV2 || !VGDir || !VS) {
                //    cout << "Error" << VP1 << ", "<< VV1 << ", "<< VG << ", "<< VA << ", "
                //         << VP2 << ", " << VV2 <<  ", "<< VGDir << ", "<< VS <<endl;
                //    continue;
                //}

                EdgeInertialGS* ei = new EdgeInertialGS(pFr2->mpImuPreintegrated);
                ei->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
                ei->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
                ei->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VGk));
                ei->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VAk));
                ei->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
                ei->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));
                ei->setVertex(6,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VGDir));
                ei->setVertex(7,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VS));

                if (bRobust) {
                    g2o::RobustKernelHuber *rki = new g2o::RobustKernelHuber;
                    ei->setRobustKernel(rki);
                    rki->setDelta(thHuber);
                }

                optimizer.addEdge(ei);
            }
        }
    }

    void MyOptimizer::addMapPointEdges(g2o::SparseOptimizer& optimizer, std::vector<EORB_SLAM::EvFramePtr>& vpEvFrames,
                                       std::vector<cv::Point3f>& vPts3d, std::vector<int>& vbValidPts3d, const int vfactorPose,
                                       const int maxId, const double thHuber, const bool bRobust, const bool inertial) {

        // Set MapPoint vertices and edges
        const int maxFrId = static_cast<int>(std::max(vpEvFrames.back()->mnId, vpEvFrames.size()));
        const int nMPs = vPts3d.size();
        EvFramePtr pIniFrame = vpEvFrames[0];
        //vector<int> vMatches12 = pIniFrame->getMatches();
        //vector<cv::KeyPoint> currKpts = kfCur.getAllUndistKPtsMono();
        GeometricCamera* pCamera = pIniFrame->mpCamera;

        int nEdges = 0;
        for (size_t i = 0; i < nMPs; i++) {

            if (vbValidPts3d[i] < 0)
                continue;

            nEdges = 0;
            const int p3dId = maxId + static_cast<int>(i) + 1; // +1 seems redundant

            g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
            vPoint->setEstimate(Converter::toVector3d(vPts3d[i]));
            vPoint->setId(p3dId);
            vPoint->setFixed(false);
            vPoint->setMarginalized(true);
            optimizer.addVertex(vPoint);

            // What about ini frame??? -> remember to set matches for that too
            for (int j = 0; j < vpEvFrames.size(); j++) {

                EvFramePtr pFri = vpEvFrames[j];

                int currMchIdx = i;

                // Always add ref. frame edges
                if (j != 0) {
                    currMchIdx = pFri->getMatches(i);
                    if (currMchIdx < 0 || currMchIdx >= nMPs)
                        continue;
                }

                nEdges++;
                const int frId = vfactorPose * maxFrId + pFri->mnId;

                Eigen::Matrix<double, 2, 1> obs;
                cv::KeyPoint kpUn = pFri->getUndistKPtMono(currMchIdx);
                obs << kpUn.pt.x, kpUn.pt.y;

                const float &invSigma2 = pFri->getKPtInvLevelSigma2(currMchIdx);

                if (inertial) {

                    EdgeMono* e = new EdgeMono(0);

                    //g2o::OptimizableGraph::Vertex* VP = ;
                    //if(bAllFixed)
                    //    if(!VP->fixed())
                    //        bAllFixed=false;

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(p3dId)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(frId)));
                    e->setMeasurement(obs);
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    if (bRobust) {
                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuber);
                    }
                    optimizer.addEdge(e);
                }
                else {
                    ORB_SLAM3::EdgeSE3ProjectXYZ *e = new ORB_SLAM3::EdgeSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(p3dId)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(frId)));
                    e->setMeasurement(obs);
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);
                    e->pCamera = pCamera;

                    if (bRobust) {
                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuber);
                    }
                    optimizer.addEdge(e);
                }
            }

            if(nEdges==0) {

                optimizer.removeVertex(vPoint);
                vbValidPts3d[i] = false;
            }
            //else {
            //vbNotIncludedMP[i]=false;
            //nVertexMP++;
            //}
        }
    }

    // Only for IMU case that we have prior poses
    void MyOptimizer::addMedianDepthEdges(g2o::SparseOptimizer& optimizer, std::vector<EORB_SLAM::EvFramePtr>& vpEvFrames,
                                          const double scale, const int vfactorPose, const int maxId,
                                          const double thHuber, const bool bRobust, const bool fixedScale) {

        //const int numFrames = vpEvFrames.size();
        const int maxFrId = static_cast<int>(std::max(vpEvFrames.back()->mnId, vpEvFrames.size()));

        // Add the only median depth vertex
        VertexScale* VS = new VertexScale(scale);
        VS->setId(maxId+1);
        VS->setFixed(fixedScale);
        optimizer.addVertex(VS);

        int nEdges = 0;
        EvFramePtr pIniFrame = vpEvFrames[0];
        cv::Mat Rwc0 = pIniFrame->GetRotationInverse();
        cv::Mat twc0 = pIniFrame->GetCameraCenter();

        vector<cv::KeyPoint> vIniPts2d = pIniFrame->getAllUndistKPtsMono();

        for (size_t i = 0; i < vIniPts2d.size(); i++) {

            nEdges = 0;

            cv::KeyPoint currKpt = vIniPts2d[i];
            cv::Mat Xc0 = pIniFrame->mpCamera->unprojectMat(currKpt.pt);
            cv::Mat cvXw = Rwc0 * Xc0 + twc0;
            Eigen::Vector3d Xw = Converter::toVector3d(cvXw);


            for (size_t j = 0; j < vpEvFrames.size(); j++) {

                EvFramePtr pFri = vpEvFrames[j];

                int currMchIdx = i;

                if (j > 0) {
                    currMchIdx = pFri->getMatches(i);
                    if (currMchIdx < 0)
                        continue;
                }

                nEdges++;

                const int frId = vfactorPose * maxFrId + static_cast<int>(pFri->mnId);

                Eigen::Matrix<double, 2, 1> obs;
                cv::KeyPoint kpUn = pFri->getUndistKPtMono(currMchIdx);
                obs << kpUn.pt.x, kpUn.pt.y;

                const float &invSigma2 = pFri->getKPtInvLevelSigma2(currMchIdx);

                EdgePoseMedDepth *emd = new EdgePoseMedDepth(Xw);

                emd->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(frId)));
                emd->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(maxId + 1)));
                emd->setMeasurement(obs);
                emd->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                if (bRobust) {
                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    emd->setRobustKernel(rk);
                    rk->setDelta(thHuber);
                }

                optimizer.addEdge(emd);
            }

            //if(nEdges==0) {
            //    optimizer.removeVertex(vPoint);
            //    vbValidPts3d[i] = false;
            //}
        }
    }

    /* -------------------------------------------------------------------------------------------------------------- */

    void MyOptimizer::recoverPose(const g2o::SparseOptimizer &optimizer, std::vector<EORB_SLAM::EvFramePtr> &vpEvFrames,
                                  const int vfactorPose) {

        const int maxFrId = static_cast<int>(std::max(vpEvFrames.back()->mnId, vpEvFrames.size()));

        for (int i = 0; i < vpEvFrames.size(); i++) {

            EvFramePtr pFri = vpEvFrames[i];
            const int frId = vfactorPose * maxFrId + pFri->mnId;

            const g2o::VertexSE3Expmap *vSE3_recov = static_cast<const g2o::VertexSE3Expmap *>(optimizer.vertex(frId));
            g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
            cv::Mat pose = ORB_SLAM3::Converter::toCvMat(SE3quat_recov);

            pFri->SetPose(pose);
        }
    }

    void MyOptimizer::recoverPoseIMU(const g2o::SparseOptimizer &optimizer, std::vector<EORB_SLAM::EvFramePtr> &vpEvFrames,
                                     InitInfoImuPtr& pInfoImu, const int vfactorPose, const int vfactorBias,
                                     const int vfactorGS, const bool singleBias) {

        const int maxFrId = static_cast<int>(std::max(vpEvFrames.back()->mnId, vpEvFrames.size()));

        for (int i = 0; i < vpEvFrames.size(); i++) {

            EvFramePtr pFri = vpEvFrames[i];
            const int currId = pFri->mnId;

            const VertexPose *VP = static_cast<const VertexPose *>(optimizer.vertex(vfactorPose*maxFrId+currId));
            const VertexVelocity *VV = static_cast<const VertexVelocity*>(optimizer.vertex((vfactorPose+1)*maxFrId+currId));

            cv::Mat Tcw = Converter::toCvSE3(VP->estimate().Rcw[0], VP->estimate().tcw[0]);
            pFri->SetPose(Tcw);
            pFri->SetVelocity(Converter::toCvMat(VV->estimate()));

            if (!singleBias) {

                const VertexGyroBias* VG = static_cast<const VertexGyroBias*>(optimizer.vertex(vfactorBias*maxFrId+currId));
                const VertexAccBias* VA = static_cast<const VertexAccBias*>(optimizer.vertex((vfactorBias+1)*maxFrId+currId));

                Vector6d vb;
                vb << VG->estimate(), VA->estimate();
                IMU::Bias b (vb[3],vb[4],vb[5],vb[0],vb[1],vb[2]);

                pFri->SetNewBias(b);
            }
        }

        if (singleBias) {

            const VertexGyroBias* VG = static_cast<const VertexGyroBias*>(optimizer.vertex(vfactorBias*maxFrId+0));
            const VertexAccBias* VA = static_cast<const VertexAccBias*>(optimizer.vertex(vfactorBias*maxFrId+1));

            Vector6d vb;
            vb << VG->estimate(), VA->estimate();
            IMU::Bias b(vb[3],vb[4],vb[5],vb[0],vb[1],vb[2]);

            pInfoImu->setLastImuBias(b);
        }

        if (vfactorGS >= 0) {

            const VertexGDir* VGDir = static_cast<const VertexGDir*>(optimizer.vertex(vfactorGS*maxFrId+0));
            const VertexScale* VS = static_cast<const VertexScale*>(optimizer.vertex(vfactorGS*maxFrId+1));

            // Rgw = Rwg^T
            cv::Mat orbRwg = Converter::toCvMat(VGDir->estimate().Rwg);
            pInfoImu->setLastRwg(orbRwg);
            pInfoImu->setLastScale(VS->estimate());
        }
    }

    void MyOptimizer::recoverMapPoints(const g2o::SparseOptimizer& optimizer, std::vector<cv::Point3f>& vPts3d,
                                       std::vector<int>& vbValidPts3d, const int maxId) {

        for (int i = 0; i < vPts3d.size(); i++) {

            if (vbValidPts3d[i] < 0)
                continue;

            const int id = maxId + i + 1;

            if (optimizer.vertex(id)) {

                const g2o::VertexSBAPointXYZ *vPtXyz = static_cast<const g2o::VertexSBAPointXYZ *>(optimizer.vertex(id));
                vPts3d[i] = cv::Point3f(Converter::toCvMat(vPtXyz->estimate()).clone());
            }
        }
    }

} // EORB_SLAM


