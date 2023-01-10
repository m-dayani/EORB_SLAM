//
// Created by root on 3/1/21.
//

#ifndef ORB_SLAM3_MYOPTIMIZER_H
#define ORB_SLAM3_MYOPTIMIZER_H

#include "Optimizer.h"


namespace EORB_SLAM {

    typedef std::map<double, std::pair<ORB_SLAM3::KeyFrame*, int>> PoseConstraintList;
    typedef std::map<double, std::pair<ORB_SLAM3::KeyFrame*, int>>::const_iterator PoseConstraintListIter;
    typedef std::map<ORB_SLAM3::KeyFrame*, std::pair<int, g2o::Sim3>> EventVertexPoseList;
    typedef std::map<ORB_SLAM3::MapPoint*, std::pair<unsigned long, g2o::Sim3>> EventVertexMPList;

    class MyOptimizer : public ORB_SLAM3::Optimizer {
    public:

        int static PoseOptimization(ORB_SLAM3::Frame* pFrame, EvFrame& currEvFrame);

        void static StructureBA(std::vector<ORB_SLAM3::KeyFrame*>& vpKFs, int nIterations = 5, bool structOnly = false,
                                bool *pbStopFlag= nullptr, unsigned long nLoopKF= 0, bool bRobust = true);

        // Event-Image bundle adjustment (Synch. Event Tracker)
        void static BundleAdjustment(const std::vector<ORB_SLAM3::KeyFrame*> &vpKF, const std::vector<ORB_SLAM3::MapPoint*> &vpMP,
                                     const std::vector<ORB_SLAM3::KeyFrame*> &vpEvRefKF, int nIterations = 5,
                                     bool *pbStopFlag=nullptr, unsigned long nLoopKF=0, bool bRobust = true);

        void static GlobalBundleAdjustment(ORB_SLAM3::Map* pMap, const std::vector<ORB_SLAM3::KeyFrame*> &vpEvKFs,
                int nIterations=5, bool *pbStopFlag=nullptr, unsigned long nLoopKF=0, bool bRobust = true);


        void static getLocalOptKFs(ORB_SLAM3::KeyFrame* pKF, std::vector<ORB_SLAM3::KeyFrame*> &vpOptimizableKFs, int maxSz, bool bMergeMaps);
        void static getCovisOptKFs(ORB_SLAM3::KeyFrame* pKF, std::list<ORB_SLAM3::KeyFrame*> &lpOptVisKFs, int maxSz, bool bMergeMaps);
        void static getFixedKFs(ORB_SLAM3::KeyFrame* pKF, std::vector<ORB_SLAM3::KeyFrame*> &vpOptimizableKFs,
                                const std::list<ORB_SLAM3::MapPoint*>& lLocalMapPoints,
                                std::list<ORB_SLAM3::KeyFrame*> &lFixedKeyFrames, int maxSz, bool bMergeMaps);
        void static getLocalOptMPts(ORB_SLAM3::KeyFrame* pKF, const std::vector<ORB_SLAM3::KeyFrame*> &vpOptimizableKFs,
                                    const std::list<ORB_SLAM3::KeyFrame*> &lpOptVisKFs,
                                    std::list<ORB_SLAM3::MapPoint*>& lLocalMapPoints, bool bMergeMaps);
        bool static checkKeyFrameIntegrity(std::vector<ORB_SLAM3::KeyFrame*> &vpOptimizableKFs,
                                           std::list<ORB_SLAM3::KeyFrame*> &lpOptVisKFs,
                                           std::list<ORB_SLAM3::KeyFrame*> &lFixedKeyFrames, ulong& maxId);

        void static LocalInertialBA(ORB_SLAM3::KeyFrame* pKF, bool *pbStopFlag, ORB_SLAM3::Map *pMap, int& num_fixedKF,
                                    int& num_OptKF, int& num_MPs, int& num_edges, int nMaxOpt = 10, int nIter = 10,
                                    bool bLarge = false, bool bRecInit = false, bool bMergeMaps = false);

        void static LocalBundleAdjustment(ORB_SLAM3::KeyFrame* pKF, bool *pbStopFlag, ORB_SLAM3::Map *pMap, int& num_fixedKF,
                                          int& num_OptKF, int& num_MPs, int& num_edges);

        void static LocalBundleAdjustment22(ORB_SLAM3::KeyFrame* pKF, bool *pbStopFlag, ORB_SLAM3::Map *pMap, int& num_fixedKF,
                                            int& num_OptKF, int& num_MPs, int& num_edges);



        // Merges and Optimizes all poses from Event Atlas & Image Atlas
        void static MergeVisualEvent(ORB_SLAM3::Atlas* pAtlasORB, const std::vector<ORB_SLAM3::KeyFrame*> &vpEvRefKF,
                                     ORB_SLAM3::Atlas* pAtlasMerge, int nIterations = 5, bool bRobust = true);


        double static OptimInitV(std::vector<EORB_SLAM::EvFramePtr>& vpEvFrames, std::vector<cv::Point3f>& vPts3d,
                               std::vector<int>& vbValidPts3d, int nIterations = 10, bool bRobust = true,
                               bool addPrior = false);

        double static OptimInitVI(std::vector<EORB_SLAM::EvFramePtr>& vpEvFrames, std::vector<cv::Point3f>& vPts3d,
                                std::vector<int>& vbValidPts3d, double& medDepth,
                                InitInfoImuPtr& pInfoImu, int nIterations = 10, bool bRobust = true,
                                bool addPrior = false, bool addInertial = true, bool addGDirScale = false,
                                bool singleBias = true);

        // Optimize Sim3 motion between tiny L1 event frames (for MC-Image recovery)
        double static optimizeSim3L1(std::vector<EvFramePtr>& vpKfs, double& sc0, float th2, int nIter, bool fixedScale = false);

        // Minimize matched KPts distance using a 2D transformation (SE2)
        static void optimize2D(const vector<EORB_SLAM::EvFrame>& vEvFrames, cv::Mat& paramsTrans2d,
                               int maxIterations = 10, bool verbose = false);

    protected:

        static void addPoseVertices(g2o::SparseOptimizer& optimizer, const vector<EORB_SLAM::EvFramePtr>& vpEvFrames,
                                    int &vfactor, bool addInertial = false);

        static void addBiasIMU(g2o::SparseOptimizer& optimizer, const vector<EORB_SLAM::EvFramePtr>& vpEvFrames,
                               const InitInfoImuPtr& pInfoImu, int &vfactor, bool singleBias = true);

        static void addScaleGDirIMU(g2o::SparseOptimizer& optimizer, const InitInfoImuPtr& pInfoImu,
                                    int maxFrId, int &vfactor);

        static void addPriorEdges(g2o::SparseOptimizer& optimizer, const vector<EORB_SLAM::EvFramePtr>& vpEvFrames,
                                  int vfactorPose, int &vfactor, double thHuber, bool robust = false);

        static void addPriorImuEdges(g2o::SparseOptimizer& optimizer, const vector<EORB_SLAM::EvFramePtr>& vpEvFrames,
                                     int vfactorPose, int vfactorBias, double thHuber, bool robust = false,
                                     bool singleBias = true);

        static void addImuEdges(g2o::SparseOptimizer& optimizer, const vector<EORB_SLAM::EvFramePtr>& vpEvFrames,
                                const InitInfoImuPtr& pInfoImu, int vfactorPose, int vfactorBias, int vfactorGS,
                                double thHuber, bool robust = false, bool singleBias = true);

        static void addMapPointEdges(g2o::SparseOptimizer& optimizer, std::vector<EORB_SLAM::EvFramePtr>& vpEvFrames,
                                     std::vector<cv::Point3f>& vPts3d, std::vector<int>& vbValidPts3d,
                                     int vfactorPose, int maxId, double thHuber, bool bRobust = true, bool inertial = false);

        static void addMedianDepthEdges(g2o::SparseOptimizer& optimizer, std::vector<EORB_SLAM::EvFramePtr>& vpEvFrames,
                                        double scale, int vfactorPose, int maxId, double thHuber, bool bRobust = true,
                                        bool fixedScale = false);

        static int addEventVertexPose(g2o::SparseOptimizer& optimizer, EventVertexPoseList& mEvVertex, g2o::Sim3& Twowe,
                ORB_SLAM3::KeyFrame* pEvKF, PoseConstraintList& orbConstraints, unsigned long currId, float thTs = 0.01);

        //TODO: Maybe add some IMU Gyro & Acc random walk constraints

        /* ---------------------------------------------------------------------------------------------------------- */

        static void recoverPose(const g2o::SparseOptimizer& optimizer, std::vector<EORB_SLAM::EvFramePtr>& vpEvFrames,
                                int vfactorPose);

        static void recoverPoseIMU(const g2o::SparseOptimizer& optimizer, std::vector<EORB_SLAM::EvFramePtr>& vpEvFrames,
                                   InitInfoImuPtr& pInfoImu, int vfactorPose, int vfactorBias, int vfactorGS, bool singleBias = true);

        static void recoverMapPoints(const g2o::SparseOptimizer& optimizer, std::vector<cv::Point3f>& vPts3d,
                                     std::vector<int>& vbValidPts3d, int maxId);

        /* ---------------------------------------------------------------------------------------------------------- */

        static void stitchPoseConstraints(const std::vector<ORB_SLAM3::KeyFrame*>& vpKFs, g2o::SparseOptimizer& optimizer,
                ORB_SLAM3::Atlas* pAtlasMerge, map<ORB_SLAM3::KeyFrame*, unsigned long>& mAllVertices,
                PoseConstraintList& mLastSearchConsts, unsigned long& currId, double& minTs, double& maxTs,
                float thTs = 0.01, bool fixedRef = false);

        static int findNearestPose(const PoseConstraintList& mPoseList, ORB_SLAM3::KeyFrame* pKF,
                PoseConstraintListIter& firstConst, PoseConstraintListIter& secondConst, float thTs = 0.01);
    };

} // EORB_SLAM

#endif //ORB_SLAM3_MYOPTIMIZER_H
