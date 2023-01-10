//
// Created by root on 2/24/21.
//

#ifndef ORB_SLAM3_MIXEDMATCHER_H
#define ORB_SLAM3_MIXEDMATCHER_H

#include "ORBmatcher.h"
#include "MixedFrame.h"
#include "MixedKeyFrame.h"


namespace EORB_SLAM {

    class MixedMatcher : public ORB_SLAM3::ORBmatcher {
    public:
        explicit MixedMatcher(float nnratio=0.6, bool checkOri=true);

        // Search matches between Frame keypoints and projected MapPoints. Returns number of matches
        // Used to track the local map (Tracking)
        int SearchByProjection(MixedFrame &F, const std::vector<ORB_SLAM3::MapPoint*> &vpMapPoints,
                float th=3, bool bFarPoints = false, float thFarPoints = 50.0f);

        // Project MapPoints tracked in last frame into the current frame and search matches.
        // Used to track from previous frame (Tracking)
        int SearchByProjection(MixedFrame &CurrentFrame, const MixedFrame &LastFrame, float th, bool bMono);

        // Project MapPoints seen in KeyFrame into the Frame and search matches.
        // Used in relocalisation (Tracking)
        int SearchByProjection(MixedFrame &CurrentFrame, MixedKeyFrame* pKF,
                const std::set<ORB_SLAM3::MapPoint*> &sAlreadyFound, float th, int ORBdist);

        // Project MapPoints using a Similarity Transformation and search matches.
        // Used in loop detection (Loop Closing)
        int SearchByProjection(MixedKeyFrame* pKF, const cv::Mat& Scw, const std::vector<ORB_SLAM3::MapPoint*> &vpPoints,
                std::vector<ORB_SLAM3::MapPoint*> &vpMatched, int th, float ratioHamming=1.0);

        // Project MapPoints using a Similarity Transformation and search matches.
        // Used in Place Recognition (Loop Closing and Merging)
        int SearchByProjection(MixedKeyFrame* pKF, const cv::Mat& Scw, const std::vector<ORB_SLAM3::MapPoint*> &vpPoints,
                const std::vector<ORB_SLAM3::KeyFrame*> &vpPointsKFs, std::vector<ORB_SLAM3::MapPoint*> &vpMatched,
                std::vector<ORB_SLAM3::KeyFrame*> &vpMatchedKF, int th, float ratioHamming=1.0);

        // Search matches between MapPoints in a KeyFrame and ORB in a Frame.
        // Brute force constrained to ORB that belong to the same vocabulary node (at a certain level)
        // Used in Relocalisation and Loop Detection
        int SearchByBoW(MixedKeyFrame *pKF, MixedFrame &F, std::vector<ORB_SLAM3::MapPoint*> &vpMapPointMatches);
        int SearchByBoW(MixedKeyFrame *pKF1, MixedKeyFrame* pKF2, std::vector<ORB_SLAM3::MapPoint*> &vpMatches12);

        // Matching for the Map Initialization (only used in the monocular case)
        int SearchForInitialization(MixedFrame &F1, MixedFrame &F2, std::vector<cv::Point2f> &vbPrevMatched,
                std::vector<int> &vnMatches12, int windowSize=10);

        // Matching to triangulate new MapPoints. Check Epipolar Constraint.
        int SearchForTriangulation(MixedKeyFrame *pKF1, MixedKeyFrame* pKF2, const cv::Mat& F12,
                                   std::vector<pair<size_t, size_t> > &vMatchedPairs, bool bOnlyStereo, bool bCoarse = false);

        // Project MapPoints into KeyFrame and search for duplicated MapPoints.
        int Fuse(MixedKeyFrame* pKF, const vector<ORB_SLAM3::MapPoint *> &vpMapPoints, float th=3.0, bool bRight = false);

        // Project MapPoints into KeyFrame using a given Sim3 and search for duplicated MapPoints.
        int Fuse(MixedKeyFrame* pKF, const cv::Mat& Scw, const std::vector<ORB_SLAM3::MapPoint*> &vpPoints,
                float th, vector<ORB_SLAM3::MapPoint *> &vpReplacePoint);

    protected:
        //static bool isORBMapPoint(ORB_SLAM3::MapPoint *pMP);
    };

} // EORB_SLAM

#endif //ORB_SLAM3_MIXEDMATCHER_H
