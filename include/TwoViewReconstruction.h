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

#ifndef TwoViewReconstruction_H
#define TwoViewReconstruction_H

#include<opencv2/opencv.hpp>

#include <unordered_set>

namespace ORB_SLAM3 {

#define DEF_TH_HF_RATIO 0.45f
#define DEF_MIN_PARALLAX 1.0f
#define DEF_MIN_PARALLAX_CRT 0.99998
#define DEF_MIN_TRIANGULATED 50
#define DEF_TH_CHISQ_SCORE 5.991f
#define DEF_TH_CHISQ_F 3.841f
#define DEF_TH_R_MIN_GOOD 0.9f
#define DEF_TH_R_MAX_GOOD_F 0.7f
#define DEF_TH_R_MAX_GOOD_H 0.75f
#define DEF_TH_R_D_H 1.00001
#define DEF_MAX_RANSAC_ITER 200
#define DEF_2VR_SIGMA 1.f

struct Params2VR {
    float mThHFRatio = DEF_TH_HF_RATIO;

    float mThChiSqScore = DEF_TH_CHISQ_SCORE;
    float mThChiSqF = DEF_TH_CHISQ_F;

    float mDefMinParallax = DEF_MIN_PARALLAX;
    float mMinParallaxCRT = DEF_MIN_PARALLAX_CRT;

    int mMinTriangulated = DEF_MIN_TRIANGULATED;

    float mThMinGoodR = DEF_TH_R_MIN_GOOD;
    float mThMaxGoodR_F = DEF_TH_R_MAX_GOOD_F;
    float mThMaxGoodR_H = DEF_TH_R_MAX_GOOD_H;

    float mThRDHomo = DEF_TH_R_D_H;

    float mSigma = DEF_2VR_SIGMA;
    int mMaxRansacIter = DEF_MAX_RANSAC_ITER;
};

struct ReconstInfo {

    int mnBestGood = 0;
    int mnMinGood = 0;
    int mnSecondBest = 0;
    float mBestParallax = -1.f;
    float mSecondBestPar = -1.f;

    bool isHomography = true;

    float mHScore = 0.0;
    float mFScore = 0.0;
    float mHFRatio = 0.0;

    unsigned char mnSimilar = 0;
    std::vector<int> mvnGood = std::vector<int>(8,0);

    std::string print() const;
};

class TwoViewReconstruction
{
    typedef std::pair<int,int> Match;

public:

    // Fix the reference frame
    explicit TwoViewReconstruction(const cv::Mat& k, float sigma = DEF_2VR_SIGMA, int iterations = DEF_MAX_RANSAC_ITER);

    explicit TwoViewReconstruction(const cv::Mat& k, const Params2VR& params2VR);

    // Computes in parallel a fundamental matrix and a homography
    // Selects a model and tries to recover the motion and the structure from motion
    bool Reconstruct(const std::vector<cv::KeyPoint>& vKeys1, const std::vector<cv::KeyPoint>& vKeys2,
                     const std::vector<int> &vMatches12, cv::Mat &R21, cv::Mat &t21,
                     std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated);

    // Sloppy version of above method (returns something no matter if is true or not!)
    static ReconstInfo dummyReconstInfo;
    bool Reconstruct(const std::vector<cv::KeyPoint>& vKeys1, const std::vector<cv::KeyPoint>& vKeys2,
            const std::vector<int> &vMatches12, std::vector<cv::Mat>& R21, std::vector<cv::Mat>& t21,
            std::vector<std::vector<cv::Point3f>>& vP3D, std::vector<std::vector<bool>>& vbTriangulated,
            std::vector<bool> &vbTransInliers, ReconstInfo& reconstInfo = dummyReconstInfo);

    static void Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);

    Params2VR mParams2VR;

private:

    void FindHomography(std::vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21);
    void FindFundamental(std::vector<bool> &vbInliers, float &score, cv::Mat &F21);

    cv::Mat ComputeH21(const std::vector<cv::Point2f> &vP1, const std::vector<cv::Point2f> &vP2);
    cv::Mat ComputeF21(const std::vector<cv::Point2f> &vP1, const std::vector<cv::Point2f> &vP2);

    float CheckHomography(const cv::Mat &H21, const cv::Mat &H12, std::vector<bool> &vbMatchesInliers, float sigma);

    float CheckFundamental(const cv::Mat &F21, std::vector<bool> &vbMatchesInliers, float sigma);

    bool ReconstructF(std::vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated,
                      float minParallax, int minTriangulated);

    bool ReconstructH(std::vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated,
                      float minParallax, int minTriangulated);

    bool resolveReconstStatF(ReconstInfo& reconstInfo, const int& N,
            const float& minParallax, const int& minTriangulated) const;

    bool ReconstructF(std::vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
            std::vector<cv::Mat>& R21, std::vector<cv::Mat>& t21, std::vector<std::vector<cv::Point3f>>& vP3D,
            std::vector<std::vector<bool>>& vbTriangulated, std::vector<bool> &vbTransInliers, float minParallax,
            int minTriangulated, ReconstInfo& reconstInfo = dummyReconstInfo);

    bool resolveReconstStatH(ReconstInfo& reconstInfo, const int& N,
                             const float& minParallax, const int& minTriangulated) const;

    bool ReconstructH(std::vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
            std::vector<cv::Mat>& R21, std::vector<cv::Mat>& t21, std::vector<std::vector<cv::Point3f>>& vP3D,
            std::vector<std::vector<bool>>& vbTriangulated, std:: vector<bool> &vbTransInliers, float minParallax,
            int minTriangulated, ReconstInfo& reconstInfo = dummyReconstInfo);

    void Normalize(const std::vector<cv::KeyPoint> &vKeys, std::vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T);


    int CheckRT(const cv::Mat &R, const cv::Mat &t, const std::vector<cv::KeyPoint> &vKeys1, const std::vector<cv::KeyPoint> &vKeys2,
                       const std::vector<Match> &vMatches12, std::vector<bool> &vbInliers,
                       const cv::Mat &K, std::vector<cv::Point3f> &vP3D, float th2, std::vector<bool> &vbGood, float &parallax);

    void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t);


    // Keypoints from Reference Frame (Frame 1)
    std::vector<cv::KeyPoint> mvKeys1;

    // Keypoints from Current Frame (Frame 2)
    std::vector<cv::KeyPoint> mvKeys2;

    // Current Matches from Reference to Current
    std::vector<Match> mvMatches12;
    std::vector<bool> mvbMatched1;

    // Calibration
    cv::Mat mK;

    // Standard Deviation and Variance
    float mSigma, mSigma2;

    // Ransac max iterations
    int mMaxIterations;

    // Ransac sets
    std::vector<std::vector<size_t> > mvSets;
};

} //namespace ORB_SLAM

#endif // TwoViewReconstruction_H
