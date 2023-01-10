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

#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv/cv.h>


namespace ORB_SLAM3
{
#define DEF_STD_GAUSS_KER 2
#define DEF_DESC_LEN 32
#define DEF_IMAGE_WIDTH 752

struct ORBxParams {

    ORBxParams() : nfeatures(0), scaleFactor(1), nlevels(1), iniThFAST(10), minThFAST(7), edgeTh(19), patchSize(31) {}
    ORBxParams(int _nfeatures, float _scaleFactor, int _nlevels, int _iniThFAST, int _minThFAST);
    ORBxParams(int _nfeatures, float _scaleFactor, int _nlevels, int _iniThFAST, int _minThFAST, int _edgeTh, const cv::Size& imSz);

    int nfeatures;
    float scaleFactor;
    int nlevels;
    int iniThFAST;
    int minThFAST;
    int edgeTh;
    int patchSize;
    cv::Size imSize;
};

class ExtractorNode
{
public:
    ExtractorNode():bNoMore(false){}

    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

    std::vector<cv::KeyPoint> vKeys;
    cv::Point2i UL, UR, BL, BR;
    std::list<ExtractorNode>::iterator lit;
    bool bNoMore;
};

class ORBextractor
{
public:
    
    //enum {HARRIS_SCORE=0, FAST_SCORE=1 };

    ORBextractor(const ORBxParams& paramsORB);

    ~ORBextractor()= default;

    // Compute the ORB features and descriptors on an image.
    // ORB are dispersed on the image using an octree.
    // Mask is ignored in the current implementation.
    int operator()( cv::InputArray _image, cv::InputArray _mask,
                    std::vector<cv::KeyPoint>& _keypoints,
                    cv::OutputArray _descriptors, std::vector<int> &vLappingArea);

    // Only detect ORB features (no descriptors)
    int operator()( cv::InputArray _image, cv::InputArray _mask,
                    std::vector<cv::KeyPoint>& _keypoints, std::vector<int> &vLappingArea);

    int inline GetLevels() const{
        return nlevels;}

    float inline GetScaleFactor() const{
        return static_cast<float>(scaleFactor);}

    std::vector<float> inline GetScaleFactors(){
        return mvScaleFactor;
    }

    std::vector<float> inline GetInverseScaleFactors(){
        return mvInvScaleFactor;
    }

    std::vector<float> inline GetScaleSigmaSquares(){
        return mvLevelSigma2;
    }

    std::vector<float> inline GetInverseScaleSigmaSquares(){
        return mvInvLevelSigma2;
    }

    std::vector<cv::Mat> mvImagePyramid;

    int GetNumFeatures() const { return nfeatures; }

    void AssignKPtLevelByBestDesc(const cv::Mat& refDescs, const cv::Mat& trackedImage, std::vector<cv::KeyPoint>& trackedKPts);
    void ComputeTrackedKPtsDesc(const cv::Mat& trackedImage, const std::vector<cv::KeyPoint>& trackedKPts, cv::Mat& refDescs);

protected:

    void ComputePyramid(const cv::Mat& image);
    void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);    
    static std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
            const int &maxX, const int &minY, const int &maxY, const int &N, const int &level, const int& nFeatures);

    void ComputeKeyPointsOld(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);
    std::vector<cv::Point> pattern;

    int nfeatures;
    double scaleFactor;
    int nlevels;
    int iniThFAST;
    int minThFAST;

    std::vector<int> mnFeaturesPerLevel;

    std::vector<int> umax;

    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;    
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;
};

} //namespace ORB_SLAM

#endif

