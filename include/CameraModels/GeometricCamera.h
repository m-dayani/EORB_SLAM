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

#ifndef CAMERAMODELS_GEOMETRICCAMERA_H
#define CAMERAMODELS_GEOMETRICCAMERA_H

#include <opencv2/core/core.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include <utility>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/assume_abstract.hpp>

#include <Eigen/Geometry>

#include "TwoViewReconstruction.h"


namespace ORB_SLAM3 {
    class GeometricCamera {

        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int version)
        {
            ar & mnId;
            ar & mnType;
            ar & mvParameters;
        }


    public:
        GeometricCamera() : tvr(nullptr) {

            mnId=nNextId++;

            mK = cv::Mat::eye(3, 3, CV_32F);
            mD = cv::Mat::zeros(4, 1, CV_32F);
            mR = cv::Mat::eye(3, 3, CV_32F);
            mP = mK.clone();
        }

        explicit GeometricCamera(std::vector<float>  _vParameters) : mvParameters(std::move(_vParameters)), tvr(nullptr) {

            assert(mvParameters.size() >= 4);

            mnId=nNextId++;

            mK = (cv::Mat_<float>(3, 3)
                    << mvParameters[0], 0.f, mvParameters[2], 0.f, mvParameters[1], mvParameters[3], 0.f, 0.f, 1.f);
            mD = cv::Mat::zeros(4, 1, CV_32F);
            mR = cv::Mat::eye(3, 3, CV_32F);
            mP = mK.clone();
        }

        GeometricCamera(std::vector<float>  _vParameters, TwoViewReconstruction* _tvr) :
                mvParameters(std::move(_vParameters)), tvr(_tvr) {

            assert(mvParameters.size() >= 4);

            mnId=nNextId++;
            
            mK = (cv::Mat_<float>(3, 3)
                    << mvParameters[0], 0.f, mvParameters[2], 0.f, mvParameters[1], mvParameters[3], 0.f, 0.f, 1.f);
            mD = cv::Mat::zeros(4, 1, CV_32F);
            mR = cv::Mat::eye(3, 3, CV_32F);
            mP = mK.clone();
        }

        virtual ~GeometricCamera() = default;

        virtual cv::Point2f project(const cv::Point3f &p3D) = 0;
        virtual cv::Point2f project(const cv::Mat& m3D) = 0;
        virtual Eigen::Vector2d project(const Eigen::Vector3d & v3D) = 0;
        virtual cv::Mat projectMat(const cv::Point3f& p3D) = 0;

        virtual float uncertainty2(const Eigen::Matrix<double,2,1> &p2D) = 0;

        virtual cv::Point3f unproject(const cv::Point2f &p2D) = 0;
        virtual cv::Mat unprojectMat(const cv::Point2f &p2D) = 0;

        virtual cv::Mat projectJac(const cv::Point3f &p3D) = 0;
        virtual Eigen::Matrix<double,2,3> projectJac(const Eigen::Vector3d& v3D) = 0;

        virtual cv::Mat unprojectJac(const cv::Point2f &p2D) = 0;

        virtual bool ReconstructWithTwoViews(const std::vector<cv::KeyPoint>& vKeys1, const std::vector<cv::KeyPoint>& vKeys2,
                const std::vector<int> &vMatches12, cv::Mat &R21, cv::Mat &t21, std::vector<cv::Point3f> &vP3D,
                std::vector<bool> &vbTriangulated) = 0;

        virtual bool ReconstructWithTwoViews(const std::vector<cv::KeyPoint>& vKeys1,
                const std::vector<cv::KeyPoint>& vKeys2, const std::vector<int> &vMatches12,
                std::vector<cv::Mat> &R21, std::vector<cv::Mat> &t21, std::vector<std::vector<cv::Point3f>> &vP3D,
                std::vector<std::vector<bool>> &vbTriangulated, std::vector<bool> &vbTransInliers,
                ReconstInfo& reconstInfo) = 0;

        virtual std::vector<cv::KeyPoint> UndistortKeyPoints(const std::vector<cv::KeyPoint>& vKPts) = 0;

        /*virtual bool ReconstructWithTwoViews(const std::vector<cv::KeyPoint>& vKeys1, const std::vector<cv::KeyPoint>& vKeys2,
                const std::vector<int> &vMatches12, cv::Mat &R21, cv::Mat &t21, std::vector<cv::Point3f> &vP3D,
                std::vector<bool> &vbTriangulated, const Params2VR& params2Vr) = 0;
        virtual bool ReconstructWithTwoViews(const std::vector<cv::KeyPoint>& vKeys1, const std::vector<cv::KeyPoint>& vKeys2,
                const std::vector<int> &vMatches12, cv::Mat &R21, cv::Mat &t21, std::vector<cv::Point3f> &vP3D,
                std::vector<bool> &vbTriangulated, std::vector<bool> &vbTransInliers, const Params2VR& params2Vr) = 0;*/

        virtual cv::Mat toK() = 0;

        virtual bool epipolarConstrain(GeometricCamera* otherCamera, const cv::KeyPoint& kp1, const cv::KeyPoint& kp2,
                const cv::Mat& R12, const cv::Mat& t12, float sigmaLevel, float unc) = 0;

        float getParameter(const int i){return mvParameters[i];}
        void setParameter(const float p, const size_t i){mvParameters[i] = p;}

        size_t size(){return mvParameters.size();}

        virtual bool matchAndtriangulate(const cv::KeyPoint& kp1, const cv::KeyPoint& kp2, GeometricCamera* pOther,
                                 cv::Mat& Tcw1, cv::Mat& Tcw2, float sigmaLevel1, float sigmaLevel2,
                                 cv::Mat& x3Dtriangulated) = 0;

        unsigned int GetId() { return mnId; }

        unsigned int GetType() { return mnType; }

        const unsigned int CAM_PINHOLE = 0;
        const unsigned int CAM_FISHEYE = 1;

        static long unsigned int nNextId;

    protected:
        std::vector<float> mvParameters;

        cv::Mat mK, mD, mR, mP;

        unsigned int mnId{};

        unsigned int mnType{};

        TwoViewReconstruction* tvr;
    };
}


#endif //CAMERAMODELS_GEOMETRICCAMERA_H
