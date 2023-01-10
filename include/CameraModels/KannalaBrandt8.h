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

#ifndef CAMERAMODELS_KANNALABRANDT8_H
#define CAMERAMODELS_KANNALABRANDT8_H


#include <cassert>
#include <vector>
#include <opencv2/core/core.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/vector.hpp>

#include "GeometricCamera.h"


namespace ORB_SLAM3 {

#define KB8_DEF_PRECISION 1e-6
#define KB8_DEF_TH_EPC 0.0001f
#define KB8_DEF_MIN_PLX 0.9998
#define KB8_DEF_CHISQ_COEF 5.991

    class KannalaBrandt8 final : public GeometricCamera {

    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        ar & boost::serialization::base_object<GeometricCamera>(*this);
        ar & const_cast<float&>(precision);
    }

    public:
        KannalaBrandt8();
        explicit KannalaBrandt8(const std::vector<float>& _vParameters);
        KannalaBrandt8(const std::vector<float>& _vParameters, float _precision);
        KannalaBrandt8(const std::vector<float>& _vParameters, TwoViewReconstruction* _tvr);
        KannalaBrandt8(const std::vector<float>& _vParameters, const cv::Mat& _cvRmat,
                const cv::Mat& _cvPmat, TwoViewReconstruction* _tvr);
        explicit KannalaBrandt8(KannalaBrandt8* pKannala);

        ~KannalaBrandt8() override {
            delete tvr;
            tvr = nullptr;
        }

        cv::Point2f project(const cv::Point3f &p3D);
        cv::Point2f project(const cv::Mat& m3D);
        Eigen::Vector2d project(const Eigen::Vector3d & v3D);
        cv::Mat projectMat(const cv::Point3f& p3D);

        float uncertainty2(const Eigen::Matrix<double,2,1> &p2D);

        cv::Point3f unproject(const cv::Point2f &p2D);
        cv::Mat unprojectMat(const cv::Point2f &p2D);

        cv::Mat projectJac(const cv::Point3f &p3D);
        Eigen::Matrix<double,2,3> projectJac(const Eigen::Vector3d& v3D);

        cv::Mat unprojectJac(const cv::Point2f &p2D);

        bool ReconstructWithTwoViews(const std::vector<cv::KeyPoint>& vKeys1, const std::vector<cv::KeyPoint>& vKeys2,
                const std::vector<int> &vMatches12, cv::Mat &R21, cv::Mat &t21, std::vector<cv::Point3f> &vP3D,
                std::vector<bool> &vbTriangulated) override;

        bool ReconstructWithTwoViews(const std::vector<cv::KeyPoint>& vKeys1,
                const std::vector<cv::KeyPoint>& vKeys2, const std::vector<int> &vMatches12,
                std::vector<cv::Mat> &R21, std::vector<cv::Mat> &t21, std::vector<std::vector<cv::Point3f>> &vP3D,
                std::vector<std::vector<bool>> &vbTriangulated, std::vector<bool> &vbTransInliers,
                ReconstInfo& reconstInfo) override;

        std::vector<cv::KeyPoint> UndistortKeyPoints(const std::vector<cv::KeyPoint>& vKPts) override;

        /*bool ReconstructWithTwoViews(const std::vector<cv::KeyPoint>& vKeys1, const std::vector<cv::KeyPoint>& vKeys2,
                const std::vector<int> &vMatches12, cv::Mat &R21, cv::Mat &t21, std::vector<cv::Point3f> &vP3D,
                std::vector<bool> &vbTriangulated, const Params2VR& params2Vr) override;
        bool ReconstructWithTwoViews(const std::vector<cv::KeyPoint>& vKeys1, const std::vector<cv::KeyPoint>& vKeys2,
                const std::vector<int> &vMatches12, cv::Mat &R21, cv::Mat &t21, std::vector<cv::Point3f> &vP3D,
                std::vector<bool> &vbTriangulated, std::vector<bool> &vbTransInliers, const Params2VR& params2Vr) override ;*/

        cv::Mat toK();

        bool epipolarConstrain(GeometricCamera* pCamera2, const cv::KeyPoint& kp1, const cv::KeyPoint& kp2,
                const cv::Mat& R12, const cv::Mat& t12, float sigmaLevel, float unc);

        float TriangulateMatches(GeometricCamera* pCamera2, const cv::KeyPoint& kp1, const cv::KeyPoint& kp2,
                const cv::Mat& R12, const cv::Mat& t12, float sigmaLevel, float unc, cv::Mat& p3D);

        std::vector<int> mvLappingArea;

        bool matchAndtriangulate(const cv::KeyPoint& kp1, const cv::KeyPoint& kp2, GeometricCamera* pOther,
                cv::Mat& Tcw1, cv::Mat& Tcw2, float sigmaLevel1, float sigmaLevel2,
                cv::Mat& x3Dtriangulated);

        friend std::ostream& operator<<(std::ostream& os, const KannalaBrandt8& kb);
        friend std::istream& operator>>(std::istream& is, KannalaBrandt8& kb);
    private:
        const float precision;

        //Parameters vector corresponds to
        //[fx, fy, cx, cy, k0, k1, k2, k3]

        void Triangulate(const cv::Point2f &p1, const cv::Point2f &p2, const cv::Mat &Tcw1, const cv::Mat &Tcw2,cv::Mat &x3D);
    };
}

//BOOST_CLASS_EXPORT_KEY(ORBSLAM2::KannalaBrandt8)

#endif //CAMERAMODELS_KANNALABRANDT8_H
