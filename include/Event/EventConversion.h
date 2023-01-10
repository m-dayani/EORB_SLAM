/**
* 
*/


#ifndef EVENT_CONVERSION_H
#define EVENT_CONVERSION_H

#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "EventData.h"
#include "DataStore.h"
#include "MyCalibrator.h"

#include "Converter.h"
#include "GeometricCamera.h"


namespace EORB_SLAM
{
#ifndef DEF_PATCH_SIZE_STD
#define DEF_PATCH_SIZE_STD 30
#endif

    //void im2sensor_pinhole(const float& x, const float& y, const cv::Mat& K, float& xs, float& ys);
    //void sensor2im_pinhole(const float& xs, const float& ys, const cv::Mat& K, float& x, float& y);

    void breakFloatCoords(float X, float Y, int &x, int &y, float &xRes, float &yRes);
    //bool isInImage(int x, int y, int imWidth, int imHeight);
    float exp_XY2f(float x, float y, float sig2);

    void normalizeImage(const cv::Mat& inputIm, cv::Mat& outputIm, float maxVal, float minVal);

    class EvImConverter
    {
    public:
        //EvImConverter() {}

        static float measureImageFocus(const cv::Mat& image);
        static float measureImageFocusLocal(const cv::Mat& image, bool avg = true);
        static float measureImageFocusGlobal(const cv::Mat& image);

        static float imageMeanLocal(const cv::Mat& image, bool avg = true);
        static float imageMean(const cv::Mat& image, bool global = false, bool avg = true);

        static cv::Mat ev2im(const std::vector<EventData> &vEvData, unsigned imWidth, unsigned imHeight,
                             bool pol = false, bool normalized = true);

        static cv::Mat ev2im_gauss(const std::vector<EventData> &vEvData, unsigned imWidth, unsigned imHeight,
                                   float sigma, bool pol = false, bool normalized = true);

        /*static cv::Mat ev2im_gauss_gray(const std::vector<EventData> &vEvData, unsigned imWidth, unsigned imHeight,
                                        float sigma, bool pol = false, bool normalized = true);*/

        static cv::Mat ev2mci_gg_f(const std::vector<EventData> &vEvData, ORB_SLAM3::GeometricCamera* pCamera,
                                   const cv::Mat& Tcw, float medDepth, unsigned imWidth,
                                   unsigned imHeight, float imSigma, bool pol = false, bool normalized = true);

        static Eigen::Matrix<double, 1, 6> ev2mci_gg_f_jac(const std::vector<EventData> &vEvData,
                ORB_SLAM3::GeometricCamera* pCamera, const g2o::VertexSE3Expmap* vSE3, float medDepth, unsigned imWidth,
                unsigned imHeight, float imSigma, bool pol = false, bool global = false);

        static cv::Mat ev2mci_gg_f(const std::vector<EventData> &vEvData, ORB_SLAM3::GeometricCamera* pCamera,
                                   const cv::Mat& params2D, unsigned imWidth, unsigned imHeight, float sigma,
                                   bool pol = false, bool normalized = true);

        static cv::Mat ev2mci_gg_f(const std::vector<EventData> &vEvData, ORB_SLAM3::GeometricCamera* pCamera,
                                   const cv::Mat& Tcw, const MyDepthMap& depthMapObj, unsigned imWidth,
                                   unsigned imHeight, float imSigma, bool pol = false, bool normalized = true);
    protected:

        static float resolvePolarity(bool withPol, bool evPol);
    private:

    };

}// namespace EORB_SLAM

#endif // EVIMCONVERTER_H
