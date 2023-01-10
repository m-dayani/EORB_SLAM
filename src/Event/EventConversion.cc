/**
* Covert Events to Image representation
*/

#include "EventConversion.h"
#include "Pinhole.h"

using namespace std;


namespace EORB_SLAM
{

    /*void im2sensor_pinhole(const float& x, const float& y, const cv::Mat& K, float& xs, float& ys) {

        xs = (x - K.at<float>(0,2)) / K.at<float>(0,0);
        ys = (y - K.at<float>(1,2)) / K.at<float>(1,1);
    }

    void sensor2im_pinhole(const float& xs, const float& ys, const cv::Mat& K, float& x, float& y) {

        x = xs * K.at<float>(0,0) + K.at<float>(0,2);
        y = ys * K.at<float>(1,1) + K.at<float>(1,2);
    }*/

    inline float EvImConverter::resolvePolarity(const bool withPol, const bool evPol)
    {
        float res = (withPol && !evPol) ? -1.0f : 1.0f;
        return res;
    }

    void resolveMinMaxVals(const float newVal, const float oldMin, const float oldMax,
                           float& newMin, float& newMax)
    {
        if (newVal > oldMax)
            newMax = newVal;
        if (newVal < oldMin)
            newMin = newVal;
    }

    /*bool isInImage(const int x, const int y, const int imWidth, const int imHeight)
    {
        return (x >= 0 && x < imWidth) && (y >= 0 && y < imHeight);
    }*/

    int roundFloatCoord(const float currX)
    {
        return static_cast<int>(roundf(currX));
    }

    void breakFloatCoords(const float X, const float Y, int &x, int &y, float &xRes, float &yRes)
    {
        x = static_cast<int>(floor(X));
        xRes = X - float(x);
        y = static_cast<int>(floor(Y));
        yRes = Y - float(y);
    }

    float exp_XY2f(const float x, const float y, const float sig2)
    {
        float dd = powf(x, 2) + powf(y, 2);
        dd /= 2.0f * sig2;
        float val = expf(-dd) / (2.0f*float(CV_PI)*sig2);
        return val;
    }

    void normalizeImage(const cv::Mat& inputIm, cv::Mat& outputIm, const float maxVal, const float minVal) {

        float alpha = 255.f / (maxVal - minVal);
        float betta = - minVal * alpha;
        inputIm.convertTo(outputIm, CV_8UC1, alpha, betta);
    }

    float EvImConverter::measureImageFocus(const cv::Mat &image) {

        return measureImageFocusLocal(image, true);
    }

    float EvImConverter::measureImageFocusLocal(const cv::Mat &image, const bool avg) {

        float localStd = 0.f;
        int localStdCnt = 0;
        int patchSize = DEF_PATCH_SIZE_STD;
        int imRows = image.rows;
        int imCols = image.cols;
        int nCells = (imRows/patchSize) * (imCols/patchSize);
        vector<float> vLocalStd;
        vLocalStd.reserve(nCells);

        for (int i = 0; i < imRows; i += patchSize) {
            for (int j = 0; j < imCols; j += patchSize) {

                int maxRow = min(i+patchSize, imRows);
                int maxCol = min(j+patchSize, imCols);
                cv::Scalar imMean, imStd;
                cv::meanStdDev(image.rowRange(i,maxRow).colRange(j, maxCol), imMean, imStd);
                float currStd = imStd[0];
                localStd += currStd;
                localStdCnt++;
                vLocalStd.push_back(currStd);
            }
        }

        if (avg) {
            return localStd / static_cast<float>(localStdCnt);
        }
        else {
            sort(vLocalStd.begin(), vLocalStd.end());
            return vLocalStd[localStdCnt/2];
        }
    }

    float EvImConverter::measureImageFocusGlobal(const cv::Mat &image) {

        cv::Scalar imMean, imStd;
        cv::meanStdDev(image, imMean, imStd);

        return imStd[0];
    }

    float EvImConverter::imageMeanLocal(const cv::Mat &image, const bool avg) {

        float localAvg = 0.f;
        int localCnt = 0;
        int patchSize = DEF_PATCH_SIZE_STD;
        int imRows = image.rows;
        int imCols = image.cols;
        int nCells = (imRows/patchSize) * (imCols/patchSize);
        vector<float> vLocalAvg;
        vLocalAvg.reserve(nCells);

        for (int i = 0; i < imRows; i += patchSize) {
            for (int j = 0; j < imCols; j += patchSize) {

                int maxRow = min(i+patchSize, imRows);
                int maxCol = min(j+patchSize, imCols);
                cv::Scalar imMean, imStd;
                cv::meanStdDev(image.rowRange(i,maxRow).colRange(j, maxCol), imMean, imStd);
                float currMean = imMean[0];
                localAvg += currMean;
                localCnt++;
                vLocalAvg.push_back(currMean);
            }
        }

        if (avg) {
            return localAvg / static_cast<float>(localCnt);
        }
        else {
            sort(vLocalAvg.begin(), vLocalAvg.end());
            return vLocalAvg[localCnt/2];
        }
    }

    float EvImConverter::imageMean(const cv::Mat &image, const bool global, const bool avg) {

        if (global) {
            cv::Scalar m = cv::mean(image);
            return m[0];
        }
        else {
            return imageMeanLocal(image, avg);
        }
    }

    /**
    * If pol, discreminate different event polarities
    * else consider all the same
    * TODO: currently, this function makes a very noisy image for
    *	floating point event coordinates (Why????)
    * WARNING: Only use this with int coordinates indices.
    */
    cv::Mat EvImConverter::ev2im(const std::vector<EventData> &vEvData,
                                 unsigned imWidth, unsigned imHeight, bool pol, bool normalized)
    {
        float maxVal = -1000000.0;
        float minVal = 0.0;

        cv::Mat image = cv::Mat::zeros(imHeight, imWidth, CV_32FC1);
        size_t evCount = vEvData.size();

        //std::cout << "Converting " << evCount << " events" << std::endl;
        //std::cout << "image size: (" << imHeight << ',' << imWidth << ")\n";

        for (size_t i = 0; i < evCount; i++)
        {
            EventData evData = vEvData[i];

            float polSign = resolvePolarity(pol, evData.p);
            int pX = roundFloatCoord(evData.x);
            int pY = roundFloatCoord(evData.y);
            //cout << "round(x,y): (" << pX << ", " << pY << ")\n";
            if (!MyCalibrator::isInImage(pX, pY, imWidth, imHeight))
                continue;

            float newVal = image.at<float>(pY, pX) + (polSign*0.001f);
            // Beware of mat type saturation
            //if (checkCvTypeBounds(newVal, CV_32FC1))
            image.at<float>(pY, pX) = newVal;

            resolveMinMaxVals(newVal, minVal, maxVal, minVal, maxVal);

            //if (i == 0)
            //	evData.print();
        }
        if (normalized && maxVal > minVal) {

            normalizeImage(image, image, maxVal, minVal);
        }

        return image.clone();
    }

    //TODO: Use better min/max Vals (like above)
    cv::Mat EvImConverter::ev2im_gauss(const std::vector<EventData> &vEvData,
                                       const unsigned imWidth, const unsigned imHeight,
                                       const float sigma, const bool pol, const bool normalized)
    {
        float maxVal = -1000000.0;
        float minVal = 0.0;
        float sig2 = powf(sigma, 2);
        int lenHalfWin = static_cast<int>(ceil(sigma*3.0));

        cv::Mat image = cv::Mat::zeros(imHeight, imWidth, CV_32FC1);
        size_t evCount = vEvData.size();

        //std::cout << "Converting " << evCount << " events" << std::endl;
        //std::cout << "image size: (" << imHeight << ',' << imWidth << ")\n";
        //std::cout << "Window half length is: " << lenHalfWin << std::endl;

        for (unsigned int k = 0; k < evCount; k++)
        {
            EventData evData = vEvData[k];

            int xIdx, yIdx;
            float xRes, yRes;
            breakFloatCoords(evData.x, evData.y, xIdx, yIdx, xRes, yRes);

            for (int i = -lenHalfWin; i <= lenHalfWin; i++)
            {
                for (int j = -lenHalfWin; j <= lenHalfWin; j++)
                {
                    int xnIdx = xIdx + i;
                    int ynIdx = yIdx + j;

                    float val = exp_XY2f(i - xRes, j - yRes, sig2);

                    if (!MyCalibrator::isInImage(xnIdx, ynIdx, imWidth, imHeight)) {
                        continue;
                    }

                    float polSign = resolvePolarity(pol, evData.p);
                    float newVal = image.at<float>(ynIdx, xnIdx) + polSign * val;
                    //if (checkCvTypeBounds(newVal, CV_32FC1))
                    image.at<float>(ynIdx, xnIdx) = newVal;

                    resolveMinMaxVals(newVal, minVal, maxVal, minVal, maxVal);

                    //if (i == 0)
                    //	evData.print();
                }
            }
        }
        if (normalized) {
            normalizeImage(image, image, maxVal, minVal);
        }

        return image.clone();
    }

    /*cv::Mat
    EvImConverter::ev2im_gauss_gray(const vector<EventData> &vEvData, unsigned int imWidth, unsigned int imHeight,
                                    float sigma, bool pol) {
        cv::Mat image = ev2im_gauss(vEvData, imWidth, imHeight, sigma, pol, true);
        image.convertTo(image, CV_8UC1, 255, 0);
        return image.clone();
    }*/

    cv::Mat
    EvImConverter::ev2mci_gg_f(const vector<EventData> &vEvData, ORB_SLAM3::GeometricCamera* pCamera,
                               const cv::Mat& Tcw, const float medDepth, const unsigned imWidth,
                               const unsigned imHeight, const float sigma, const bool pol, const bool normalized) {

        float maxVal = -1000000.0;
        float minVal = 0.0;
        float sig2 = powf(sigma, 2);
        int lenHalfWin = static_cast<int>(ceil(sigma*3.0));

        cv::Mat image = cv::Mat::zeros(static_cast<int>(imHeight), static_cast<int>(imWidth), CV_32FC1);

        size_t evCount = vEvData.size();
        if (evCount <= 0) {
            LOG(ERROR) << "EvImConverter::ev2mci_gg_f: no events.\n";
            return image;
        }

        cv::Mat R = Tcw.rowRange(0,3).colRange(0,3).clone();
        cv::Mat t = Tcw.rowRange(0,3).col(3).clone();

        Eigen::Matrix<double,3,3> RR = ORB_SLAM3::Converter::toMatrix3d(R);
        Eigen::Matrix<double,3,1> tt = ORB_SLAM3::Converter::toVector3d(t);

        Eigen::AngleAxisd omega(RR);
        double t1 = vEvData.back().ts;
        double DT = t1 - vEvData[0].ts;
        double invDT = 1.0/DT;

        for (unsigned int k = 0; k < evCount; k++)
        {
            EventData evData = vEvData[k];

            float ex = evData.x, ey = evData.y;
            double etRate = (t1-evData.ts)*invDT;

            cv::Point3f cvP3d = pCamera->unproject(cv::Point2f(ex, ey));

            // Transform events to sensor coord. sys.
            Eigen::Matrix<double,3,1> P3D = ORB_SLAM3::Converter::toVector3d(cvP3d);

            Eigen::Matrix<double,3,3> newR;
            newR = Eigen::AngleAxisd(omega.angle()*etRate, omega.axis());
            Eigen::Matrix<double,3,1> newT = tt * etRate;

            Eigen::Matrix<double,3,1> newPt = medDepth * newR * P3D + newT;

            // Convert points back to image plane
            Eigen::Vector2d uv = pCamera->project(newPt);

            int xIdx, yIdx;
            float xRes, yRes;
            breakFloatCoords(uv[0], uv[1], xIdx, yIdx, xRes, yRes);

            for (int i = -lenHalfWin; i <= lenHalfWin; i++)
            {
                for (int j = -lenHalfWin; j <= lenHalfWin; j++)
                {
                    int xnIdx = xIdx + i;
                    int ynIdx = yIdx + j;

                    float val = exp_XY2f(i - xRes, j - yRes, sig2);

                    if (!MyCalibrator::isInImage(xnIdx, ynIdx, imWidth, imHeight)) {
                        continue;
                    }

                    float polSign = resolvePolarity(pol, evData.p);
                    float newVal = image.at<float>(ynIdx, xnIdx) + polSign * val;
                    //if (checkCvTypeBounds(newVal, CV_32FC1))
                    image.at<float>(ynIdx, xnIdx) = newVal;

                    resolveMinMaxVals(newVal, minVal, maxVal, minVal, maxVal);
                }
            }
        }
        if (normalized) {
            normalizeImage(image, image, maxVal, minVal);
        }

        return image.clone();
    }

    cv::Mat
    EvImConverter::ev2mci_gg_f(const std::vector<EventData> &vEvData, ORB_SLAM3::GeometricCamera* pCamera,
                               const cv::Mat& params2D, const unsigned imWidth, const unsigned imHeight,
                               const float sigma, const bool pol, const bool normalized) {

        float maxVal = -1000000.0;
        float minVal = 0.0;
        float sig2 = powf(sigma, 2);
        int lenHalfWin = static_cast<int>(ceil(sigma*3.0));

        cv::Mat image = cv::Mat::zeros(imHeight, imWidth, CV_32FC1);

        size_t evCount = vEvData.size();
        if (evCount <= 0) {
            LOG(ERROR) << "EvImConverter::ev2mci_gg_f: no events.\n";
            return image;
        }

        double t1 = vEvData.back().ts;
        auto DT = static_cast<float>(t1 - vEvData[0].ts);
        float invDT = 1.f/DT;

        float omega0 = params2D.at<float>(0, 0) * invDT;
        float vx0 = params2D.at<float>(1, 0) * invDT;
        float vy0 = params2D.at<float>(2, 0) * invDT;

        float sc = 1.f;
        if (params2D.rows > 3) {
            sc = params2D.at<float>(3, 0);
        }
        float scDiff = 1.f - sc;

        for (unsigned int k = 0; k < evCount; k++)
        {
            EventData evData = vEvData[k];

            float ex = evData.x, ey = evData.y;
            auto tk = static_cast<float>(t1-evData.ts);

            // Transform events to sensor coord. sys.
            cv::Point3f cvP3d = pCamera->unproject(cv::Point2f(ex, ey));

            float xp, yp;
            float theta_k = tk * omega0;
            float currSc = scDiff * (1 - tk * invDT) + sc;

            xp = currSc * (cvP3d.x * cos(theta_k) - cvP3d.y * sin(theta_k)) + vx0 * tk;
            yp = currSc * (cvP3d.x * sin(theta_k) + cvP3d.y * cos(theta_k)) + vy0 * tk;

            // Convert points back to image plane
            cv::Point2f uv = pCamera->project(cv::Point3f(xp, yp, cvP3d.z));

            int xIdx, yIdx;
            float xRes, yRes;
            breakFloatCoords(uv.x, uv.y, xIdx, yIdx, xRes, yRes);

            for (int i = -lenHalfWin; i <= lenHalfWin; i++)
            {
                for (int j = -lenHalfWin; j <= lenHalfWin; j++)
                {
                    int xnIdx = xIdx + i;
                    int ynIdx = yIdx + j;

                    float val = exp_XY2f(i - xRes, j - yRes, sig2);

                    if (!MyCalibrator::isInImage(xnIdx, ynIdx, imWidth, imHeight)) {
                        continue;
                    }

                    float polSign = resolvePolarity(pol, evData.p);
                    float newVal = image.at<float>(ynIdx, xnIdx) + polSign * val;
                    //if (checkCvTypeBounds(newVal, CV_32FC1))
                    image.at<float>(ynIdx, xnIdx) = newVal;

                    resolveMinMaxVals(newVal, minVal, maxVal, minVal, maxVal);

                    //if (i == 0)
                    //	evData.print();
                }
            }
        }
        if (normalized) {
            normalizeImage(image, image, maxVal, minVal);
        }

        return image.clone();
    }

    cv::Mat
    EvImConverter::ev2mci_gg_f(const vector<EventData> &vEvData, ORB_SLAM3::GeometricCamera* pCamera,
                               const cv::Mat& Tcw, const MyDepthMap& depthMapObj, const unsigned imWidth,
                               const unsigned imHeight, const float sigma, const bool pol, const bool normalized) {

        float maxVal = -1000000.0;
        float minVal = 0.0;
        float sig2 = powf(sigma, 2);
        int lenHalfWin = static_cast<int>(ceil(sigma*3.0));

        cv::Mat image = cv::Mat::zeros(static_cast<int>(imHeight), static_cast<int>(imWidth), CV_32FC1);

        size_t evCount = vEvData.size();
        if (evCount <= 0) {
            LOG(ERROR) << "EvImConverter::ev2mci_gg_f: no events.\n";
            return image;
        }

        cv::Mat R = Tcw.rowRange(0,3).colRange(0,3).clone();
        cv::Mat t = Tcw.rowRange(0,3).col(3).clone();

        Eigen::Matrix<double,3,3> RR = ORB_SLAM3::Converter::toMatrix3d(R);
        Eigen::Matrix<double,3,1> tt = ORB_SLAM3::Converter::toVector3d(t);

        Eigen::AngleAxisd omega(RR);
        double t1 = vEvData.back().ts;
        double DT = t1 - vEvData[0].ts;
        double invDT = 1.0/DT;

        for (unsigned int k = 0; k < evCount; k++)
        {
            EventData evData = vEvData[k];

            float ex = evData.x, ey = evData.y;
            double etRate = (t1-evData.ts)*invDT;

            cv::Point3f cvP3d = pCamera->unproject(cv::Point2f(ex, ey));

            // Transform events to sensor coord. sys.
            Eigen::Matrix<double,3,1> P3D = ORB_SLAM3::Converter::toVector3d(cvP3d);

            Eigen::Matrix<double,3,3> newR;
            newR = Eigen::AngleAxisd(omega.angle()*etRate, omega.axis());
            Eigen::Matrix<double,3,1> newT = tt * etRate;

            Eigen::Matrix<double,3,1> newPt = depthMapObj.getDepthLinInterp(ex,ey) * newR * P3D + newT;

            // Convert points back to image plane
            Eigen::Vector2d uv = pCamera->project(newPt);

            int xIdx, yIdx;
            float xRes, yRes;
            breakFloatCoords(uv[0], uv[1], xIdx, yIdx, xRes, yRes);

            for (int i = -lenHalfWin; i <= lenHalfWin; i++)
            {
                for (int j = -lenHalfWin; j <= lenHalfWin; j++)
                {
                    int xnIdx = xIdx + i;
                    int ynIdx = yIdx + j;

                    float val = exp_XY2f(i - xRes, j - yRes, sig2);

                    if (!MyCalibrator::isInImage(xnIdx, ynIdx, imWidth, imHeight)) {
                        continue;
                    }

                    float polSign = resolvePolarity(pol, evData.p);
                    float newVal = image.at<float>(ynIdx, xnIdx) + polSign * val;
                    //if (checkCvTypeBounds(newVal, CV_32FC1))
                    image.at<float>(ynIdx, xnIdx) = newVal;

                    resolveMinMaxVals(newVal, minVal, maxVal, minVal, maxVal);
                }
            }
        }
        if (normalized) {
            normalizeImage(image, image, maxVal, minVal);
        }

        return image.clone();
    }

    Eigen::Matrix<double, 1, 6>
    EvImConverter::ev2mci_gg_f_jac(const vector<EventData> &vEvData, ORB_SLAM3::GeometricCamera* pCamera,
            const g2o::VertexSE3Expmap* vSE3, const float medDepth, const unsigned imWidth,
            const unsigned imHeight, const float sigma, const bool pol, const bool global) {

        float sig2 = powf(sigma, 2);
        float invSig2 = 1.f / sig2;
        int lenHalfWin = static_cast<int>(ceil(sigma*3.0));

        size_t evCount = vEvData.size();
        if (evCount <= 0) {
            LOG(ERROR) << "EvImConverter::ev2mci_gg_f_jac: no events.\n";
            return Eigen::Matrix<double, 1, 6>::Zero();
        }

        cv::Mat I = cv::Mat::zeros(static_cast<int>(imHeight), static_cast<int>(imWidth), CV_32FC1);
        cv::Mat Iwx = cv::Mat::zeros(static_cast<int>(imHeight), static_cast<int>(imWidth), CV_32FC1);
        cv::Mat Iwy = cv::Mat::zeros(static_cast<int>(imHeight), static_cast<int>(imWidth), CV_32FC1);
        cv::Mat Iwz = cv::Mat::zeros(static_cast<int>(imHeight), static_cast<int>(imWidth), CV_32FC1);
        cv::Mat Ivx = cv::Mat::zeros(static_cast<int>(imHeight), static_cast<int>(imWidth), CV_32FC1);
        cv::Mat Ivy = cv::Mat::zeros(static_cast<int>(imHeight), static_cast<int>(imWidth), CV_32FC1);
        cv::Mat Ivz = cv::Mat::zeros(static_cast<int>(imHeight), static_cast<int>(imWidth), CV_32FC1);

        Eigen::Matrix<double,3,3> R = vSE3->estimate().rotation().toRotationMatrix();
        Eigen::Matrix<double,3,1> t = vSE3->estimate().translation();
        Eigen::AngleAxisd omega(R);

        double t1 = vEvData.back().ts;
        double DT = t1 - vEvData[0].ts;
        double invDT = 1.0/DT;

        for (unsigned int k = 0; k < evCount; k++)
        {
            // Retrieve e_k data
            EventData ev_k = vEvData[k];

            float exk = ev_k.x, eyk = ev_k.y;
            double etRate = (t1-ev_k.ts)*invDT;

            // Reconstruct 3D point
            cv::Point3f cvP3d = pCamera->unproject(cv::Point2f(exk, eyk));

            Eigen::Matrix<double,3,1> X3D = ORB_SLAM3::Converter::toVector3d(cvP3d);
            X3D = medDepth * X3D;

            // Construct Rk, tk (transformations for kth event)
            Eigen::Matrix<double,3,3> Rk;
            Rk = Eigen::AngleAxisd(omega.angle()*etRate, omega.axis());
            Eigen::Matrix<double,3,1> tk = t * etRate;

            // Map 3D point
            Eigen::Matrix<double,3,1> xyz_trans = Rk * X3D + tk;

            // Calculate Jacobian for 2D image point
            double X = xyz_trans[0];
            double Y = xyz_trans[1];
            double Z = xyz_trans[2];

            Eigen::Matrix<double,3,6> SE3deriv;
            SE3deriv << 0.f, Z,   -Y, 1.f, 0.f, 0.f,
                    -Z , 0.f, X, 0.f, 1.f, 0.f,
                    Y ,  -X , 0.f, 0.f, 0.f, 1.f;
            // Account for kth event timestamp
            SE3deriv *= etRate;

            Eigen::Matrix<double, 2, 6> JP2D = -pCamera->projectJac(xyz_trans) * SE3deriv;

            // Project 3D point back to image plane
            Eigen::Vector2d uv = pCamera->project(xyz_trans);

            // Sample image point to reconstruct jacobian images
            int xIdx, yIdx;
            float xRes, yRes;
            breakFloatCoords(uv[0], uv[1], xIdx, yIdx, xRes, yRes);

            for (int i = -lenHalfWin; i <= lenHalfWin; i++)
            {
                for (int j = -lenHalfWin; j <= lenHalfWin; j++)
                {
                    // Calculate Gaussian value at pixel and gradients
                    int xnIdx = xIdx + i;
                    int ynIdx = yIdx + j;

                    if (!MyCalibrator::isInImage(xnIdx, ynIdx, imWidth, imHeight)) {
                        continue;
                    }

                    float val = exp_XY2f(i - xRes, j - yRes, sig2);

                    float gx = invSig2 * (i - xRes) * val;
                    float gy = invSig2 * (j - yRes) * val;

                    Eigen::Matrix<double, 1, 2> grad_g;
                    grad_g << gx, gy;

                    Eigen::Matrix<double, 1, 6> JIx = grad_g * JP2D;

                    float polSign = resolvePolarity(pol, ev_k.p);

                    // Assign values to each gradient image
                    I.at<float>(ynIdx, xnIdx) = I.at<float>(ynIdx, xnIdx) + polSign * val;

                    Iwx.at<float>(ynIdx, xnIdx) = Iwx.at<float>(ynIdx, xnIdx) + polSign * JIx[0];
                    Iwy.at<float>(ynIdx, xnIdx) = Iwy.at<float>(ynIdx, xnIdx) + polSign * JIx[1];
                    Iwz.at<float>(ynIdx, xnIdx) = Iwz.at<float>(ynIdx, xnIdx) + polSign * JIx[2];
                    Ivx.at<float>(ynIdx, xnIdx) = Ivx.at<float>(ynIdx, xnIdx) + polSign * JIx[3];
                    Ivy.at<float>(ynIdx, xnIdx) = Ivy.at<float>(ynIdx, xnIdx) + polSign * JIx[4];
                    Ivz.at<float>(ynIdx, xnIdx) = Ivz.at<float>(ynIdx, xnIdx) + polSign * JIx[5];
                }
            }
        }

        // Multiply images and average each to calculate final value of jac.
        cv::Mat I2wx = I.mul(Iwx);
        cv::Mat I2wy = I.mul(Iwy);
        cv::Mat I2wz = I.mul(Iwz);
        cv::Mat I2vx = I.mul(Ivx);
        cv::Mat I2vy = I.mul(Ivy);
        cv::Mat I2vz = I.mul(Ivz);

        Eigen::Matrix<double, 1, 6> jac;
        jac[0] = -imageMean(I2wx, global) * 2;
        jac[1] = -imageMean(I2wy, global) * 2;
        jac[2] = -imageMean(I2wz, global) * 2;
        jac[3] = -imageMean(I2vx, global) * 2;
        jac[4] = -imageMean(I2vy, global) * 2;
        jac[5] = -imageMean(I2vz, global) * 2;

        return jac;
    }


    //TODO:	Implement an upsample-downsample method and see if it's better

} //namespace ORB_SLAM