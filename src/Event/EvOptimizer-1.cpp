//
// Created by root on 1/13/21.
//

#include "EvOptimizer.h"

#include <utility>

using namespace std;
using namespace cv;


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

} // EORB_SLAM


