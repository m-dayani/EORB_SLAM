//
// Created by root on 11/22/21.
//

#ifndef ORB_SLAM3_FEATURETRACK_H
#define ORB_SLAM3_FEATURETRACK_H

#include <vector>
#include <map>
#include <algorithm>

#include <opencv2/core.hpp>

#include "MapPoint.h"


namespace EORB_SLAM {

#define DEF_TH_LOST_FEATURE 3

    class FeatureTrack {
    public:
        explicit FeatureTrack(int id);
        FeatureTrack(ulong id, ulong frameIdx, const cv::KeyPoint& refKpt);
        ~FeatureTrack();

        bool isValid() const { return mbIsValid; }
        bool isLost() const { return mbIsLost; }
        bool validFrame(ulong frame);

        bool getFeature(ulong frame, cv::KeyPoint& kpt);
        bool getFeatureAndMapPoint(ulong frame, cv::KeyPoint& kpt, ORB_SLAM3::MapPoint*& pMpt);
        void updateTrackedFeature(ulong frame, const cv::KeyPoint& kpt, int stat);
        void updateTrackValidity(ulong frame, int stat);
        void updateMapPoint(ulong frame, ORB_SLAM3::MapPoint* pMpt);

        int findTrackById(const std::vector<std::shared_ptr<FeatureTrack>>& vpSearchFtTracks);

        void removeTrack();

        static unsigned matchFeatureTracks(const EvFramePtr& pRefFrame, EvFramePtr& pCurFrame);

        static void assignFeaturesToGrid(const std::vector<cv::KeyPoint>& vkpts, int patchSz,
                                         std::vector<std::vector<std::vector<const cv::KeyPoint*>>>& grid);
        static void selectNewKPtsUniform(const std::vector<cv::KeyPoint>& vkpts0, const std::vector<cv::KeyPoint>& vkpts1,
                                         const cv::Size& imSize, int patchSz, int nPts, std::vector<cv::KeyPoint>& result,
                                         bool respFilter = false);

        static void getTracksReconstSFM(const std::vector<std::shared_ptr<FeatureTrack>>& vFtTracks,
                                        ulong minFrames, ulong maxFrames, std::vector<cv::Mat>& vvTracks);

        static void mapToVectorTracks(const std::map<ulong, std::shared_ptr<FeatureTrack>>& mTracks,
                                      std::vector<std::shared_ptr<FeatureTrack>>& vFtTracks);

        static float calcAreaFeatureTracks(ulong frId, const std::vector<FeatureTrackPtr>& vpTracks, bool only3D = true);

        ulong getId() const { return mnId; }

        ulong getFrameIdMax();
        ulong getFrameIdMin();

    protected:

    private:
        const ulong mnId;

        int mLostIdx;
        bool mbIsValid;
        bool mbIsLost;

        std::map<ulong, cv::KeyPoint> mmKeyPoints;

        ORB_SLAM3::MapPoint* mpMapPoint;
    };

    typedef std::shared_ptr<FeatureTrack> FeatureTrackPtr;

} // EORB_SLAM


#endif //ORB_SLAM3_FEATURETRACK_H
