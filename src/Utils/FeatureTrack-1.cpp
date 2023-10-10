//
// Created by root on 11/22/21.
//

#include "FeatureTrack.h"

using namespace std;
using namespace ORB_SLAM3;


namespace EORB_SLAM {

    FeatureTrack::FeatureTrack(const int id) :
            mnId(id), mLostIdx(0), mbIsValid(true), mbIsLost(false), mpMapPoint(nullptr)
    {
        //mmKeyPoints.reserve(500);
    }

    FeatureTrack::FeatureTrack(const ulong id, const ulong frameIdx, const cv::KeyPoint &refKpt) : FeatureTrack(id)
    {
        mmKeyPoints.insert(make_pair(frameIdx, refKpt));
    }

    FeatureTrack::~FeatureTrack() {
        mpMapPoint = nullptr;
    }

    bool FeatureTrack::validFrame(const ulong frame) {

        // Do we need to check for begin() too??
        return mmKeyPoints.find(frame) != mmKeyPoints.end();
    }

    ulong FeatureTrack::getFrameIdMax() {

        if (mmKeyPoints.empty())
            return 0;
        return mmKeyPoints.rbegin()->first;
    }

    ulong FeatureTrack::getFrameIdMin() {

        if (mmKeyPoints.empty())
            return 0;
        return mmKeyPoints.begin()->first;
    }

    bool FeatureTrack::getFeature(ulong frame, cv::KeyPoint &kpt) {

        if (!validFrame(frame)) {
            return false;
        }
        kpt = mmKeyPoints[frame];
        return true;
    }

    bool FeatureTrack::getFeatureAndMapPoint(ulong frame, cv::KeyPoint &kpt, ORB_SLAM3::MapPoint *&pMpt) {

        if (!validFrame(frame)) {
            return false;
        }
        kpt = mmKeyPoints[frame];
        pMpt = mpMapPoint;
        return true;
    }

    void FeatureTrack::updateTrackValidity(ulong frame, int stat) {

        if (validFrame(frame)) {

            if (stat < 0) {
                if (!mbIsLost)
                    mbIsLost = true;
                mLostIdx++;
                if (mLostIdx > DEF_TH_LOST_FEATURE)
                    mbIsValid = false;
            }
            else if (isValid()) {
                mbIsLost = false;
                mLostIdx = 0;
            }
        }
    }

    void FeatureTrack::updateTrackedFeature(ulong frame, const cv::KeyPoint &kpt, int stat) {

        if (stat >= 0) {
            if (mmKeyPoints.find(frame) != mmKeyPoints.end()) {
                mmKeyPoints[frame] = kpt;
            }
            else {
                mmKeyPoints.insert(make_pair(frame, kpt));
            }
        }
        else if (mmKeyPoints.find(frame) == mmKeyPoints.end() && this->isValid()) {
            // in case this track is still valid and we have a new frame, add the point
            cv::KeyPoint prevKpt;
            this->getFeature(mmKeyPoints.rbegin()->first, prevKpt);
            mmKeyPoints.insert(make_pair(frame, prevKpt));
        }
        this->updateTrackValidity(frame, stat);
    }

    void FeatureTrack::updateMapPoint(ulong frame, ORB_SLAM3::MapPoint *pMpt) {

        if (validFrame(frame)) {
            mpMapPoint = pMpt;
        }
    }

    struct comp_tracks_by_id
    {
        const ulong keyTracker;
        explicit comp_tracks_by_id(const ulong& key): keyTracker(key) {}

        bool operator()(const FeatureTrackPtr& pTrack) const {
            return keyTracker == pTrack->getId();
        }
    };

    int FeatureTrack::findTrackById(const std::vector<std::shared_ptr<FeatureTrack> > &vpSearchFtTracks) {

        auto itr = std::find_if(vpSearchFtTracks.begin(), vpSearchFtTracks.end(), comp_tracks_by_id(mnId));
        if (itr == vpSearchFtTracks.end()) {
            return -1;
        }
        else {
            return std::distance(vpSearchFtTracks.begin(), itr);
        }
    }

    void FeatureTrack::removeTrack() {

        //if (mpMapPoint && !mpMapPoint->isBad())
        //    mpMapPoint->SetBadFlag();
    }

    unsigned FeatureTrack::matchFeatureTracks(const EvFramePtr &pRefFrame, EvFramePtr &pCurFrame) {

        size_t nRefKpts = pRefFrame->numAllKPts();
        vector<FeatureTrackPtr> vpCurTracks = pCurFrame->getAllFeatureTracks();
        const int szCurTracks = vpCurTracks.size();
        const ulong curFrId = pCurFrame->mnId;

        unsigned nMatches = 0;
        vector<int> vMatches12(nRefKpts, -1);

        for (size_t i = 0; i < nRefKpts; i++) {

            FeatureTrackPtr pRefTrack = pRefFrame->getFeatureTrack(i);

            if (pRefTrack && pRefTrack->isValid() && !pRefTrack->isLost()) { // TODO: Maybe better to check valid frame

                int matchedIdx = pRefTrack->findTrackById(vpCurTracks);

                if (matchedIdx >= 0 && matchedIdx < szCurTracks) {

                    vMatches12[i] = matchedIdx;

                    MapPoint* pMpt = nullptr;
                    cv::KeyPoint kpt;
                    vpCurTracks[matchedIdx]->getFeatureAndMapPoint(curFrId, kpt, pMpt);
                    pCurFrame->setMapPoint(matchedIdx, pMpt);

                    nMatches++;
                }
            }
        }

        pCurFrame->setMatches(vMatches12);
        pCurFrame->setNumMatches(nMatches);

        return nMatches;
    }

    void FeatureTrack::assignFeaturesToGrid(const std::vector<cv::KeyPoint> &vkpts, const int patchSz,
                                            std::vector<std::vector<std::vector<const cv::KeyPoint*> > > &grid) {

        if (grid.empty() || grid[0].empty()) {
            LOG(WARNING) << "FeatureTrack::assignFeaturesToGrid: Empty input grid, abort";
            return;
        }

        auto pSz = static_cast<float>(patchSz);
        const int nRows = grid.size();
        const int nCols = grid[0].size();

        for (const auto & kpt : vkpts) {

            const int ci = static_cast<int>(kpt.pt.x / pSz);
            const int ri = static_cast<int>(kpt.pt.y / pSz);

            if (ri >= 0 && ri < nRows && ci >= 0 && ci < nCols)
                grid[ri][ci].push_back(&kpt);
        }
    }

    bool comp_pkp_response(const cv::KeyPoint* kp0, const cv::KeyPoint* kp1) {
        return kp0->response > kp1->response;
    }

    bool comp_kp_response(const cv::KeyPoint& kp0, const cv::KeyPoint& kp1) {
        return kp0.response > kp1.response;
    }

    void FeatureTrack::selectNewKPtsUniform(const std::vector<cv::KeyPoint> &vkpts0, const std::vector<cv::KeyPoint> &vkpts1,
                                            const cv::Size& imSize, const int patchSz, const int nPts,
                                            std::vector<cv::KeyPoint> &result, const bool respFilter) {

        const int nRows = imSize.height / patchSz;
        const int nCols = imSize.width / patchSz;
        const int nGrid = nRows * nCols;
        const int nFtsPerCell = std::ceil(nPts / nGrid);

        float medRespPoints1 = 1.f;
        if (respFilter) {
            vector<cv::KeyPoint> vkpts1_copy = vkpts1;
            sort(vkpts1_copy.begin(), vkpts1_copy.end(), comp_kp_response);
            medRespPoints1 = vkpts1_copy[vkpts1_copy.size()/2].response;
        }

        // Assign key points to grid
        vector<vector<vector<const cv::KeyPoint*>>> grid0(nRows, vector<vector<const cv::KeyPoint*>>(nCols));
        vector<vector<vector<const cv::KeyPoint*>>> grid1(nRows, vector<vector<const cv::KeyPoint*>>(nCols));
        for (int i = 0; i < nRows; i++)
            for (int j = 0; j < nCols; j++) {
                grid0[i][j].reserve(nFtsPerCell);
                grid1[i][j].reserve(nFtsPerCell);
            }

        assignFeaturesToGrid(vkpts0, patchSz, grid0);
        assignFeaturesToGrid(vkpts1, patchSz, grid1);

        // Pick points from low density areas
        result.reserve(nPts);

        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {

                vector<const cv::KeyPoint*> vKptsCell0 = grid0[i][j];
                const int gridCellSize = vKptsCell0.size();

                if (gridCellSize < nFtsPerCell) {

                    const int nRes = nFtsPerCell - gridCellSize;

                    vector<const cv::KeyPoint*> vKptsCell1 = grid1[i][j];
                    std::sort(vKptsCell1.begin(), vKptsCell1.end(), comp_pkp_response);

                    for (int k = 0; k < vKptsCell1.size() && k < nRes; k++) {

                        cv::KeyPoint currKpt1 = *(vKptsCell1[k]);

                        if (respFilter && currKpt1.response < medRespPoints1)
                            continue;

                        result.push_back(currKpt1);
                    }
                }
            }
        }
    }

    void FeatureTrack::getTracksReconstSFM(const std::vector<std::shared_ptr<FeatureTrack> > &vFtTracks,
                                           ulong minFrames, ulong maxFrames, std::vector<cv::Mat> &vvTracks) {

        const size_t nFrames = maxFrames - minFrames + 1;
        const int nTracks = vFtTracks.size();
        vector<int> vvTracksValidity(nTracks, 0);

        vvTracks.reserve(nFrames);

        // check tracks validity in terms of num. views
        for (size_t fr = minFrames, i = 0; i < nFrames; fr++, i++) {
            for (size_t tr = 0; tr < nTracks; tr++) {

                FeatureTrackPtr pTrack = vFtTracks[tr];

                if (pTrack && pTrack->isValid() && pTrack->validFrame(fr)) {
                    vvTracksValidity[tr] += 1;
                }
            }
        }

        for (size_t fr = minFrames, i = 0; i < nFrames; fr++, i++) {

            cv::Mat_<double> frame(2, nTracks);

            for (size_t tr = 0; tr < nTracks; tr++) {

                if (vvTracksValidity[tr] < 2)
                    continue;

                FeatureTrackPtr pTrack = vFtTracks[tr];

                if (pTrack && pTrack->isValid() && pTrack->validFrame(fr)) {

                    cv::KeyPoint kpt;
                    pTrack->getFeature(fr, kpt);
                    frame(0,tr) = kpt.pt.x;
                    frame(1,tr) = kpt.pt.y;
                }
                else {
                    frame(0,tr) = -1;
                    frame(1,tr) = -1;
                }
            }
            vvTracks.emplace_back(frame);
        }
    }

    void FeatureTrack::mapToVectorTracks(const std::map<ulong, std::shared_ptr<FeatureTrack> > &mTracks,
                                         std::vector<std::shared_ptr<FeatureTrack> > &vFtTracks) {

        vFtTracks.clear();
        vFtTracks.reserve(mTracks.size());

        for (const auto& ppTrack : mTracks) {

            FeatureTrackPtr pTrack = ppTrack.second;

            if (pTrack && pTrack->isValid()) {
                vFtTracks.push_back(pTrack);
            }
        }
    }

    float FeatureTrack::calcAreaFeatureTracks(const ulong frId, const std::vector<FeatureTrackPtr> &vpTracks, const bool only3D) {

        float minX = MAXFLOAT, maxX = 0, minY = MAXFLOAT, maxY = 0;
        bool firstIter = true;

        for (const auto& pTrack : vpTracks) {

            MapPoint* pMpt;
            cv::KeyPoint kpt;

            if (pTrack && pTrack->isValid() && pTrack->getFeatureAndMapPoint(frId, kpt, pMpt)) {

                if (only3D && (!pMpt || pMpt->isBad()))
                    continue;

                float currX = kpt.pt.x;
                float currY = kpt.pt.y;

                if (firstIter) {
                    minX = currX;
                    maxX = currX;
                    minY = currY;
                    maxY = currY;
                    firstIter = false;
                }
                else {
                    if (minX > currX)
                        minX = currX;
                    if (maxX < currX)
                        maxX = currX;
                    if (minY > currY)
                        minY = currY;
                    if (maxY < currY)
                        maxY = currY;
                }
            }
        }

        return abs(maxX - minX) * abs(maxY - minY);
    }


} // EORB_SLAM
