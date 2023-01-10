//
// Created by root on 2/24/21.
//

#include "MixedMatcher.h"
#include<climits>

#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#include<stdint-gcc.h>

using namespace std;
using namespace ORB_SLAM3;


namespace EORB_SLAM {

    MixedMatcher::MixedMatcher(float nnratio, bool checkOri): ORB_SLAM3::ORBmatcher(nnratio, checkOri) {}

    int MixedMatcher::SearchForInitialization(MixedFrame &F1, MixedFrame &F2, vector<cv::Point2f> &vbPrevMatched,
                                              vector<int> &vnMatches12, int windowSize)
    {
        int nmatches=0;
        vnMatches12 = vector<int>(F1.numAllKPts(),-1);

        vector<int> rotHist[HISTO_LENGTH];
        for(int i=0;i<HISTO_LENGTH;i++)
            rotHist[i].reserve(500);
        const float factor = 1.0f/HISTO_LENGTH;

        vector<int> vMatchedDistance(F2.numAllKPts(),INT_MAX);
        vector<int> vnMatches21(F2.numAllKPts(),-1);

        for(size_t i1=0, iend1=F1.numAllKPts(); i1<iend1; i1++)
        {
            //cv::KeyPoint kp1 = F1.getUndistKPtMono(i1);
            //int level1 = kp1.octave;
            int level1 = F1.getKPtLevelMono(i1);
            if(level1>0)
                continue;

            vector<size_t> vIndices2 = F2.GetFeaturesInArea(vbPrevMatched[i1].x,vbPrevMatched[i1].y, windowSize,level1,level1);

            if(vIndices2.empty())
                continue;

            const bool isORB1 = F1.isORBDescValid(i1);

            cv::Mat d1 = F1.getDescriptorMono(i1);

            int bestDist = INT_MAX;
            int bestDist2 = INT_MAX;
            int bestIdx2 = -1;

            for(vector<size_t>::iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
            {
                size_t i2 = *vit;

                const bool isORB2 = F2.isORBDescValid(i2);

                // If descriptors are not compatible, abort
                if (isORB1 != isORB2)
                    continue;

                cv::Mat d2 = F2.getDescriptorMono(i2);

                int dist = DescriptorDistance(d1,d2);

                if(vMatchedDistance[i2]<=dist)
                    continue;

                if(dist<bestDist)
                {
                    bestDist2=bestDist;
                    bestDist=dist;
                    bestIdx2=i2;
                }
                else if(dist<bestDist2)
                {
                    bestDist2=dist;
                }
            }

            if(bestDist<=TH_LOW)
            {
                if(bestDist<(float)bestDist2*mfNNratio)
                {
                    if(vnMatches21[bestIdx2]>=0)
                    {
                        vnMatches12[vnMatches21[bestIdx2]]=-1;
                        nmatches--;
                    }
                    vnMatches12[i1]=bestIdx2;
                    vnMatches21[bestIdx2]=i1;
                    vMatchedDistance[bestIdx2]=bestDist;
                    nmatches++;

                    if(mbCheckOrientation)
                    {
                        float rot = F1.getUndistKPtMono(i1).angle-F2.getUndistKPtMono(bestIdx2).angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(i1);
                    }
                }
            }

        }

        if(mbCheckOrientation)
        {
            int ind1=-1;
            int ind2=-1;
            int ind3=-1;

            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

            for(int i=0; i<HISTO_LENGTH; i++)
            {
                if(i==ind1 || i==ind2 || i==ind3)
                    continue;
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    int idx1 = rotHist[i][j];
                    if(vnMatches12[idx1]>=0)
                    {
                        vnMatches12[idx1]=-1;
                        nmatches--;
                    }
                }
            }

        }

        //Update prev matched
        for(size_t i1=0, iend1=vnMatches12.size(); i1<iend1; i1++)
            if(vnMatches12[i1]>=0)
                vbPrevMatched[i1]=F2.getUndistKPtMono(vnMatches12[i1]).pt;

        return nmatches;
    }

    // Note: This can only match ORB features because we compute BoW for ORB features
    int MixedMatcher::SearchByBoW(MixedKeyFrame* pKF, MixedFrame &F, vector<ORB_SLAM3::MapPoint*> &vpMapPointMatches)
    {
        const vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();

        vpMapPointMatches = vector<MapPoint*>(F.numAllKPts(),static_cast<MapPoint*>(NULL));

        const DBoW2::FeatureVector &vFeatVecKF = pKF->mFeatVec;

        int nmatches=0;

        vector<int> rotHist[HISTO_LENGTH];
        for(int i=0;i<HISTO_LENGTH;i++)
            rotHist[i].reserve(500);
        const float factor = 1.0f/HISTO_LENGTH;

        // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
        DBoW2::FeatureVector::const_iterator KFit = vFeatVecKF.begin();
        DBoW2::FeatureVector::const_iterator Fit = F.mFeatVec.begin();
        DBoW2::FeatureVector::const_iterator KFend = vFeatVecKF.end();
        DBoW2::FeatureVector::const_iterator Fend = F.mFeatVec.end();

        while(KFit != KFend && Fit != Fend)
        {
            if(KFit->first == Fit->first)
            {
                const vector<unsigned int> vIndicesKF = KFit->second;
                const vector<unsigned int> vIndicesF = Fit->second;

                for(size_t iKF=0; iKF<vIndicesKF.size(); iKF++)
                {
                    const unsigned int realIdxKF = vIndicesKF[iKF];

                    if (!pKF->isORBDescValid(realIdxKF))
                        continue;

                    MapPoint* pMP = vpMapPointsKF[realIdxKF];

                    if(!pMP)
                        continue;

                    if(pMP->isBad())
                        continue;

                    const cv::Mat &dKF= pKF->getDescriptorMono(realIdxKF);

                    int bestDist1=256;
                    int bestIdxF =-1 ;
                    int bestDist2=256;

                    int bestDist1R=256;
                    int bestIdxFR =-1 ;
                    int bestDist2R=256;

                    for(size_t iF=0; iF<vIndicesF.size(); iF++)
                    {
                        if(F.numKPtsLeft() == -1){
                            const unsigned int realIdxF = vIndicesF[iF];

                            if (!F.isORBDescValid(realIdxF))
                                continue;

                            if(vpMapPointMatches[realIdxF])
                                continue;

                            const cv::Mat &dF = F.getDescriptorMono(realIdxF);

                            const int dist =  DescriptorDistance(dKF,dF);

                            if(dist<bestDist1)
                            {
                                bestDist2=bestDist1;
                                bestDist1=dist;
                                bestIdxF=realIdxF;
                            }
                            else if(dist<bestDist2)
                            {
                                bestDist2=dist;
                            }
                        }
                        else{
                            const unsigned int realIdxF = vIndicesF[iF];

                            if(vpMapPointMatches[realIdxF])
                                continue;

                            const cv::Mat &dF = F.getORBDescriptor(realIdxF);

                            const int dist =  DescriptorDistance(dKF,dF);

                            if(realIdxF < F.numKPtsLeft() && dist<bestDist1){
                                bestDist2=bestDist1;
                                bestDist1=dist;
                                bestIdxF=realIdxF;
                            }
                            else if(realIdxF < F.numKPtsLeft() && dist<bestDist2){
                                bestDist2=dist;
                            }

                            if(realIdxF >= F.numKPtsLeft() && dist<bestDist1R){
                                bestDist2R=bestDist1R;
                                bestDist1R=dist;
                                bestIdxFR=realIdxF;
                            }
                            else if(realIdxF >= F.numKPtsLeft() && dist<bestDist2R){
                                bestDist2R=dist;
                            }
                        }

                    }

                    if(bestDist1<=TH_LOW)
                    {
                        if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                        {
                            vpMapPointMatches[bestIdxF]=pMP;

                            const cv::KeyPoint &kp =
                                    (!pKF->mpCamera2) ? pKF->getUndistKPtMono(realIdxKF) :
                                    (realIdxKF >= pKF -> numAllKPtsLeft()) ? pKF -> getKPtRight(realIdxKF - pKF -> numAllKPtsLeft())
                                                                           : pKF -> getDistKPtMono(realIdxKF);

                            if(mbCheckOrientation)
                            {
                                cv::KeyPoint Fkp =
                                        (!pKF->mpCamera2 || F.numKPtsLeft() == -1) ? F.getUndistKPtMono(bestIdxF) : // TODO: Check original
                                        (bestIdxF >= F.numKPtsLeft()) ? F.getKPtRight(bestIdxF - F.numKPtsLeft())
                                                                      : F.getDistKPtMono(bestIdxF);

                                float rot = kp.angle-Fkp.angle;
                                if(rot<0.0)
                                    rot+=360.0f;
                                int bin = round(rot*factor);
                                if(bin==HISTO_LENGTH)
                                    bin=0;
                                assert(bin>=0 && bin<HISTO_LENGTH);
                                rotHist[bin].push_back(bestIdxF);
                            }
                            nmatches++;
                        }

                        if(bestDist1R<=TH_LOW)
                        {
                            if(static_cast<float>(bestDist1R)<mfNNratio*static_cast<float>(bestDist2R) || true)
                            {
                                vpMapPointMatches[bestIdxFR]=pMP;

                                const cv::KeyPoint &kp =
                                        (!pKF->mpCamera2) ? pKF->getUndistKPtMono(realIdxKF) :
                                        (realIdxKF >= pKF -> numAllKPtsLeft()) ? pKF -> getKPtRight(realIdxKF - pKF -> numAllKPtsLeft())
                                                                               : pKF -> getDistKPtMono(realIdxKF);

                                if(mbCheckOrientation)
                                {
                                    cv::KeyPoint Fkp =
                                            (!F.mpCamera2) ? F.getUndistKPtMono(bestIdxFR) : // TODO: Check original
                                            (bestIdxFR >= F.numKPtsLeft()) ? F.getKPtRight(bestIdxFR - F.numKPtsLeft())
                                                                           : F.getDistKPtMono(bestIdxFR);

                                    float rot = kp.angle-Fkp.angle;
                                    if(rot<0.0)
                                        rot+=360.0f;
                                    int bin = round(rot*factor);
                                    if(bin==HISTO_LENGTH)
                                        bin=0;
                                    assert(bin>=0 && bin<HISTO_LENGTH);
                                    rotHist[bin].push_back(bestIdxFR);
                                }
                                nmatches++;
                            }
                        }
                    }

                }

                KFit++;
                Fit++;
            }
            else if(KFit->first < Fit->first)
            {
                KFit = vFeatVecKF.lower_bound(Fit->first);
            }
            else
            {
                Fit = F.mFeatVec.lower_bound(KFit->first);
            }
        }

        if(mbCheckOrientation)
        {
            int ind1=-1;
            int ind2=-1;
            int ind3=-1;

            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

            for(int i=0; i<HISTO_LENGTH; i++)
            {
                if(i==ind1 || i==ind2 || i==ind3)
                    continue;
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    vpMapPointMatches[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                    nmatches--;
                }
            }
        }

        return nmatches;
    }

    int MixedMatcher::SearchByBoW(MixedKeyFrame *pKF1, MixedKeyFrame *pKF2, vector<ORB_SLAM3::MapPoint *> &vpMatches12)
    {
        //const vector<cv::KeyPoint> &vKeysUn1 = pKF1->getAllUndistKPtsMono();
        const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
        const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
        //const cv::Mat &Descriptors1 = pKF1->getAllORBDescriptors();

        //const vector<cv::KeyPoint> &vKeysUn2 = pKF2->getAllUndistKPtsMono();
        const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
        const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
        //const cv::Mat &Descriptors2 = pKF2->getAllORBDescriptors();

        vpMatches12 = vector<MapPoint*>(vpMapPoints1.size(),static_cast<MapPoint*>(NULL));
        vector<bool> vbMatched2(vpMapPoints2.size(),false);

        vector<int> rotHist[HISTO_LENGTH];
        for(int i=0;i<HISTO_LENGTH;i++)
            rotHist[i].reserve(500);

        const float factor = 1.0f/HISTO_LENGTH;

        int nmatches = 0;

        DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
        DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
        DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
        DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

        while(f1it != f1end && f2it != f2end)
        {
            if(f1it->first == f2it->first)
            {
                for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
                {
                    const size_t idx1 = f1it->second[i1];
                    if(!pKF1->isORBDescValid(idx1) || (pKF1 -> numAllKPtsLeft() != -1 && idx1 >= pKF1 -> numAllKPts())){
                        continue;
                    }

                    MapPoint* pMP1 = vpMapPoints1[idx1];
                    if(!pMP1)
                        continue;
                    if(pMP1->isBad())
                        continue;

                    const cv::Mat &d1 = pKF1->getDescriptorMono(idx1);

                    int bestDist1=256;
                    int bestIdx2 =-1 ;
                    int bestDist2=256;

                    for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                    {
                        const size_t idx2 = f2it->second[i2];

                        if(!pKF2->isORBDescValid(idx2) || (pKF2 -> numAllKPtsLeft() != -1 && idx2 >= pKF2 -> numAllKPts())){
                            continue;
                        }

                        MapPoint* pMP2 = vpMapPoints2[idx2];

                        if(vbMatched2[idx2] || !pMP2)
                            continue;

                        if(pMP2->isBad())
                            continue;

                        const cv::Mat &d2 = pKF2->getDescriptorMono(idx2);

                        int dist = DescriptorDistance(d1,d2);

                        if(dist<bestDist1)
                        {
                            bestDist2=bestDist1;
                            bestDist1=dist;
                            bestIdx2=idx2;
                        }
                        else if(dist<bestDist2)
                        {
                            bestDist2=dist;
                        }
                    }

                    if(bestDist1<TH_LOW)
                    {
                        if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                        {
                            vpMatches12[idx1]=vpMapPoints2[bestIdx2];
                            vbMatched2[bestIdx2]=true;

                            if(mbCheckOrientation)
                            {
                                float rot = pKF1->getDistKPtMono(idx1).angle-pKF2->getUndistKPtMono(bestIdx2).angle;
                                if(rot<0.0)
                                    rot+=360.0f;
                                int bin = round(rot*factor);
                                if(bin==HISTO_LENGTH)
                                    bin=0;
                                assert(bin>=0 && bin<HISTO_LENGTH);
                                rotHist[bin].push_back(idx1);
                            }
                            nmatches++;
                        }
                    }
                }

                f1it++;
                f2it++;
            }
            else if(f1it->first < f2it->first)
            {
                f1it = vFeatVec1.lower_bound(f2it->first);
            }
            else
            {
                f2it = vFeatVec2.lower_bound(f1it->first);
            }
        }

        if(mbCheckOrientation)
        {
            int ind1=-1;
            int ind2=-1;
            int ind3=-1;

            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

            for(int i=0; i<HISTO_LENGTH; i++)
            {
                if(i==ind1 || i==ind2 || i==ind3)
                    continue;
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    vpMatches12[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                    nmatches--;
                }
            }
        }

        return nmatches;
    }

    int MixedMatcher::SearchByProjection(MixedFrame &F, const vector<ORB_SLAM3::MapPoint*> &vpMapPoints, const float th,
                                         const bool bFarPoints, const float thFarPoints)
    {
        int nmatches=0, left = 0, right = 0;

        const bool bFactor = th!=1.0;

        for(size_t iMP=0; iMP<vpMapPoints.size(); iMP++)
        {
            MapPoint* pMP = vpMapPoints[iMP];
            if(!pMP->mbTrackInView && !pMP->mbTrackInViewR)
                continue;

            if(bFarPoints && pMP->mTrackDepth>thFarPoints)
                continue;

            if(pMP->isBad())
                continue;

            if(pMP->mbTrackInView)
            {
                const int &nPredictedLevel = pMP->mnTrackScaleLevel;

                // The size of the window will depend on the viewing direction
                float r = RadiusByViewingCos(pMP->mTrackViewCos);

                if(bFactor)
                    r*=th;

                float fScaleFactor = F.getORBScaleFactor(nPredictedLevel);
                if (!pMP->isORBMapPoint()) {
                    fScaleFactor = F.getAKAZEScaleFactor(nPredictedLevel);
                }

                const vector<size_t> vIndices = F.GetFeaturesInArea(pMP->mTrackProjX,pMP->mTrackProjY,r*fScaleFactor,
                                            nPredictedLevel-1,nPredictedLevel);

                if(!vIndices.empty()){

                    const bool isORBMP = pMP->isORBMapPoint();

                    const cv::Mat MPdescriptor = pMP->GetDescriptor();

                    int bestDist=256;
                    int bestLevel= -1;
                    int bestDist2=256;
                    int bestLevel2 = -1;
                    int bestIdx =-1 ;

                    // Get best and second matches with near keypoints
                    for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
                    {
                        const size_t idx = *vit;

                        if(F.getMapPoint(idx))
                            if(F.getMapPoint(idx)->Observations()>0)
                                continue;

                        if(F.numKPtsLeft() == -1 && F.mvuRight[idx]>0)
                        {
                            const float er = fabs(pMP->mTrackProjXR-F.mvuRight[idx]);
                            if(er>r*fScaleFactor)
                                continue;
                        }

                        const bool isORBPt = F.isORBDescValid(idx);

                        if (isORBMP != isORBPt)
                            continue;

                        const cv::Mat &d = F.getDescriptorMono(idx);

                        const int dist = DescriptorDistance(MPdescriptor,d);

                        if(dist<bestDist)
                        {
                            bestDist2=bestDist;
                            bestDist=dist;
                            bestLevel2 = bestLevel;
                            //bestLevel = (F.numKPtsLeft() == -1) ? F.getUndistKPtMono(idx).octave
                            //                                    : (idx < F.numKPtsLeft()) ? F.getDistKPtMono(idx).octave
                            //                                                              : F.getKPtRight(idx - F.numKPtsLeft()).octave;
                            bestLevel = F.getKPtLevelMono(idx);
                            bestIdx=idx;
                        }
                        else if(dist<bestDist2)
                        {
                            //bestLevel2 = (F.numKPtsLeft() == -1) ? F.getUndistKPtMono(idx).octave
                            //                                     : (idx < F.numKPtsLeft()) ? F.getDistKPtMono(idx).octave
                            //                                                               : F.getKPtRight(idx - F.numKPtsLeft()).octave;
                            bestLevel2 = F.getKPtLevelMono(idx);
                            bestDist2=dist;
                        }
                    }

                    // Apply ratio to second match (only if best and second are in the same scale level)
                    if(bestDist<=TH_HIGH)
                    {
                        if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                            continue;

                        if(bestLevel!=bestLevel2 || bestDist<=mfNNratio*bestDist2){
                            F.setMapPoint(bestIdx, pMP);

                            if(F.numKPtsLeft() != -1 && F.mvLeftToRightMatch[bestIdx] != -1){ //Also match with the stereo observation at right camera
                                F.setMapPoint(F.mvLeftToRightMatch[bestIdx] + F.numKPtsLeft(), pMP);
                                nmatches++;
                                right++;
                            }

                            nmatches++;
                            left++;
                        }
                    }
                }
            }

            if(F.numKPtsLeft() != -1 && pMP->mbTrackInViewR){
                const int &nPredictedLevel = pMP->mnTrackScaleLevelR;
                if(nPredictedLevel != -1){
                    float r = RadiusByViewingCos(pMP->mTrackViewCosR);

                    float fScaleFactor = F.getORBScaleFactor(nPredictedLevel);
                    if (!pMP->isORBMapPoint()) {
                        fScaleFactor = F.getAKAZEScaleFactor(nPredictedLevel);
                    }

                    const vector<size_t> vIndices = F.GetFeaturesInArea(pMP->mTrackProjXR,pMP->mTrackProjYR,r*fScaleFactor,
                                                nPredictedLevel-1, nPredictedLevel,true);

                    if(vIndices.empty())
                        continue;

                    const cv::Mat MPdescriptor = pMP->GetDescriptor();

                    int bestDist=256;
                    int bestLevel= -1;
                    int bestDist2=256;
                    int bestLevel2 = -1;
                    int bestIdx =-1 ;

                    // Get best and second matches with near keypoints
                    for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
                    {
                        const size_t idx = *vit;

                        if(F.getMapPoint(idx + F.numKPtsLeft()))
                            if(F.getMapPoint(idx + F.numKPtsLeft())->Observations()>0)
                                continue;


                        const cv::Mat &d = F.getORBDescriptor(idx + F.numKPtsLeft());

                        const int dist = DescriptorDistance(MPdescriptor,d);

                        if(dist<bestDist)
                        {
                            bestDist2=bestDist;
                            bestDist=dist;
                            bestLevel2 = bestLevel;
                            bestLevel = F.getKPtRight(idx).octave;
                            bestIdx=idx;
                        }
                        else if(dist<bestDist2)
                        {
                            bestLevel2 = F.getKPtRight(idx).octave;
                            bestDist2=dist;
                        }
                    }

                    // Apply ratio to second match (only if best and second are in the same scale level)
                    if(bestDist<=TH_HIGH)
                    {
                        if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                            continue;

                        if(F.numKPtsLeft() != -1 && F.mvRightToLeftMatch[bestIdx] != -1){ //Also match with the stereo observation at right camera
                            F.setMapPoint(F.mvRightToLeftMatch[bestIdx], pMP);
                            nmatches++;
                            left++;
                        }


                        F.setMapPoint(bestIdx + F.numKPtsLeft(), pMP);
                        nmatches++;
                        right++;
                    }
                }
            }
        }
        return nmatches;
    }

    int MixedMatcher::SearchByProjection(MixedFrame &CurrentFrame, const MixedFrame &LastFrame, const float th, const bool bMono)
    {
        int nmatches = 0;

        // Rotation Histogram (to check rotation consistency)
        vector<int> rotHist[HISTO_LENGTH];
        for(int i=0;i<HISTO_LENGTH;i++)
            rotHist[i].reserve(500);
        const float factor = 1.0f/HISTO_LENGTH;

        const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
        const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);

        const cv::Mat twc = -Rcw.t()*tcw;

        const cv::Mat Rlw = LastFrame.mTcw.rowRange(0,3).colRange(0,3);
        const cv::Mat tlw = LastFrame.mTcw.rowRange(0,3).col(3);

        const cv::Mat tlc = Rlw*twc+tlw;

        const bool bForward = tlc.at<float>(2)>CurrentFrame.mb && !bMono;
        const bool bBackward = -tlc.at<float>(2)>CurrentFrame.mb && !bMono;

        for(int i=0; i<LastFrame.numAllKPts(); i++)
        {
            MapPoint* pMP = LastFrame.getMapPoint(i);
            if(pMP)
            {
                if(!LastFrame.getMPOutlier(i))
                {
                    const bool isORBMP = LastFrame.isORBDescValid(i);

                    // Project
                    cv::Mat x3Dw = pMP->GetWorldPos();
                    cv::Mat x3Dc = Rcw*x3Dw+tcw;

                    const float xc = x3Dc.at<float>(0);
                    const float yc = x3Dc.at<float>(1);
                    const float invzc = 1.0/x3Dc.at<float>(2);

                    if(invzc<0)
                        continue;

                    cv::Point2f uv = CurrentFrame.mpCamera->project(x3Dc);

                    if(uv.x<CurrentFrame.mnMinX || uv.x>CurrentFrame.mnMaxX)
                        continue;
                    if(uv.y<CurrentFrame.mnMinY || uv.y>CurrentFrame.mnMaxY)
                        continue;

                    //int nLastOctave = (LastFrame.numKPtsLeft() == -1 || i < LastFrame.numKPtsLeft()) ? LastFrame.getDistKPtMono(i).octave
                    //                                                                                 : LastFrame.getKPtRight(i - LastFrame.numKPtsLeft()).octave;
                    int nLastOctave = LastFrame.getKPtLevelMono(i);

                    // Search in a window. Size depends on scale
                    float fScaleFactor = CurrentFrame.getORBScaleFactor(nLastOctave);
                    if (!isORBMP) {
                        fScaleFactor = CurrentFrame.getAKAZEScaleFactor(nLastOctave);
                    }
                    float radius = th*fScaleFactor; //getORBScaleFactor(nLastOctave);

                    vector<size_t> vIndices2;

                    if(bForward)
                        vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x,uv.y, radius, nLastOctave);
                    else if(bBackward)
                        vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x,uv.y, radius, 0, nLastOctave);
                    else
                        vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x,uv.y, radius, nLastOctave-1, nLastOctave+1);

                    if(vIndices2.empty())
                        continue;

                    const cv::Mat dMP = pMP->GetDescriptor();

                    int bestDist = 256;
                    int bestIdx2 = -1;

                    for(vector<size_t>::const_iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
                    {
                        const size_t i2 = *vit;

                        if(CurrentFrame.getMapPoint(i2))
                            if(CurrentFrame.getMapPoint(i2)->Observations()>0)
                                continue;

                        if(CurrentFrame.numKPtsLeft() == -1 && CurrentFrame.mvuRight[i2]>0)
                        {
                            const float ur = uv.x - CurrentFrame.mbf*invzc;
                            const float er = fabs(ur - CurrentFrame.mvuRight[i2]);
                            if(er>radius)
                                continue;
                        }

                        const bool isORBPt = CurrentFrame.isORBDescValid(i2);

                        if (isORBMP != isORBPt)
                            continue;

                        const cv::Mat &d = CurrentFrame.getDescriptorMono(i2);

                        const int dist = DescriptorDistance(dMP,d);

                        if(dist<bestDist)
                        {
                            bestDist=dist;
                            bestIdx2=i2;
                        }
                    }

                    if(bestDist<=TH_HIGH)
                    {
                        CurrentFrame.setMapPoint(bestIdx2, pMP);
                        nmatches++;

                        if(mbCheckOrientation)
                        {
                            cv::KeyPoint kpLF = (LastFrame.numKPtsLeft() == -1) ? LastFrame.getUndistKPtMono(i)
                                                                                : (i < LastFrame.numKPtsLeft()) ? LastFrame.getDistKPtMono(i)
                                                                                                                : LastFrame.getKPtRight(i - LastFrame.numKPtsLeft());

                            cv::KeyPoint kpCF = (CurrentFrame.numKPtsLeft() == -1) ? CurrentFrame.getUndistKPtMono(bestIdx2)
                                                                                   : (bestIdx2 < CurrentFrame.numKPtsLeft()) ? CurrentFrame.getDistKPtMono(bestIdx2)
                                                                                                                             : CurrentFrame.getKPtRight(bestIdx2 - CurrentFrame.numKPtsLeft());
                            float rot = kpLF.angle-kpCF.angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdx2);
                        }
                    }
                    if(CurrentFrame.numKPtsLeft() != -1){
                        cv::Mat x3Dr = CurrentFrame.mTrl.colRange(0,3).rowRange(0,3) * x3Dc + CurrentFrame.mTrl.col(3);

                        cv::Point2f uv = CurrentFrame.mpCamera->project(x3Dr);

                        //int nLastOctave = (LastFrame.numKPtsLeft() == -1 || i < LastFrame.numKPtsLeft()) ? LastFrame.getDistKPtMono(i).octave
                        //                                                                                 : LastFrame.getKPtRight(i - LastFrame.numKPtsLeft()).octave;
                        int nLastOctave = LastFrame.getKPtLevelMono(i);

                        // Search in a window. Size depends on scale
                        float fScaleFactor = CurrentFrame.getORBScaleFactor(nLastOctave);
                        if (!isORBMP) {
                            fScaleFactor = CurrentFrame.getAKAZEScaleFactor(nLastOctave);
                        }
                        float radius = th*fScaleFactor; //getORBScaleFactor(nLastOctave);

                        vector<size_t> vIndices2;

                        if(bForward)
                            vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x,uv.y, radius, nLastOctave, -1,true);
                        else if(bBackward)
                            vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x,uv.y, radius, 0, nLastOctave, true);
                        else
                            vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x,uv.y, radius, nLastOctave-1, nLastOctave+1, true);

                        const cv::Mat dMP = pMP->GetDescriptor();

                        int bestDist = 256;
                        int bestIdx2 = -1;

                        for(vector<size_t>::const_iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
                        {
                            const size_t i2 = *vit;
                            if(CurrentFrame.getMapPoint(i2 + CurrentFrame.numKPtsLeft()))
                                if(CurrentFrame.getMapPoint(i2 + CurrentFrame.numKPtsLeft())->Observations()>0)
                                    continue;

                            const cv::Mat &d = CurrentFrame.getORBDescriptor(i2 + CurrentFrame.numKPtsLeft());

                            const int dist = DescriptorDistance(dMP,d);

                            if(dist<bestDist)
                            {
                                bestDist=dist;
                                bestIdx2=i2;
                            }
                        }

                        if(bestDist<=TH_HIGH)
                        {
                            CurrentFrame.setMapPoint(bestIdx2 + CurrentFrame.numKPtsLeft(), pMP);
                            nmatches++;
                            if(mbCheckOrientation)
                            {
                                cv::KeyPoint kpLF = (LastFrame.numKPtsLeft() == -1) ? LastFrame.getUndistKPtMono(i)
                                                                                    : (i < LastFrame.numKPtsLeft()) ? LastFrame.getDistKPtMono(i)
                                                                                                                    : LastFrame.getKPtRight(i - LastFrame.numKPtsLeft());

                                cv::KeyPoint kpCF = CurrentFrame.getKPtRight(bestIdx2);

                                float rot = kpLF.angle-kpCF.angle;
                                if(rot<0.0)
                                    rot+=360.0f;
                                int bin = round(rot*factor);
                                if(bin==HISTO_LENGTH)
                                    bin=0;
                                assert(bin>=0 && bin<HISTO_LENGTH);
                                rotHist[bin].push_back(bestIdx2  + CurrentFrame.numKPtsLeft());
                            }
                        }

                    }
                }
            }
        }

        //Apply rotation consistency
        if(mbCheckOrientation)
        {
            int ind1=-1;
            int ind2=-1;
            int ind3=-1;

            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

            for(int i=0; i<HISTO_LENGTH; i++)
            {
                if(i!=ind1 && i!=ind2 && i!=ind3)
                {
                    for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                    {
                        CurrentFrame.setMapPoint(rotHist[i][j], static_cast<MapPoint*>(NULL));
                        nmatches--;
                    }
                }
            }
        }

        return nmatches;
    }

    int MixedMatcher::SearchByProjection(MixedFrame &CurrentFrame, MixedKeyFrame *pKF, const set<ORB_SLAM3::MapPoint*> &sAlreadyFound,
                                         const float th , const int ORBdist)
    {
        int nmatches = 0;

        const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
        const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);
        const cv::Mat Ow = -Rcw.t()*tcw;

        // Rotation Histogram (to check rotation consistency)
        vector<int> rotHist[HISTO_LENGTH];
        for(int i=0;i<HISTO_LENGTH;i++)
            rotHist[i].reserve(500);
        const float factor = 1.0f/HISTO_LENGTH;

        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMPs[i];

            if(pMP)
            {
                if(!pMP->isBad() && !sAlreadyFound.count(pMP))
                {
                    //Project
                    cv::Mat x3Dw = pMP->GetWorldPos();
                    cv::Mat x3Dc = Rcw*x3Dw+tcw;

                    const cv::Point2f uv = CurrentFrame.mpCamera->project(x3Dc);

                    if(uv.x<CurrentFrame.mnMinX || uv.x>CurrentFrame.mnMaxX)
                        continue;
                    if(uv.y<CurrentFrame.mnMinY || uv.y>CurrentFrame.mnMaxY)
                        continue;

                    // Compute predicted scale level
                    cv::Mat PO = x3Dw-Ow;
                    float dist3D = cv::norm(PO);

                    const float maxDistance = pMP->GetMaxDistanceInvariance();
                    const float minDistance = pMP->GetMinDistanceInvariance();

                    // Depth must be inside the scale pyramid of the image
                    if(dist3D<minDistance || dist3D>maxDistance)
                        continue;

                    int nPredictedLevel = pMP->PredictScale(dist3D,&CurrentFrame);

                    float fScaleFactor = CurrentFrame.getORBScaleFactor(nPredictedLevel);
                    if (!pMP->isORBMapPoint()) {
                        fScaleFactor = CurrentFrame.getAKAZEScaleFactor(nPredictedLevel);
                    }
                    // Search in a window
                    const float radius = th*fScaleFactor;

                    const vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x, uv.y, radius,
                            nPredictedLevel-1, nPredictedLevel+1);

                    if(vIndices2.empty())
                        continue;

                    const bool isORBMP = pKF->isORBDescValid(i);

                    const cv::Mat dMP = pMP->GetDescriptor();

                    int bestDist = 256;
                    int bestIdx2 = -1;

                    for(vector<size_t>::const_iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
                    {
                        const size_t i2 = *vit;
                        if(CurrentFrame.getMapPoint(i2))
                            continue;

                        const bool isORBPt = CurrentFrame.isORBDescValid(i2);

                        if (isORBMP != isORBPt)
                            continue;

                        const cv::Mat &d = CurrentFrame.getDescriptorMono(i2);

                        const int dist = DescriptorDistance(dMP,d);

                        if(dist<bestDist)
                        {
                            bestDist=dist;
                            bestIdx2=i2;
                        }
                    }

                    if(bestDist<=ORBdist)
                    {
                        CurrentFrame.setMapPoint(bestIdx2,pMP);
                        nmatches++;

                        if(mbCheckOrientation)
                        {
                            float rot = pKF->getUndistKPtMono(i).angle-CurrentFrame.getUndistKPtMono(bestIdx2).angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdx2);
                        }
                    }

                }
            }
        }

        if(mbCheckOrientation)
        {
            int ind1=-1;
            int ind2=-1;
            int ind3=-1;

            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

            for(int i=0; i<HISTO_LENGTH; i++)
            {
                if(i!=ind1 && i!=ind2 && i!=ind3)
                {
                    for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                    {
                        CurrentFrame.setMapPoint(rotHist[i][j], NULL);
                        nmatches--;
                    }
                }
            }
        }

        return nmatches;
    }

    int MixedMatcher::SearchByProjection(MixedKeyFrame* pKF, const cv::Mat& Scw, const vector<ORB_SLAM3::MapPoint*> &vpPoints,
                                       vector<ORB_SLAM3::MapPoint*> &vpMatched, int th, float ratioHamming)
    {
        // Get Calibration Parameters for later projection
        const float &fx = pKF->fx;
        const float &fy = pKF->fy;
        const float &cx = pKF->cx;
        const float &cy = pKF->cy;

        // Decompose Scw
        cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
        const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
        cv::Mat Rcw = sRcw/scw;
        cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
        cv::Mat Ow = -Rcw.t()*tcw;

        // Set of MapPoints already found in the KeyFrame
        set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
        spAlreadyFound.erase(static_cast<MapPoint*>(NULL));

        int nmatches=0;

        // For each Candidate MapPoint Project and Match
        for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
        {
            MapPoint* pMP = vpPoints[iMP];

            // Discard Bad MapPoints and already found
            if(pMP->isBad() || spAlreadyFound.count(pMP))
                continue;

            const bool isORBMP = pMP->isORBMapPoint();

            // Get 3D Coords.
            cv::Mat p3Dw = pMP->GetWorldPos();

            // Transform into Camera Coords.
            cv::Mat p3Dc = Rcw*p3Dw+tcw;

            // Depth must be positive
            if(p3Dc.at<float>(2)<0.0)
                continue;

            // Project into Image
            const float x = p3Dc.at<float>(0);
            const float y = p3Dc.at<float>(1);
            const float z = p3Dc.at<float>(2);

            const cv::Point2f uv = pKF->mpCamera->project(cv::Point3f(x,y,z));

            // Point must be inside the image
            if(!pKF->IsInImage(uv.x,uv.y))
                continue;

            // Depth must be inside the scale invariance region of the point
            const float maxDistance = pMP->GetMaxDistanceInvariance();
            const float minDistance = pMP->GetMinDistanceInvariance();
            cv::Mat PO = p3Dw-Ow;
            const float dist = cv::norm(PO);

            if(dist<minDistance || dist>maxDistance)
                continue;

            // Viewing angle must be less than 60 deg
            cv::Mat Pn = pMP->GetNormal();

            if(PO.dot(Pn)<0.5*dist)
                continue;

            int nPredictedLevel = pMP->PredictScale(dist,pKF);

            // Search in a radius
            float fScaleFactor = pKF->getORBScaleFactor(nPredictedLevel);
            if (!isORBMP) {
                fScaleFactor = pKF->getAKAZEScaleFactor(nPredictedLevel);
            }
            const float radius = th*fScaleFactor;

            const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv.x,uv.y,radius);

            if(vIndices.empty())
                continue;

            // Match to the most similar keypoint in the radius
            const cv::Mat dMP = pMP->GetDescriptor();

            int bestDist = 256;
            int bestIdx = -1;
            for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
            {
                const size_t idx = *vit;
                if(vpMatched[idx])
                    continue;

                const int &kpLevel= pKF->getKPtLevelMono(idx);

                if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                    continue;

                const bool isORBPt = pKF->isORBDescValid(idx);

                if (isORBMP != isORBPt)
                    continue;

                const cv::Mat &dKF = pKF->getDescriptorMono(idx);

                const int dist = DescriptorDistance(dMP,dKF);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            if(bestDist<=TH_LOW*ratioHamming)
            {
                vpMatched[bestIdx]=pMP;
                nmatches++;
            }

        }

        return nmatches;
    }

    int MixedMatcher::SearchByProjection(MixedKeyFrame* pKF, const cv::Mat& Scw, const std::vector<ORB_SLAM3::MapPoint*> &vpPoints,
                                       const std::vector<ORB_SLAM3::KeyFrame*> &vpPointsKFs, std::vector<ORB_SLAM3::MapPoint*> &vpMatched,
                                       std::vector<ORB_SLAM3::KeyFrame*> &vpMatchedKF, int th, float ratioHamming)
    {
        // Get Calibration Parameters for later projection
        const float &fx = pKF->fx;
        const float &fy = pKF->fy;
        const float &cx = pKF->cx;
        const float &cy = pKF->cy;

        // Decompose Scw
        cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
        const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
        cv::Mat Rcw = sRcw/scw;
        cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
        cv::Mat Ow = -Rcw.t()*tcw;

        // Set of MapPoints already found in the KeyFrame
        set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
        spAlreadyFound.erase(static_cast<MapPoint*>(NULL));

        int nmatches=0;

        // For each Candidate MapPoint Project and Match
        for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
        {
            MapPoint* pMP = vpPoints[iMP];
            KeyFrame* pKFi = vpPointsKFs[iMP];
            auto* pMixedKFi = dynamic_cast<MixedKeyFrame*>(pKFi);

            // All things must be mixed
            if (!pMixedKFi)
                continue;

            // Discard Bad MapPoints and already found
            if(pMP->isBad() || spAlreadyFound.count(pMP))
                continue;

            const bool isORBMP = pMP->isORBMapPoint();

            // Get 3D Coords.
            cv::Mat p3Dw = pMP->GetWorldPos();

            // Transform into Camera Coords.
            cv::Mat p3Dc = Rcw*p3Dw+tcw;

            // Depth must be positive
            if(p3Dc.at<float>(2)<0.0)
                continue;

            // Project into Image
            const float invz = 1/p3Dc.at<float>(2);
            const float x = p3Dc.at<float>(0)*invz;
            const float y = p3Dc.at<float>(1)*invz;

            const float u = fx*x+cx;
            const float v = fy*y+cy;

            // Point must be inside the image
            if(!pKF->IsInImage(u,v))
                continue;

            // Depth must be inside the scale invariance region of the point
            const float maxDistance = pMP->GetMaxDistanceInvariance();
            const float minDistance = pMP->GetMinDistanceInvariance();
            cv::Mat PO = p3Dw-Ow;
            const float dist = cv::norm(PO);

            if(dist<minDistance || dist>maxDistance)
                continue;

            // Viewing angle must be less than 60 deg
            cv::Mat Pn = pMP->GetNormal();

            if(PO.dot(Pn)<0.5*dist)
                continue;

            int nPredictedLevel = pMP->PredictScale(dist,pKF);

            // Search in a radius
            float fScaleFactor = pKF->getORBScaleFactor(nPredictedLevel);
            if (!isORBMP) {
                fScaleFactor = pKF->getAKAZEScaleFactor(nPredictedLevel);
            }
            const float radius = th*fScaleFactor;

            const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

            if(vIndices.empty())
                continue;

            // Match to the most similar keypoint in the radius
            const cv::Mat dMP = pMP->GetDescriptor();

            int bestDist = 256;
            int bestIdx = -1;
            for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
            {
                const size_t idx = *vit;
                if(vpMatched[idx])
                    continue;

                const int &kpLevel= pKF->getKPtLevelMono(idx);

                if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                    continue;

                const bool isORBPt = pKF->isORBDescValid(idx);

                if (isORBPt != isORBMP)
                    continue;

                const cv::Mat &dKF = pKF->getDescriptorMono(idx);

                const int dist = DescriptorDistance(dMP,dKF);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            if(bestDist<=TH_LOW*ratioHamming)
            {
                vpMatched[bestIdx] = pMP;
                vpMatchedKF[bestIdx] = pKFi;
                nmatches++;
            }

        }

        return nmatches;
    }

    int MixedMatcher::SearchForTriangulation(MixedKeyFrame *pKF1, MixedKeyFrame *pKF2, const cv::Mat& F12,
                                           vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo, const bool bCoarse)
    {
        const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
        const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

        //Compute epipole in second image
        cv::Mat Cw = pKF1->GetCameraCenter();
        cv::Mat R2w = pKF2->GetRotation();
        cv::Mat t2w = pKF2->GetTranslation();
        cv::Mat C2 = R2w*Cw+t2w;

        cv::Point2f ep = pKF2->mpCamera->project(C2);

        cv::Mat R1w = pKF1->GetRotation();
        cv::Mat t1w = pKF1->GetTranslation();

        cv::Mat R12;
        cv::Mat t12;

        cv::Mat Rll,Rlr,Rrl,Rrr;
        cv::Mat tll,tlr,trl,trr;

        GeometricCamera* pCamera1 = pKF1->mpCamera, *pCamera2 = pKF2->mpCamera;

        if(!pKF1->mpCamera2 && !pKF2->mpCamera2){
            R12 = R1w*R2w.t();
            t12 = -R1w*R2w.t()*t2w+t1w;
        }
        else{
            Rll = pKF1->GetRotation() * pKF2->GetRotation().t();
            Rlr = pKF1->GetRotation() * pKF2->GetRightRotation().t();
            Rrl = pKF1->GetRightRotation() * pKF2->GetRotation().t();
            Rrr = pKF1->GetRightRotation() * pKF2->GetRightRotation().t();

            tll = pKF1->GetRotation() * (-pKF2->GetRotation().t() * pKF2->GetTranslation()) + pKF1->GetTranslation();
            tlr = pKF1->GetRotation() * (-pKF2->GetRightRotation().t() * pKF2->GetRightTranslation()) + pKF1->GetTranslation();
            trl = pKF1->GetRightRotation() * (-pKF2->GetRotation().t() * pKF2->GetTranslation()) + pKF1->GetRightTranslation();
            trr = pKF1->GetRightRotation() * (-pKF2->GetRightRotation().t() * pKF2->GetRightTranslation()) + pKF1->GetRightTranslation();
        }

        // Find matches between not tracked keypoints
        // Matching speed-up by ORB Vocabulary
        // Compare only ORB that share the same node

        int nmatches=0;
        vector<bool> vbMatched2(pKF2->numAllKPts(),false);
        vector<int> vMatches12(pKF1->numAllKPts(),-1);

        vector<int> rotHist[HISTO_LENGTH];
        for(int i=0;i<HISTO_LENGTH;i++)
            rotHist[i].reserve(500);

        const float factor = 1.0f/HISTO_LENGTH;

        DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
        DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
        DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
        DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

        while(f1it!=f1end && f2it!=f2end)
        {
            if(f1it->first == f2it->first)
            {
                for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
                {
                    const size_t idx1 = f1it->second[i1];

                    // Match only ORB Descriptors
                    if (!pKF1->isORBDescValid(idx1))
                        continue;

                    MapPoint* pMP1 = pKF1->GetMapPoint(idx1);

                    // If there is already a MapPoint skip
                    if(pMP1)
                    {
                        continue;
                    }

                    const bool bStereo1 = (!pKF1->mpCamera2 && pKF1->mvuRight[idx1]>=0);

                    if(bOnlyStereo)
                        if(!bStereo1)
                            continue;


                    const cv::KeyPoint &kp1 = (pKF1 -> numAllKPtsLeft() == -1) ? pKF1->getUndistKPtMono(idx1)
                                                                               : (idx1 < pKF1 -> numAllKPtsLeft()) ? pKF1 -> getDistKPtMono(idx1)
                                                                                                                   : pKF1 -> getKPtRight(idx1 - pKF1 -> numAllKPtsLeft());

                    const bool bRight1 = !(pKF1->numAllKPtsLeft() == -1 || idx1 < pKF1->numAllKPtsLeft());

                    //if(bRight1) continue;
                    const cv::Mat &d1 = pKF1->getDescriptorMono(idx1);

                    int bestDist = TH_LOW;
                    int bestIdx2 = -1;

                    for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                    {
                        size_t idx2 = f2it->second[i2];

                        if (!pKF2->isORBDescValid(idx2))
                            continue;

                        MapPoint* pMP2 = pKF2->GetMapPoint(idx2);

                        // If we have already matched or there is a MapPoint skip
                        if(vbMatched2[idx2] || pMP2)
                            continue;

                        const bool bStereo2 = (!pKF2->mpCamera2 &&  pKF2->mvuRight[idx2]>=0);

                        if(bOnlyStereo)
                            if(!bStereo2)
                                continue;

                        const cv::Mat &d2 = pKF2->getDescriptorMono(idx2);

                        const int dist = DescriptorDistance(d1,d2);

                        if(dist>TH_LOW || dist>bestDist)
                            continue;

                        const cv::KeyPoint &kp2 = (pKF2 -> numAllKPtsLeft() == -1) ? pKF2->getUndistKPtMono(idx2)
                                                                                   : (idx2 < pKF2 -> numAllKPtsLeft()) ? pKF2 -> getDistKPtMono(idx2)
                                                                                                                       : pKF2 -> getKPtRight(idx2 - pKF2 -> numAllKPtsLeft());
                        const bool bRight2 = !(pKF2->numAllKPtsLeft() == -1 || idx2 < pKF2->numAllKPtsLeft());

                        if(!bStereo1 && !bStereo2 && !pKF1->mpCamera2)
                        {
                            const float distex = ep.x-kp2.pt.x;
                            const float distey = ep.y-kp2.pt.y;
                            if(distex*distex+distey*distey<100*pKF2->getORBScaleFactor(kp2.octave))
                            {
                                continue;
                            }
                        }

                        if(pKF1->mpCamera2 && pKF2->mpCamera2){
                            if(bRight1 && bRight2){
                                R12 = Rrr;
                                t12 = trr;

                                pCamera1 = pKF1->mpCamera2;
                                pCamera2 = pKF2->mpCamera2;
                            }
                            else if(bRight1 && !bRight2){
                                R12 = Rrl;
                                t12 = trl;

                                pCamera1 = pKF1->mpCamera2;
                                pCamera2 = pKF2->mpCamera;
                            }
                            else if(!bRight1 && bRight2){
                                R12 = Rlr;
                                t12 = tlr;

                                pCamera1 = pKF1->mpCamera;
                                pCamera2 = pKF2->mpCamera2;
                            }
                            else{
                                R12 = Rll;
                                t12 = tll;

                                pCamera1 = pKF1->mpCamera;
                                pCamera2 = pKF2->mpCamera;
                            }

                        }

                        if(pCamera1->epipolarConstrain(pCamera2,kp1,kp2,R12,t12,pKF1->getORBLevelSigma2(kp1.octave),
                                                       pKF2->getORBLevelSigma2(kp2.octave))||bCoarse) // MODIFICATION_2
                        {
                            bestIdx2 = idx2;
                            bestDist = dist;
                        }
                    }

                    if(bestIdx2>=0)
                    {
                        const cv::KeyPoint &kp2 = (pKF2 -> numAllKPtsLeft() == -1) ? pKF2->getUndistKPtMono(bestIdx2)
                                                                                   : (bestIdx2 < pKF2 -> numAllKPtsLeft()) ? pKF2 -> getDistKPtMono(bestIdx2)
                                                                                                                           : pKF2 -> getKPtRight(bestIdx2 - pKF2 -> numAllKPtsLeft());
                        vMatches12[idx1]=bestIdx2;
                        nmatches++;

                        if(mbCheckOrientation)
                        {
                            float rot = kp1.angle-kp2.angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(idx1);
                        }
                    }
                }

                f1it++;
                f2it++;
            }
            else if(f1it->first < f2it->first)
            {
                f1it = vFeatVec1.lower_bound(f2it->first);
            }
            else
            {
                f2it = vFeatVec2.lower_bound(f1it->first);
            }
        }

        if(mbCheckOrientation)
        {
            int ind1=-1;
            int ind2=-1;
            int ind3=-1;

            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

            for(int i=0; i<HISTO_LENGTH; i++)
            {
                if(i==ind1 || i==ind2 || i==ind3)
                    continue;
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    vMatches12[rotHist[i][j]]=-1;
                    nmatches--;
                }
            }

        }

        vMatchedPairs.clear();
        vMatchedPairs.reserve(nmatches);

        for(size_t i=0, iend=vMatches12.size(); i<iend; i++)
        {
            if(vMatches12[i]<0)
                continue;
            vMatchedPairs.push_back(make_pair(i,vMatches12[i]));
        }

        return nmatches;
    }

    int MixedMatcher::Fuse(MixedKeyFrame *pKF, const vector<ORB_SLAM3::MapPoint *> &vpMapPoints, const float th, const bool bRight)
    {
        cv::Mat Rcw,tcw, Ow;
        GeometricCamera* pCamera;

        if(bRight){
            Rcw = pKF->GetRightRotation();
            tcw = pKF->GetRightTranslation();
            Ow = pKF->GetRightCameraCenter();

            pCamera = pKF->mpCamera2;
        }
        else{
            Rcw = pKF->GetRotation();
            tcw = pKF->GetTranslation();
            Ow = pKF->GetCameraCenter();

            pCamera = pKF->mpCamera;
        }

        const float &fx = pKF->fx;
        const float &fy = pKF->fy;
        const float &cx = pKF->cx;
        const float &cy = pKF->cy;
        const float &bf = pKF->mbf;

        int nFused=0;

        const int nMPs = vpMapPoints.size();

        // For debbuging
        int count_notMP = 0, count_bad=0, count_isinKF = 0, count_negdepth = 0, count_notinim = 0,
            count_dist = 0, count_normal=0, count_notidx = 0, count_thcheck = 0;

        for(int i=0; i<nMPs; i++)
        {
            MapPoint* pMP = vpMapPoints[i];

            if(!pMP)
            {
                count_notMP++;
                continue;
            }

            /*if(pMP->isBad() || pMP->IsInKeyFrame(pKF))
                continue;*/
            if(pMP->isBad())
            {
                count_bad++;
                continue;
            }
            else if(pMP->IsInKeyFrame(pKF))
            {
                count_isinKF++;
                continue;
            }

            const bool isORBMP = pMP->isORBMapPoint();

            cv::Mat p3Dw = pMP->GetWorldPos();
            cv::Mat p3Dc = Rcw*p3Dw + tcw;

            // Depth must be positive
            if(p3Dc.at<float>(2)<0.0f)
            {
                count_negdepth++;
                continue;
            }

            const float invz = 1/p3Dc.at<float>(2);
            const float x = p3Dc.at<float>(0);
            const float y = p3Dc.at<float>(1);
            const float z = p3Dc.at<float>(2);

            const cv::Point2f uv = pCamera->project(cv::Point3f(x,y,z));

            // Point must be inside the image
            if(!pKF->IsInImage(uv.x,uv.y))
            {
                count_notinim++;
                continue;
            }

            const float ur = uv.x-bf*invz;

            const float maxDistance = pMP->GetMaxDistanceInvariance();
            const float minDistance = pMP->GetMinDistanceInvariance();
            cv::Mat PO = p3Dw-Ow;
            const float dist3D = cv::norm(PO);

            // Depth must be inside the scale pyramid of the image
            if(dist3D<minDistance || dist3D>maxDistance)
            {
                count_dist++;
                continue;
            }

            // Viewing angle must be less than 60 deg
            cv::Mat Pn = pMP->GetNormal();

            if(PO.dot(Pn)<0.5*dist3D)
            {
                count_normal++;
                continue;
            }

            int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

            // Search in a radius
            float fScaleFactor = pKF->getORBScaleFactor(nPredictedLevel);
            if (!isORBMP) {
                fScaleFactor = pKF->getAKAZEScaleFactor(nPredictedLevel);
            }
            const float radius = th*fScaleFactor;

            const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv.x,uv.y,radius,bRight);

            if(vIndices.empty())
            {
                count_notidx++;
                continue;
            }

            // Match to the most similar keypoint in the radius
            const cv::Mat dMP = pMP->GetDescriptor();

            int bestDist = 256;
            int bestIdx = -1;
            for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
            {
                size_t idx = *vit;

                const bool isORBPt = pKF->isORBDescValid(idx);

                if (isORBMP != isORBPt)
                    continue;

                const cv::KeyPoint &kp = (pKF -> numAllKPtsLeft() == -1) ? pKF->getUndistKPtMono(idx)
                                                                         : (!bRight) ? pKF -> getDistKPtMono(idx)
                                                                                     : pKF -> getKPtRight(idx);

                const int &kpLevel= pKF->getKPtLevelMono(idx);

                if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                    continue;

                if(pKF->mvuRight[idx]>=0)
                {
                    // Check reprojection error in stereo
                    const float &kpx = kp.pt.x;
                    const float &kpy = kp.pt.y;
                    const float &kpr = pKF->mvuRight[idx];
                    const float ex = uv.x-kpx;
                    const float ey = uv.y-kpy;
                    const float er = ur-kpr;
                    const float e2 = ex*ex+ey*ey+er*er;

                    if(e2*pKF->getKPtInvLevelSigma2(idx)>7.8)
                        continue;
                }
                else
                {
                    const float &kpx = kp.pt.x;
                    const float &kpy = kp.pt.y;
                    const float ex = uv.x-kpx;
                    const float ey = uv.y-kpy;
                    const float e2 = ex*ex+ey*ey;

                    if(e2*pKF->getKPtInvLevelSigma2(idx)>5.99)
                        continue;
                }

                if(bRight) idx += pKF->numAllKPtsLeft();

                const cv::Mat &dKF = pKF->getDescriptorMono(idx);

                const int dist = DescriptorDistance(dMP,dKF);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            // If there is already a MapPoint replace otherwise add new measurement
            if(bestDist<=TH_LOW)
            {
                MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
                if(pMPinKF)
                {
                    if(!pMPinKF->isBad())
                    {
                        if(pMPinKF->Observations()>pMP->Observations())
                            pMP->Replace(pMPinKF);
                        else
                            pMPinKF->Replace(pMP);
                    }
                }
                else
                {
                    pMP->AddObservation(pKF,bestIdx);
                    pKF->AddMapPoint(pMP,bestIdx);
                }
                nFused++;
            }
            else
                count_thcheck++;

        }

        /*cout << "count_notMP = " << count_notMP << endl;
        cout << "count_bad = " << count_bad << endl;
        cout << "count_isinKF = " << count_isinKF << endl;
        cout << "count_negdepth = " << count_negdepth << endl;
        cout << "count_notinim = " << count_notinim << endl;
        cout << "count_dist = " << count_dist << endl;
        cout << "count_normal = " << count_normal << endl;
        cout << "count_notidx = " << count_notidx << endl;
        cout << "count_thcheck = " << count_thcheck << endl;
        cout << "tot fused points: " << nFused << endl;*/
        return nFused;
    }

    int MixedMatcher::Fuse(MixedKeyFrame *pKF, const cv::Mat& Scw, const vector<ORB_SLAM3::MapPoint *> &vpPoints,
            float th, vector<ORB_SLAM3::MapPoint *> &vpReplacePoint)
    {
        // Get Calibration Parameters for later projection
        const float &fx = pKF->fx;
        const float &fy = pKF->fy;
        const float &cx = pKF->cx;
        const float &cy = pKF->cy;

        // Decompose Scw
        cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
        const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
        cv::Mat Rcw = sRcw/scw;
        cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
        cv::Mat Ow = -Rcw.t()*tcw;

        // Set of MapPoints already found in the KeyFrame
        const set<MapPoint*> spAlreadyFound = pKF->GetMapPoints();

        int nFused=0;

        const int nPoints = vpPoints.size();

        // For each candidate MapPoint project and match
        for(int iMP=0; iMP<nPoints; iMP++)
        {
            MapPoint* pMP = vpPoints[iMP];

            // Discard Bad MapPoints and already found
            if(pMP->isBad() || spAlreadyFound.count(pMP))
                continue;

            const bool isORBMP = pMP->isORBMapPoint();

            // Get 3D Coords.
            cv::Mat p3Dw = pMP->GetWorldPos();

            // Transform into Camera Coords.
            cv::Mat p3Dc = Rcw*p3Dw+tcw;

            // Depth must be positive
            if(p3Dc.at<float>(2)<0.0f)
                continue;

            // Project into Image
            const float x = p3Dc.at<float>(0);
            const float y = p3Dc.at<float>(1);
            const float z = p3Dc.at<float>(2);

            const cv::Point2f uv = pKF->mpCamera->project(cv::Point3f(x,y,z));

            // Point must be inside the image
            if(!pKF->IsInImage(uv.x,uv.y))
                continue;

            // Depth must be inside the scale pyramid of the image
            const float maxDistance = pMP->GetMaxDistanceInvariance();
            const float minDistance = pMP->GetMinDistanceInvariance();
            cv::Mat PO = p3Dw-Ow;
            const float dist3D = cv::norm(PO);

            if(dist3D<minDistance || dist3D>maxDistance)
                continue;

            // Viewing angle must be less than 60 deg
            cv::Mat Pn = pMP->GetNormal();

            if(PO.dot(Pn)<0.5*dist3D)
                continue;

            // Compute predicted scale level
            const int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

            // Search in a radius
            float fScaleFactor = pKF->getORBScaleFactor(nPredictedLevel);
            if (!isORBMP) {
                fScaleFactor = pKF->getAKAZEScaleFactor(nPredictedLevel);
            }
            const float radius = th*fScaleFactor;

            const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv.x,uv.y,radius);

            if(vIndices.empty())
                continue;

            // Match to the most similar keypoint in the radius
            const cv::Mat dMP = pMP->GetDescriptor();

            int bestDist = INT_MAX;
            int bestIdx = -1;
            for(vector<size_t>::const_iterator vit=vIndices.begin(); vit!=vIndices.end(); vit++)
            {
                const size_t idx = *vit;

                const bool isORBPt = pKF->isORBDescValid(idx);

                if (isORBPt != isORBMP)
                    continue;

                const int &kpLevel = pKF->getKPtLevelMono(idx);

                if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                    continue;

                const cv::Mat &dKF = pKF->getDescriptorMono(idx);

                int dist = DescriptorDistance(dMP,dKF);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            // If there is already a MapPoint replace otherwise add new measurement
            if(bestDist<=TH_LOW)
            {
                MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
                if(pMPinKF)
                {
                    if(!pMPinKF->isBad())
                        vpReplacePoint[iMP] = pMPinKF;
                }
                else
                {
                    pMP->AddObservation(pKF,bestIdx);
                    pKF->AddMapPoint(pMP,bestIdx);
                }
                nFused++;
            }
        }

        return nFused;
    }

} // EORB_SLAM

