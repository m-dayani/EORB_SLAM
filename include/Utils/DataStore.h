/**
* Data Management: Load Datasets
* Author: M. Dayani
*/

#ifndef ORB_SLAM3_DATASTORE_H
#define ORB_SLAM3_DATASTORE_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <utility>
#include <vector>
#include <string>
#include <memory>

#include <boost/filesystem.hpp>
#include <boost/any.hpp>

#include <glog/logging.h>

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/core/persistence.hpp>

#include "MyDataTypes.h"
#include "MyYamlParser.h"
#include "MyParameters.h"
#include "MyCalibrator.h"


//using namespace std;

namespace EORB_SLAM {

    class BaseLoader {
    public:
        enum PathStat {
            INVALID,
            FILE,
            FILE_CSV,
            DIR
        };
        enum LoadStat {
            BAD_PATH,
            BAD_DATA,
            READY,
            GOOD
        };
        enum DsFormat {
            NOT_SUPPORTED,
            EUROC,
            EV_ETHZ,
            EV_MVSEC
        };

        BaseLoader();
        virtual ~BaseLoader() = default;

        static bool checkDirectory(const std::string &strPath);
        static bool checkFile(const std::string &strPath);
        static bool checkFileAndExtension(const std::string &strPath, const std::string &ext);
        static bool checkExtension(const boost::filesystem::path &p, const std::string &ext) {
            return p.has_extension() && p.extension().string() == ext;
        }
        static bool isComment(const std::string &txt);

        static BaseLoader::DsFormat mapDsFormats(const std::string& strFormat);
        static std::string mapDsFormats(BaseLoader::DsFormat dsFormat);

        unsigned long getCurrByteIdx() const { return this->mnCurrByteIdx; }

        virtual void reset() = 0;

    protected:
        bool checkTxtStream();

        void openFile();
        void openNextDirFile();
        void loadTxtFile();

        template<typename T>
        unsigned long getTxtData(unsigned long chunkSize, std::vector<T> &outData) {

            if (!this->mTxtDataFile.is_open()) {

                LOG(ERROR) << "Text data file is not open\n";
                return 0;
            }

            std::string line;

            unsigned long dtCount = 0;

            //If data manager constantly supply the same outData,
            // all data will be stacked together.
            if (chunkSize > 0)
                outData.reserve(chunkSize);

            while (this->checkTxtStream())
            {
                if (chunkSize > 0 && dtCount >= chunkSize)
                    break;

                getline(this->mTxtDataFile, line);
                this->mnCurrByteIdx += line.length();

                if (isComment(line))
                    continue;

                boost::any res = parseLine(line);

                outData.push_back(boost::any_cast<T>(res));

                dtCount++;
            }
            return dtCount;
        }

        virtual boost::any parseLine(const std::string &dataStr) = 0;

        std::string mDataPath;

        std::ifstream mTxtDataFile;

        std::vector<std::string> mvFileNames;
        unsigned int mnFiles;
        unsigned int mnCurrFileIdx;

        //Number of bytes read from the file
        //This can be used to seek text file
        unsigned long mnCurrByteIdx;

        PathStat mPathStat;
        LoadStat mLoadStat;

        double mTsFactor;
    private:
    };

    class ImageDataStore : public BaseLoader {
    public:
        ImageDataStore() = default;
        ImageDataStore(const std::string &filePath, const std::string &imBase, double tsFactor);
        ~ImageDataStore() override;

        std::vector<ImageData> getImageData() { return mvImageData; }
        void getImage(unsigned idx, cv::Mat &image, double &ts);

        unsigned getNumFiles() { return this->mvImageData.size(); }
        std::string getFileName(size_t idx, bool fullPath = true);
        double getTimeStamp(size_t idx);

        void reset() override;

    protected:
        boost::any parseLine(const std::string &evStr) override;
    private:

        std::string mImagesBasePath;
        std::vector<ImageData> mvImageData;
    };

    class ImuDataStore : public BaseLoader {
    public:
        ImuDataStore() = default;
        ImuDataStore(const std::string &filePath, bool _gFirst, double tsFactor);
        ~ImuDataStore() override;

        void setGyroOrder(bool _gFirst) { gFirst = _gFirst; }

        unsigned long getNextChunk(size_t offset, unsigned long chunkSize, std::vector<ImuData> &outData);
        // This will change object's mSIdx internally
        // Be careful about ts units, some datasets are in sec others are in nsecs!
        unsigned long getNextChunk(double tsEndNsec, std::vector<ORB_SLAM3::IMU::Point> &vImuMeas, double tsFactor = 1);
        unsigned long getNumData() { return mvImuData.size(); }

        void incIdx();
        void decIdx();
        void resetIdx() { mSIdx = 0; }

        void reset() override;

    protected:
        boost::any parseLine(const std::string &evStr) override;
    private:
        std::vector<ImuData> mvImuData;
        size_t mSIdx = 0;
        // For EuRoC type data gyro is first then accel
        // For Ethz public event data the ordering is reverse
        bool gFirst = true;
    };

    class GtDataStore : public BaseLoader {
    public:
        GtDataStore() = default;
        GtDataStore(const std::string &filePath, bool _qwFirst, bool _posFirst, double tsFactor);
        ~GtDataStore() override;

        void setPosOrder(bool _posFirst) { posFirst = _posFirst; }
        void setQwOrder(bool _qwFirst) { qwFirst = _qwFirst; }

        unsigned long getNextChunk(size_t offset, unsigned long chunkSize, std::vector<GtDataQuat> &outData);
        unsigned long getNumData() { return mvGtData.size(); }

        void reset() override;

    protected:
        boost::any parseLine(const std::string &evStr) override;
    private:
        std::vector<GtDataQuat> mvGtData;
        // For EuRoC type data qw is the first in quat data
        // For Ethz public event data the ordering is reverse
        // For both first is p, then q
        bool posFirst = true;
        bool qwFirst = true;
    };

    typedef std::unique_ptr<ImageDataStore> ImDsUnPtr;
    typedef std::unique_ptr<ImuDataStore> ImuDsUnPtr;
    typedef std::unique_ptr<GtDataStore> GtDsUnPtr;

    class EurocLoader {
    public:
        EurocLoader();
        explicit EurocLoader(const std::string &fSettings);
        virtual ~EurocLoader() = default;

        //Getters/Setters
        std::string getDatasetName() { return mDsName; }
        SensorConfigPtr getConfigStat() const { return mpConfig; }
        //CamParamsPtr getCamParams() const { return mpCamParams; }
        CamParamsPtr getCamParams() { return mpCamParams; }
        MyCalibPtr getCamCalibrator(bool right = false) const;
        PairCalibPtr getPairCamCalibrator() { return mppCalib; }
        MixedFtsParamsPtr getORBParams() const { return mpORBParams; }
        IMUParamsPtr getIMUParams() const { return mpIMUParams; }
        ViewerParamsPtr getViewerParams() const { return mpViewerParams; }

        unsigned int getNumSequences() const { return mSeqCount; }
        unsigned int getNumTargetSequences() const;
        unsigned int getMaxNumIter() const { return mnMaxIter; }
        std::string getSequenceName() const;
        virtual std::string getTrajFileName();
        virtual std::string getTrajFileName(uint seq);
        std::string getPathOrbVoc() { return mPathOrbVoc; }

        //Utils
        bool isGood() const;
        bool isInertial();
        void resetSequences();
        void incSequence();
        void decSequence();
        bool checkSequence(unsigned int seq) const;

        virtual std::string printLoaderStat() const;

        //Image
        unsigned int getNumImages(bool right = false);
        ulong getNumTotalImages();
        void getImage(size_t seqIdx, cv::Mat &image, double &ts, bool right = false);
        double getImageTime(size_t seqIdx, bool right = false);
        std::string getImageFileName(size_t seqIdx, bool fullName = true, bool right = false);

        //IMU
        // initTs is in (sec)
        void initImu(double initTs, int seqIdx = -1, bool right = false);
        unsigned int getNextImu(double ts, std::vector<ORB_SLAM3::IMU::Point> &vImuMeas, bool right = false);

        virtual void resetCurrSequence();

    protected:
        std::string getSequencePath();

        virtual bool resolveDsFormat(const cv::FileStorage &fsSettings);
        virtual void resolveDsPaths(const cv::FileNode &dsPathNode);
        virtual bool parseParameters(const cv::FileStorage &fsSettings);
        virtual void loadSequence(const std::string &dsRoot, const std::string &seqPath, size_t idx);
        virtual bool checkLoadStat();

        bool resolveSeqInfo(const cv::FileStorage &fsSettings);
        bool parseSettings(const std::string &settingsFile);
        void loadData();
        bool updateLoadStat();

        // The first two can be stereo in general (euroc supports stereo)
        std::vector<std::pair<ImDsUnPtr, ImDsUnPtr>> mvpImDs;
        std::vector<std::pair<ImuDsUnPtr, ImuDsUnPtr>> mvpImuDs;
        std::vector<GtDsUnPtr> mvpGtDs;

        BaseLoader::LoadStat mLoadStat;
        BaseLoader::DsFormat mDsFormat;
        std::string mDsName;

        SensorConfigPtr mpConfig;
        CamParamsPtr mpCamParams;
        MixedFtsParamsPtr mpORBParams;
        IMUParamsPtr mpIMUParams;
        ViewerParamsPtr mpViewerParams;

        std::string mPathSettings;
        std::string mPathDsRoot;
        std::string mPathOrbVoc;

        std::pair<std::string, std::string> mPathImFile;
        std::pair<std::string, std::string> mPathImBase;

        std::pair<std::string, std::string> mPathImu;

        std::string mPathGt;

        // Sequence count counts the size of active dataStores (like mImDs)
        unsigned int mSeqCount;
        int mSeqTarget;
        std::vector<std::string> mSeqNames;
        // If mSeqCount == 1, mSeqIdx must always be zero
        unsigned int mSeqIdx;
        unsigned int mnMaxIter;

        PairCalibPtr  mppCalib;

    private:

    };

} //namespace EORB_SLAM

#endif //ORB_SLAM3_DATASTORE_H
