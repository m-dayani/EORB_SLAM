//
// Created by root on 12/24/20.
//

#include "EventLoader.h"

using namespace boost::filesystem;
using namespace std;


namespace EORB_SLAM {

    /* =============================== EventDataStore ================================ */

    EventDataStore::EventDataStore(const string &evPath, const double tsFactor) : mLastEvTs(0.0)
    {
        boost::filesystem::path p(evPath);  // avoid repeated path construction below
        mTsFactor = tsFactor;

        try {
            if (exists(p))    // does path p actually exist?
            {
                this->mDataPath = evPath;
                if (is_regular_file(p))        // is path p a regular file?
                {
                    mPathStat = PathStat::FILE;
                    if (checkExtension(p, ".txt")) {
                        mLoadStat = LoadStat::READY;
                    }
                    else {
                        mLoadStat = LoadStat::BAD_PATH;
                    }
                }
                else if (is_directory(p))      // is path p a directory?
                {
                    mPathStat = PathStat::DIR;

                    for (auto&& x : boost::filesystem::directory_iterator(p))
                    {
                        const boost::filesystem::path& evFilePath = x.path();
                        if (checkExtension(evFilePath, ".txt"))
                        {
                            mvFileNames.push_back(evFilePath.string());
                            mnFiles++;
                        }
                    }
                    if (mnFiles > 0) {
                        mLoadStat = LoadStat::READY;
                        std::sort(mvFileNames.begin(), mvFileNames.end());
                    }
                    else {
                        mLoadStat = LoadStat::BAD_PATH;
                    }
                }
                else {
                    cout << p << " exists, but is not a regular file or directory\n";
                }

                this->loadTxtFile();
            }
            else {
                cerr << p << " does not exist\n";
            }
        }
        catch (const boost::filesystem::filesystem_error& ex)
        {
            cerr << ex.what() << '\n';
        }

    }

    EventDataStore::~EventDataStore()
    {
        if (mTxtDataFile.is_open())
        {
            mTxtDataFile.close();
        }
    }

    boost::any EventDataStore::parseLine(const string &evStr) {

        std::istringstream stream(evStr);

        double ts;
        float x, y;
        bool p;

        stream >> ts >> x >> y >> p;

        EventData evData(ts/mTsFactor, x, y, p);
        return boost::any(evData);
    }

    EventData EventDataStore::parseLine(const std::string &evStr, const cv::Mat& K,
            const cv::Mat& distCoefs, const cv::Mat& rectMat) {

        std::istringstream stream(evStr);

        double ts;
        float x, y;
        bool p;

        stream >> ts >> x >> y >> p;

        cv::Point2f evPt(x, y);
        MyCalibrator::undistPointPinhole(evPt, evPt, K, distCoefs, rectMat);

        return {ts/mTsFactor, evPt.x, evPt.y, p};
    }

    EventData EventDataStore::parseLine(const std::string &evStr, const MyCalibPtr& pCalib) {

        std::istringstream stream(evStr);

        double ts;
        float x, y;
        bool p;

        stream >> ts >> x >> y >> p;

        cv::Point2f evPt(x, y);
        pCalib->undistPointMaps(evPt, evPt);

        return {ts/mTsFactor, evPt.x, evPt.y, p};
    }

    unsigned long EventDataStore::getEventChunk(const unsigned long chunkSize, vector<EventData> &evBuffer) {

        return getTxtData(chunkSize, evBuffer);
    }

    unsigned long EventDataStore::getEventChunk(const double tsEnd, vector<EventData> &evBuffer) {

        if (!this->mTxtDataFile.is_open()) {

            LOG(ERROR) << "Text data file is not open\n";
            return 0;
        }

        string line;

        unsigned long dtCount = 0;

        //If data manager constantly supply the same outData,
        // all data will be stacked together.
        if (tsEnd <= mLastEvTs) {

            LOG(WARNING) << "Same event timestamp supplied: " << tsEnd << endl;
            return 0;
        }

        evBuffer.reserve(DEF_L1_CHUNK_SIZE * DEF_L1_NUM_LOOP);

        while (this->checkTxtStream())
        {
            getline(this->mTxtDataFile, line);
            this->mnCurrByteIdx += line.length();

            if (isComment(line))
                continue;

            EventData currEv = boost::any_cast<EventData>(this->parseLine(line));
            evBuffer.push_back(currEv);

            dtCount++;

            if (currEv.ts >= tsEnd) {
                mLastEvTs = currEv.ts;
                break;
            }
        }
        return dtCount;
    }

    unsigned long EventDataStore::getEventChunkRectified(unsigned long chunkSize, vector<EventData> &evBuffer,
            const cv::Mat& K, const cv::Mat& distCoefs, const cv::Mat& rectMat,
            const cv::Scalar& imageSize, const bool checkInImage) {

        if (!this->mTxtDataFile.is_open()) {

            LOG(ERROR) << "Text data file is not open\n";
            return 0;
        }

        string line;

        unsigned long dtCount = 0;

        //If data manager constantly supply the same outData,
        // all data will be stacked together.
        if (chunkSize > 0)
            evBuffer.reserve(chunkSize);

        while (this->checkTxtStream())
        {
            if (chunkSize > 0 && dtCount >= chunkSize)
                break;

            getline(this->mTxtDataFile, line);
            this->mnCurrByteIdx += line.length();

            if (isComment(line))
                continue;

            EventData currEv = this->parseLine(line, K, distCoefs, rectMat);

            if (checkInImage && !MyCalibrator::isInImage(currEv.x, currEv.y, imageSize))
                continue;

            evBuffer.push_back(currEv);

            dtCount++;
        }
        return dtCount;
    }

    unsigned long EventDataStore::getEventChunkRectified(double tsEnd, vector<EventData> &evBuffer,
            const cv::Mat& K, const cv::Mat& distCoefs, const cv::Mat& rectMat,
            const cv::Scalar& imageSize, const bool checkInImage) {

        if (!this->mTxtDataFile.is_open()) {

            LOG(ERROR) << "Text data file is not open\n";
            return 0;
        }

        string line;

        unsigned long dtCount = 0;

        //If data manager constantly supply the same outData,
        // all data will be stacked together.
        if (tsEnd <= mLastEvTs) {
            LOG(WARNING) << "Same event timestamp supplied: " << tsEnd << endl;
            return 0;
        }
        evBuffer.reserve(DEF_L1_CHUNK_SIZE * DEF_L1_NUM_LOOP);

        while (this->checkTxtStream())
        {
            getline(this->mTxtDataFile, line);
            this->mnCurrByteIdx += line.length();

            if (isComment(line))
                continue;

            EventData currEv = this->parseLine(line, K, distCoefs, rectMat);

            if (checkInImage && !MyCalibrator::isInImage(currEv.x, currEv.y, imageSize))
                continue;

            evBuffer.push_back(currEv);

            dtCount++;

            if (currEv.ts >= tsEnd) {
                mLastEvTs = currEv.ts;
                break;
            }
        }
        return dtCount;
    }

    unsigned long EventDataStore::getEventChunkRectified(unsigned long chunkSize, vector<EventData> &evBuffer,
                                                         const MyCalibPtr& pCalib, const bool checkInImage) {

        if (!this->mTxtDataFile.is_open()) {

            LOG(ERROR) << "Text data file is not open\n";
            return 0;
        }

        string line;

        unsigned long dtCount = 0;

        //If data manager constantly supply the same outData,
        // all data will be stacked together.
        if (chunkSize > 0)
            evBuffer.reserve(chunkSize);

        while (this->checkTxtStream())
        {
            if (chunkSize > 0 && dtCount >= chunkSize)
                break;

            getline(this->mTxtDataFile, line);
            this->mnCurrByteIdx += line.length();

            if (isComment(line))
                continue;

            EventData currEv = this->parseLine(line, pCalib);

            if (checkInImage && !pCalib->isInImage(currEv.x, currEv.y))
                continue;

            evBuffer.push_back(currEv);

            dtCount++;

            //DLOG_EVERY_N(INFO, 1000) << "Current ev. ts: " << currEv.ts << "\n";
        }
        return dtCount;
    }

    unsigned long EventDataStore::getEventChunkRectified(double tsEnd, vector<EventData> &evBuffer,
                                                         const MyCalibPtr& pCalib, const bool checkInImage) {

        if (!this->mTxtDataFile.is_open()) {

            LOG(ERROR) << "Text data file is not open\n";
            return 0;
        }

        string line;

        unsigned long dtCount = 0;

        //If data manager constantly supply the same outData,
        // all data will be stacked together.
        if (tsEnd <= mLastEvTs) {
            LOG(WARNING) << "Same event timestamp supplied: " << tsEnd << endl;
            return 0;
        }
        evBuffer.reserve(DEF_L1_CHUNK_SIZE * DEF_L1_NUM_LOOP);

        while (this->checkTxtStream())
        {
            getline(this->mTxtDataFile, line);
            this->mnCurrByteIdx += line.length();

            if (isComment(line))
                continue;

            EventData currEv = this->parseLine(line, pCalib);

            if (checkInImage && !pCalib->isInImage(currEv.x, currEv.y))
                continue;

            evBuffer.push_back(currEv);

            dtCount++;

            if (currEv.ts >= tsEnd) {
                mLastEvTs = currEv.ts;
                break;
            }
        }
        return dtCount;
    }

    void EventDataStore::reset() {

        if (mLoadStat != GOOD && mLoadStat != READY) {
            LOG(WARNING) << "EventDataStore::reset(): Load stat is not good.\n";
            return;
        }

        mLastEvTs = 0.0;
        mnCurrByteIdx = 0;
        mnCurrFileIdx = 0;

        if (mTxtDataFile.is_open()) {

            mTxtDataFile.close();
        }

        // loadTxtFile() only works in READY state!
        if (mLoadStat == GOOD) {
            mLoadStat = READY;
        }
        this->loadTxtFile();
    }

    /* ================================ EvEthzLoader ================================= */

    EvEthzLoader::EvEthzLoader(const string &fSettings) {

        //check settings file exists and is valid
        path pathSettings = path(fSettings);
        try {
            if (exists(pathSettings) && is_regular_file(pathSettings) &&
                (BaseLoader::checkExtension(pathSettings, ".yaml") ||
                 BaseLoader::checkExtension(pathSettings, ".yml"))) {

                mPathSettings = fSettings;
                //parse ds settings
                bool res = this->parseSettings(fSettings);
                if (res) {
                    //load data
                    this->loadData();
                }
            }
            else {
                cerr << "Ds root path or settings are not valid.\n";
            }
            // Update variables
            if (!this->updateLoadStat()) {
                cerr << "Load state: BAD_DATA, check dataset settings.\n";
            }
        }
        catch (const boost::filesystem::filesystem_error& ex)
        {
            cerr << ex.what() << '\n';
        }
    }

    bool EvEthzLoader::resolveDsFormat(const cv::FileStorage &fsSettings) {

        string dsFormat = MyYamlParser::parseString(fsSettings, "DS.format",
                BaseLoader::mapDsFormats(BaseLoader::EV_ETHZ));
        mDsFormat = BaseLoader::mapDsFormats(dsFormat);
        return (mDsFormat == BaseLoader::EV_ETHZ);
    }

    void EvEthzLoader::resolveDsPaths(const cv::FileNode &dsPathNode) {

        // These will be checked in each loader class
        // We don't load traj. path and construct it based on other info
        mPathImFile.first = MyYamlParser::parseString(dsPathNode, "imageFile", string("images.txt"));
        mPathImBase.first = MyYamlParser::parseString(dsPathNode, "imageBase", string(""));

        mPathEvents = MyYamlParser::parseString(dsPathNode, "events", string("events.txt"));

        mPathImu.first = MyYamlParser::parseString(dsPathNode, "imu", string("imu.txt"));

        mPathGt = MyYamlParser::parseString(dsPathNode, "gt", string("groundtruth.txt"));
    }

    bool EvEthzLoader::parseParameters(const cv::FileStorage &fsSettings) {

        bool res = EurocLoader::parseParameters(fsSettings);
        if (!res) {
            return false;
        }
        mpEventParams = std::make_shared<EvParams>();
        res = mpEventParams->parseParams(fsSettings);
        if (!res) {
            cerr << "*Missing event parameters.\n";
            return false;
        }
        return true;
    }

    // Deal with timestamp units from the beginning
    //TODO: Change this to use DataStore's methods for loading non-event ds.
    void EvEthzLoader::loadSequence(const string &dsRoot, const string &sqPath, const size_t idx) {

        if (mvpEvDs.empty()) {
            mvpEvDs.resize(mSeqCount);
        }
        float tsFactor = 1.0f;
        string seqPath = dsRoot + '/' + sqPath + '/';

        this->mvpEvDs[idx] = std::unique_ptr<EventDataStore>(
                new EventDataStore(seqPath + mPathEvents, tsFactor));

        this->mvpImDs[idx].first = std::unique_ptr<ImageDataStore>(
                new ImageDataStore(seqPath + mPathImFile.first, dsRoot + '/' + sqPath, tsFactor));
        // The event public dataset doesn't support stereo
        this->mvpImDs[idx].second = nullptr;

        this->mvpImuDs[idx].first = std::unique_ptr<ImuDataStore>(
                new ImuDataStore(seqPath + mPathImu.first, false, tsFactor));
        this->mvpImuDs[idx].second = nullptr;

        this->mvpGtDs[idx] = std::unique_ptr<GtDataStore>(
                new GtDataStore(seqPath + mPathGt, false, true, tsFactor));

        // If this is inertial, init imuDs
        if (this->isInertial()) {
            double tFrame = this->mvpImDs[idx].first->getTimeStamp(0);
            this->initImu(tFrame, idx);
        }
    }

    EvEthzLoader::~EvEthzLoader() = default;

    bool EvEthzLoader::checkLoadStat() {
        bool res = EurocLoader::checkLoadStat();
        if (res) {
            size_t imDsSz = mvpImDs.size();
            size_t evDsSz = mvpEvDs.size();
            if (imDsSz == evDsSz) {
                mLoadStat = BaseLoader::GOOD;
                return true;
            }
        }
        mLoadStat = BaseLoader::BAD_DATA;
        return false;
    }

    string EvEthzLoader::getTrajFileName() {

        string trajFileName = EurocLoader::getTrajFileName();

        if (mpConfig->isEvent() && !mpEventParams->missParams) {

            trajFileName += "_etm"+to_string(mpEventParams->l2TrackMode);
            // L2 Feature detection mode: FAST
            trajFileName += "_ftdtm"+to_string(mpEventParams->detMode);
            if (mpEventParams->trackTinyFrames) { // Track L1 frames in L2
                trajFileName += "_tTiFr";
            }
            if (mpEventParams->l1FixedWinSz) {
                trajFileName += "_fxWin";
            }
            if (mpEventParams->continTracking) {
                trajFileName += "_cont";
            }
        }
        
        return trajFileName;
    }

    string EvEthzLoader::printLoaderStat() const {

        ostringstream oss;

        oss << EurocLoader::printLoaderStat();

        oss << "\n# Event data info:\n";
        oss << "Num. event ds: " << mvpEvDs.size() << endl;
        oss << "Events path: " << mPathEvents << endl;
        oss << "# Event parameters:\n";
        if (mpEventParams) {
            oss << mpEventParams->printParams();
        }

        return oss.str();
    }

    // TODO: Do I need to save K & DistCoefs in separate vars???
    unsigned long EvEthzLoader::getNextEvents(unsigned long chunkSize, vector<EventData> &evs,
            const bool undistPoints, const bool checkInImage) {

        if (!checkSequence(mSeqIdx)) {
            LOG(ERROR) << "EvEthzLoader::getNextEvents: Wrong Sequence number\n";
            return 0;
        }

        if (undistPoints) {
            if (mppCalib.first) {
                return mvpEvDs[mSeqIdx]->getEventChunkRectified(chunkSize, evs, mppCalib.first, checkInImage);
            }
            else {
                LOG(ERROR) << "EvEthzLoader::getNextEvents: Cannot find calibrator!\n";
                return 0;
            }
        }
        else {
            return mvpEvDs[mSeqIdx]->getEventChunk(chunkSize, evs);
        }
    }

    unsigned long EvEthzLoader::getNextEvents(const double tsEnd, vector<EventData> &evs,
            const bool undistPoints, const bool checkInImage) {

        if (!checkSequence(mSeqIdx)) {
            LOG(ERROR) << "EvEthzLoader::getNextEvents: Wrong Sequence number\n";
            return 0;
        }

        if (undistPoints) {
            if (mppCalib.first) {
                return mvpEvDs[mSeqIdx]->getEventChunkRectified(tsEnd, evs, mppCalib.first, checkInImage);
            }
            else {
                LOG(ERROR) << "EvEthzLoader::getNextEvents: Cannot find calibrator!\n";
                return 0;
            }
        }
        else {
            return mvpEvDs[mSeqIdx]->getEventChunk(tsEnd, evs);
        }
    }

    void EvEthzLoader::resetCurrSequence() {

        EurocLoader::resetCurrSequence();

        if (!checkSequence(mSeqIdx)) {
            LOG(ERROR) << "EvEthzLoader::resetCurrSequence: Wrong Sequence number\n";
            return;
        }
        mvpEvDs[mSeqIdx]->reset();
    }

    bool EvEthzLoader::EoSEQ(const EORB_SLAM::SensorConfigPtr &dsConf, const unsigned int nImages, const int ni) {

        if (dsConf->isImage()) {
            
            return ni >= nImages;
        }
        else {
            // In case of event only, loop finishes when the event buffer (file) is exhausted
            return false;
        }
    }

    bool EvEthzLoader::EoSEQ(const unsigned int nImages, const int ni) {

        return EoSEQ(mpConfig, nImages, ni);
    }
}