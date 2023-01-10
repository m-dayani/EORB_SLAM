/**
* Event Data Management Implementation
* Managing event data load/store
* Author: M. Dayani
*/

#include "DataStore.h"

#include <memory>

using namespace boost::filesystem;
using namespace std;

namespace EORB_SLAM {

    /* ================================= BaseLoader ================================== */

    BaseLoader::BaseLoader() :
            mnFiles(0), mnCurrFileIdx(0), mnCurrByteIdx(0), mPathStat(PathStat::INVALID),
            mLoadStat(LoadStat::BAD_PATH), mTsFactor(1.0)
    {}

    BaseLoader::DsFormat BaseLoader::mapDsFormats(const string& strFormat) {

        if (strFormat == "euroc")
            return BaseLoader::DsFormat::EUROC;
        else if (strFormat == "ev_ethz")
            return BaseLoader::DsFormat::EV_ETHZ;
        else if (strFormat == "ev_mvsec")
            return BaseLoader::DsFormat::EV_MVSEC;
        else
            return BaseLoader::DsFormat::NOT_SUPPORTED;
    }

    string BaseLoader::mapDsFormats(const BaseLoader::DsFormat dsFormat) {

        switch (dsFormat) {
            case BaseLoader::EUROC:
                return "euroc";
            case BaseLoader::EV_ETHZ:
                return "ev_ethz";
            case BaseLoader::EV_MVSEC:
                return "ev_mvsec";
            case BaseLoader::NOT_SUPPORTED:
            default:
                return "not_supported";
        }
    }

    void BaseLoader::openFile()
    {
        this->mTxtDataFile.open(this->mDataPath);
        if (mTxtDataFile.is_open()) {
            mLoadStat = LoadStat::GOOD;
        }
        else {
            mLoadStat = LoadStat::BAD_DATA;
        }
    }

    void BaseLoader::openNextDirFile()
    {
        if (this->mTxtDataFile.is_open()) {
            this->mTxtDataFile.close();
            this->mnCurrByteIdx = 0;
        }
        if (mnCurrFileIdx < mvFileNames.size())
        {
            this->mTxtDataFile.open(mvFileNames[mnCurrFileIdx]);
            if (mTxtDataFile.is_open()) {
                mnCurrFileIdx++;
                mLoadStat = LoadStat::GOOD;
            }
            else {
                mLoadStat = LoadStat::BAD_DATA;
            }
        }
    }

    void BaseLoader::loadTxtFile()
    {
        if (mPathStat == PathStat::FILE && mLoadStat == LoadStat::READY)
        {
            this->openFile();
        }
        else if (mPathStat == PathStat::DIR && mLoadStat == LoadStat::READY)
        {
            this->openNextDirFile();
        }
    }

    bool BaseLoader::checkTxtStream()
    {
        if (this->mPathStat == PathStat::FILE || this->mPathStat == PathStat::FILE_CSV)
        {
            return this->mTxtDataFile.good();
        }
        else if (this->mPathStat == PathStat::DIR)
        {
            if (this->mTxtDataFile) {
                return true;
            }
            else {
                this->openNextDirFile();
                return this->mTxtDataFile.good();
            }
        }
        return false;
    }

    bool BaseLoader::isComment(const string &txt) {
        size_t idx = txt.find_first_not_of(" \t\n");
        return txt[idx] == '#';
    }

//    template<typename T>
//    unsigned long BaseLoader::getTxtData(const unsigned long chunkSize, vector<T> &outData)

    bool BaseLoader::checkFileAndExtension(const string &strPath, const string &ext) {
        path p(strPath);
        return exists(p) && is_regular_file(p) && p.has_extension() && p.extension().string() == ext;
    }

    bool BaseLoader::checkDirectory(const string &strPath) {
        path p(strPath);
        return exists(p) && is_directory(p);
    }

    bool BaseLoader::checkFile(const string &strPath) {
        path p(strPath);
        return exists(p) && is_regular_file(p);
    }

    /* =============================== ImageDataStore ================================ */

    ImageDataStore::ImageDataStore(const string &imTxtFile, const string &imBase, const double tsFactor)
    {
        boost::filesystem::path imFiles(imTxtFile);
        boost::filesystem::path imBasePath(imBase);
        mTsFactor = tsFactor;

        try {
            // Check image file names
            if (exists(imFiles) && is_regular_file(imFiles) && is_directory(imBasePath))
            {
                this->mDataPath = imTxtFile;
                this->mImagesBasePath = imBase;
                this->mPathStat = PathStat::FILE;

                if (checkExtension(imFiles, ".txt")) {
                    mLoadStat = LoadStat::READY;
                }
                else if (checkExtension(imFiles, ".csv")) {
                    mPathStat = PathStat::FILE_CSV;
                    mLoadStat = LoadStat::READY;
                }
                else {
                    mLoadStat = LoadStat::BAD_PATH;
                }
            }
            else {
                cerr << imFiles << " does not exist\n";
            }
            if (mLoadStat == LoadStat::READY) {
                this->openFile();
                this->getTxtData(0, mvImageData);
                // Check if number of images in directory matches
                // the number of file names in image file.
                if (!mvImageData.empty()) {
                    boost::filesystem::path firstIm = boost::filesystem::path(imBase + '/' + mvImageData[0].uri);
                    bool res = is_regular_file(firstIm) && checkExtension(firstIm, ".png");
                    if (res) {
                        mnFiles = mvImageData.size();
                        mLoadStat = LoadStat::GOOD;
                    }
                    else
                        mLoadStat = LoadStat::BAD_DATA;
                }
                else {
                    cerr << "Image file names is empty\n";
                }
            }
        }
        catch (const boost::filesystem::filesystem_error& ex)
        {
            cerr << ex.what() << '\n';
        }

    }

    ImageDataStore::~ImageDataStore() {
        if (mTxtDataFile && mTxtDataFile.is_open())
        {
            mTxtDataFile.close();
        }
    }

    boost::any ImageDataStore::parseLine(const string &evStr) {

        std::istringstream stream(evStr);

        double ts;
        char c;
        string s;

        if (mPathStat == PathStat::FILE)
            stream >> ts >> s;
        else if (mPathStat == PathStat::FILE_CSV)
            stream >> ts >> c >> s;

        ImageData imData(ts/mTsFactor, s);
        return boost::any(imData);
    }

    void ImageDataStore::getImage(unsigned int idx, cv::Mat &image, double &ts) {
        image = cv::Mat();
        ts = 0.0;
        if (idx >= mvImageData.size() || idx < 0)
            return;
        image = cv::imread(mImagesBasePath+'/'+mvImageData[idx].uri, cv::IMREAD_UNCHANGED);
        ts = mvImageData[idx].ts;
    }

    string ImageDataStore::getFileName(size_t idx, bool fullPath) {
        if (idx < 0 || idx >= mvImageData.size())
            return string();
        if (fullPath)
            return mImagesBasePath + '/' + mvImageData[idx].uri;
        else
            return mvImageData[idx].uri;
    }

    double ImageDataStore::getTimeStamp(size_t idx) {
        if (idx >= mvImageData.size())
            return 0.0;
        return this->mvImageData[idx].ts;
    }

    // Nothing to do!
    void ImageDataStore::reset() {}

    /* ================================ ImuDataStore ================================= */

    ImuDataStore::ImuDataStore(const string &filePath, bool _gFirst, const double tsFactor) : gFirst(_gFirst) {

        boost::filesystem::path imuFile(filePath);
        mTsFactor = tsFactor;

        try {
            // Check image file names
            if (exists(imuFile) && is_regular_file(imuFile))
            {
                this->mDataPath = filePath;
                this->mPathStat = PathStat::FILE;

                if (checkExtension(imuFile, ".txt")) {
                    mLoadStat = LoadStat::READY;
                }
                else if (checkExtension(imuFile, ".csv")) {
                    mPathStat = PathStat::FILE_CSV;
                    mLoadStat = LoadStat::READY;
                }
                else {
                    mLoadStat = LoadStat::BAD_PATH;
                }

                if (mLoadStat == LoadStat::READY) {
                    this->openFile();
                    this->getTxtData(0, mvImuData);
                    if (!mvImuData.empty())
                        mLoadStat = LoadStat::GOOD;
                    else
                        mLoadStat = LoadStat::BAD_DATA;
                }
            }
            else {
                cerr << imuFile << " does not exist\n";
            }
        }
        catch (const boost::filesystem::filesystem_error& ex)
        {
            cerr << ex.what() << '\n';
        }
    }

    ImuDataStore::~ImuDataStore() {
        if (mTxtDataFile && mTxtDataFile.is_open())
        {
            mTxtDataFile.close();
        }
    }

    boost::any ImuDataStore::parseLine(const string &evStr) {

        std::istringstream stream(evStr);

        double ts;
        char c1, c2, c3, c4, c5, c6;
        float gx, gy, gz, ax, ay, az;

        if (mPathStat == PathStat::FILE) {
            if (this->gFirst)
                stream >> ts >> gx >> gy >> gz >> ax >> ay >> az;
            else
                stream >> ts >> ax >> ay >> az >> gx >> gy >> gz;
        }
        else if (mPathStat == PathStat::FILE_CSV) {
            if (this->gFirst)
                stream >> ts >> c1 >> gx >> c2 >> gy >> c3 >> gz >> c4 >> ax >> c5 >> ay >> c6 >> az;
            else
                stream >> ts >> c1 >> ax >> c2 >> ay >> c3 >> az >> c4 >> gx >> c5 >> gy >> c6 >> gz;
        }

        ImuData imuData(ts/mTsFactor, gx, gy, gz, ax, ay, az);
        return boost::any(imuData);
    }

    unsigned long ImuDataStore::getNextChunk(size_t offset, unsigned long chunkSize, vector<ImuData> &outData) {

        if (offset + chunkSize >= mvImuData.size())
            return 0;
        outData.resize(chunkSize);
        unsigned long dtCount = 0;
        for (; dtCount < chunkSize; dtCount++) {
            outData[dtCount] = mvImuData[offset+dtCount];
        }
        return dtCount;
    }

    unsigned long ImuDataStore::getNextChunk(double tsEnd,
            vector<ORB_SLAM3::IMU::Point> &vImuMeas, const double tsFactor) {

        size_t vImuSize = mvImuData.size();
        unsigned long imuCnt = 0;
        if (mSIdx < 0 || mSIdx >= vImuSize)
            return imuCnt;

        const double invTsFactor = 1.0 / tsFactor;

        while (mSIdx < vImuSize && mvImuData[mSIdx].ts * invTsFactor <= tsEnd) {

            vImuMeas.push_back(mvImuData[mSIdx].toImuPoint(tsFactor));
            this->incIdx();
            imuCnt++;
        }
        return imuCnt;
    }

    void ImuDataStore::incIdx() {
        if (mSIdx < mvImuData.size())
            mSIdx++;
    }

    void ImuDataStore::decIdx() {
        if (mSIdx > 0)
            mSIdx--;
    }

    void ImuDataStore::reset() {

        mSIdx = 0;
    }

    /* ================================= GtDataStore ================================= */


    GtDataStore::GtDataStore(const string &filePath, bool _qwFirst, bool _posFirst, const double tsFactor) :
        posFirst(_posFirst), qwFirst(_qwFirst) {

        boost::filesystem::path gtFile(filePath);
        mTsFactor = tsFactor;

        try {
            // Check image file names
            if (exists(gtFile) && is_regular_file(gtFile))
            {
                this->mDataPath = filePath;
                this->mPathStat = PathStat::FILE;

                if (checkExtension(gtFile, ".txt")) {
                    mLoadStat = LoadStat::READY;
                }
                else if (checkExtension(gtFile, ".csv")) {
                    mPathStat = PathStat::FILE_CSV;
                    mLoadStat = LoadStat::READY;
                }
                else {
                    mLoadStat = LoadStat::BAD_PATH;
                }

                if (mLoadStat == LoadStat::READY) {
                    this->openFile();
                    this->getTxtData(0, mvGtData);
                    if (!mvGtData.empty())
                        mLoadStat = LoadStat::GOOD;
                    else
                        mLoadStat = LoadStat::BAD_DATA;
                }
            }
            else {
                cerr << gtFile << " does not exist\n";
            }
        }
        catch (const boost::filesystem::filesystem_error& ex)
        {
            cerr << ex.what() << '\n';
        }
    }

    GtDataStore::~GtDataStore() {
        if (mTxtDataFile && mTxtDataFile.is_open())
        {
            mTxtDataFile.close();
        }
    }

    boost::any GtDataStore::parseLine(const string &evStr) {

        std::istringstream stream(evStr);

        double ts;
        char c1, c2, c3, c4, c5, c6, c7;
        float qw, qx, qy, qz, px, py, pz;

        if (mPathStat == PathStat::FILE) {
            if (this->qwFirst) {
                if (this->posFirst)
                    stream >> ts >> px >> py >> pz >> qw >> qx >> qy >> qz;
                else
                    stream >> ts >> qw >> qx >> qy >> qz >> px >> py >> pz;
            }
            else {
                if (this->posFirst)
                    stream >> ts >> px >> py >> pz >> qx >> qy >> qz >> qw;
                else
                    stream >> ts >> qx >> qy >> qz >> qw >> px >> py >> pz;
            }
        }
        else if (mPathStat == PathStat::FILE_CSV) {
            if (this->qwFirst) {
                if (this->posFirst)
                    stream >> ts >> c1 >> px >> c2 >> py >> c3 >> pz >> c4 >> qw >> c5 >> qx >> c6 >> qy >> c7 >> qz;
                else
                    stream >> ts >> c4 >> qw >> c5 >> qx >> c6 >> qy >> c7 >> qz >> c1 >> px >> c2 >> py >> c3 >> pz;
            } else {
                if (this->posFirst)
                    stream >> ts >> c1 >> px >> c2 >> py >> c3 >> pz >> c5 >> qx >> c6 >> qy >> c7 >> qz >> c4 >> qw;
                else
                    stream >> ts >> c5 >> qx >> c6 >> qy >> c7 >> qz >> c4 >> qw >> c1 >> px >> c2 >> py >> c3 >> pz;
            }
        }

        GtDataQuat gtData(ts/mTsFactor, px, py, pz, qw, qx, qy, qz);
        return boost::any(gtData);
    }

    unsigned long GtDataStore::getNextChunk(size_t offset, unsigned long chunkSize, vector<GtDataQuat> &outData) {

        if (offset + chunkSize >= mvGtData.size())
            return 0;
        outData.resize(chunkSize);
        unsigned long dtCount = 0;
        for (; dtCount < chunkSize; dtCount++) {
            outData[dtCount] = mvGtData[offset+dtCount];
        }
        return dtCount;
    }

    void GtDataStore::reset() {}

    /* ================================= EurocLoader ================================= */

    EurocLoader::EurocLoader() :
            mLoadStat(BaseLoader::LoadStat::BAD_PATH), mDsFormat(BaseLoader::DsFormat::EUROC),
            mpConfig(nullptr), mpCamParams(nullptr), mSeqCount(0), mSeqTarget(0), mSeqIdx(0),
            mnMaxIter(1)
    {}

    EurocLoader::EurocLoader(const string &fSettings) : EurocLoader() {

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

    bool EurocLoader::checkLoadStat() {

        unsigned imDsSize = mvpImDs.size();
        unsigned imuDsSize = mvpImuDs.size();
        unsigned gtDsSize = mvpGtDs.size();

        bool cond = imDsSize == imuDsSize && imDsSize == gtDsSize && imDsSize > 0;
        if (cond) {
            mSeqCount = imDsSize;
            mLoadStat = BaseLoader::GOOD;
        }
        else {
            mSeqCount = 0;
            mLoadStat = BaseLoader::BAD_DATA;
        }
        return cond;
    }

    bool EurocLoader::updateLoadStat() {

        return this->checkLoadStat();
    }

    bool EurocLoader::resolveDsFormat(const cv::FileStorage &fsSettings) {

        string dsFormat = MyYamlParser::parseString(fsSettings, "DS.format",
                BaseLoader::mapDsFormats(BaseLoader::EUROC));
        mDsFormat = BaseLoader::mapDsFormats(dsFormat);
        return (mDsFormat == BaseLoader::EUROC);
    }

    void EurocLoader::resolveDsPaths(const cv::FileNode &dsPathNode) {

        // These will be checked in each loader class
        // We don't load traj. path and construct it based on other info

        string cam0File = MyYamlParser::parseString(dsPathNode, "imageFile", string("mav0/cam0/data.csv"));
        string cam0Base = MyYamlParser::parseString(dsPathNode, "imageBase", string("mav0/cam0/data"));
        string cam1File = MyYamlParser::parseString(dsPathNode, "imageFileRight", string("mav0/cam1/data.csv"));
        string cam1Base = MyYamlParser::parseString(dsPathNode, "imageBaseRight", string("mav0/cam1/data"));
        mPathImFile = make_pair(cam0File, cam1File);
        mPathImBase = make_pair(cam0Base, cam1Base);

        string imu0File = MyYamlParser::parseString(dsPathNode, "imu", string("mav0/im0/data.csv"));
        string imu1File = MyYamlParser::parseString(dsPathNode, "imuRight", "");
        mPathImu = make_pair(imu0File, imu1File);

        mPathGt = MyYamlParser::parseString(dsPathNode, "gt", string("mav0/state_groundtruth_estimate0/data.csv"));
    }

    /**
     * This will set 3 internal class variables:
     *  mSeqTarget, mSeqCount, mSeqNames
     * @param fsSettings
     * @return
     */
    bool EurocLoader::resolveSeqInfo(const cv::FileStorage &fsSettings) {

        // Load sequence info
        mSeqTarget = MyYamlParser::parseInt(fsSettings, "DS.Seq.target", 0);

        vector<string> seqNames;
        unsigned int seqCount;
        seqCount = MyYamlParser::parseStringSequence(fsSettings, "DS.Seq.names", seqNames);

        if (seqCount) {
            // Check target sequence path(s) exist.
            if (mSeqTarget >= 0) {
                if (mSeqTarget < seqCount) {

                    string seqPath = mPathDsRoot + '/' + seqNames[mSeqTarget];
                    if (!BaseLoader::checkDirectory(seqPath)) {
                        cerr << "Failed to find sequence: " << seqPath << endl;
                        return false;
                    }
                    else {
                        mSeqCount = 1;
                        mSeqNames.resize(mSeqCount);
                        mSeqNames[0] = seqNames[mSeqTarget];
                        mSeqTarget = 0;
                        mSeqIdx = 0;
                    }
                }
                else {
                    cerr << "Target sequence number is outside range.\n";
                    return false;
                }
            }
            else {
                // If some sequences does not exist, we still want to
                // be able to work with existing sequences
                for (size_t seq = 0; seq < seqCount; seq++) {
                    string seqPath = mPathDsRoot + '/' + seqNames[seq];
                    if (!BaseLoader::checkDirectory(seqPath)) {
                        cerr << "Failed to find sequence: " << seqPath << endl;
                    }
                    else {
                        mSeqNames.push_back(seqNames[seq]);
                    }
                }
                mSeqCount = mSeqNames.size();
                if (!mSeqCount)
                    return false;
                mSeqIdx = 0;
            }
        }
        else {
            cerr << "Empty sequence names.\n";
            return false;
        }
        return true;
    }

    bool EurocLoader::parseSettings(const string &settingsFile) {

        cv::FileStorage fsSettings(settingsFile.c_str(), cv::FileStorage::READ);
        if(!fsSettings.isOpened()) {

            cerr << "Failed to open settings file at: " << settingsFile << endl;
            return false;
        }

        // Check DS format
        if (!this->resolveDsFormat(fsSettings)) {

            cerr << "Dataset format is not correct.\n";
            return false;
        }

        // Retrieve DS name
        mDsName = MyYamlParser::parseString(fsSettings, "DS.name", "euroc");

        // Check DS Sensor configuration
        string dsConf = MyYamlParser::parseString(fsSettings, "DS.config",
                MySensorConfig::mapConfig(MySensorConfig::IDLE));
        mpConfig = std::make_shared<MySensorConfig>();
        mpConfig->setConfig(MySensorConfig::mapConfig(dsConf));

        // Load important paths
        mPathOrbVoc = MyYamlParser::parseString(fsSettings, "Path.ORBvoc", string());
        if (mPathOrbVoc.empty() || !BaseLoader::checkFileAndExtension(mPathOrbVoc, ".txt")) {
            cerr << "Failed to find vocabulary: " << mPathOrbVoc << endl;
            return false;
        }
        cv::FileNode dsPathNode = fsSettings["Path.DS"];
        mPathDsRoot = MyYamlParser::parseString(dsPathNode, "root", string());
        if (mPathDsRoot.empty() || !BaseLoader::checkDirectory(mPathDsRoot)) {
            cerr << "Failed to find dataset root path: " << mPathDsRoot << endl;
            return false;
        }

        this->resolveDsPaths(dsPathNode);

        // Resolve max number of iterations
        mnMaxIter = MyYamlParser::parseInt(fsSettings, "DS.nMaxIter", 1);

        bool res = this->resolveSeqInfo(fsSettings);
        if (!res) {
            cerr << "No viable sequence found.\n";
            return false;
        }

        res = this->parseParameters(fsSettings);
        if (!res) {
            cerr << "Cannot parse important parameters.\n";
            return false;
        }

        fsSettings.release();
        return true;
    }

    bool EurocLoader::parseParameters(const cv::FileStorage &fsSettings) {

        if (!mpConfig) {
            LOG(ERROR) << "EurocLoader::parseParameters: No sensor configuration is loaded\n";
            return false;
        }

        // Parse camera parameters
        mpCamParams = std::make_shared<MyCamParams>();
        bool res = MyParameters::parseCameraParams(fsSettings, *(mpCamParams),
                                              mpConfig->isStereo(), mpConfig->isRGBD());
        if (!res) {
            LOG(ERROR) << "EurocLoader::parseParameters: Failed to load camera parameters.\n";
            return false;
        }
        else {
            DLOG(INFO) << "Camera Parameters are good, creating Calib object...\n";
            mppCalib.first = make_shared<MyCalibrator>(mpCamParams->mK, mpCamParams->mDistCoef,
                    cv::Scalar(mpCamParams->mImWidth, mpCamParams->mImHeight),
                    mpCamParams->mR, mpCamParams->mP, mpCamParams->isFisheye());
            mppCalib.second = nullptr;
            if (mpConfig->isStereo() && mpCamParams->mLinkedCam) {
                MyCamParams* pCam2Params = mpCamParams->mLinkedCam;
                mppCalib.second = make_shared<MyCalibrator>(pCam2Params->mK, pCam2Params->mDistCoef,
                        cv::Scalar(pCam2Params->mImWidth, pCam2Params->mImHeight),
                        pCam2Params->mR, pCam2Params->mP, pCam2Params->isFisheye());
            }
        }

        // Parse ORB-Extractor paramaters
        mpORBParams = std::make_shared<MixedFtsParams>();
        res = MyParameters::parseFeatureParams(fsSettings, *(mpORBParams));
        if (mpConfig->isImage() && !res) {
            LOG(ERROR) << "EurocLoader::parseParameters: Failed to load ORB-Extractor parameters.\n";
            return false;
        }

        // Parse IMU parameters
        mpIMUParams = std::make_shared<MyIMUSettings>();
        res = MyParameters::parseIMUParams(fsSettings, *(mpIMUParams));
        if (mpConfig->isInertial() && !res) {
            LOG(ERROR) << "EurocLoader::parseParameters: Failed to load IMU parameters.\n";
            return false;
        }

        // Parse Viewer parameters
        mpViewerParams = std::make_shared<MyViewerSettings>();
        res = MyParameters::parseViewerParams(fsSettings, *(mpViewerParams));
        if (!res) {
            LOG(ERROR) << "EurocLoader::parseParameters: Failed to load Viewer parameters.\n";
        }

        return true;
    }

    void EurocLoader::loadData() {

        if (mSeqTarget < 0) {
            mSeqCount = mSeqNames.size();
        }
        else {
            mSeqCount = 1;
        }
        if (mSeqCount) {
            //mvpEvDs.resize(mSeqCount);
            mvpImDs.resize(mSeqCount);
            mvpImuDs.resize(mSeqCount);
            mvpGtDs.resize(mSeqCount);
            if (mSeqTarget < 0) {
                for (size_t seq = 0; seq < mSeqCount; seq++) {
                    loadSequence(mPathDsRoot, mSeqNames[seq], seq);
                }
            }
            else if (mSeqTarget < mSeqNames.size()) {
                loadSequence(mPathDsRoot, mSeqNames[mSeqTarget], 0);
            }
        }
    }

    // Deal with timestamp units from the beginning
    void EurocLoader::loadSequence(const string &dsRoot, const string &sqPath, const size_t idx) {

        float tsFactor = 1e9;
        string seqPath = dsRoot + '/' + sqPath + '/';

        this->mvpImDs[idx].first = std::unique_ptr<ImageDataStore>(
                new ImageDataStore(seqPath + mPathImFile.first, seqPath + mPathImBase.first, tsFactor));
        if (mpConfig->isStereo()) {
            this->mvpImDs[idx].second = std::unique_ptr<ImageDataStore>(
                    new ImageDataStore(seqPath + mPathImFile.second, seqPath + mPathImBase.second, tsFactor));
        }

        this->mvpImuDs[idx].first = std::unique_ptr<ImuDataStore>(
                new ImuDataStore(seqPath + mPathImu.first, true, tsFactor));
        if (mpConfig->isStereo() && !mPathImu.second.empty()) {
            this->mvpImuDs[idx].second = std::unique_ptr<ImuDataStore>(
                    new ImuDataStore(seqPath + mPathImu.second, true, tsFactor));
        }

        this->mvpGtDs[idx] = std::unique_ptr<GtDataStore>(
                new GtDataStore(seqPath + mPathGt, true, true, tsFactor));

        // If this is inertial, init imuDs
        if (this->isInertial()) {
            double tFrame = this->mvpImDs[idx].first->getTimeStamp(0);
            this->initImu(tFrame, idx);

            if (mvpImDs[idx].second) {
                tFrame = this->mvpImDs[idx].second->getTimeStamp(0);
                this->initImu(tFrame, idx, true);
            }
        }
    }

    MyCalibPtr EurocLoader::getCamCalibrator(const bool right) const {

        if (right)
            return mppCalib.second;
        else
            return mppCalib.first;
    }

    unsigned int EurocLoader::getNumImages(const bool right) {

        if (!checkSequence(mSeqIdx) || this->mvpImDs.empty())
            return 0;

        if (right && mvpImDs[mSeqIdx].second)
            return mvpImDs[mSeqIdx].second->getNumFiles();
        else if (mvpImDs[mSeqIdx].first)
            return this->mvpImDs[mSeqIdx].first->getNumFiles();
    }

    void EurocLoader::getImage(const size_t idx, cv::Mat &image, double &ts, const bool right) {
        if (!checkSequence(mSeqIdx))
            return;
        if (right)
            this->mvpImDs[mSeqIdx].second->getImage(idx, image, ts);
        else
            this->mvpImDs[mSeqIdx].first->getImage(idx, image, ts);
    }

    string EurocLoader::getImageFileName(const size_t idx, bool fullName, const bool right) {
        if (!checkSequence(mSeqIdx))
            return string();
        if (right)
            return this->mvpImDs[mSeqIdx].second->getFileName(idx, fullName);
        else
            return this->mvpImDs[mSeqIdx].first->getFileName(idx, fullName);
    }

    double EurocLoader::getImageTime(size_t idx, const bool right) {
        if (!checkSequence(mSeqIdx))
            return 0.0;
        if (right)
            return this->mvpImDs[mSeqIdx].second->getTimeStamp(idx);
        else
            return this->mvpImDs[mSeqIdx].first->getTimeStamp(idx);
    }

    // Replicate what is done in ORB_SLAM3 main inertial progs.
    void EurocLoader::initImu(const double initTs, const int seqIdx, const bool right) {

        if ((mSeqTarget < 0 && (seqIdx >= mSeqCount || seqIdx < 0)) || (mSeqTarget >= 0 && seqIdx != 0)) {
            return;
        }

        vector<ORB_SLAM3::IMU::Point> vImuMeas;

        if (right && this->mvpImuDs[seqIdx].second) {
            this->mvpImuDs[seqIdx].second->getNextChunk(initTs, vImuMeas);
            if (!vImuMeas.empty())
                this->mvpImuDs[seqIdx].second->decIdx();
        }
        else if (this->mvpImuDs[seqIdx].first) {
            this->mvpImuDs[seqIdx].first->getNextChunk(initTs, vImuMeas);
            if (!vImuMeas.empty())
                this->mvpImuDs[seqIdx].first->decIdx();
        }
    }

    unsigned int EurocLoader::getNextImu(const double ts, vector<ORB_SLAM3::IMU::Point> &vImuMeas, const bool right) {

        if (!checkSequence(mSeqIdx))
            return 0;
        if (right && this->mvpImuDs[mSeqIdx].second)
            return this->mvpImuDs[mSeqIdx].second->getNextChunk(ts, vImuMeas);
        else if (this->mvpImuDs[mSeqIdx].first)
            return this->mvpImuDs[mSeqIdx].first->getNextChunk(ts, vImuMeas);
    }

    string EurocLoader::getSequencePath() {

        if (mSeqNames.empty() || mSeqTarget < 0 || mSeqTarget >= mSeqNames.size())
            return string();
        return mPathDsRoot + '/' + mSeqNames[mSeqTarget];
    }

    void EurocLoader::resetSequences() {
        if (this->mSeqTarget < 0)
            this->mSeqIdx = 0;
    }

    void EurocLoader::incSequence() {
        if (this->mSeqTarget < 0 && this->mSeqIdx < mSeqCount-1)
            this->mSeqIdx++;
    }

    void EurocLoader::decSequence() {
        if (this->mSeqTarget < 0 && this->mSeqIdx > 1)
            this->mSeqIdx--;
    }

    bool EurocLoader::checkSequence(const unsigned int seq) const {
        if (mLoadStat == BaseLoader::GOOD) {
            if (mSeqTarget < 0) {
                return seq >= 0 && seq < mSeqCount;
            } else {
                return seq == 0;
            }
        }
        return false;
    }

    string EurocLoader::getTrajFileName() {

        string trajFileName = this->getTrajFileName(mSeqIdx);

        if (mpORBParams && !mpORBParams->missParams && mpORBParams->detMode != 0) {
            trajFileName += "_mfts";
        }

        return trajFileName;
    }

    string EurocLoader::getTrajFileName(const uint seq) {

        string seqName;
        if (!mSeqNames.empty() && checkSequence(seq))
            seqName = mSeqNames[seq];

        return BaseLoader::mapDsFormats(mDsFormat)+'_'+mpConfig->toDsStr()+'_'+seqName;
    }

    ulong EurocLoader::getNumTotalImages() {

        unsigned int numIm = 0;
        if (mSeqTarget < 0) {
            for (auto & mvpImD : mvpImDs) {
                numIm += mvpImD.first->getNumFiles();
                if (mpConfig->isStereo())
                    numIm += mvpImD.second->getNumFiles();
            }
        }
        else {
            numIm = this->getNumImages();
            if (mpConfig->isStereo())
                numIm += this->getNumImages(true);
        }
        return numIm;
    }

    unsigned int EurocLoader::getNumTargetSequences() const {

        if (mSeqTarget < 0)
            return this->getNumSequences();
        else {
            if (checkSequence(mSeqIdx))
                return 1;
            else
                return 0;
        }
    }

    string EurocLoader::getSequenceName() const {

        if (!checkSequence(mSeqIdx))
            return string();
        return mSeqNames[mSeqIdx];
    }

    bool EurocLoader::isGood() const {

        return mLoadStat == BaseLoader::GOOD;
    }

    bool EurocLoader::isInertial() {

        return mpConfig->isInertial();
    }

    string EurocLoader::printLoaderStat() const {

        ostringstream oss;

        if (this->isGood()) {
            oss << "# Loader state is good.\n";
        }
        else {
            oss << "# Loader state is not good.\n";
        }
        oss << "# Loader format is: " << BaseLoader::mapDsFormats(mDsFormat) << endl;
        oss << "# Sensor configuration: " << mpConfig->toStr() << endl;

        oss << "# Data Loaders stats: \n";
        oss << "Num. image ds: " << mvpImDs.size() << endl;
        oss << "Num. IMU ds: " << mvpImuDs.size() << endl;
        oss << "Num. GT ds: " << mvpGtDs.size() << endl;

        oss << "# Paths stat:\n";
        oss << "Settings file path: " << mPathSettings << endl;
        oss << "Ds root path: " << mPathDsRoot << endl;
        oss << "ORB vocab path: " << mPathOrbVoc << endl;

        oss << "Left image file path: " << mPathImFile.first << endl;
        oss << "Left image base path: " << mPathImBase.first << endl;
        oss << "Right image file path: " << mPathImFile.second << endl;
        oss << "Right image base path: " << mPathImBase.second << endl;

        oss << "Left IMU file path: " << mPathImu.first << endl;
        oss << "Right IMU file path: " << mPathImu.second << endl;

        oss << "Gt file path: " << mPathGt << endl;

        oss << "# Sequence info:\n";
        oss << "Sequence count: " << mSeqCount << endl;
        oss << "Current seq index: " << mSeqIdx << endl;
        oss << "Target sequence: " << mSeqTarget << endl;
        oss << "Sequence names: [";
        char delim = ',';
        for (int i = 0; i < mSeqNames.size(); i++) {
            if (i == mSeqNames.size()-1) {
                oss << mSeqNames[i];
                break;
            }
            oss << mSeqNames[i] << delim;
        }
        oss << "]\n";

        oss << "# Camera parameters:\n";
        if (mpCamParams) {
            if (mpConfig->isMonocular()) {
                oss << mpCamParams->print();
            } else {
                oss << mpCamParams->printStereo();
            }
        }

        oss << "# Feature detector and extractor parameters:\n";
        if (mpORBParams) {
            oss << mpORBParams->printParams();
        }

        oss << "# IMU parameters:\n";
        if (mpIMUParams) {
            oss << mpIMUParams->printParams();
        }

        oss << "# Viewer parameters:\n";
        if (mpViewerParams) {
            oss << mpViewerParams->printParams();
        }

        return oss.str();
    }

    void EurocLoader::resetCurrSequence() {

        if (!checkSequence(mSeqIdx))
            return;
        this->mvpImDs[mSeqIdx].first->reset();
        this->mvpImuDs[mSeqIdx].first->reset();
        this->mvpGtDs[mSeqIdx]->reset();

        if (mpConfig->isStereo()) {
            this->mvpImDs[mSeqIdx].second->reset();
            if (mvpImuDs[mSeqIdx].second)
                mvpImuDs[mSeqIdx].second->reset();
        }
    }

} //namespace EORB_SLAM



