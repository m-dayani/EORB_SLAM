#!/bin/bash

# Install build essential
sudo apt update -y
sudo apt install -y build-essential
sudo apt-get install -y manpages-dev
sudo apt install -y software-properties-common
gcc --version

# On Ubuntu 20.04, you need to install gcc and g++ 7
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt update -y
sudo apt install -y gcc-7 g++-7

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 --slave /usr/bin/g++ g++ /usr/bin/g++-9 --slave /usr/bin/gcov gcov /usr/bin/gcov-9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 100 --slave /usr/bin/g++ g++ /usr/bin/g++-7 --slave /usr/bin/gcov gcov /usr/bin/gcov-7

gcc --version

# Install the necessary tools
sudo apt install -y make cmake pkg-config unzip yasm git gfortran nano wget curl

cd /home
mkdir dep

# Get Pangolin
cd /home/dep
git clone --recursive https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin

# Install dependencies (as described above, or your preferred method)
# WARNING: This requires user interaction, so make it independent of user
sed -i 's/(install/(install -y/g' ./scripts/install_prerequisites.sh
./scripts/install_prerequisites.sh recommended

# Configure and build
cmake -B build
cmake --build build

# with Ninja for faster builds (sudo apt install ninja-build)
# cmake -B build -GNinja
# cmake --build build

# GIVEME THE PYTHON STUFF!!!! (Check the output to verify selected python version)
# cmake --build build -t pypangolin_pip_install

# Run me some tests! (Requires Catch2 which must be manually installed on Ubuntu.)
ctest
sudo cmake --install build
cd .. #/home


# Install Eigen3
sudo apt install -y libeigen3-dev

# Install Ceres-solver 1.14
# NOTE: CERES MUST BE INSTALLED BEFORE OPENCV
# CMake
sudo apt install -y cmake
# google-glog + gflags
sudo apt install -y libgoogle-glog-dev libgflags-dev
# Use ATLAS for BLAS & LAPACK
sudo apt install -y libatlas-base-dev
# Eigen3
sudo apt install -y libeigen3-dev
# SuiteSparse (optional)
sudo apt install -y libsuitesparse-dev

# clone ceres
wget ceres-solver.org/ceres-solver-1.14.0.tar.gz
tar -xvf ceres-solver-1.14.0.tar.gz
mkdir ceres-bin
cd ceres-bin

# build and install
cmake ../ceres-solver-1.14.0
make -j3
make test
sudo make install

cd .. #/home/dep


# Install Boost library
sudo apt update -y
sudo apt-get install -y libboost-all-dev


# Install OpenCV 3.4.1 (the last step)
sudo apt update -y

# Image I/O libs
Sudo add-apt-repository 'deb http://security.ubuntu.com/ubuntu xenial-security main'
sudo add-apt-repository -y ppa:linuxuprising/libpng12
sudo apt update -y
sudo apt install -y libpng12-0
sudo apt install -y libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev

# Video Libs - FFMPEG, GSTREAMER, x264 and so on
sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev
sudo apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt install -y libxvidcore-dev x264 libx264-dev libfaac-dev libmp3lame-dev libtheora-dev libvorbis-dev

# Cameras programming interface libs
sudo apt install -y libdc1394-22 libdc1394-22-dev libxine2-dev libv4l-dev v4l-utils

# GTK lib for the graphical user functionalites coming from OpenCV highghui module
sudo apt install -y libgtk-3-dev

# Python libraries for python2 and python3
sudo apt install -y python2-dev python3-dev python3-pip

curl https://bootstrap.pypa.io/pip/2.7/get-pip.py --output get-pip.py
sudo python2 get-pip.py

sudo -H pip2 install -U pip numpy
sudo -H pip3 install -U pip numpy

# Parallelism library C++ for CPU
sudo apt install -y libtbb-dev

# Optimization libraries for OpenCV
sudo apt install -y libatlas-base-dev gfortran

# Optional libraries:
sudo apt install -y libprotobuf-dev protobuf-compiler
sudo apt install -y libgoogle-glog-dev libgflags-dev
sudo apt install -y libgphoto2-dev libeigen3-dev libhdf5-dev doxygen

# Install other packages
apt install -y libfcitx-qt5-1 libvtk6-dev vtk7 python3-vtk7 tcl-vtk7

# clone and compile
mkdir opencv
cd opencv
wget https://github.com/opencv/opencv/archive/3.4.1.tar.gz
tar -xzvf 3.4.1.tar.gz
cd opencv-3.4.1
wget https://github.com/opencv/opencv_contrib/archive/3.4.1.tar.gz
tar -xzvf 3.4.1.tar.gz
mkdir build
cd build

# Config
cmake \
-D WITH_1394=ON \
-D WITH_AVFOUNDATION=OFF \
-D WITH_CARBON=OFF \
-D WITH_CAROTENE=OFF \
-D WITH_CPUFEATURES=OFF \
-D WITH_VTK=ON \
-D WITH_CUDA=OFF \
-D WITH_CUFFT=OFF \
-D WITH_CUBLAS=OFF \
-D WITH_NVCUVID=OFF \
-D WITH_EIGEN=ON \
-D WITH_VFW=OFF \
-D WITH_FFMPEG=ON \
-D WITH_GSTREAMER=ON \
-D WITH_GTK=ON \
-D WITH_IPP=ON \
-D WITH_HALIDE=ON \
-D WITH_INF_ENGINE=ON \
-D WITH_JASPER=ON \
-D WITH_JPEG=ON \
-D WITH_WEBP=ON \
-D WITH_OPENEXR=ON \
-D WITH_OPENGL=ON \
-D WITH_OPENVX=ON \
-D WITH_OPENNI=OFF \
-D WITH_OPENNI2=OFF \
-D WITH_PNG=ON \
-D WITH_GDCM=ON \
-D WITH_PVAPI=ON \
-D WITH_GIGEAPI=ON \
-D WITH_ARAVIS=ON \
-D WITH_QT=OFF \
-D WITH_WIN32UI=OFF \
-D WITH_QUICKTIME=OFF \
-D WITH_QTKIT=OFF \
-D WITH_TBB=ON \
-D WITH_OPENMP=ON \
-D WITH_CSTRIPES=ON \
-D WITH_PTHREADS_PF=ON \
-D WITH_TIFF=ON \
-D WITH_UNICAP=ON \
-D WITH_V4L=ON \
-D WITH_LIBV4L=ON \
-D WITH_DSHOW=ON \
-D WITH_MSMF=OFF \
-D WITH_XIMEA=ON \
-D WITH_XINE=ON \
-D WITH_CLP=ON \
-D WITH_OPENCL=ON \
-D WITH_OPENCL_SVM=ON \
-D WITH_OPENCLAMDFFT=ON \
-D WITH_OPENCLAMDBLAS=ON \
-D WITH_DIRECTX=OFF \
-D WITH_INTELPERC=OFF \
-D WITH_MATLAB=ON \
-D WITH_VA=ON \
-D WITH_VA_INTEL=ON \
-D WITH_MFX=ON \
-D WITH_GDAL=ON \
-D WITH_GPHOTO2=ON \
-D WITH_LAPACK=ON \
-D WITH_ITT=ON \
-D BUILD_opencv_apps=ON \
-D BUILD_opencv_world=OFF \
-D BUILD_opencv_calib3d=ON \
-D BUILD_opencv_core=ON \
-D BUILD_opencv_cudaarathm=OFF \
-D BUILD_opencv_cudabgsegm=OFF \
-D BUILD_opencv_cudacodec=OFF \
-D BUILD_opencv_cudafeatures2d=0FF \
-D BUILD_opencv_cuclafilters=OFF \
-D BUILD_opencv_cuciaimgproc=OFF \
-D BUILD_opencv_cudalegacy=OFF \
-D BUILD_opencv_cudaobjdetect=OFF \
-D BUILD_opencv_cudaoptflow=OFF \
-D BUILD_opencv_cudastereo=OFF \
-D BUILD_opencv_cudawarping=OFF \
-D BUILD_opencv_cudev=OFF \
-D BUILD_opencv_features2d=ON \
-D BUILD_opencv_flann=ON \
-D BUILD_opencv_hal=ON \
-D BUILD_opencv_highgui=ON \
-D BUILD_opencv_imgcodecs=ON \
-D BUILD_opencv_imgproc=ON \
-D BUILD_opencv_ml=ON \
-D BUILD_opencv_objdetect=ON \
-D BUILD_opencv_photo=ON \
-D BUILD_opencv_sharp=ON \
-D BUILD_opencv_dnn=ON \
-D BUILD_opencv_js=OFF \
-D BUILD_opencv_aruco=ON \
-D BUILD_opencv_bgsegm=ON \
-D BUILD_opencv_bioinspired=ON \
-D BUILD_opencv_ccalib=ON \
-D BUILD_opencv_cnn_3dobj=ON \
-D BUILD_opencv_cvv=ON \
-D BUILD_opencv_datasets=ON \
-D BUILD_opencv_dnn_objdetect=ON \
-D BUILD_opencv_dnns_easily_fooled=ON \
-D BUILD_opencv_dpm=ON \
-D BUILD_opencv_face=ON \
-D BUILD_opencv_fuzzy=ON \
-D BUILD_opencv_freetype=ON \
-D BUILD_opencv_line_descriptor=ON \
-D BUILD_opencv_matlab=ON \
-D BUILD_opencv_optflow=ON \
-D BUILD_opencv_ovis=ON \
-D BUILD_opencv_plot=ON \
-D BUILD_opencv_reg=ON \
-D BUILD_opencv_rgbd=ON \
-D BUILD_opencv_saliency=ON \
-D BUILD_opencv_sfm=ON \
-D BUILD_opencv_stereo=ON \
-D BUILD_opencv_structured_light=ON \
-D BUILD_opencv_surface_matching=ON \
-D BUILD_opencv_text=ON \
-D BUILD_opencv_tracking=ON \
-D BUILD_opencv_xfeatures2d=ON \
-D BUILD_opencv_ximgproc=ON \
-D BUILD_opencv_xobjdetect=ON \
-D BUILD_opencv_xphoto=ON \
-D BUILD_ANDROID_EXAMPLES=OFF \
-D BUILD_DOCS=ON \
-D BUILD_EXAMPLES=ON \
-D BUILD_PACKAGE=ON \
-D BUILD_PERF_TESTS=OFF \
-D BUILD_TESTS=OFF \
-D BUILD_WITH_DEBUG_INFO=OFF \
-D BUILD_WITH_STATIC_CRT=OFF \
-D BUILD_WITH_DYNAMIC_IPP=OFF \
-D BUILD_FAT_JAVA_LIB=OFF \
-D BUILD_ANDROID_SERVICE=OFF \
-D BUILD_CUDA_STUBS=OFF \
-D BUILD_JAVA=OFF \
-D INSTALL_CREATE_DISTRIB=OFF \
-D INSTALL_C_EXAMPLES=ON \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D INSTALL_ANDROID_EXAMPLES=OFF \
-D INSTALL_TO_MANGLED_PATHS=OFF \
-D INSTALL_TESTS=OFF \
-D ENABLE_FAST_MATH=1 \
-D ENABLE_NEON:BOOL=ON \
-D BUILD_opencv_python2=ON \
-D BUILD_opencv_python3=ON \
-D BUILD_NEW_PYTHON_SUPPORT=ON \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D OPENCV_EXTRA_MODULES_PATH=$PWD/../opencv_contrib-3.4.1/modules \
-D CMAKE_BUILD_TYPE=RELEASE ..


# Deal with an error in one of the source files
sed -i 's/char\* str = PyString_AsString(obj);/const char\* str = PyString_AsString(obj);/g' ../modules/python/src2/cv2.cpp

# Build and install
make -j8
sudo make install

sudo /bin/bash -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/opencv.conf'
sudo ldconfig

cd ../../.. #/home/dep


# clone and build the EORB-SLAM algorithm
cd .. #/home
git clone https://github.com/m-dayani/EORB_SLAM.git
cd EORB_SLAM

cd Thirdparty/g2o
mkdir build
cd build
cmake ..
make -j4

cd ../../DBoW2 #Thirdparty/DBoW2
mkdir build
cd build
cmake ..
make -j4

cd ../../.. #EORB_SLAM
mkdir build
cd build
#sed -i 's/++11/++14/g' ../CMakeLists.txt
cmake ..
make -j4














