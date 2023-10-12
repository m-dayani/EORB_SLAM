FROM ubuntu:20.04

# RUN mkdir -p /home/$(uname -r)/build

# Install build essential
RUN apt update -y
RUN apt install -y build-essential
RUN apt-get install -y manpages-dev
RUN apt install -y software-properties-common && gcc --version

# On Ubuntu 20.04, you need to install gcc and g++ 7
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test && apt update -y
RUN apt install -y gcc-7 g++-7

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 --slave /usr/bin/g++ g++ /usr/bin/g++-9 --slave /usr/bin/gcov gcov /usr/bin/gcov-9
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 100 --slave /usr/bin/g++ g++ /usr/bin/g++-7 --slave /usr/bin/gcov gcov /usr/bin/gcov-7

RUN gcc --version

# Install other tools
RUN apt install -y make cmake pkg-config unzip yasm git gfortran nano wget curl

# Get Pangolin
RUN cd /home && mkdir dep && cd dep
RUN git clone --recursive https://github.com/stevenlovegrove/Pangolin.git
RUN cd Pangolin

# Install dependencies (as described above, or your preferred method)
# WARNING: This requires user interaction, so make it independent of user
RUN sed -i 's/(install/(install -y/g' ./scripts/install_prerequisites.sh
# Also, docker container is run as root by default so there is no need to: sudo
RUN sed -i 's/sudo/ /g' ./scripts/install_prerequisites.sh
RUN ./scripts/install_prerequisites.sh recommended

# Configure and build
RUN cmake -B build && cmake --build build

# Run me some tests! (Requires Catch2 which must be manually installed on Ubuntu.)
RUN ctest && cmake --install build && cd ..

# Install Eigen3
RUN apt install -y libeigen3-dev

# Install Ceres-solver 1.14
# NOTE: CERES MUST BE INSTALLED BEFORE OPENCV
# CMake
RUN apt install -y cmake
# google-glog + gflags
RUN apt install -y libgoogle-glog-dev libgflags-dev
# Use ATLAS for BLAS & LAPACK
RUN apt install -y libatlas-base-dev
# Eigen3
RUN apt install -y libeigen3-dev
# SuiteSparse (optional)
RUN apt install -y libsuitesparse-dev

# clone ceres
RUN wget ceres-solver.org/ceres-solver-1.14.0.tar.gz
RUN tar -xvf ceres-solver-1.14.0.tar.gz
RUN mkdir ceres-bin && cd ceres-bin

# build and install
RUN cmake ../ceres-solver-1.14.0
RUN make -j3 && make test && make install && cd .. #/home/dep


# Install Boost library
RUN apt update -y && apt install -y libboost-all-dev


# Install OpenCV 3.4.1 (the last step)
RUN apt update -y

# Image I/O libs
RUN add-apt-repository -y 'deb http://security.ubuntu.com/ubuntu xenial-security main'
RUN add-apt-repository -y ppa:linuxuprising/libpng12
RUN apt update -y
RUN apt install -y libpng12-0
RUN apt install -y libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev

# Video Libs - FFMPEG, GSTREAMER, x264 and so on
RUN apt install -y libavcodec-dev libavformat-dev libswscale-dev
RUN apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
RUN apt install -y libxvidcore-dev x264 libx264-dev libfaac-dev libmp3lame-dev libtheora-dev libvorbis-dev

# Cameras programming interface libs
RUN apt install -y libdc1394-22 libdc1394-22-dev libxine2-dev libv4l-dev v4l-utils

# GTK lib for the graphical user functionalites coming from OpenCV highghui module
RUN apt install -y libgtk-3-dev

# Python libraries for python2 and python3
RUN apt install -y python2-dev python3-dev python3-pip

RUN curl https://bootstrap.pypa.io/pip/2.7/get-pip.py --output get-pip.py
RUN python2 get-pip.py

RUN -H pip2 install -U pip numpy
RUN -H pip3 install -U pip numpy

# Parallelism library C++ for CPU
RUN apt install -y libtbb-dev

# Optimization libraries for OpenCV
RUN apt install -y libatlas-base-dev gfortran

# Optional libraries:
RUN apt install -y libprotobuf-dev protobuf-compiler
RUN apt install -y libgoogle-glog-dev libgflags-dev
RUN apt install -y libgphoto2-dev libeigen3-dev libhdf5-dev doxygen

# Install other packages
RUN install -y libfcitx-qt5-1 libvtk6-dev vtk7 python3-vtk7 tcl-vtk7

# clone and compile
RUN mkdir opencv && cd opencv
RUN wget https://github.com/opencv/opencv/archive/3.4.1.tar.gz
RUN tar -xzvf 3.4.1.tar.gz && cd opencv-3.4.1
RUN wget https://github.com/opencv/opencv_contrib/archive/3.4.1.tar.gz
RUN tar -xzvf 3.4.1.tar.gz && mkdir build && cd build

# Config
RUN cmake \
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
RUN sed -i 's/char\* str = PyString_AsString(obj);/const char\* str = PyString_AsString(obj);/g' ../modules/python/src2/cv2.cpp

# Build and install
RUN make -j8 && make install

RUN /bin/bash -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/opencv.conf' && ldconfig

RUN cd ../../../.. #/home


# clone and build the EORB-SLAM algorithm
RUN git clone https://github.com/m-dayani/EORB_SLAM.git && cd EORB_SLAM

RUN cd Thirdparty/g2o && mkdir build && cd build && cmake .. && make -j4 && cd ../../DBoW2 #Thirdparty/DBoW2

RUN mkdir build && cd build && cmake .. && make -j4 && cd ../../.. #EORB_SLAM

RUN mkdir build && cd build && cmake .. && make -j4


CMD ["/bin/bash"]
