FROM ubuntu:20.04

# RUN mkdir -p /home/$(uname -r)/build

# Install build essential
RUN apt update -y
RUN apt install -y build-essential
RUN apt-get install -y manpages-dev && gcc --version

# On Ubuntu 20.04, you need to install gcc and g++ 7
RUN apt install -y software-properties-common
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test && apt update -y
RUN apt install -y gcc-7 g++-7

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 --slave /usr/bin/g++ g++ /usr/bin/g++-9 --slave /usr/bin/gcov gcov /usr/bin/gcov-9
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 100 --slave /usr/bin/g++ g++ /usr/bin/g++-7 --slave /usr/bin/gcov gcov /usr/bin/gcov-7

RUN gcc --version

# Install other tools
RUN apt install -y git make cmake

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
RUN cmake -B build
RUN cmake --build build

# Run me some tests! (Requires Catch2 which must be manually installed on Ubuntu.)
RUN ctest && cmake --install build && cd ..

# OpenCV


CMD ["/bin/bash"]
