# EORB-SLAM: An Event-based ORB-SLAM

This project is a feature-based odometry/SLAM algorithm that is used to estimate the 6DoF pose of a robot and reconstruct the 3D point cloud of the scene using a variety of sensors. This project is built on an older version of the [ORB-SLAM](https://github.com/UZ-SLAMLab/ORB_SLAM3) algorithm. It extends ORB-SLAM, which allows simultaneous localization and mapping using a monocular [DAVIS event camera](https://en.wikipedia.org/wiki/Event_camera). In addition to the configurations in the original work, this project supports Event-only, Event-IMU, Event-Image, and Event-Image-IMU modes. Please refer to [this article](https://arxiv.org/abs/2301.00618) for an explanation of the event-based configuration (no image).

**Contributions**:

- Support for DAVIS event cameras
- Added these modes to the original algorithm: Event-only, Event-Image (monocular), Event-Image-Inertial (monocular), Event-Inertial (monocular)
- Dataset loader can automatically load different datasets and configuration parameters
- Introducing sensor configuration object
- Mixed key point tracking (in addition to ORB features, the proposed algorithm can use both the AKAZE and ORB features for image-based SLAM)

## Dependencies

Since this project is an extension of the ORB-SLAM algorithm, it inherits all its dependencies in addition to new ones. These instructions are for Ubuntu 20.04 LTS. A successful build requires a stable internet connection (downloads more than 2 GB of data!) and a powerful machine with about 16 GB RAM.

### C++11 or C++0x Compiler

[Install g++-7 and gcc-7 along with version 9 on Ubuntu 20.04](https://vegastack.com/tutorials/how-to-install-gcc-compiler-on-ubuntu-20-04/)

### Install the necessary tools

```bash
sudo apt install -y make cmake pkg-config unzip yasm git gfortran nano wget curl
```

### Pangolin

[Pangolin project page](https://github.com/stevenlovegrove/Pangolin)

### Eigen3

`sudo apt install libeigen3-dev`

### Ceres-solver 1.14

Follow [these steps](http://ceres-solver.org/installation.html) to install Ceres-solver 1.14. Note the version; it won't work with newer versions! (Also see [this post in Medium](https://yunusmuhammad007.medium.com/jetson-tk1-install-ceres-solver-2-2-68787e237649))

### Boost

`sudo apt-get install -y libboost-all-dev`

### The OpenCV hell!

Follow the steps in [Install OpenCV-3.4.1 with CUDA](https://gist.github.com/raulqf/a3caa97db3f8760af33266a1475d0e5e) and [Install OpenCV-3.4.1 on Ubuntu 17.10](https://gist.github.com/okanon/c09669f3ff3351c864742bc2754b01ea) to install `OpenCV 3.4.1` on your machine.

### ROS (optional)

The original code uses ROS `melodic` on Ubuntu 18.04. However, ROS `noetic` is supported on Ubuntu 20.04. Integration of this project with ROS is possible, but it requires compiling and installing the core project first and then linking it to your ROS logic.

## Installation

Once you installed all the dependencies, you can download and install the project:

```bash
git clone https://github.com/m-dayani/EORB_SLAM.git
cd EORB_SLAM

mkdir build
cd build
cmake ..
make -j4
```

To make life easier for you, I included a bash script (`build_eorb_slam.sh`) and a Docker file that contains all the necessary commands to download and install the required packages.

## Usage

1. Download an event-based dataset:
    - [The Public Event dataset](https://rpg.ifi.uzh.ch/davis_data.html)
    - [The MVSEC dataset](https://daniilidis-group.github.io/mvsec/)
2. You can also use a non-event-based dataset to explore the visual and inertial modes:
    - [The EuRoC dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)
3. Extract the ORB Vocabulary file in the Vocabulary directory
4. Edit a configuration file (listed in `Examples/Event/*.yaml`)
5. Disable the visualization mode in the configuration file if running a Docker container.
6. Run an example:
  `./Examples/Event/fmt_ev_ethz ./Examples/Event/EvETHZ.yaml`

## Troubleshooting

1. Are all dependencies installed correctly?
2. Can you install the [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)?
3. Do you use `g++ 7.5` and `gcc 7.5` to compile packages?
4. Is the Ceres version `1.14`?
5. Did you build and install the OpenCV last so that it compiles the library with the necessary packages (`sfm::reconstruct`)?

## Issues

**DISCLAIMER**:
Although this app is tested on various emulated and physical devices, some specific hardware or unpredicted practical situations might affect its performance. Use it at your own risk!

- Generally, __this app is not stable__. There are some issues in the ORB-SLAM3 algorithm itself. The original algorithm is designed for larger images of datasets like `EuRoC`. If you try to run it on small images of an event-based dataset (especially the hdr_boxes of The Public Event dataset) the backend (optimizer) throws unhandled exceptions. Such optimization exceptions might randomly happen for the EORB-SLAM algorithm.
- For performance, most main components of this algorithm are executed in parallel threads. Unexpected race conditions and deadlocks are other issues of this algorithm.


