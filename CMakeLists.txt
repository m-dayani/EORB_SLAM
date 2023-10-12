cmake_minimum_required(VERSION 2.8)
project(ORB_SLAM3)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("-- Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wno-deprecated -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-deprecated -O3 -march=native")

# Check C++14 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++14.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Ceres REQUIRED)

set(OpenCV_DIR /usr/local/share/OpenCV)
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

MESSAGE("-- OPENCV VERSION: ${OpenCV_VERSION}")
#message(STATUS "OpenCV library status:")
#message(STATUS "    config: ${OpenCV_DIR}")
#message(STATUS "    version: ${OpenCV_VERSION}")
#message(STATUS "    libraries: ${OpenCV_LIBS}")
#message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

find_package(Pangolin REQUIRED)

#find_package(Glog REQUIRED)


include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${PROJECT_SOURCE_DIR}/include/CameraModels
${PROJECT_SOURCE_DIR}/include/Utils
${PROJECT_SOURCE_DIR}/include/IMU
${PROJECT_SOURCE_DIR}/include/Event
${PROJECT_SOURCE_DIR}/include/EventTest
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
${CERES_INCLUDE_DIRS}
${OpenCV_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Atlas.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/PnPsolver.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Initializer.cc
src/Viewer.cc
        src/IMU/ImuTypes.cc
        src/IMU/IMU_Manager.cpp
src/G2oTypes.cc
src/TwoViewReconstruction.cc
src/CameraModels/Pinhole.cpp
src/CameraModels/KannalaBrandt8.cpp
src/OptimizableTypes.cpp
src/MLPnPsolver.cpp
src/MixedFrame.cpp
        src/MixedKeyFrame.cpp
        src/MixedMatcher.cpp
        src/Utils/MyDataTypes.cpp
src/Utils/DataStore.cpp
        src/Utils/TumRgbdLoader.cpp
        src/Utils/KittiLoader.cpp
src/Utils/MyYamlParser.cpp
        src/Utils/MyParameters.cpp
        src/Utils/Visualization.cpp
        src/Utils/MyRandGen.cpp
        src/Utils/MyOptimizer.cpp
        src/Utils/MyCalibrator.cpp
        src/Utils/MyOptimTypes.cpp
        src/Utils/MyFrameDrawer.cpp
        src/Utils/FeatureTrack.cpp
        src/Event/EventData.cpp
        src/Event/EventLoader.cpp
src/Event/EventConversion.cc
        src/Event/EventFrame.cpp
src/Event/KLT_Tracker.cpp
src/Event/EvTrackManager.cpp
        src/Event/EvAsynchTracker.cpp
        src/Event/EvAsynchTrackerU.cpp
        src/Event/EvImBuilder.cpp
        src/Event/EvOptimizer.cpp
        src/Event/EvBaseTracker.cpp
        src/Event/EvLocalMapping.cpp
        src/Event/EvSynchTracker.cpp
        src/Event/EvSynchTrackerU.cpp
include/System.h
include/Tracking.h
include/LocalMapping.h
include/LoopClosing.h
include/ORBextractor.h
include/ORBmatcher.h
include/FrameDrawer.h
include/Converter.h
include/MapPoint.h
include/KeyFrame.h
include/Atlas.h
include/Map.h
include/MapDrawer.h
include/Optimizer.h
include/PnPsolver.h
include/Frame.h
include/KeyFrameDatabase.h
include/Sim3Solver.h
include/Initializer.h
include/Viewer.h
        include/IMU/ImuTypes.h
        include/IMU/IMU_Manager.h
include/G2oTypes.h
include/CameraModels/GeometricCamera.h
include/CameraModels/Pinhole.h
include/CameraModels/KannalaBrandt8.h
include/OptimizableTypes.h
include/MLPnPsolver.h
include/TwoViewReconstruction.h
include/MixedFrame.h
        include/MixedKeyFrame.h
        include/MixedMatcher.h
        include/Utils/MyDataTypes.h
include/Utils/DataStore.h
        include/Utils/TumRgbdLoader.h
        include/Utils/KittiLoader.h
include/Utils/MyYamlParser.h
        include/Utils/MyParameters.h
        include/Utils/Visualization.h
        include/Utils/MyRandGen.h
        include/Utils/MyOptimizer.h
        include/Utils/MyCalibrator.h
        include/Utils/MyOptimTypes.h
        include/Utils/MyFrameDrawer.h
        include/Utils/FeatureTrack.h
        include/Event/EventData.h
        include/Event/EventLoader.h
include/Event/EventConversion.h
        include/Event/EventFrame.h
include/Event/KLT_Tracker.h
include/Event/EvTrackManager.h
        include/Event/EvAsynchTracker.h
        include/Event/EvAsynchTrackerU.h
        include/Event/EvImBuilder.h
        include/Event/EvOptimizer.h
        include/Event/EvBaseTracker.h
        include/Event/EvLocalMapping.h
        include/Event/EvSynchTracker.h
        include/Event/EvSynchTrackerU.h
        include/compiler_options.h)

add_subdirectory(Thirdparty/g2o)


target_link_libraries(${PROJECT_NAME}
${EIGEN3_LIBS}
${CERES_LIBRARIES}
${OpenCV_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
-lboost_system
-lboost_filesystem
-lboost_serialization
-lcrypto
)


# Build examples

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)
#add_executable(rgbd_tum
#Examples/RGB-D/rgbd_tum.cc)
#target_link_libraries(rgbd_tum ${PROJECT_NAME})


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo)

#add_executable(stereo_kitti
#Examples/Stereo/stereo_kitti.cc)
#target_link_libraries(stereo_kitti ${PROJECT_NAME})

add_executable(stereo_euroc
Examples/Stereo/stereo_euroc.cc)
target_link_libraries(stereo_euroc ${PROJECT_NAME})

#add_executable(stereo_tum_vi
#Examples/Stereo/stereo_tum_vi.cc)
#target_link_libraries(stereo_tum_vi ${PROJECT_NAME})


#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo-Inertial)
#
#add_executable(stereo_inertial_euroc
#Examples/Stereo-Inertial/stereo_inertial_euroc.cc)
#target_link_libraries(stereo_inertial_euroc ${PROJECT_NAME})
#
#
#add_executable(stereo_inertial_tum_vi
#Examples/Stereo-Inertial/stereo_inertial_tum_vi.cc)
#target_link_libraries(stereo_inertial_tum_vi ${PROJECT_NAME})


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)

add_executable(mono_tum
Examples/Monocular/mono_tum.cc)
target_link_libraries(mono_tum ${PROJECT_NAME})

add_executable(mono_kitti
Examples/Monocular/mono_kitti.cc)
target_link_libraries(mono_kitti ${PROJECT_NAME})

add_executable(mono_euroc
Examples/Monocular/mono_euroc.cc)
target_link_libraries(mono_euroc ${PROJECT_NAME})

add_executable(mono_tum_vi
Examples/Monocular/mono_tum_vi.cc)
target_link_libraries(mono_tum_vi ${PROJECT_NAME})


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular-Inertial)

add_executable(mono_inertial_euroc
Examples/Monocular-Inertial/mono_inertial_euroc.cc)
target_link_libraries(mono_inertial_euroc ${PROJECT_NAME})

add_executable(mono_inertial_tum_vi
Examples/Monocular-Inertial/mono_inertial_tum_vi.cc)
target_link_libraries(mono_inertial_tum_vi ${PROJECT_NAME})


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Event)

add_executable(fmt_euroc
Examples/Event/fmt_euroc.cpp)
target_link_libraries(fmt_euroc ${PROJECT_NAME})

add_executable(fmt_ev_ethz
Examples/Event/fmt_ev_ethz.cpp)
target_link_libraries(fmt_ev_ethz ${PROJECT_NAME})

add_executable(fmt_tum_rgbd
Examples/Event/fmt_tum_rgbd.cpp)
target_link_libraries(fmt_tum_rgbd ${PROJECT_NAME})

add_executable(fmt_kitti
Examples/Event/fmt_kitti.cpp)
target_link_libraries(fmt_kitti ${PROJECT_NAME})


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Test)

#add_executable(glog_test
#Examples/Test/glog_test.cpp)
#target_link_libraries(glog_test ${PROJECT_NAME})

#add_executable(static_fun_redef
#Examples/Test/static_fun_redef.cpp)
#target_link_libraries(static_fun_redef ${PROJECT_NAME})

#add_executable(stl_containers
#Examples/Test/stl_containers.cpp)
#target_link_libraries(stl_containers ${PROJECT_NAME})

#add_executable(cv_sfm_reconst
#Examples/Test/cv_sfm_reconst.cpp)
#target_link_libraries(cv_sfm_reconst ${PROJECT_NAME})

#add_executable(events_to_image
#Examples/Test/events_to_image.cc)
#target_link_libraries(events_to_image ${PROJECT_NAME})
#
#add_executable(load_data
#Examples/Test/load_data.cpp)
#target_link_libraries(load_data ${PROJECT_NAME})
#
#add_executable(ft_extract_match
#Examples/Test/ft_extract_match.cpp)
#target_link_libraries(ft_extract_match ${PROJECT_NAME})




