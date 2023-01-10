cmake_minimum_required(VERSION 2.8)
project(EORB_SLAM_ROS)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Debug)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall -Wno-deprecated -Wno-unused -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-deprecated -Wno-unused -O0")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")


# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(ORB_SLAM3 REQUIRED)

find_package(catkin REQUIRED COMPONENTS
        roscpp
        rosbag
        sensor_msgs
        dvs_msgs
        cv_bridge
        std_msgs
        image_transport)

include_directories(
${ORB_SLAM3_INCLUDE_DIRS}
${PROJECT_SOURCE_DIR}/include/ROS
${catkin_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME}
#ROS-Utils
src/ROS/RosBagStore.cpp
include/ROS/RosBagStore.h
)

target_link_libraries(${PROJECT_NAME}
${ORB_SLAM3_LIBRARIES}
${catkin_LIBRARIES}
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Event)

# Requires ROS
add_executable(fmt_ev_mvsec
Examples/Event/fmt_ev_mvsec.cpp)
target_link_libraries(fmt_ev_mvsec ${PROJECT_NAME})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Test)

# Requires ROS
add_executable(load_bag_data
Examples/Test/load_bag_data.cpp)
target_link_libraries(load_bag_data ${PROJECT_NAME})




