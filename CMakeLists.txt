cmake_minimum_required(VERSION 3.10)
project(AR3)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_PREFIX_PATH /Users/apple/Documents/opencv3.4/opencv-3.4.1/release)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDES})


add_executable(AR3 main.cpp FaceClassifier.cpp FaceClassifier.h)
target_link_libraries(AR3 ${OpenCV_LIBS})