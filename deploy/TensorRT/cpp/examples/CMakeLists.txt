cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(examples)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(LINK_OPTIONS
  bytetrack
  ${OpenCV_LIBS}
  Eigen3::Eigen
)

add_executable(mot_eval mot_eval/mot_eval.cpp)

target_include_directories(mot_eval PRIVATE mot_eval)

target_link_libraries(mot_eval ${LINK_OPTIONS})
