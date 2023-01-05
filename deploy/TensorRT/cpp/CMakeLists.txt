cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(bytetrack)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Set the bytetrack include directory and link
set(BYTETRACK_PUBLIC_HEADERS_DIR "include/")

set(CMAKE_CXX_FLAGiS "${CMAKE_CXX_FLAGS} -O3 -Wall")

find_package(OpenCV REQUIRED)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
set(Eigen3_LIBS Eigen3::Eigen)

file(GLOB BYTETRACK_SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp)

add_library(bytetrack SHARED ${BYTETRACK_SRCS})

target_link_libraries(bytetrack PUBLIC ${OpenCV_LIBS} ${Eigen3_LIBS})
target_include_directories(bytetrack PUBLIC ${BYTETRACK_PUBLIC_HEADERS_DIR})

message(STATUS "CMAKE INSTALL INCLUDE DIR: " ${CMAKE_INSTALL_INCLUDEDIR})

include(GNUInstallDirs)

install(DIRECTORY "include/" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS bytetrack LIBRARY DESTINATION ${CMAKE_ISNTALL_LIBDIR})
