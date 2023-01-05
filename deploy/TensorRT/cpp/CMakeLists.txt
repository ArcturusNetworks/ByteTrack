cmake_minimum_required(VERSION 3.10)

project(bytetrack)

#add_definitions(-std=c++11)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Set the bytetrack include directory and link
set(BYTETRACK_PUBLIC_HEADERS_DIR "include/bytetrack")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

find_package(OpenCV REQUIRED)
#target_include_directories(bytetrack ${OpenCV_INCLUDE_DIRS})

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
set(Eigen3_LIBS Eigen3::Eigen)

file(GLOB My_Source_Files ${PROJECT_SOURCE_DIR}/src/*.cpp)

add_library(bytetrack SHARED ${My_Source_Files})

target_link_libraries(bytetrack PUBLIC ${OpenCV_LIBS} ${Eigen3_LIBS})
target_include_directories(bytetrack PUBLIC ${BYTETRACK_PUBLIC_HEADERS_DIR})

add_definitions(-O2 -pthread)

message(STATUS "CMAKE INSTALL INCLUDE DIR: " ${CMAKE_INSTALL_INCLUDEDIR})

include(GNUInstallDirs)

install(DIRECTORY "include/bytetrack" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS bytetrack LIBRARY DESTINATION ${CMAKE_ISNTALL_LIBDIR})
