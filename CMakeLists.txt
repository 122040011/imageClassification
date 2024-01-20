cmake_minimum_required(VERSION 2.8)
project(Project3002)
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( Project3002 main.cpp
        main.h)
target_link_libraries( Project3002  ${OpenCV_LIBS})