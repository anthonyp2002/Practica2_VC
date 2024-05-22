cmake_minimum_required(VERSION 3.10)
project(Task2-Part-3)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(Task2-Part-3 Task2-Part-3.cpp)
target_link_libraries(Task2-Part-3 ${OpenCV_LIBS})

