cmake_minimum_required(VERSION 3.26.3)
project("peddet")
set(CMAKE_BUILD_TYPE DEBUG)

# project
message(STATUS "    PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
include_directories(${PROJECT_SOURCE_DIR}/include/)
add_library(inference SHARED
    ${PROJECT_SOURCE_DIR}/include/inference.h
    ${PROJECT_SOURCE_DIR}/src/inference.cpp
)
add_library(utils SHARED
    ${PROJECT_SOURCE_DIR}/include/utils.h
    ${PROJECT_SOURCE_DIR}/src/utils.cpp
)
add_executable(peddet ${PROJECT_SOURCE_DIR}/src/main.cpp)
target_link_libraries(peddet inference)
target_link_libraries(peddet utils)

#boost
set(BOOST_ROOT /usr/include/boost)
find_package(Boost REQUIRED COMPONENTS system filesystem)
include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(peddet ${Boost_SYSTEM_LIBRARY} ${Boost_FILESYSTEM_LIBRARY})

#opencv
target_link_libraries(peddet opencv_core opencv_imgproc opencv_highgui opencv_ml opencv_video)

# onnxruntime
set(ONNXRUNTIME_ROOT_PATH ./onnxruntime/)
set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime/core/session)
set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT_PATH}/build/Linux/Release/libonnxruntime.so)
include_directories(${ONNXRUNTIME_INCLUDE_DIRS})
target_link_libraries(peddet ${ONNXRUNTIME_LIB})











