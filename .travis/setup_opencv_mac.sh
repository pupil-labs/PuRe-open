#!/bin/bash
set -e

mkdir -p dependencies
cd dependencies

# Opencv
echo "Checking OpenCV cache..."
if [[ -d opencv ]]
then
    echo "Found OpenCV cache. Build configuration:"
    opencv/bin/opencv_version -v
else
    echo "OpenCV cache missing. Rebuilding..."
    wget -q -O opencv.zip https://github.com/opencv/opencv/archive/4.2.0.zip
    unzip -q opencv.zip
    cd opencv-4.2.0
    mkdir -p build
    cd build
    cmake ..\
        -DCMAKE_BUILD_TYPE=Release\
        -DCMAKE_INSTALL_PREFIX=../../opencv\
        -DBUILD_LIST=core,highgui,videoio,imgcodecs,imgproc,video\
        -DBUILD_opencv_world=ON\
        -DBUILD_EXAMPLES=OFF\
        -DBUILD_DOCS=OFF\
        -DBUILD_PERF_TESTS=OFF\
        -DBUILD_TESTS=OFF\
        -DBUILD_opencv_java=OFF\
        -DBUILD_opencv_python=OFF\
        -DWITH_OPENMP=ON\
        -DWITH_IPP=ON\
        -DWITH_CSTRIPES=ON\
        -DWITH_OPENCL=ON\
        -DWITH_TBB=OFF\
        -DWITH_CUDA=OFF
    make -j2 && make install
    cd ../..
    rm -rf opencv.zip
    rm -rf opencv-4.2.0
fi
