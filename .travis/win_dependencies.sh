#!/bin/bash
set -e

mkdir -p dependencies
cd dependencies

# Opencv
echo "\e[34mChecking OpenCV cache...\e[0m"
if [[ -d opencv ]]
then
    echo "\e[32mFound OpenCV cache. Build configuration:\e[0m"
    opencv/x64/vc15/bin/opencv_version_win32.exe
else
    echo "\e[33mOpenCV cache missing. Rebuilding...\e[0m"
    wget -q -O opencv.zip https://github.com/opencv/opencv/archive/4.2.0.zip
    unzip -q opencv.zip
    cd opencv-4.2.0
    mkdir -p build
    cd build
    # MSMF: see https://github.com/skvark/opencv-python/issues/263
    # CUDA/TBB: turned off because not easy to install on Windows and we cannot easily
    # ship this with the wheel.
    cmake ..\
        -G"Visual Studio 15 2017 Win64"\
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
        -DWITH_CUDA=OFF\
        -DWITH_TBB=OFF\
        -DWITH_MSMF=OFF
    cmake --build . --target INSTALL --config Release --parallel
    cd ../..
    rm -rf opencv.zip
    rm -rf opencv-4.2.0
fi
