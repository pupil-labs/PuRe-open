#!/bin/bash
set -ev

cmake --version

# NOTE: Remove all unnecessary stuff (install files etc.) since we want to cache the
# dependencies and keep the cache small.

mkdir -p dependencies
cd dependencies

# Opencv
wget -O opencv.zip https://github.com/opencv/opencv/archive/3.4.5.zip
unzip -q opencv.zip
cd opencv-3.4.5
mkdir -p build
cd build
cmake ..\
    -G"Visual Studio 15 2017 Win64"\
    -DCMAKE_BUILD_TYPE=Release\
    -DCMAKE_INSTALL_PREFIX=../../opencv\
    -DBUILD_LIST=core,highgui,videoio,imgcodecs,imgproc,video\
    -DBUILD_opencv_world=ON\
    -DWITH_TBB=ON\
    -DWITH_OPENMP=ON\
    -DWITH_IPP=ON\
    -DCMAKE_BUILD_TYPE=RELEASE\
    -DBUILD_EXAMPLES=OFF\
    -DWITH_NVCUVID=ON\
    -DWITH_CUDA=ON\
    -DBUILD_DOCS=OFF\
    -DBUILD_PERF_TESTS=OFF\
    -DBUILD_TESTS=OFF\
    -DWITH_CSTRIPES=ON\
    -DWITH_OPENCL=ON
cmake --build . --target INSTALL --config Release --parallel
cd ../..
rm -rf opencv.zip
rm -rf opencv-3.4.5

# # Opencv
# # Downloading the precompiled installer for windows and executing via cmd is actually
# # faster than compiling yourself. Note: The .exe is a 7-zip self extracting archive,
# # which explains the non-exe-standard cli argument -y. See here:
# # https://sevenzip.osdn.jp/chm/cmdline/switches/sfx.htm
# wget https://sourceforge.net/projects/opencvlibrary/files/3.4.5/opencv-3.4.5-vc14_vc15.exe
# ./opencv-3.4.5-vc14_vc15.exe -y
# mv opencv opencvfull
# mv opencvfull/build opencv
# rm -rf opencvfull
# rm -rf opencv-3.4.5-vc14_vc15.exe
