#!/bin/bash
set -e

mkdir -p dependencies
cd dependencies

# Opencv
if [[ -d opencv ]]
then
    echo "Found cached OpenCV."
else
    echo "Rebuilding OpenCV cache..."
    wget -O -q opencv.zip https://github.com/opencv/opencv/archive/4.2.0.zip
    unzip -q opencv.zip
    cd opencv-4.2.0
    mkdir -p build
    cd build
    # MSMF: see https://github.com/skvark/opencv-python/issues/263
    cmake ..\
        -G"Visual Studio 15 2017 Win64"\
        -DCMAKE_BUILD_TYPE=Release\
        -DCMAKE_INSTALL_PREFIX=../../opencv\
        -DBUILD_LIST=core,highgui,videoio,imgcodecs,imgproc,video\
        -DBUILD_opencv_world=ON\
        -DWITH_TBB=ON\
        -DWITH_OPENMP=ON\
        -DWITH_IPP=ON\
        -DBUILD_EXAMPLES=OFF\
        -DWITH_NVCUVID=ON\
        -DWITH_CUDA=ON\
        -DBUILD_DOCS=OFF\
        -DBUILD_PERF_TESTS=OFF\
        -DBUILD_TESTS=OFF\
        -DWITH_CSTRIPES=ON\
        -DWITH_OPENCL=ON\
        -DWITH_MSMF=OFF
    cmake --build . --target INSTALL --config Release --parallel
    cd ../..
    rm -rf opencv.zip
    rm -rf opencv-4.2.0
fi

# Python
echo "Installing Python ${PYTHON_VERSION} with choco..."
choco install python --version $PYTHON_VERSION
python -m pip install -U pip
