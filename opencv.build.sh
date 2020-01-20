#!/bin/bash
set -e
rm -R build || true
rm -R install || true
mkdir build
mkdir install
pushd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_opencv_apps=OFF \
    -DBUILD_LIST=core,highgui,videoio,imgcodecs,imgproc,video \
    -DWITH_TBB=ON \
    -DWITH_OPENMP=ON \
    -DWITH_IPP=ON \
    -DWITH_NVCUVID=ON \
    -DWITH_CUDA=ON \
    -DWITH_CSTRIPES=ON \
    -DWITH_OPENCL=ON \
    -DWITH_GTK=ON
make -j8
sudo make install
popd