#!/bin/bash
set -e

# Opencv
echo "Checking OpenCV cache..."
if [[ -d dependencies/opencv ]]
then
    echo "Found OpenCV cache. Build configuration:"
    dependencies/opencv/bin/opencv_version -v
else
    echo "OpenCV cache missing. Rebuilding..."
    chmod +x .travis/setup_opencv_manylinux.sh
    docker run --rm -v `pwd`:/io quay.io/pypa/manylinux2014_x86_64 /io/.travis/setup_opencv_manylinux.sh
fi
