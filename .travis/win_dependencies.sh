#!/bin/bash
set -ev

if [[ -d dependencies/opencv ]]
then
    echo "Found cached OpenCV."
    exit 0
fi
echo "Rebuilding OpenCV cache."

mkdir -p dependencies
cd dependencies

# Opencv
# Downloading the precompiled installer for windows and executing via cmd is actually
# faster than compiling yourself. Note: The .exe is a 7-zip self extracting archive,
# which explains the non-exe-standard cli argument -y. See here:
# https://sevenzip.osdn.jp/chm/cmdline/switches/sfx.htm
wget -q https://sourceforge.net/projects/opencvlibrary/files/4.2.0/opencv-4.2.0-vc14_vc15.exe
./opencv-4.2.0-vc14_vc15.exe -y
mv opencv opencvfull
mv opencvfull/build opencv
rm -rf opencvfull
rm -rf opencv-4.2.0-vc14_vc15.exe
