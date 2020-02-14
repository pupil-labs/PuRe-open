#!/bin/bash
set -e

python_bin_dir="$(echo /opt/python/cp${PY_MM}*/bin)"
echo Activating Python binaries at $python_bin_dir
export PATH=$python_bin_dir:$PATH

pip install -U pip


export OpenCV_DIR=/io/dependencies/opencv
pip wheel /io -w /io/raw_wheels --no-deps

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/io/dependencies/opencv/lib64
for whl in /io/raw_wheels/*.whl; do
    auditwheel repair "$whl" --plat manylinux2014_x86_64 -w /io/dist/
done

find /io/dist

for whl in /io/dist/*.whl; do
    pip install "$whl"
done

# test
python -c "from pure_detector import PuReDetector"

# create sdist for deployment
pip install pep517
python -m pep517.build --source --out-dir /io/dist /io
