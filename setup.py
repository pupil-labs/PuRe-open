import os
import platform
import sysconfig

import numpy as np
from Cython.Build import cythonize
from setuptools import find_packages, setup, Extension
from pathlib import Path

package_dir = "src"
package = "pure_detector"

install_requires = [
    "numpy",
]

########################################################################################
# Setup Libraries

include_dirs = []
libraries = []
library_dirs = []

# Cross-platform setup
include_dirs += [
    "PuRe",
    np.get_include(),
]

# Platform-specific setup
if platform.system() == "Windows":
    # legacy fallback for old manual windows setup
    OPENCV = "C:\\Users\\PFA\\Downloads\\opencv\\build"
    include_dirs.append(f"{OPENCV}\\include")
    library_dirs.append(f"{OPENCV}\\x64\\vc14\\lib")
    libraries.append("opencv_world412")

else:
    # Opencv
    opencv_include_dirs = [
        "/usr/local/opt/opencv/include",  # old opencv brew (v3)
        "/usr/local/opt/opencv@3/include",  # new opencv@3 brew
        "/usr/local/include/opencv4",  # new opencv brew (v4)
    ]
    opencv_library_dirs = [
        "/usr/local/opt/opencv/lib",  # old opencv brew (v3)
        "/usr/local/opt/opencv@3/lib",  # new opencv@3 brew
        "/usr/local/lib",  # new opencv brew (v4)
    ]
    opencv_libraries = [
        "opencv_core",
        "opencv_highgui",
        "opencv_videoio",
        "opencv_imgcodecs",
        "opencv_imgproc",
        "opencv_video",
    ]
    # Check if OpenCV has been installed through ROS
    opencv_core_found = any(
        os.path.isfile(path + "/libopencv_core.so") for path in opencv_library_dirs
    )
    if not opencv_core_found:
        ros_dists = ["kinetic", "jade", "indigo"]
        for ros_dist in ros_dists:
            ros_candidate_path = "/opt/ros/" + ros_dist + "/lib"
            if os.path.isfile(ros_candidate_path + "/libopencv_core3.so"):
                opencv_library_dirs = [ros_candidate_path]
                opencv_include_dirs = [
                    "/opt/ros/" + ros_dist + "/include/opencv-3.1.0-dev"
                ]
                opencv_libraries = [lib + "3" for lib in opencv_libraries]
                break
    include_dirs += opencv_include_dirs
    library_dirs += opencv_library_dirs
    libraries += opencv_libraries

    # Eigen
    include_dirs += [
        "/usr/local/include/eigen3",
        "/usr/include/eigen3",
    ]

    # Ceres
    libraries.append("ceres")

########################################################################################
# Setup Compile Args

extra_compile_args = []
extra_compile_args += [
    "-w",  # suppress all warnings (TODO: Why do we do this?)
]
if platform.system() == "Windows":
    # NOTE: c++11 is not available as compiler flag on MSVC
    extra_compile_args += [
        "-W3", # NOTE: warnings are not applied to underlying cpp only, but also to cython autogerated cpp code!
        "-O2",  # best speed optimization for MSVC
        "-D_USE_MATH_DEFINES",  # for M_PI
        "-DCMAKE_GENERATOR_PLATFORM=x64",
    ]
else:
    extra_compile_args += [
        "-std=c++11",
        # apply all recommended speed optimization (note -O3 is typically not recommeded
        # as it heavily relies on well-written code)
        "-O2",
    ]


########################################################################################
# Extension specs

# TODO: Cython recommends to include the generated cpp files in the source distribution
# and try to build from those first, only regenerating the cpp files from cython as a
# fallback. We don't do this currently, but since we are going to ship wheels, it won't
# be so bad since most users can just install the wheels. Read about this here:
# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#distributing-cython-modules

extensions = [
    Extension(
        name="pure_detector.pure_detector",
        sources=[
            f"{package_dir}/pure_detector/pure_detector.pyx",
            "PuRe/pure.cpp"
        ],
        language="c++",
        extra_compile_args=extra_compile_args,
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
    ),
]
########################################################################################
# Setup Script

print("INCLUDE DIRS:")
print(*(f" - {v}\n" for v in include_dirs), sep="")

print("LIBRARY DIRS:")
print(*(f" - {v}\n" for v in library_dirs), sep="")

print("LIBRARIES:")
print(*(f" - {v}\n" for v in libraries), sep="")

print("COMPILE ARGS:")
print(*(f" - {v}\n" for v in extra_compile_args), sep="")

if __name__ == "__main__":
    setup(
        author="Pupil Labs",
        author_email="info@pupil-labs.com",
        extras_require={"dev": ["pytest", "tox"]},
        ext_modules=cythonize(extensions, quiet=True, nthreads=8),
        install_requires=install_requires,
        license="GNU",
        name="pure_detector",
        packages=find_packages(package_dir),
        package_dir={"": package_dir},
        version="0.2",
    )
