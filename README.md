# pfa-pupil-detector

Open source implementation of the pupil detection algorithm described in the paper ["PuRe: Robust pupil detection for real-time pervasive eye tracking"](https://www.sciencedirect.com/science/article/pii/S1077314218300146).

## Building from source

### Setup OpenCV Dependency
This project depends on [OpenCV 4+](https://opencv.org/).
When building from source, you have to make sure that OpenCV can be found. 

#### Ubuntu
You can install OpenCV via the terminal with:

```bash
sudo apt install libopencv-dev
```

#### MacOS
You can install OpenCV with [brew](https://brew.sh/):
```bash
brew install opencv
```

#### Windows
- Download latest OpenCV 4.x release for Windows from https://opencv.org/releases/
- Run the downloaded `opencv-*.exe`. Extract to a place where you want to install OpenCV. This should NOT be a temporary location! Choose e.g. `C:/opencv`.
- Create a new environment variable `OpenCV_DIR` that points to the `build` folder of the just installed OpenCV, e.g. to `C:/opencv/build`.
- Add the following location to your PATH environment variable: `%OpenCV_DIR%/x64/vc15/bin`.

#### Custom OpenCV Built from Source
The build script uses CMAKE's `find_package(OpenCV)`, which should work fine with the most common installation procedures for OpenCV. If you decide to compile OpenCV from source, make sure to use CMAKE, set the `OpenCV_DIR` environment variable and include the shared libraries in your `PATH`.

### Build and Install with pip
Clone the repository and run
```
pip install .
```

## Project Structure

### C++ Core
All code for the detector is located in [cpp](./cpp) and includes a `CMakeLists.txt` for building a static library. In [examples/cpp](./examples/cpp) you can find an example for how to use the code in a C++ project.

### Python Wrapper
The python library works by wrapping a Cython interface around the C++ code. The Cython wrapper can be found in [src/pure_detector](./src/pure_detector). Both C++ and Cython are compiled via CMake with the help of the [scikit-build build system](https://scikit-build.readthedocs.io/en/latest/). After installing with `pip` you can run the examples in [examples/python](./examples/python).