# Pupil-PuRe

Open source implementation of the pupil detection algorithm described in the paper ["PuRe: Robust pupil detection for real-time pervasive eye tracking"](https://www.sciencedirect.com/science/article/pii/S1077314218300146).

## Usage

Here is some quickstart code to get you started. Have a look at the [Python example](./examples/python/main.py) for a working example.

```python
from pupil_pure import PuReDetector

detector = PuReDetector()

img = ... # read some eye image here

result = detector.detect(img)
```

The result will be a dictionary in the following form:
```python
{
    'confidence': 0.9491466283798218,                       # detection confidence
    'diameter': 92.72294616699219,                          # pupil diameter
    'location': (347.7300720214844, 259.9238586425781)      # pupil location (center)
    'ellipse': {                                            # ellipse parameters
        'angle': 137.4583740234375,
        'axes': (78.5771255493164, 92.72294616699219),
        'center': (347.7300720214844, 259.9238586425781)
    },
}
```


## Building from source

### Setup OpenCV Dependency
This project depends on [OpenCV](https://opencv.org/). We currently support OpenCV3 and OpenCV4.
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
- Download latest OpenCV release for Windows from https://opencv.org/releases/
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
The python library works by wrapping a Cython interface around the C++ code. The Cython wrapper can be found in [src/pupil_pure](./src/pupil_pure). Both C++ and Cython are compiled via CMake with the help of the [scikit-build build system](https://scikit-build.readthedocs.io/en/latest/). After installing with `pip` you can run the examples in [examples/python](./examples/python).

### Wheel Building Notes
Building wheels works on travis via macOS, Windows and Linux VMs. OpenCV is built on travis and included in the wheel as dynamic library.

In order to speed up builds, the built OpenCV is stored in travis caches. Caches can be cleared by including the following line in the commit message:
`[travis: clear-cache]` or more specifically e.g. `[travis: clear-cache win]`. The caches are checked, loaded and rebuilt in a setup stage and made available via travis workspaces.

Including the shared library in the wheel works differently depending on the platform:
- For Windows, the DLLs are added to the `package_data` in `setup.py`. Since they live in the same folder as the python extension module, they will be loaded without problems.
- For Linux the wheels are built on the officially recommended [manylinux](https://github.com/pypa/manylinux) image. The shared library is afterwards patched into the wheel with [auditwheel](https://github.com/pypa/auditwheel) as also recommended on the manylinux docs.
- For macOS we use [delocate](https://github.com/matthew-brett/delocate) to patch the libraries into the wheel and adjust the `install_name` accordingly.
