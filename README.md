# pfa-pupil-detector

Open source implementation of the pupil detection algorithm described in the paper ["PuRe: Robust pupil detection for real-time pervasive eye tracking"](https://www.sciencedirect.com/science/article/pii/S1077314218300146).

## Building from source

### Setup OpenCV Dependency
When building from source, you have to make sure that OpenCV can be found.

#### Ubuntu
You can install OpenCV via the terminal with:

```bash
apt get install libopencv-dev
```

#### Windows
- Download latest OpenCV 3.x release for Windows from https://opencv.org/releases/
- Run the downloaded `opencv-*.exe`. Extract to a place where you want to install OpenCV. This should NOT be a temporary location! Choose e.g. `C:/OpenCV`.
- Create a new environment variable `OpenCV_DIR` that points to the `build` folder of the just installed OpenCV, e.g. to `C:/OpenCV/build`.
- Add the following location to your PATH environment variable: `%OpenCV_DIR%/x64/vc15/bin`.

#### Custom OpenCV Built from Source
The build script uses CMAKE's `find_package(OpenCV)`, which should work fine with the most common installation procedures for OpenCV. If you decide to compile OpenCV from source, make sure to use CMAKE, set the `OpenCV_DIR` environment variable and include the shared libraries in your `PATH`.

### Install with pip
Clone the repository and run
```
pip install .
```