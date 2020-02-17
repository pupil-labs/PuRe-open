Build the project with cmake.

Unix:
```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

Windows (remember to set OpenCV_DIR)
```bat
mkdir build
cd build
cmake .. -G "Visual Studio 15 Win64" -DOpenCV_DIR=C:\opencv\build
cmake --build . --config Release
```