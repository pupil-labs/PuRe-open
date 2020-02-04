Unix:
```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

Windows:
```bat
mkdir build
cd build
cmake .. -G "Visual Studio 15 Win64" -DOpenCV_DIR=C:\work\opencv\build
cmake --build . --config Release
```