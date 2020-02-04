Unix:
```bash
mkdir build
cd build
cmake ..
make
```

Windows:
```bat
mkdir build
cd build
cmake .. -G "Visual Studio 15 Win64" -DOpenCV_DIR=C:\work\opencv\build
cmake --build . --config Release
```