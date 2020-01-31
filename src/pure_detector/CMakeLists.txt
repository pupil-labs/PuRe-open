cmake_minimum_required (VERSION 3.8)


find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)
find_package(NumPy REQUIRED)
find_package(OpenCV REQUIRED)

add_cython_target(pure_detector CXX)
add_library(pure_detector MODULE pure_detector pure.cpp)
python_extension_module(pure_detector)

include_directories(${NumPy_INCLUDE_DIRS})
include_directories(".")
target_link_libraries(pure_detector ${OpenCV_LIBS})

install(TARGETS pure_detector LIBRARY DESTINATION pure_detector)