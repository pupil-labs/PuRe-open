cmake_minimum_required (VERSION 3.8)

if(NOT TARGET pure_cpp_static)
    message(FATAL_ERROR "Could not find pure_cpp_static target!")
endif()

find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)
find_package(NumPy REQUIRED)
find_package(OpenCV REQUIRED)

# Note: add_cython_target does not actually add a target, but fills a variable with the
# corresponding compiled source file, e.g. here 'pure_detector.cxx'. If only the name is
# specified, it will look for a cython file with the same base name: 'pure_detecot.pyx'.
add_cython_target(pure_detector CXX PY3)
# Then we create a module library from this source file and wrap it with settings for
# creating a python extension.
add_library(pure_detector MODULE ${pure_detector})
python_extension_module(pure_detector)

# linker settings
set(CMAKE_MACOSX_RPATH FALSE)
target_link_libraries(pure_detector pure_cpp_static)
target_link_libraries(pure_detector ${OpenCV_LIBS})
include_directories(${NumPy_INCLUDE_DIRS})

# we need to install the compiled cython lib into the package dir in /src !!
file(RELATIVE_PATH package_dir ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS pure_detector LIBRARY DESTINATION ${package_dir})