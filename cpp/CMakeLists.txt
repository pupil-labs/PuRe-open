cmake_minimum_required (VERSION 3.8)

project(pure_cpp)

# C++17 required
if (APPLE)
	# We target 10.12 on travis with LLVM 4.0, which does not offer c++17 yet, so we
	# need to use c++1z.
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1z")
else ()
	set(CMAKE_CXX_STANDARD 17)
	set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
endif()

# Find OpenCV. Needs to be on cmake search path, or specify OpenCV_DIR variable.
find_package(OpenCV REQUIRED)

# static library for general purpose
add_library(pure_cpp_static STATIC pure.cpp pure.hpp)
target_include_directories(pure_cpp_static PUBLIC .)
target_link_libraries(pure_cpp_static ${OpenCV_LIBS})

if(MSVC)
	add_definitions(-D_USE_MATH_DEFINES)    # for M_PI in <cmath>
endif()