cmake_minimum_required (VERSION 3.8)

project(pure_cpp_example)

# C++17 required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Find OpenCV. Needs to be on cmake search path, or specify OpenCV_DIR variable.
find_package(OpenCV REQUIRED)
# Adding the subdirectory will add the library target as well
add_subdirectory(
	${CMAKE_CURRENT_SOURCE_DIR}/../../cpp	# where to load from
	${CMAKE_CURRENT_BINARY_DIR}/cpp			# where to store it while building
)

if(NOT TARGET pure_cpp_static)
    message(FATAL_ERROR "Could not find pure_cpp_static target!")
endif()

# add and link executable example
add_executable (main main.cpp)
target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(main pure_cpp_static)


# Compiler (platform) specific settings
if(MSVC)
	message(STATUS "Detected MSVC compiler")

	# Compiler flags for Windows MSVC
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")	# O2 is best speed optimization
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Od")		# debug: no optimization
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")					# MSVC warning levels: W3 < W4 < Wall (Wall is way to verbose!)
	add_definitions(-D_USE_MATH_DEFINES)							# for M_PI in <cmath>

elseif(CMAKE_COMPILER_IS_GNUCXX)
	message(STATUS "Detected g++ compiler: ${CMAKE_CXX_COMPILER}")

	# Compiler flags for GCC
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")	# best optimization for runtime speed
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")		# debug: no optimization
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")					# enable warnings
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")			# compile with c++17 support
	
else()
	message(SEND_ERROR "Unsupported compiler! ${CMAKE_CXX_COMPILER}")
endif()
