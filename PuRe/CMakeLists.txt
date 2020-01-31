cmake_minimum_required (VERSION 3.8)

# C++17 required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Find OpenCV. Needs to be on cmake search path, or specify OpenCV_DIR variable.
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Add all source files to main target
file(GLOB MAIN_SRC
	"*.hpp"
	"*.cpp"
)
add_executable (main ${MAIN_SRC})
message(STATUS "Detected source files: ${MAIN_SRC}")

# Link against OpenCV
target_link_libraries(main ${OpenCV_LIBS})


# Compiler (platform) specific settings
if(MSVC)
	message(STATUS "Detected MSVC compiler")

	# Compiler flags for Windows MSVC
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")					# MSVC warning levels: W3 < W4 < Wall (Wall is way to verbose!)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")					# O2 is best speed optimization
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_USE_MATH_DEFINES")	# for M_PI in <cmath>
	# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Od")					# debug: no optimization

elseif(CMAKE_COMPILER_IS_GNUCXX)
	message(STATUS "Detected g++ compiler: ${CMAKE_CXX_COMPILER}")

	# Compiler flags for GCC
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")			# enable warnings
	# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")			# best optimization for runtime speed
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")	# compile with c++17 support
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")			# debug: no optimization
	
else()
	message(SEND_ERROR "Unsupported compiler! ${CMAKE_CXX_COMPILER}")
endif()


message(STATUS "CXX_FLAGS: ${CMAKE_CXX_FLAGS}")




