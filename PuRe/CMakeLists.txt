cmake_minimum_required (VERSION 3.8)

file(GLOB MAIN_SRC
	"*.hpp"
	"*.cpp"
)

message(STATUS "Detected source files: ${MAIN_SRC}")

add_executable (main ${MAIN_SRC})

if(MSVC)

	message(STATUS "Detected MSVC compiler")

	set(OPENCV_INSTALL_LOCATION "NOT SET" CACHE PATH "Path to installed OpenCV folder.")

	message(STATUS "Checking config...")
	message(STATUS "  - OPENCV_INSTALL_LOCATION: ${OPENCV_INSTALL_LOCATION}")
	if(OPENCV_INSTALL_LOCATION STREQUAL "NOT SET")
		message(FATAL_ERROR "Need to set OPENCV_INSTALL_LOCATION!")
	endif()

	# Compiler flags for Windows MSVC
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4") # MSVC warning levels: W3 < W4 < Wall
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2") # O2 is best speed optimization
	# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Od") # Od disables all optimization for debugging
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_USE_MATH_DEFINES") # for M_PI in <cmath

	message(STATUS "CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

	set(OpenCV_DIR "${OPENCV_INSTALL_LOCATION}\\build\\x64\\vc15\\lib")

	add_custom_command(TARGET main POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_if_different "${OPENCV_INSTALL_LOCATION}\\build\\x64\\vc15\\bin\\opencv_world345.dll" $<TARGET_FILE_DIR:main>
	COMMAND ${CMAKE_COMMAND} -E copy_if_different "${OPENCV_INSTALL_LOCATION}\\build\\x64\\vc15\\bin\\opencv_world345.pdb" $<TARGET_FILE_DIR:main>
	COMMAND ${CMAKE_COMMAND} -E copy_if_different "${OPENCV_INSTALL_LOCATION}\\build\\x64\\vc15\\bin\\opencv_world345d.dll" $<TARGET_FILE_DIR:main>
	COMMAND ${CMAKE_COMMAND} -E copy_if_different "${OPENCV_INSTALL_LOCATION}\\build\\x64\\vc15\\bin\\opencv_world345d.pdb" $<TARGET_FILE_DIR:main>
	COMMAND ${CMAKE_COMMAND} -E copy_if_different "${OPENCV_INSTALL_LOCATION}\\build\\bin\\opencv_ffmpeg345.dll" $<TARGET_FILE_DIR:main>
)


elseif(CMAKE_COMPILER_IS_GNUCXX)

	message(STATUS "Detected g++ compiler: ${CMAKE_CXX_COMPILER}")

	set(CMAKE_CXX_STANDARD 17)
	set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

	# Compiler flags for GCC
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall") 		# enable warnings
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")			# best optimization for runtime speed
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")	# compile with c++17 support
	# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")   		# debug: no optimization

	message(STATUS "CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

else()

	message(SEND_ERROR "Unsupported compiler! ${CMAKE_CXX_COMPILER}")

endif()


find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")



target_link_libraries(main ${OpenCV_LIBS})


