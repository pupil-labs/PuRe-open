cmake_minimum_required (VERSION 3.8)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)


set(OPENCV_INSTALL_LOCATION "NOT SET" CACHE PATH "Path to installed OpenCV folder.")

message(STATUS "Checking config...")
message(STATUS "  - OPENCV_INSTALL_LOCATION: ${OPENCV_INSTALL_LOCATION}")
if(OPENCV_INSTALL_LOCATION STREQUAL "NOT SET")
	message(FATAL_ERROR "Need to set OPENCV_INSTALL_LOCATION!")
endif()

if(MSVC)

	message(STATUS "Detected MSVC compiler")

	# Compiler flags for Windows MSVC
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4") # MSVC warning levels: W3 < W4 < Wall
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2") # O2 is best speed optimization
	# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Od") # Od disables all optimization for debugging

	message(STATUS "CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

	set(OpenCV_DIR "${OPENCV_INSTALL_LOCATION}\\build\\x64\\vc15\\lib")
	find_package(OpenCV REQUIRED)
	message(STATUS "OpenCV library status:")
	message(STATUS "    config: ${OpenCV_DIR}")
	message(STATUS "    version: ${OpenCV_VERSION}")
	message(STATUS "    libraries: ${OpenCV_LIBS}")
	message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")


else()

	message(SEND_ERROR "Compiler other than MSVC detected. Not yet supported!")

endif()


file(GLOB MAIN_SRC
	"*.hpp"
	"*.cpp"
)

message(STATUS "Detected source files: ${MAIN_SRC}")

add_executable (main ${MAIN_SRC})
target_link_libraries(main ${OpenCV_LIBS})

add_custom_command(TARGET main POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_if_different "${OPENCV_INSTALL_LOCATION}\\build\\x64\\vc15\\bin\\opencv_world411.dll" $<TARGET_FILE_DIR:main>
	COMMAND ${CMAKE_COMMAND} -E copy_if_different "${OPENCV_INSTALL_LOCATION}\\build\\x64\\vc15\\bin\\opencv_world411.pdb" $<TARGET_FILE_DIR:main>
	COMMAND ${CMAKE_COMMAND} -E copy_if_different "${OPENCV_INSTALL_LOCATION}\\build\\x64\\vc15\\bin\\opencv_world411d.dll" $<TARGET_FILE_DIR:main>
	COMMAND ${CMAKE_COMMAND} -E copy_if_different "${OPENCV_INSTALL_LOCATION}\\build\\x64\\vc15\\bin\\opencv_world411d.pdb" $<TARGET_FILE_DIR:main>
	COMMAND ${CMAKE_COMMAND} -E copy_if_different "${OPENCV_INSTALL_LOCATION}\\build\\bin\\opencv_videoio_ffmpeg411_64.dll" $<TARGET_FILE_DIR:main>
)
