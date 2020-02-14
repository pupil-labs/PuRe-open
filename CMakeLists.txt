cmake_minimum_required (VERSION 3.8)

project("PuRe")

if (APPLE)
	# We target 10.12 on travis with LLVM 4.0, which does not offer c++17 yet, so we
	# need to use c++1z.
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1z")
endif ()

# include static cpp library target
add_subdirectory("cpp")

# include cython wrapping
add_subdirectory("src/pure_detector")
