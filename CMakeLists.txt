cmake_minimum_required (VERSION 3.8)

project("PuRe")

if (APPLE)

elseif (UNIX)
    # required for building on manylinyx images
    set (CMAKE_POSITION_INDEPENDENT_CODE TRUE)
elseif (WIN32)

endif()

# include static cpp library target
add_subdirectory("cpp")

# include cython wrapping
add_subdirectory("src/pure_detector")