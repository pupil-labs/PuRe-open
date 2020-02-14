cmake_minimum_required (VERSION 3.8)

project("PuRe")

# include static cpp library target
add_subdirectory("cpp")

# include cython wrapping
add_subdirectory("src/pure_detector")