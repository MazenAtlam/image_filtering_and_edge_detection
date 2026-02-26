cmake_minimum_required(VERSION 3.15)
project(Backend)

# 1. Require pybind11 using local subdirectory (circumvents Windows FetchContent MAX_PATH issues)
add_subdirectory(pybind11)

find_package(OpenCV REQUIRED)

# 2. Build your C++ code into a Python module
pybind11_add_module(backend main_binding.cpp)
target_include_directories(backend PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(backend PRIVATE ${OpenCV_LIBS})