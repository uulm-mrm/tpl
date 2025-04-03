cmake_minimum_required(VERSION 3.25)
project(tplcpp LANGUAGES CXX CUDA)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(OpenGL_GL_PREFERENCE GLVND)

# supress warnings from eigen
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -diag-suppress 20012")

find_package(Eigen3 QUIET)
find_package(pybind11 QUIET)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

include(FetchContent)

if(NOT pybind11_FOUND)
    # if we did not find pybind11 as systems include
    # download it from the inter-webs ...
    FetchContent_Declare(
        pybind
        GIT_REPOSITORY "https://github.com/pybind/pybind11"
        GIT_TAG "v2.12"
    )
    message(STATUS "Loading pybind ...")
    FetchContent_MakeAvailable(pybind)
endif()

if(NOT Eigen3_FOUND)
    # same for eigen ...
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY "https://gitlab.com/libeigen/eigen"
        GIT_TAG "21ae2afd4edaa1b69782c67a54182d34efe43f9c"
    )
    message(STATUS "Loading eigen ...")
    FetchContent_Populate(eigen)
endif()

# configuring extension module

pybind11_add_module(${PROJECT_NAME} MODULE
    src/utils.cpp
    src/poly_interp.cu
    src/poly_sampling.cpp
    src/idm_sampling.cpp
    src/dyn_prog/common.cu
    src/dyn_prog/env.cu
    src/dyn_prog/lat_lon_planner.cu
    src/dyn_prog/lon_planner.cu
    src/dyn_prog/poly_lat_planner.cu
    src/dyn_prog/poly_planner.cu
    src/dyn_prog/occupancy_renderer.cu
    src/dyn_prog/frame_buffer.cpp
    src/python/bindings.cu
    src/python/bindings_dp_common.cu
    src/python/bindings_dp_env.cu
    src/python/bindings_dp_lat_lon.cu
    src/python/bindings_dp_lon.cu
    src/python/bindings_dp_poly.cu
    src/python/bindings_dp_poly_lat.cu
    src/python/bindings_poly_sampling.cu
    src/python/bindings_idm_sampling.cpp
    src/python/binding_helpers.cpp)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES native)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${PROJECT_NAME} PUBLIC
    gomp
    stdc++fs
    pybind11::module
    ${CUDA_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${OPENGL_egl_LIBRARY}
    ${GLEW_LIBRARIES})

target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC 
    ${eigen_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
	${CUDA_INCLUDE_DIRS}
    include)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)

target_compile_options(${PROJECT_NAME} PUBLIC
                        -O3
                        -ffast-math
                        -fopenmp
                        -Wall
                        -Wextra
                        -Wunreachable-code
                         # caused by pybind, cannot do anything about that
                        -Wno-unused-parameter
                        -Wno-array-bounds
                        -Wno-stringop-overread)
