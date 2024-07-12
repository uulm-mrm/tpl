cmake_minimum_required(VERSION 3.10)
project(genopt@CODE_HASH)

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)

add_library(${PROJECT_NAME} SHARED optim.c)

target_include_directories(${PROJECT_NAME} PRIVATE
        ${PYTHON_INCLUDE_DIRS}
        "@NUMPY_INCLUDES"
    )

target_link_libraries(${PROJECT_NAME} ${PYTHON_LIBRARIES})

target_compile_options(${PROJECT_NAME} PRIVATE
                        -UNDEBUG
                        -O3
                        -ffast-math
                        -march=native)

set_target_properties(${PROJECT_NAME} PROPERTIES
    PREFIX ""
    OUTPUT_NAME "${PROJECT_NAME}"
    LINKER_LANGUAGE C)
