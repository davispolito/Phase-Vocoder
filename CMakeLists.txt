cmake_minimum_required(VERSION 3.1)

project(cuda-fft)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Enable C++11 for host code
set(CMAKE_CXX_STANDARD 11)

list(APPEND CUDA_NVCC_FLAGS_DEBUG -G -g)
list(APPEND CUDA_NVCC_FLAGS_RELWITHDEBUGINFO -lineinfo)

# Crucial magic for CUDA linking
find_package(Threads REQUIRED)
find_package(CUDA 10 REQUIRED)


set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
endif()

include_directories(.)
include_directories(karnel)
cuda_add_library(kernel
	"karnel/cufft_.cu"
	"karnel/cufft_.h"
	"karnel/common.h"
	"karnel/common.cu"
    "karnel/kernel.h"
    "karnel/kernel.cu"
	OPTIONS -arch=sm_53
	)

cuda_add_executable(${CMAKE_PROJECT_NAME}
    	"src/main.cpp"
    "src/testing_helpers.hpp"
        "src/phaseVocoder.cpp"
        "src/phaseVocoder.h"
        "src/Sine.h"
	"src/RtAudio.cpp"
	"src/RtAudio.h"
    "src/RtError.h"
    "src/io.h"
    "src/io.cpp"
    "src/AudioFile.h"
    )
SET(GCC_COVERAGE_LINK_FLAGS "-lasound")
find_package(ALSA REQUIRED)
include_directories(${ALSA_INCLUDE_DIRS})
if(WIN32)
    # Set up include and lib paths
    set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE FILEPATH "Host side compiler used by NVCC" FORCE)
endif(WIN32)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
target_link_libraries(${CMAKE_PROJECT_NAME}
    kernel
    ${CORELIBS}
    "/usr/local/cuda/lib64/libcufft.so"
    "/usr/local/cuda/lib64/libcudadevrt.a"
    ${ALSA_LIBRARIES}
    )

