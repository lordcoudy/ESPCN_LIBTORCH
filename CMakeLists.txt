cmake_minimum_required(VERSION 3.27)
project(ESPCN CXX)

set(CMAKE_CXX_STANDARD 17)

# Where to find CMake modules and OpenCV
set(OpenCV_DIR YOUR_PATH_TO_OpenCV)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
set(Argparse_DIR "argparse-3.0")
set(Torch_DIR YOUR_PATH_TO_LIBTORCH)

find_package(Torch REQUIRED PATHS ${Torch_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ${Argparse_DIR})

add_executable(ESPCN main.cpp)

# add libs you need
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)

# linking
target_link_libraries(ESPCN ${OpenCV_LIBS} "${TORCH_LIBRARIES}")

set_target_properties(ESPCN PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_ARCHITECTURES "75")

if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET ESPCN
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:ESPCN>)
endif (MSVC)