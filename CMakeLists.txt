cmake_minimum_required(VERSION 3.22)
project(ESPCN_LIBTORCH LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(ESPCN_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)

if(MSVC)
  add_compile_options(/W4)
  if(ESPCN_WARNINGS_AS_ERRORS)
    add_compile_options(/WX)
  endif()
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
  if(ESPCN_WARNINGS_AS_ERRORS)
    add_compile_options(-Werror)
  endif()
endif()

find_package(Torch REQUIRED)

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)

add_executable(espcn_libtorch
  src/main.cpp
  src/config.cpp
  src/espcn_model.cpp
  src/icnr.cpp
  src/scheduler.cpp
  src/transforms.cpp
  src/dataset.cpp
  src/train.cpp
  src/tune.cpp
  src/eval.cpp
)

target_include_directories(espcn_libtorch PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

target_link_libraries(espcn_libtorch PRIVATE ${TORCH_LIBRARIES})
target_link_libraries(espcn_libtorch PRIVATE ${OpenCV_LIBS})

target_compile_definitions(espcn_libtorch PRIVATE
  $<$<CONFIG:Release>:NDEBUG>
)

set_property(TARGET espcn_libtorch PROPERTY CXX_STANDARD 17)

if(APPLE)
  set_target_properties(espcn_libtorch PROPERTIES
    BUILD_RPATH "@loader_path"
  )
endif()
