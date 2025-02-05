cmake_minimum_required(VERSION 3.16)

project(hello_onnxruntime VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 REQUIRED COMPONENTS Widgets)
find_package(PNG)
find_package(JPEG)
qt_standard_project_setup()

set(ONNXRUNTIME_ROOTDIR "./onnxruntime-osx-arm64-1.20.1")
include_directories("${ONNXRUNTIME_ROOTDIR}/include")

link_directories("${ONNXRUNTIME_ROOTDIR}/lib")

add_subdirectory(fns_candy_style_transfer)
add_subdirectory(model_explorer)