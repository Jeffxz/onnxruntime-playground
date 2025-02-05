cmake_minimum_required(VERSION 3.16)

project(fns_candy_style_transfer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 REQUIRED COMPONENTS Core)
qt_standard_project_setup()

qt_add_executable(fns_candy_style_transfer main.cpp)

target_link_libraries(fns_candy_style_transfer PRIVATE Qt6::Core)
target_sources(fns_candy_style_transfer PRIVATE image_file_libpng.cpp)
target_include_directories(fns_candy_style_transfer PRIVATE ${PROJECT_SOURCE_DIR}/include ${PNG_INCLUDE_DIRS})
target_link_libraries(fns_candy_style_transfer PRIVATE onnxruntime ${PNG_LIBRARIES} ${WIL_LIB})
if(PNG_LIBDIR)
  target_link_directories(fns_candy_style_transfer PRIVATE ${PNG_LIBDIR})
endif()
