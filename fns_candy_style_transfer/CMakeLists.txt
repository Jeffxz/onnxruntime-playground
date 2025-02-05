cmake_minimum_required(VERSION 3.16)

project(fns_candy_style_transfer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 REQUIRED COMPONENTS Core)
qt_standard_project_setup()

SET(IMAGE_PNG_SRC ../utils/image_file_libpng.cpp)
SET(UTILS_HEADER ../utils/)

qt_add_executable(fns_candy_style_transfer main.cpp ${IMAGE_PNG_SRC})

target_link_libraries(fns_candy_style_transfer PRIVATE Qt6::Core)
target_include_directories(fns_candy_style_transfer PRIVATE ${PROJECT_SOURCE_DIR}/include ${PNG_INCLUDE_DIRS} ${UTILS_HEADER})
target_link_libraries(fns_candy_style_transfer PRIVATE onnxruntime ${PNG_LIBRARIES} ${WIL_LIB})
if(PNG_LIBDIR)
  target_link_directories(fns_candy_style_transfer PRIVATE ${PNG_LIBDIR})
endif()
