cmake_minimum_required(VERSION 3.16)

project(yolo_face VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 REQUIRED COMPONENTS Core)
qt_standard_project_setup()

qt_add_executable(yolo_face main.cpp)

target_link_libraries(yolo_face PRIVATE Qt6::Core)
target_sources(yolo_face PRIVATE ../utils/image_file_libpng.cpp)
target_include_directories(yolo_face PRIVATE ${PROJECT_SOURCE_DIR}/include ${PNG_INCLUDE_DIRS} ../utils/)
target_link_libraries(yolo_face PRIVATE onnxruntime ${PNG_LIBRARIES} ${WIL_LIB})
if(PNG_LIBDIR)
  target_link_directories(yolo_face PRIVATE ${PNG_LIBDIR})
endif()
