cmake_minimum_required(VERSION 3.16)

project(face_analyzer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 REQUIRED COMPONENTS Core)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
qt_standard_project_setup()

qt_add_executable(face_analyzer main.cpp)

target_link_libraries(face_analyzer PRIVATE Qt6::Core)
target_sources(face_analyzer PRIVATE
  ./inference_manager.cpp
  ./face_detector.cpp
  ../utils/image_file_opencv.cpp
  ../utils/utils.cpp
  )
target_include_directories(face_analyzer PRIVATE ${PROJECT_SOURCE_DIR}/include ${PNG_INCLUDE_DIRS} ../utils/)
target_link_libraries(face_analyzer PRIVATE onnxruntime ${PNG_LIBRARIES} ${OpenCV_LIBS})
if(PNG_LIBDIR)
  target_link_directories(face_analyzer PRIVATE ${PNG_LIBDIR})
endif()
