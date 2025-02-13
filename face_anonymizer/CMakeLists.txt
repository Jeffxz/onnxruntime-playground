cmake_minimum_required(VERSION 3.16)

project(face_anonymizer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 REQUIRED COMPONENTS Core)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
qt_standard_project_setup()

qt_add_executable(face_anonymizer main.cpp)

target_link_libraries(face_anonymizer PRIVATE Qt6::Core)
target_sources(face_anonymizer PRIVATE
  ./inference_manager.cpp
  ./face_analyzer.cpp
  ./face_detector.cpp
  ./face_helper.cpp
  ../utils/image_file_opencv.cpp
  ../utils/utils.cpp
)
target_include_directories(face_anonymizer PRIVATE ${PROJECT_SOURCE_DIR}/include ${PNG_INCLUDE_DIRS} ../utils/)
target_link_libraries(face_anonymizer PRIVATE onnxruntime ${PNG_LIBRARIES} ${OpenCV_LIBS})
if(PNG_LIBDIR)
  target_link_directories(face_anonymizer PRIVATE ${PNG_LIBDIR})
endif()
