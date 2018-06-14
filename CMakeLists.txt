cmake_minimum_required(VERSION 3.10)

project(face_recognition LANGUAGES CXX)

find_package(OpenCV REQUIRED)

target_include_directories(face_recognition PUBLIC ${OpenCV_INCLUDE_DIRS})

target_link_libraries(face_recognition PUBLIC ${OpenCV_LIBS})
