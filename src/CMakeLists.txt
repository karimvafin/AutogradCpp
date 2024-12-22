cmake_minimum_required(VERSION 3.20)

file(GLOB_RECURSE SOURCES *.cpp)

add_library(${PROJECT_NAME} STATIC ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)