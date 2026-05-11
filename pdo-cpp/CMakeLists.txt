cmake_minimum_required(VERSION 3.20)
project(pdo_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

file(GLOB_RECURSE PDO_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

add_library(pdo STATIC ${PDO_SOURCES})
target_include_directories(pdo
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(pdo_main ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(pdo_main PRIVATE pdo)
