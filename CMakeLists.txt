cmake_minimum_required(VERSION 3.16)
project(zonos2_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math")

# Use system ggml
find_package(PkgConfig REQUIRED)
pkg_check_modules(GGML REQUIRED ggml)

add_library(zonos2 SHARED
    src/zonos2_model.cpp
    src/zonos2_loader.cpp
    src/zonos2_generate.cpp
)
target_include_directories(zonos2 PUBLIC src ${GGML_INCLUDE_DIRS})
target_link_libraries(zonos2 PUBLIC ${GGML_LIBRARIES})

# CLI tool
add_executable(zonos2_cli src/main.cpp)
target_link_libraries(zonos2_cli zonos2)

# Test tool
add_executable(test_forward src/test_forward.cpp)
target_link_libraries(test_forward zonos2)

# Verification tool
add_executable(test_verify src/test_verify.cpp)
target_link_libraries(test_verify zonos2)

# Quick embedding test
add_executable(quick_embed src/quick_embed.cpp)
target_link_libraries(quick_embed zonos2)
