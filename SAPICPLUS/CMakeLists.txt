cmake_minimum_required(VERSION 3.16)
project(SapiRecorderBalanced VERSION 8.0.0 LANGUAGES CXX)

# C++17 standard required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -O2)
elseif(MSVC)
    add_compile_options(/W4 /O2)
endif()

# Find required packages
find_package(SDL2 REQUIRED)
find_package(SDL2_ttf REQUIRED)
find_package(Threads REQUIRED)

# Main executable
add_executable(SapiRecorderBalanced
    SapiRecorderBalanced.cpp
)

# Include directories
target_include_directories(SapiRecorderBalanced PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${SDL2_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(SapiRecorderBalanced PRIVATE
    ${SDL2_LIBRARIES}
    SDL2_ttf::SDL2_ttf
    Threads::Threads
)

# On some systems, SDL2_ttf may need different linking
if(NOT TARGET SDL2_ttf::SDL2_ttf)
    find_library(SDL2_TTF_LIBRARY NAMES SDL2_ttf)
    target_link_libraries(SapiRecorderBalanced PRIVATE ${SDL2_TTF_LIBRARY})
endif()

# Platform-specific settings
if(WIN32)
    # Windows-specific
    target_compile_definitions(SapiRecorderBalanced PRIVATE _USE_MATH_DEFINES)
elseif(APPLE)
    # macOS-specific
    find_library(COCOA_LIBRARY Cocoa)
    target_link_libraries(SapiRecorderBalanced PRIVATE ${COCOA_LIBRARY})
endif()

# Installation
install(TARGETS SapiRecorderBalanced
    RUNTIME DESTINATION bin
)

# CPack for packaging
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
