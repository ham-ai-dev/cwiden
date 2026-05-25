cmake_minimum_required(VERSION 3.25)
project(cwiden VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ===========================================================================
# Dependencies
# ===========================================================================

# FTXUI (via vcpkg)
find_package(ftxui CONFIG REQUIRED)

# FFTW3 single-precision (system install)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3F REQUIRED fftw3f)

# HackRF (system install)
find_path(HACKRF_INCLUDE_DIR
    NAMES libhackrf/hackrf.h
    PATHS /usr/include /usr/local/include
)
find_library(HACKRF_LIBRARY
    NAMES hackrf
    PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu
)

if(NOT HACKRF_INCLUDE_DIR OR NOT HACKRF_LIBRARY)
    message(FATAL_ERROR
        "libhackrf not found. Install with:\n"
        "  sudo apt install libhackrf-dev")
endif()

message(STATUS "FFTW3F: ${FFTW3F_LIBRARIES}")
message(STATUS "HackRF include: ${HACKRF_INCLUDE_DIR}")
message(STATUS "HackRF library: ${HACKRF_LIBRARY}")

# ===========================================================================
# cwiden executable
# ===========================================================================
add_executable(cwiden
    src/main.cpp
    src/hackrf_source.cpp
    src/channelizer.cpp
    src/cw_classifier.cpp
    src/signal_tracker.cpp
    src/audio_output.cpp
    src/feature_log.cpp
    src/tui.cpp
)

target_include_directories(cwiden PRIVATE
    src/include
    ${HACKRF_INCLUDE_DIR}
    ${FFTW3F_INCLUDE_DIRS}
)

target_link_libraries(cwiden PRIVATE
    ftxui::screen
    ftxui::dom
    ftxui::component
    ${HACKRF_LIBRARY}
    ${FFTW3F_LIBRARIES}
    pthread
    m
)

# ===========================================================================
# Testing
# ===========================================================================
enable_testing()

add_executable(test_classifier
    test/test_classifier.cpp
    src/cw_classifier.cpp
    src/channelizer.cpp
)

target_include_directories(test_classifier PRIVATE
    src/include
    ${FFTW3F_INCLUDE_DIRS}
)

target_link_libraries(test_classifier PRIVATE
    ${FFTW3F_LIBRARIES}
    pthread
    m
)

add_test(NAME classifier_tests COMMAND test_classifier)
