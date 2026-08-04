cmake_minimum_required(VERSION 3.20)
project(szca_onnx_engine VERSION 5.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags.
# -march=native is not universally supported (e.g. some Apple-clang/arm64
# toolchains reject it), so only add it when the compiler accepts it.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()
# Sanitizers only in Debug builds (default build type is Release).
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address,undefined")

# ONNX Runtime.
# Search common prefixes plus the Homebrew prefix (via `brew --prefix`) so the
# real ORT path is compiled when the package is installed. The Homebrew bottle
# nests the C API header under include/onnxruntime, so that dir is searched too.
execute_process(COMMAND brew --prefix
                OUTPUT_VARIABLE HOMEBREW_PREFIX
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)
find_library(ONNXRUNTIME_LIB onnxruntime
    PATHS /usr/lib /usr/local/lib /opt/homebrew/lib /opt/onnxruntime/lib "${HOMEBREW_PREFIX}/lib")
find_path(ONNXRUNTIME_INCLUDE onnxruntime_c_api.h
    PATHS /usr/include /usr/local/include /opt/homebrew/include /opt/onnxruntime/include
          "${HOMEBREW_PREFIX}/include"
    PATH_SUFFIXES onnxruntime)

# SoXR (audio resampler)
find_library(SOXR_LIB soxr
    PATHS /usr/lib /usr/local/lib /opt/homebrew/lib "${HOMEBREW_PREFIX}/lib")

# Library source files (shared by the main binary and the tests).
set(LIB_SOURCES
    src/stt.cpp
    src/llm.cpp
    src/tts.cpp
    src/resampler.cpp
    src/ipc.cpp
    src/session.cpp
    src/engine.cpp
    src/ort_utils.cpp
)

find_package(Threads REQUIRED)

# Helper that applies the common dependencies to a target.
function(szca_configure_target target)
    target_include_directories(${target} PRIVATE include)
    target_link_libraries(${target} PRIVATE Threads::Threads)
    # Only enable the ORT path when BOTH the library and its C API header are
    # present; otherwise build in stub mode. (A lib without a matching header
    # would otherwise break configuration / compile against a missing API.)
    if(ONNXRUNTIME_LIB AND ONNXRUNTIME_INCLUDE)
        target_link_libraries(${target} PRIVATE ${ONNXRUNTIME_LIB})
        target_include_directories(${target} PRIVATE ${ONNXRUNTIME_INCLUDE})
        target_compile_definitions(${target} PRIVATE HAS_ONNXRUNTIME=1)
    endif()
    if(SOXR_LIB)
        target_link_libraries(${target} PRIVATE ${SOXR_LIB})
        target_compile_definitions(${target} PRIVATE HAS_SOXR=1)
    endif()
endfunction()

# Main executable
add_executable(szca_onnx_engine src/main.cpp ${LIB_SOURCES})
szca_configure_target(szca_onnx_engine)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
# H10: each tests/test_*.cpp defines its own int main(), so they cannot be
# linked into a single executable (duplicate symbol). Build one binary per
# test file and register each with CTest.
enable_testing()

set(TEST_FILES
    test_stt
    test_llm
    test_tts
    test_resampler
    test_ipc
    test_session
    test_engine
)

foreach(test_name ${TEST_FILES})
    add_executable(${test_name} tests/${test_name}.cpp ${LIB_SOURCES})
    szca_configure_target(${test_name})
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# Convenience target that runs every test binary.
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS ${TEST_FILES}
)
