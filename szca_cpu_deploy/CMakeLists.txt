cmake_minimum_required(VERSION 3.20)
project(szca_cpu_deploy VERSION 5.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force CPU-only build (no CUDA)
add_definitions(-DUSE_CPU_ONLY=1)

# C4: This deployment's main.cpp/http_server.cpp depend on the szca::Engine and
# its STT/LLM/TTS/etc. components, which physically live in the sibling
# szca_onnx_engine project. Only src/main.cpp and src/http_server.cpp actually
# exist here; the previous SOURCES/TEST_SOURCES lists referenced 8 and 10 files
# respectively that do not exist in this repo. We reference the real engine
# sources from the sibling directory and only the files that exist locally.
set(ENGINE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../szca_onnx_engine")

if(NOT EXISTS "${ENGINE_DIR}/include/engine.h")
    message(FATAL_ERROR
        "Expected sibling engine project at ${ENGINE_DIR} (provides engine.h "
        "and STT/LLM/TTS sources). Clone szca_onnx_engine next to this repo.")
endif()

# Engine library sources shared from the sibling project (stub mode when
# ONNX Runtime is not present).
set(ENGINE_SOURCES
    ${ENGINE_DIR}/src/stt.cpp
    ${ENGINE_DIR}/src/llm.cpp
    ${ENGINE_DIR}/src/tts.cpp
    ${ENGINE_DIR}/src/resampler.cpp
    ${ENGINE_DIR}/src/ipc.cpp
    ${ENGINE_DIR}/src/session.cpp
    ${ENGINE_DIR}/src/engine.cpp
    ${ENGINE_DIR}/src/ort_utils.cpp
)

# Local sources that actually exist in this repo.
set(LOCAL_SOURCES
    src/http_server.cpp
)

find_package(Threads REQUIRED)

# Main executable
add_executable(szca_cpu
    src/main.cpp
    ${LOCAL_SOURCES}
    ${ENGINE_SOURCES}
)
target_include_directories(szca_cpu PRIVATE include ${ENGINE_DIR}/include)
target_link_libraries(szca_cpu PRIVATE Threads::Threads)

# Test executable — only tests/test_e2e.cpp exists in this repo.
# It uses libcurl for its HTTP client helpers.
find_package(CURL)

add_executable(szca_cpu_tests
    tests/test_e2e.cpp
    ${LOCAL_SOURCES}
    ${ENGINE_SOURCES}
)
target_include_directories(szca_cpu_tests PRIVATE include ${ENGINE_DIR}/include)
target_link_libraries(szca_cpu_tests PRIVATE Threads::Threads)

if(CURL_FOUND)
    target_link_libraries(szca_cpu_tests PRIVATE CURL::libcurl)
else()
    # macOS ships libcurl in the SDK without a CMake config; fall back to the
    # plain link flag so the e2e test still builds.
    target_link_libraries(szca_cpu_tests PRIVATE curl)
endif()

# CTest integration
enable_testing()
add_test(NAME test_e2e COMMAND szca_cpu_tests)

# Custom test target
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS szca_cpu_tests
)
