cmake_minimum_required(VERSION 3.16)

project(voxtral_cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(VOXTRAL_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)
option(VOXTRAL_NATIVE_OPT "Enable native CPU tuning flags" ON)
option(VOXTRAL_AUTO_DETECT_CPU "Auto-detect CPU features and set ggml flags" ON)
option(VOXTRAL_AUTO_DETECT_BLAS "Auto-enable ggml BLAS if found" ON)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Threads REQUIRED)

set(GGML_BUILD_TESTS OFF CACHE BOOL "Disable ggml tests" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "Disable ggml examples" FORCE)

if(VOXTRAL_AUTO_DETECT_CPU AND NOT CMAKE_CROSSCOMPILING)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i[3-6]86")
        include(CheckCXXSourceRuns)

        function(voxtral_check_cpu_supports feature outvar)
            set(_src "
                #if !(defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
                int main() { return 1; }
                #endif
                #if defined(__GNUC__) || defined(__clang__)
                int main() { return __builtin_cpu_supports(\"${feature}\") ? 0 : 1; }
                #else
                int main() { return 1; }
                #endif
            ")
            check_cxx_source_runs("${_src}" ${outvar})
        endfunction()

        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
            voxtral_check_cpu_supports("avx"    VOXTRAL_CPU_HAS_AVX)
            voxtral_check_cpu_supports("avx2"   VOXTRAL_CPU_HAS_AVX2)
            voxtral_check_cpu_supports("fma"    VOXTRAL_CPU_HAS_FMA)
            voxtral_check_cpu_supports("f16c"   VOXTRAL_CPU_HAS_F16C)
            voxtral_check_cpu_supports("bmi2"   VOXTRAL_CPU_HAS_BMI2)
            voxtral_check_cpu_supports("avx512f" VOXTRAL_CPU_HAS_AVX512)

            set(VOXTRAL_CPU_FEATURES_DETECTED ON)
        endif()
    endif()
endif()

if(VOXTRAL_CPU_FEATURES_DETECTED)
    # Use explicit ISA flags instead of -march=native when auto-detecting.
    set(GGML_NATIVE OFF CACHE BOOL "ggml: optimize the build for the current system" FORCE)
    set(GGML_AVX    ${VOXTRAL_CPU_HAS_AVX}    CACHE BOOL "ggml: enable AVX" FORCE)
    set(GGML_AVX2   ${VOXTRAL_CPU_HAS_AVX2}   CACHE BOOL "ggml: enable AVX2" FORCE)
    set(GGML_FMA    ${VOXTRAL_CPU_HAS_FMA}    CACHE BOOL "ggml: enable FMA" FORCE)
    set(GGML_F16C   ${VOXTRAL_CPU_HAS_F16C}   CACHE BOOL "ggml: enable F16C" FORCE)
    set(GGML_BMI2   ${VOXTRAL_CPU_HAS_BMI2}   CACHE BOOL "ggml: enable BMI2" FORCE)
    set(GGML_AVX512 ${VOXTRAL_CPU_HAS_AVX512} CACHE BOOL "ggml: enable AVX512F" FORCE)
endif()

if(VOXTRAL_AUTO_DETECT_BLAS)
    if(NOT DEFINED CACHE{GGML_BLAS})
        find_package(BLAS QUIET)
        if(BLAS_FOUND)
            set(GGML_BLAS ON CACHE BOOL "ggml: use BLAS" FORCE)
        endif()
    endif()
endif()

# CUDA support
option(GGML_CUDA "ggml: use CUDA" OFF)

add_subdirectory(ggml)

add_library(voxtral_lib
    src/voxtral.cpp)

target_include_directories(voxtral_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(voxtral_lib
    PUBLIC
        ggml
        Threads::Threads
        m
        dl)

if(VOXTRAL_WARNINGS_AS_ERRORS)
    if (MSVC)
        target_compile_options(voxtral_lib PRIVATE /W4 /WX)
    else()
        target_compile_options(voxtral_lib PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
endif()

if(VOXTRAL_NATIVE_OPT AND NOT MSVC)
    target_compile_options(voxtral_lib PRIVATE -march=native -mtune=native)
endif()

add_executable(voxtral src/main.cpp)
target_link_libraries(voxtral PRIVATE voxtral_lib)

add_executable(voxtral-quantize src/voxtral-quantize.cpp)
target_link_libraries(voxtral-quantize PRIVATE ggml Threads::Threads m dl)

# Real-time transcription client (requires PulseAudio)
find_package(PkgConfig)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(PULSEAUDIO libpulse-simple)
    if(PULSEAUDIO_FOUND)
        add_executable(voxtral-realtime src/voxtral-realtime.cpp)
        target_link_libraries(voxtral-realtime PRIVATE voxtral_lib ${PULSEAUDIO_LIBRARIES})
        target_include_directories(voxtral-realtime PRIVATE ${PULSEAUDIO_INCLUDE_DIRS})

        # Optimized version with advanced features
        add_executable(voxtral-realtime-opt src/voxtral-realtime-optimized.cpp)
        target_link_libraries(voxtral-realtime-opt PRIVATE voxtral_lib ${PULSEAUDIO_LIBRARIES})
        target_include_directories(voxtral-realtime-opt PRIVATE ${PULSEAUDIO_INCLUDE_DIRS})

        message(STATUS "Building voxtral-realtime with PulseAudio support")
        message(STATUS "Building voxtral-realtime-opt (optimized version)")
    else()
        message(STATUS "PulseAudio not found, skipping voxtral-realtime")
    endif()
endif()

if(VOXTRAL_WARNINGS_AS_ERRORS)
    if (MSVC)
        target_compile_options(voxtral-quantize PRIVATE /W4 /WX)
    else()
        target_compile_options(voxtral-quantize PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
endif()

if(VOXTRAL_NATIVE_OPT AND NOT MSVC)
    target_compile_options(voxtral-quantize PRIVATE -march=native -mtune=native)
endif()

find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_custom_target(voxtral_convert
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/convert_voxtral_to_gguf.py --help
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Print converter usage"
        VERBATIM)
    add_custom_target(voxtral_quantize_all
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/quantize_all.py --help
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Print quantize_all usage"
        VERBATIM)
endif()

enable_testing()
if(Python3_Interpreter_FOUND)
    add_test(
        NAME voxtral_python_reference
        COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR} ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_voxtral_reference.py)
endif()

#add_subdirectory(examples/stream)
