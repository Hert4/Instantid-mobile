cmake_minimum_required(VERSION 3.18)
project(instantid_mobile VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Options ───────────────────────────────────────────────────────────────────
option(USE_ONNXSTREAM   "Use OnnxStream backend (low RAM)"  ON)
option(USE_ORT          "Use ONNX Runtime backend"          OFF)
option(USE_NNAPI        "Enable Android NNAPI"              OFF)
option(BUILD_ANDROID    "Build as Android JNI library"      OFF)
option(BUILD_IOS        "Build as iOS framework"            OFF)
option(BUILD_TESTS      "Build test executable"             ON)

# ── Paths ─────────────────────────────────────────────────────────────────────
set(THIRD_PARTY_DIR "${CMAKE_SOURCE_DIR}/../../third_party")
set(ONNXSTREAM_DIR  "${THIRD_PARTY_DIR}/OnnxStream/src")

# ─────────────────────────────────────────────────────────────────────────────
# Source files
# ─────────────────────────────────────────────────────────────────────────────
set(INSTANTID_SOURCES
    face_processor.cpp
    instantid_pipeline.cpp
)

set(INSTANTID_HEADERS
    face_processor.h
    instantid_pipeline.h
)

# ─────────────────────────────────────────────────────────────────────────────
# OnnxStream backend
# ─────────────────────────────────────────────────────────────────────────────
if(USE_ONNXSTREAM)
    message(STATUS "Backend: OnnxStream")

    # OnnxStream builds as a static lib
    add_subdirectory(${ONNXSTREAM_DIR}
                     ${CMAKE_BINARY_DIR}/onnxstream
                     EXCLUDE_FROM_ALL)

    set(BACKEND_LIBS onnxstream)
    set(BACKEND_INCLUDES ${ONNXSTREAM_DIR})
    add_compile_definitions(USE_ONNXSTREAM)

elseif(USE_ORT)
    message(STATUS "Backend: ONNX Runtime")

    # Find ORT — user must set ONNXRUNTIME_ROOT
    if(NOT ONNXRUNTIME_ROOT)
        message(FATAL_ERROR
            "Set ONNXRUNTIME_ROOT to your ONNX Runtime installation.\n"
            "  cmake -DONNXRUNTIME_ROOT=/path/to/onnxruntime ...")
    endif()

    set(BACKEND_LIBS ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)
    set(BACKEND_INCLUDES ${ONNXRUNTIME_ROOT}/include)

else()
    message(FATAL_ERROR "Select a backend: -DUSE_ONNXSTREAM=ON or -DUSE_ORT=ON")
endif()

# ─────────────────────────────────────────────────────────────────────────────
# Android JNI library
# ─────────────────────────────────────────────────────────────────────────────
if(BUILD_ANDROID)
    message(STATUS "Target: Android JNI")

    add_library(instantid_jni SHARED
        ${INSTANTID_SOURCES}
        android_jni.cpp
    )

    target_include_directories(instantid_jni PRIVATE
        ${BACKEND_INCLUDES}
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_link_libraries(instantid_jni PRIVATE
        ${BACKEND_LIBS}
        log          # Android log
        android      # Android native
    )

    if(USE_NNAPI)
        target_compile_definitions(instantid_jni PRIVATE USE_NNAPI)
        # NNAPI is linked via onnxruntime or OnnxStream's NNAPI EP
    endif()

    # Strip debug symbols for release APK
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_custom_command(TARGET instantid_jni POST_BUILD
            COMMAND ${CMAKE_STRIP} --strip-debug
                    $<TARGET_FILE:instantid_jni>
        )
    endif()

# ─────────────────────────────────────────────────────────────────────────────
# iOS framework
# ─────────────────────────────────────────────────────────────────────────────
elseif(BUILD_IOS)
    message(STATUS "Target: iOS framework")

    add_library(InstantIDMobile STATIC
        ${INSTANTID_SOURCES}
    )

    target_include_directories(InstantIDMobile PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${BACKEND_INCLUDES}
    )

    target_link_libraries(InstantIDMobile PRIVATE
        ${BACKEND_LIBS}
    )

    set_target_properties(InstantIDMobile PROPERTIES
        FRAMEWORK TRUE
        FRAMEWORK_VERSION A
        MACOSX_FRAMEWORK_IDENTIFIER "com.instantid.mobile"
        PUBLIC_HEADER "${INSTANTID_HEADERS}"
        XCODE_ATTRIBUTE_ENABLE_BITCODE NO
    )

# ─────────────────────────────────────────────────────────────────────────────
# Desktop test build
# ─────────────────────────────────────────────────────────────────────────────
else()
    message(STATUS "Target: Desktop (test)")

    if(BUILD_TESTS)
        add_executable(instantid_test
            ${INSTANTID_SOURCES}
            test_main.cpp
        )

        target_include_directories(instantid_test PRIVATE
            ${BACKEND_INCLUDES}
            ${CMAKE_CURRENT_SOURCE_DIR}
        )

        target_link_libraries(instantid_test PRIVATE
            ${BACKEND_LIBS}
        )

        # Compiler flags for performance
        target_compile_options(instantid_test PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -march=native>
            $<$<CXX_COMPILER_ID:MSVC>:/O2>
        )
    endif()
endif()

# ─────────────────────────────────────────────────────────────────────────────
# Build instructions summary
# ─────────────────────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "── InstantID Mobile Build Config ──")
message(STATUS "  Backend:  ${USE_ONNXSTREAM AND 'OnnxStream' OR 'ORT'}")
message(STATUS "  NNAPI:    ${USE_NNAPI}")
message(STATUS "")
message(STATUS "Android build example:")
message(STATUS "  cmake -DBUILD_ANDROID=ON -DUSE_ONNXSTREAM=ON \\")
message(STATUS "        -DCMAKE_TOOLCHAIN_FILE=\$ANDROID_NDK/build/cmake/android.toolchain.cmake \\")
message(STATUS "        -DANDROID_ABI=arm64-v8a \\")
message(STATUS "        -DANDROID_NATIVE_API_LEVEL=26 \\")
message(STATUS "        -DCMAKE_BUILD_TYPE=Release .")
message(STATUS "")
message(STATUS "iOS build example (run on macOS):")
message(STATUS "  cmake -DBUILD_IOS=ON -DUSE_ONNXSTREAM=ON \\")
message(STATUS "        -DCMAKE_SYSTEM_NAME=iOS \\")
message(STATUS "        -DCMAKE_OSX_ARCHITECTURES=arm64 .")
