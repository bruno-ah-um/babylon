cmake_minimum_required(VERSION 3.20)
project(MLIRGenerator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Prefer shared libraries to avoid PIE issues
set(BUILD_SHARED_LIBS ON)

# Find LLVM and MLIR
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Include directories
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# Add compile definitions
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

# Create the shared library
add_library(mlir_tosa_c SHARED mlir_tosa_c_api.cpp)

# Set library properties
set_target_properties(mlir_tosa_c PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    VERSION 1.0.0
    SOVERSION 1
    PUBLIC_HEADER mlir_tosa_c_api.h
)

# Link against MLIR and LLVM libraries
target_link_libraries(mlir_tosa_c
    # Core MLIR libraries (order matters for static linking)
    MLIRIR
    MLIRSupport
    MLIRParser

    # Bytecode libraries
    MLIRBytecodeWriter
    MLIRBytecodeReader

    # Dialect libraries
    MLIRFuncDialect
    MLIRTosaDialect

    # Additional MLIR libraries
    MLIRPass
    MLIRTransforms
    MLIRAsmParser

    # LLVM support libraries
    LLVMCore
    LLVMSupport
    LLVMBitWriter
    LLVMAsmParser
)

# Compilation flags
target_compile_options(mlir_tosa_c PRIVATE
    -fPIC
    -fno-rtti
    -fexceptions
)

# Installation
install(TARGETS mlir_tosa_c
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include)
