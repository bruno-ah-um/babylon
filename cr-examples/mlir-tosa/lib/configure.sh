#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}MLIR TOSA C API - Configuration Script${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Detect LLVM/MLIR installation
echo -e "${YELLOW}Detecting LLVM/MLIR installation...${NC}"

LLVM_PATHS=(
    "/usr/lib/llvm-19"
    "/usr/lib/llvm-18"
    "/usr/lib/llvm-17"
    "/usr/local/lib/llvm"
    "/opt/homebrew/opt/llvm"  # macOS Homebrew
)

LLVM_PREFIX=""
for path in "${LLVM_PATHS[@]}"; do
    if [ -f "$path/lib/cmake/mlir/MLIRConfig.cmake" ]; then
        LLVM_PREFIX="$path"
        break
    fi
done

if [ -z "$LLVM_PREFIX" ]; then
    echo -e "${RED}Error: Could not find MLIR installation!${NC}"
    echo -e "${YELLOW}Searched in:${NC}"
    for path in "${LLVM_PATHS[@]}"; do
        echo -e "  - $path"
    done
    echo -e "\n${YELLOW}Please install MLIR/LLVM or set CMAKE_PREFIX_PATH manually:${NC}"
    echo -e "  cmake -B build -DCMAKE_PREFIX_PATH=/path/to/llvm"
    exit 1
fi

echo -e "${GREEN}✓ Found LLVM/MLIR at: $LLVM_PREFIX${NC}\n"

# Check for required components
echo -e "${YELLOW}Checking MLIR components...${NC}"
MLIR_OPT="$LLVM_PREFIX/bin/mlir-opt"
MLIR_TRANSLATE="$LLVM_PREFIX/bin/mlir-translate"

if [ ! -f "$MLIR_OPT" ]; then
    echo -e "${RED}Error: mlir-opt not found at $MLIR_OPT${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Found mlir-opt${NC}"

if [ ! -f "$MLIR_TRANSLATE" ]; then
    echo -e "${RED}Error: mlir-translate not found at $MLIR_TRANSLATE${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Found mlir-translate${NC}\n"

# Clean old build directory
if [ -d "build" ]; then
    echo -e "${YELLOW}Removing old build directory...${NC}"
    rm -rf build
    echo -e "${GREEN}✓ Cleaned${NC}\n"
fi

# Configure with CMake
echo -e "${YELLOW}Configuring with CMake...${NC}"
cmake -B build -DCMAKE_PREFIX_PATH="$LLVM_PREFIX" "$@"

# Strip GCC-only flags that clang-tidy (clang) does not understand,
# so that manual `clang-tidy -p lib/build` runs work correctly.
if [ -f "build/compile_commands.json" ]; then
    sed -i 's/-fno-lifetime-dse //g' build/compile_commands.json
fi

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}Configuration successful!${NC}"
    echo -e "${GREEN}========================================${NC}\n"
    echo -e "${BLUE}Next steps:${NC}"
    echo -e "  Build the project:"
    echo -e "     ${YELLOW}cmake --build build${NC}"
    echo -e ""
else
    echo -e "\n${RED}========================================${NC}"
    echo -e "${RED}Configuration failed!${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi
