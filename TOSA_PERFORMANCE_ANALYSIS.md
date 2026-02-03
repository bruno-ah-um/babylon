# TOSA Performance Analysis: Native vs Java vs ONNX Runtime

This document summarizes the performance comparison between three implementations of a CNN feature extractor for MNIST digit recognition.

## Benchmark Configuration

- **Model**: 2-layer CNN feature extractor (Conv1 -> ReLU -> MaxPool -> Conv2 -> ReLU -> MaxPool)
- **Input**: 28x28 grayscale images (NHWC format for TOSA, NCHW for ONNX)
- **Output**: 4x4x16 feature maps (256 floats)
- **Warmup**: 20,000 iterations
- **Benchmark**: 100,000 iterations
- **CPU**: Intel Xeon Silver 4215R @ 3.20GHz (AVX-512 capable)

## Results Summary

| Implementation | Per Iteration | Throughput | vs Java Baseline |
|---------------|---------------|------------|------------------|
| **ONNX Runtime** | **72 µs** | **13,954 ops/sec** | **18.6x faster** |
| **TOSA Native** | 215 µs | 4,656 ops/sec | 6.2x faster |
| **TOSA Java** | 1,335 µs | 749 ops/sec | baseline |

## Analysis

### TOSA Java (Baseline)
Pure Java interpretation of TOSA operations. Each tensor operation is executed as nested Java loops. This is the slowest but most portable implementation.

### TOSA Native (6.2x faster than Java)
Compiles the model through MLIR to native x86_64 code:
```
TOSA dialect -> Linalg dialect -> SCF loops -> LLVM IR -> Native x86_64
```

**Compilation pipeline:**
```
mlir-opt --pass-pipeline="builtin.module(
    func.func(tosa-to-linalg-named),
    func.func(tosa-to-linalg),
    func.func(tosa-to-arith),
    func.func(tosa-to-tensor),
    func.func(linalg-fuse-elementwise-ops),
    one-shot-bufferize{bufferize-function-boundaries},
    func.func(convert-linalg-to-loops),
    func.func(lower-affine),
    func.func(arith-expand),
    func.func(convert-scf-to-cf),
    convert-arith-to-llvm,
    convert-func-to-llvm,
    convert-cf-to-llvm,
    finalize-memref-to-llvm,
    reconcile-unrealized-casts
)"
```

**Current limitations:**
- Generates scalar loops (no SIMD vectorization)
- No cache-friendly loop tiling
- No operation fusion beyond elementwise ops

### ONNX Runtime (18.6x faster than Java, 3x faster than TOSA Native)
Uses highly optimized execution providers with:
- Intel oneDNN (MKL-DNN) optimized convolution kernels
- Hand-tuned assembly for critical paths
- Automatic operator fusion
- Cache-friendly memory layouts
- Full SIMD utilization (AVX-512 on this CPU)

## Why TOSA Native is Slower than ONNX Runtime

### 1. Scalar Loops vs Vectorized Kernels
The `convert-linalg-to-loops` pass generates simple nested scalar loops:
```c
for (int n = 0; n < batch; n++)
    for (int h = 0; h < height; h++)
        for (int w = 0; w < width; w++)
            for (int c = 0; c < channels; c++)
                output[n][h][w][c] = ...  // scalar operation
```

ONNX Runtime uses vectorized kernels that process multiple elements per instruction:
```c
for (int n = 0; n < batch; n++)
    for (int h = 0; h < height; h++)
        for (int w = 0; w < width; w++)
            for (int c = 0; c < channels; c += 16)  // AVX-512 processes 16 floats
                _mm512_store_ps(&output[n][h][w][c], ...);
```

### 2. No Loop Tiling
Our loops don't have cache-friendly blocking. For convolutions, this causes cache misses when accessing weight tensors repeatedly.

### 3. Limited Operation Fusion
Only elementwise operations are fused. Convolution + ReLU could potentially be fused but aren't in the current pipeline.

## Potential Improvements

### Short-term (Clang flags)
Already applied:
- `-O3` (highest optimization)
- `-march=native` (enables AVX-512)
- `-ffast-math` (allows FP reordering)

Impact: Minimal (~5-10%) because the IR structure limits what clang can optimize.

### Medium-term (MLIR passes)
1. **Vectorization**: Use MLIR's vector dialect
   ```
   linalg-generalize-named-ops -> vectorize -> convert-vector-to-llvm
   ```

2. **Loop tiling**: Apply `affine-loop-tile` for better cache utilization

3. **Polyhedral optimization**: Use `affine-*` passes for complex loop transformations

### Long-term (Library integration)
1. **Link against oneDNN**: Use Intel's optimized convolution implementations
2. **Use cuDNN/CUDA**: For GPU acceleration
3. **OpenBLAS integration**: For matrix operations in fully-connected layers

## Running the Benchmarks

### TOSA Benchmark (Java + Native)
```bash
cd cr-examples/mlir-tosa
JAVA_HOME=/path/to/babylon/jdk mvn test -Dtest=TosaBenchmark#runFullBenchmark
```

### ONNX Runtime Benchmark
```bash
cd cr-examples/onnx
JAVA_HOME=/path/to/babylon/jdk mvn test -Dtest=OnnxBenchmark#runOnnxBenchmark
```

## Conclusion

The TOSA Native compilation provides a **6.2x speedup** over pure Java interpretation, demonstrating the value of AOT compilation through MLIR. However, there's still a **3x gap** compared to ONNX Runtime, primarily due to the lack of vectorization and optimized kernels in the current MLIR pipeline.

For production ML workloads, the path forward would be either:
1. Enhance the MLIR pipeline with vectorization passes
2. Link against optimized libraries (oneDNN, OpenBLAS)
3. Use existing ML runtimes (ONNX, TensorFlow Lite) for inference

The current implementation serves as a proof-of-concept for Java code reflection -> TOSA -> native compilation, with room for significant optimization.
