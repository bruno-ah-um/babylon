# MLIR TOSA Java Bindings - Proof of Concept

This project demonstrates Java bindings for MLIR TOSA (Tensor Operator Set Architecture) operations using code reflection and the Foreign Function & Memory API.

## Overview

This proof of concept enables you to:
1. Define tensor operations in Java
2. Use code reflection to transform them to MLIR TOSA dialect
3. Execute operations via the MLIR C API through jextract-generated bindings

## Architecture

The project is structured similarly to the ONNX example in `cr-examples/onnx`:

```
mlir-tosa/
├── lib/                          # C++ MLIR TOSA library
│   ├── mlir_tosa_c_api.h        # C API header
│   ├── mlir_tosa_c_api.cpp      # C API implementation
│   └── CMakeLists.txt           # CMake build configuration
├── src/main/java/mlir/tosa/
│   ├── Tensor.java              # Generic tensor abstraction
│   ├── TosaOperators.java       # TOSA operations (Add, Mul, etc.)
│   ├── TosaInterpreter.java     # Operation dispatch
│   ├── TosaRuntime.java         # MLIR execution via C API
│   └── TosaTransformer.java     # Code reflection transformer
└── src/test/java/mlir/tosa/
    └── TosaSimpleTest.java      # Tests for Add and Mul operations
```

## Supported Operations (Proof of Concept)

Currently implemented:
- **Add**: Element-wise addition
- **Mul**: Element-wise multiplication

## Build System

The Maven build automatically:
1. Runs `lib/configure.sh` to detect LLVM/MLIR installation
2. Compiles the C++ library with CMake
3. **Generates Java bindings using jextract** (from the C header)
4. Compiles Java sources
5. Runs tests

### Build Commands

```bash
# Full build (configure, compile C library, generate bindings, compile Java)
mvn clean compile

# Run tests
mvn test

# Clean everything (including generated bindings)
mvn clean
```

## Usage Example

### Basic Operations

```java
import mlir.tosa.Tensor;
import static mlir.tosa.TosaOperators.*;

// Create tensors
var a = Tensor.ofFlat(1.0f, 2.0f, 3.0f);
var b = Tensor.ofFlat(4.0f, 5.0f, 6.0f);

// Element-wise addition
var sum = Add(a, b);  // [5.0, 7.0, 9.0]

// Element-wise multiplication
var product = Mul(a, b);  // [4.0, 10.0, 18.0]

// Combined operations
var result = Add(Mul(a, b), c);
```

### Fluent API (Instance Methods)

The Tensor class provides a **fluent interface** for more natural syntax:

```java
var a = Tensor.ofFlat(1.0f, 2.0f, 3.0f);
var b = Tensor.ofFlat(4.0f, 5.0f, 6.0f);
var c = Tensor.ofFlat(0.5f, 1.0f, 1.5f);

// Using instance methods
var sum = a.add(b);         // [5.0, 7.0, 9.0]
var product = a.mul(b);     // [4.0, 10.0, 18.0]

// Method chaining: (a + b) * c
var result = a.add(b).mul(c);  // [2.5, 7.0, 13.5]
```

This fluent API style is more readable and follows Java conventions for mathematical operations.

### Multi-dimensional Tensors

```java
// 2x3 tensor
var matrix = Tensor.ofShape(new long[]{2, 3},
    1.0f, 2.0f, 3.0f,
    4.0f, 5.0f, 6.0f
);

var doubled = Add(matrix, matrix);
```

### Supported Element Types

- `Float` (FLOAT32)
- `Double` (FLOAT64)
- `Integer` (INT32)
- `Long` (INT64)

## How It Works

### 1. Tensor Abstraction
The `Tensor<T>` class uses Java's Foreign Function & Memory API for efficient memory management:
- `Arena` for automatic memory lifecycle
- `MemorySegment` for direct memory access
- Type-safe generic operations

### 2. TOSA Operators
Operations are defined in `TosaOperators` using the `@Quotable` annotation:
```java
@Quotable
public static <T> Tensor<T> Add(Tensor<T> input1, Tensor<T> input2) {
    Quoted quoted = (Quoted) Add(input1, input2);
    return TosaInterpreter.interpret("Add", quoted, input1, input2);
}
```

### 3. Code Reflection (Future)
When full transformation is implemented, the `TosaTransformer` will:
1. Capture Java lambda expressions as operation trees
2. Transform to MLIR TOSA IR
3. Build and compile MLIR modules
4. Execute via the native runtime

### 4. Current Execution
The proof of concept currently:
- Uses the MLIR C API to build TOSA operations
- Falls back to direct Java computation for results
- Demonstrates the integration pattern for future MLIR execution

## JExtract Integration

Java bindings are automatically regenerated whenever the C library is recompiled:

```xml
<execution>
    <id>jextract-generate-bindings</id>
    <phase>compile</phase>
    ...
    <executable>jextract</executable>
    <arguments>
        <argument>--output</argument>
        <argument>${project.basedir}/src/main/java</argument>
        <argument>--target-package</argument>
        <argument>mlir.tosa.bindings</argument>
        <argument>--library</argument>
        <argument>mlir_tosa_c</argument>
        <argument>${project.basedir}/lib/mlir_tosa_c_api.h</argument>
    </arguments>
</execution>
```

Generated bindings are placed in `src/main/java/mlir/tosa/bindings/`.

## Requirements

- **JDK 26** (with preview features enabled for code reflection)
- **jextract** (for generating Java bindings)
- **LLVM/MLIR** (version 17, 18, or 19)
- **CMake** (3.20+)
- **Maven** (3.6+)

## Testing

Run the test suite:
```bash
mvn test
```

Tests include:
- Element-wise operations (Add, Mul)
- Scalar, 1D, and 2D tensors
- Different element types (float, double, int, long)
- Shape and type validation
- Combined operations

## Future Work

### Code Reflection Integration
Implement full `@Reflect` method support:
```java
@Reflect
public Tensor<Float> computeExpression(Tensor<Float> a, Tensor<Float> b) {
    return Add(Mul(a, b), a);
}

var result = TosaTransformer.execute(
    MethodHandles.lookup(),
    () -> computeExpression(a, b)
);
```

### Additional TOSA Operations
Extend the API to support:
- Arithmetic: Sub, Negate, Reciprocal
- Math: Exp, Log, Sqrt
- Matrix: MatMul
- Activation: Relu, Sigmoid, Tanh
- Pooling: MaxPool, AvgPool
- Convolution: Conv2D

### MLIR Compilation Pipeline
Implement full MLIR lowering:
1. Transform to TOSA dialect
2. Lower to Linalg/Arith dialects
3. Compile to LLVM IR
4. JIT execution or ahead-of-time compilation

## Comparison with ONNX Example

This project follows the same pattern as `cr-examples/onnx`:

| Component | ONNX Example | TOSA Example |
|-----------|--------------|--------------|
| Tensor abstraction | `oracle.code.onnx.Tensor` | `mlir.tosa.Tensor` |
| Operators | `OnnxOperators` (150+ ops) | `TosaOperators` (Add, Mul) |
| Runtime | ONNX Runtime (via FFM) | MLIR C API (via jextract) |
| Transformer | `OnnxTransformer` | `TosaTransformer` (simplified) |
| Code generation | Python script from schemas | Manual (for POC) |

## References

- [TOSA Specification v1.0.1](https://www.mlplatform.org/tosa/tosa_spec.html)
- [MLIR TOSA Dialect](https://mlir.llvm.org/docs/Dialects/TOSA/)
- [TOSA GitHub](https://github.com/arm/tosa-specification)
- [Java FFM API](https://openjdk.org/jeps/454)
- [Code Reflection (JEP draft)](https://openjdk.org/projects/babylon/)
