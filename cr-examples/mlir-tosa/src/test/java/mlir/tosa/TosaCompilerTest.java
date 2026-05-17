package mlir.tosa;

import jdk.incubator.code.Reflect;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for TosaCompiler - compiling TOSA to native x86_64 code.
 */
public class TosaCompilerTest {

    /**
     * Simple element-wise addition for testing.
     */
    @Reflect
    public static Tensor<Float> simpleAdd(Tensor<Float> a, Tensor<Float> b) {
        return a.add(b);
    }

    @Test
    public void testCompileSimpleAdd() throws Exception {
        TosaCompiler compiler = new TosaCompiler(true);  // verbose mode

        // Compile the simpleAdd method
        var method = TosaCompilerTest.class.getMethod("simpleAdd", Tensor.class, Tensor.class);
        CompiledFunction fn = compiler.compile(method);

        assertNotNull(fn);
        assertEquals("simpleAdd", fn.name());
        assertEquals(2, fn.numParams());

        System.out.println("Compiled function: " + fn);
        System.out.println("Library path: " + fn.libraryPath());
    }

    @Test
    public void testInvokeSimpleAddScalar() throws Exception {
        TosaCompiler compiler = new TosaCompiler(true);

        var method = TosaCompilerTest.class.getMethod("simpleAdd", Tensor.class, Tensor.class);
        CompiledFunction fn = compiler.compile(method);

        // Create scalar test tensors (single element)
        // Note: TOSA generates scalar tensor code (tensor<f32>), so we test with single values
        Tensor<Float> a = Tensor.ofShape(new long[]{1}, 5.0f);
        Tensor<Float> b = Tensor.ofShape(new long[]{1}, 3.0f);

        // Invoke native function
        Tensor<Float> result = fn.invoke(a, b);

        System.out.println("Input a: " + a);
        System.out.println("Input b: " + b);
        System.out.println("Result:  " + result);

        // Verify result
        assertNotNull(result);

        // Check value: 5.0 + 3.0 = 8.0
        float actual = result.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 0);
        assertEquals(8.0f, actual, 0.001f, "Scalar add result mismatch");
    }

    @Test
    public void testNativeMatchesJava() throws Exception {
        TosaCompiler compiler = new TosaCompiler(true);

        var method = TosaCompilerTest.class.getMethod("simpleAdd", Tensor.class, Tensor.class);
        CompiledFunction fn = compiler.compile(method);

        // Test multiple input combinations
        float[][] testCases = {
            {1.0f, 2.0f},    // 1 + 2 = 3
            {-5.0f, 3.0f},   // -5 + 3 = -2
            {0.0f, 0.0f},    // 0 + 0 = 0
            {100.5f, 200.5f}, // 100.5 + 200.5 = 301
            {Float.MAX_VALUE / 2, Float.MAX_VALUE / 2}, // Large numbers
            {-1.5f, -2.5f},  // -1.5 + -2.5 = -4
        };

        System.out.println("\n=== Comparing Native vs Java Results ===\n");

        for (float[] testCase : testCases) {
            Tensor<Float> a = Tensor.ofShape(new long[]{1}, testCase[0]);
            Tensor<Float> b = Tensor.ofShape(new long[]{1}, testCase[1]);

            // Call native compiled function
            Tensor<Float> nativeResult = fn.invoke(a, b);
            float nativeValue = nativeResult.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 0);

            // Call Java implementation directly
            Tensor<Float> javaResult = simpleAdd(a, b);
            float javaValue = javaResult.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 0);

            System.out.printf("Input: %.2f + %.2f%n", testCase[0], testCase[1]);
            System.out.printf("  Java result:   %.6f%n", javaValue);
            System.out.printf("  Native result: %.6f%n", nativeValue);
            System.out.printf("  Match: %s%n%n", Math.abs(javaValue - nativeValue) < 0.0001f ? "YES ✓" : "NO ✗");

            // Assert they match
            assertEquals(javaValue, nativeValue, 0.0001f,
                String.format("Native and Java results differ for %.2f + %.2f", testCase[0], testCase[1]));
        }

        System.out.println("=== All test cases passed! Native matches Java ===");
    }

    /**
     * Simple multiplication for testing.
     */
    @Reflect
    public static Tensor<Float> simpleMul(Tensor<Float> a, Tensor<Float> b) {
        return a.mul(b);
    }

    /**
     * Simple subtraction for testing.
     */
    @Reflect
    public static Tensor<Float> simpleSub(Tensor<Float> a, Tensor<Float> b) {
        return a.sub(b);
    }

    /**
     * Test for 1D vector addition with known shapes.
     * This tests ranked tensor support with shape [4].
     */
    @Reflect
    public static Tensor<Float> vectorAdd(Tensor<Float> a, Tensor<Float> b) {
        return a.add(b);
    }

    @Test
    public void testVectorAddJava() throws Exception {
        // Test the Java-side vector add operation
        Tensor<Float> a = Tensor.ofShape(new long[]{4}, 1.0f, 2.0f, 3.0f, 4.0f);
        Tensor<Float> b = Tensor.ofShape(new long[]{4}, 10.0f, 20.0f, 30.0f, 40.0f);

        Tensor<Float> result = vectorAdd(a, b);

        System.out.println("Vector Add Test (Java):");
        System.out.println("  a = " + a);
        System.out.println("  b = " + b);
        System.out.println("  result = " + result);

        // Verify each element
        assertEquals(11.0f, result.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 0), 0.001f);
        assertEquals(22.0f, result.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 1), 0.001f);
        assertEquals(33.0f, result.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 2), 0.001f);
        assertEquals(44.0f, result.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 3), 0.001f);
    }

    @Test
    public void testVectorAddNative() throws Exception {
        TosaCompiler compiler = new TosaCompiler(true); // verbose mode

        var method = TosaCompilerTest.class.getMethod("vectorAdd", Tensor.class, Tensor.class);
        CompiledFunction fn = compiler.compile(method);

        // Create 1D test tensors
        Tensor<Float> a = Tensor.ofShape(new long[]{4}, 1.0f, 2.0f, 3.0f, 4.0f);
        Tensor<Float> b = Tensor.ofShape(new long[]{4}, 10.0f, 20.0f, 30.0f, 40.0f);

        System.out.println("\n=== Vector Add Test (Native) ===");
        System.out.println("  a = " + a);
        System.out.println("  b = " + b);

        // Call native function
        Tensor<Float> nativeResult = fn.invoke(a, b);
        System.out.println("  native result = " + nativeResult);

        // Call Java function for comparison
        Tensor<Float> javaResult = vectorAdd(a, b);
        System.out.println("  java result = " + javaResult);

        // Verify each element matches
        for (int i = 0; i < 4; i++) {
            float nativeVal = nativeResult.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
            float javaVal = javaResult.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
            assertEquals(javaVal, nativeVal, 0.001f,
                "Mismatch at index " + i + ": Java=" + javaVal + ", Native=" + nativeVal);
        }

        System.out.println("=== Vector Add: Native matches Java! ===");
    }

    /**
     * 2D matrix element-wise addition for testing.
     */
    @Reflect
    public static Tensor<Float> matrixAdd(Tensor<Float> a, Tensor<Float> b) {
        return a.add(b);
    }

    @Test
    public void testMatrixAddNative() throws Exception {
        TosaCompiler compiler = new TosaCompiler(true); // verbose mode

        var method = TosaCompilerTest.class.getMethod("matrixAdd", Tensor.class, Tensor.class);
        CompiledFunction fn = compiler.compile(method, 2); // rank=2 for 2D matrices

        // Create 2D test tensors [2x3]
        Tensor<Float> a = Tensor.ofShape(new long[]{2, 3},
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f);
        Tensor<Float> b = Tensor.ofShape(new long[]{2, 3},
            10.0f, 20.0f, 30.0f,
            40.0f, 50.0f, 60.0f);

        System.out.println("\n=== Matrix Add Test (Native) ===");
        System.out.println("  a [2x3] = " + a);
        System.out.println("  b [2x3] = " + b);

        // Call native function
        Tensor<Float> nativeResult = fn.invoke(a, b);
        System.out.println("  native result = " + nativeResult);

        // Call Java function for comparison
        Tensor<Float> javaResult = matrixAdd(a, b);
        System.out.println("  java result = " + javaResult);

        // Verify each element matches
        // Expected: [[11, 22, 33], [44, 55, 66]]
        float[] expected = {11.0f, 22.0f, 33.0f, 44.0f, 55.0f, 66.0f};
        for (int i = 0; i < 6; i++) {
            float nativeVal = nativeResult.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
            float javaVal = javaResult.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
            assertEquals(expected[i], nativeVal, 0.001f,
                "Mismatch at index " + i + ": Expected=" + expected[i] + ", Native=" + nativeVal);
            assertEquals(javaVal, nativeVal, 0.001f,
                "Java/Native mismatch at index " + i + ": Java=" + javaVal + ", Native=" + nativeVal);
        }

        System.out.println("=== Matrix Add: Native matches Java! ===");
    }

    /**
     * Test 2D matrix multiplication.
     * [2,3] @ [3,2] = [2,2]
     */
    @Reflect
    public static Tensor<Float> matrixMul(Tensor<Float> a, Tensor<Float> b) {
        return a.matmul(b);
    }

    @Test
    public void testMatrixMulJava() throws Exception {
        // Test Java-side matrix multiplication first
        // A is [2,3]: [[1,2,3], [4,5,6]]
        Tensor<Float> a = Tensor.ofShape(new long[]{2, 3},
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f);

        // B is [3,2]: [[7,8], [9,10], [11,12]]
        Tensor<Float> b = Tensor.ofShape(new long[]{3, 2},
            7.0f, 8.0f,
            9.0f, 10.0f,
            11.0f, 12.0f);

        Tensor<Float> result = matrixMul(a, b);

        System.out.println("MatMul Test (Java):");
        System.out.println("  A [2x3] = " + a);
        System.out.println("  B [3x2] = " + b);
        System.out.println("  Result [2x2] = " + result);

        // Expected: [[58, 64], [139, 154]]
        // [0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
        // [0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
        // [1,0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
        // [1,1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
        assertEquals(58.0f, result.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 0), 0.001f);
        assertEquals(64.0f, result.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 1), 0.001f);
        assertEquals(139.0f, result.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 2), 0.001f);
        assertEquals(154.0f, result.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 3), 0.001f);
    }

    @Test
    public void testNativeMatchesJavaMultipleOps() throws Exception {
        TosaCompiler compiler = new TosaCompiler(false); // non-verbose for cleaner output

        System.out.println("\n=== Testing Multiple TOSA Operations ===\n");

        // Test ADD
        {
            var method = TosaCompilerTest.class.getMethod("simpleAdd", Tensor.class, Tensor.class);
            CompiledFunction fn = compiler.compile(method);

            Tensor<Float> a = Tensor.ofShape(new long[]{1}, 7.5f);
            Tensor<Float> b = Tensor.ofShape(new long[]{1}, 2.5f);

            Tensor<Float> nativeResult = fn.invoke(a, b);
            Tensor<Float> javaResult = simpleAdd(a, b);

            float nativeVal = nativeResult.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 0);
            float javaVal = javaResult.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 0);

            System.out.printf("ADD: %.2f + %.2f = Java:%.2f, Native:%.2f [%s]%n",
                7.5f, 2.5f, javaVal, nativeVal, Math.abs(javaVal - nativeVal) < 0.0001f ? "PASS" : "FAIL");
            assertEquals(javaVal, nativeVal, 0.0001f, "ADD operation mismatch");
        }

        // Test MUL
        {
            var method = TosaCompilerTest.class.getMethod("simpleMul", Tensor.class, Tensor.class);
            CompiledFunction fn = compiler.compile(method);

            Tensor<Float> a = Tensor.ofShape(new long[]{1}, 3.0f);
            Tensor<Float> b = Tensor.ofShape(new long[]{1}, 4.0f);

            Tensor<Float> nativeResult = fn.invoke(a, b);
            Tensor<Float> javaResult = simpleMul(a, b);

            float nativeVal = nativeResult.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 0);
            float javaVal = javaResult.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 0);

            System.out.printf("MUL: %.2f * %.2f = Java:%.2f, Native:%.2f [%s]%n",
                3.0f, 4.0f, javaVal, nativeVal, Math.abs(javaVal - nativeVal) < 0.0001f ? "PASS" : "FAIL");
            assertEquals(javaVal, nativeVal, 0.0001f, "MUL operation mismatch");
        }

        // Test SUB
        {
            var method = TosaCompilerTest.class.getMethod("simpleSub", Tensor.class, Tensor.class);
            CompiledFunction fn = compiler.compile(method);

            Tensor<Float> a = Tensor.ofShape(new long[]{1}, 10.0f);
            Tensor<Float> b = Tensor.ofShape(new long[]{1}, 3.5f);

            Tensor<Float> nativeResult = fn.invoke(a, b);
            Tensor<Float> javaResult = simpleSub(a, b);

            float nativeVal = nativeResult.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 0);
            float javaVal = javaResult.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 0);

            System.out.printf("SUB: %.2f - %.2f = Java:%.2f, Native:%.2f [%s]%n",
                10.0f, 3.5f, javaVal, nativeVal, Math.abs(javaVal - nativeVal) < 0.0001f ? "PASS" : "FAIL");
            assertEquals(javaVal, nativeVal, 0.0001f, "SUB operation mismatch");
        }

        System.out.println("\n=== All operations match! ===");
    }

    /**
     * 3D batched matrix multiplication for TOSA native compilation.
     * TOSA matmul requires 3D tensors: [batch, M, K] @ [batch, K, N] -> [batch, M, N]
     */
    @Reflect
    public static Tensor<Float> batchedMatMul(Tensor<Float> a, Tensor<Float> b) {
        return a.matmul(b);
    }

    @Test
    public void testBatchedMatMulJava() throws Exception {
        // Test Java-side 3D batched matrix multiplication
        // A is [2, 2, 3]: batch=2, each matrix is 2x3
        // B is [2, 3, 2]: batch=2, each matrix is 3x2
        // Result should be [2, 2, 2]: batch=2, each matrix is 2x2

        Tensor<Float> a = Tensor.ofShape(new long[]{2, 2, 3},
            // Batch 0: [[1,2,3], [4,5,6]]
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
            // Batch 1: [[7,8,9], [10,11,12]]
            7.0f, 8.0f, 9.0f,
            10.0f, 11.0f, 12.0f);

        Tensor<Float> b = Tensor.ofShape(new long[]{2, 3, 2},
            // Batch 0: [[1,2], [3,4], [5,6]]
            1.0f, 2.0f,
            3.0f, 4.0f,
            5.0f, 6.0f,
            // Batch 1: [[7,8], [9,10], [11,12]]
            7.0f, 8.0f,
            9.0f, 10.0f,
            11.0f, 12.0f);

        Tensor<Float> result = batchedMatMul(a, b);

        System.out.println("\nBatched MatMul Test (Java):");
        System.out.println("  A [2x2x3] = " + a);
        System.out.println("  B [2x3x2] = " + b);
        System.out.println("  Result [2x2x2] = " + result);

        // Expected for batch 0: [[1,2,3] @ [[1,2],[3,4],[5,6]], [4,5,6] @ [[1,2],[3,4],[5,6]]]
        // [0,0,0] = 1*1 + 2*3 + 3*5 = 1 + 6 + 15 = 22
        // [0,0,1] = 1*2 + 2*4 + 3*6 = 2 + 8 + 18 = 28
        // [0,1,0] = 4*1 + 5*3 + 6*5 = 4 + 15 + 30 = 49
        // [0,1,1] = 4*2 + 5*4 + 6*6 = 8 + 20 + 36 = 64

        // Expected for batch 1: [[7,8,9] @ [[7,8],[9,10],[11,12]], [10,11,12] @ ...]
        // [1,0,0] = 7*7 + 8*9 + 9*11 = 49 + 72 + 99 = 220
        // [1,0,1] = 7*8 + 8*10 + 9*12 = 56 + 80 + 108 = 244
        // [1,1,0] = 10*7 + 11*9 + 12*11 = 70 + 99 + 132 = 301
        // [1,1,1] = 10*8 + 11*10 + 12*12 = 80 + 110 + 144 = 334

        float[] expected = {22.0f, 28.0f, 49.0f, 64.0f, 220.0f, 244.0f, 301.0f, 334.0f};
        for (int i = 0; i < expected.length; i++) {
            float actual = result.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
            assertEquals(expected[i], actual, 0.001f,
                "Batched matmul result mismatch at index " + i + ": expected=" + expected[i] + ", actual=" + actual);
        }

        System.out.println("=== Batched MatMul Java test passed! ===");
    }

    @Test
    public void testBatchedMatMulNative() throws Exception {
        TosaCompiler compiler = new TosaCompiler(true); // verbose mode

        var method = TosaCompilerTest.class.getMethod("batchedMatMul", Tensor.class, Tensor.class);
        CompiledFunction fn = compiler.compile(method, 3); // rank=3 for 3D batched matmul

        // Create 3D test tensors: [batch=2, M=2, K=3] @ [batch=2, K=3, N=2] -> [batch=2, M=2, N=2]
        Tensor<Float> a = Tensor.ofShape(new long[]{2, 2, 3},
            // Batch 0: [[1,2,3], [4,5,6]]
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
            // Batch 1: [[7,8,9], [10,11,12]]
            7.0f, 8.0f, 9.0f,
            10.0f, 11.0f, 12.0f);

        Tensor<Float> b = Tensor.ofShape(new long[]{2, 3, 2},
            // Batch 0: [[1,2], [3,4], [5,6]]
            1.0f, 2.0f,
            3.0f, 4.0f,
            5.0f, 6.0f,
            // Batch 1: [[7,8], [9,10], [11,12]]
            7.0f, 8.0f,
            9.0f, 10.0f,
            11.0f, 12.0f);

        System.out.println("\n=== Batched MatMul Test (Native) ===");
        System.out.println("  A [2x2x3] = " + a);
        System.out.println("  B [2x3x2] = " + b);

        // Call native function
        Tensor<Float> nativeResult = fn.invoke(a, b);
        System.out.println("  native result [2x2x2] = " + nativeResult);

        // Call Java function for comparison
        Tensor<Float> javaResult = batchedMatMul(a, b);
        System.out.println("  java result [2x2x2] = " + javaResult);

        // Verify each element matches
        float[] expected = {22.0f, 28.0f, 49.0f, 64.0f, 220.0f, 244.0f, 301.0f, 334.0f};
        for (int i = 0; i < 8; i++) {
            float nativeVal = nativeResult.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
            float javaVal = javaResult.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
            assertEquals(expected[i], nativeVal, 0.001f,
                "Native result mismatch at index " + i + ": Expected=" + expected[i] + ", Native=" + nativeVal);
            assertEquals(javaVal, nativeVal, 0.001f,
                "Java/Native mismatch at index " + i + ": Java=" + javaVal + ", Native=" + nativeVal);
        }

        System.out.println("=== Batched MatMul: Native matches Java! ===");
    }

    /**
     * Test a simple fully-connected layer: Y = X @ W + B
     * This is the key operation in MNIST's FC layers.
     * Uses 3D tensors for batched matmul: [batch=1, M, K] @ [batch=1, K, N] -> [batch=1, M, N]
     */
    @Reflect
    public static Tensor<Float> linearLayer(Tensor<Float> x, Tensor<Float> w) {
        // x: [1, 1, 3] (batch=1, input: 1 sample with 3 features)
        // w: [1, 3, 2] (batch=1, weights: 3 input features -> 2 output features)
        // result: [1, 1, 2]
        return x.matmul(w);
    }

    @Test
    public void testLinearLayerNative() throws Exception {
        TosaCompiler compiler = new TosaCompiler(true);

        var method = TosaCompilerTest.class.getMethod("linearLayer", Tensor.class, Tensor.class);
        CompiledFunction fn = compiler.compile(method, 3); // rank=3 for batched matmul

        // Input: [1, 1, 3] - single sample with 3 features
        Tensor<Float> x = Tensor.ofShape(new long[]{1, 1, 3},
            1.0f, 2.0f, 3.0f);

        // Weights: [1, 3, 2] - transform 3 features to 2 outputs
        Tensor<Float> w = Tensor.ofShape(new long[]{1, 3, 2},
            0.1f, 0.2f,   // feature 0 weights
            0.3f, 0.4f,   // feature 1 weights
            0.5f, 0.6f);  // feature 2 weights

        System.out.println("\n=== Linear Layer Test (Native) ===");
        System.out.println("  X [1x1x3] = " + x);
        System.out.println("  W [1x3x2] = " + w);

        // Native result
        Tensor<Float> nativeResult = fn.invoke(x, w);
        System.out.println("  native Y [1x1x2] = " + nativeResult);

        // Java result for comparison
        Tensor<Float> javaResult = linearLayer(x, w);
        System.out.println("  java Y [1x1x2] = " + javaResult);

        // Expected: [1*0.1+2*0.3+3*0.5, 1*0.2+2*0.4+3*0.6] = [0.1+0.6+1.5, 0.2+0.8+1.8] = [2.2, 2.8]
        float expected0 = 1.0f * 0.1f + 2.0f * 0.3f + 3.0f * 0.5f; // = 2.2
        float expected1 = 1.0f * 0.2f + 2.0f * 0.4f + 3.0f * 0.6f; // = 2.8

        float native0 = nativeResult.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 0);
        float native1 = nativeResult.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 1);
        float java0 = javaResult.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 0);
        float java1 = javaResult.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 1);

        assertEquals(expected0, native0, 0.001f, "Native output[0] mismatch");
        assertEquals(expected1, native1, 0.001f, "Native output[1] mismatch");
        assertEquals(java0, native0, 0.001f, "Java/Native output[0] mismatch");
        assertEquals(java1, native1, 0.001f, "Java/Native output[1] mismatch");

        System.out.println("  Expected: [" + expected0 + ", " + expected1 + "]");
        System.out.println("=== Linear Layer: Native matches Java! ===");
    }

    /**
     * Test linear layer with bias: Y = X @ W + B
     * This combines matmul and add operations.
     */
    @Reflect
    public static Tensor<Float> linearLayerWithBias(Tensor<Float> x, Tensor<Float> w, Tensor<Float> b) {
        return x.matmul(w).add(b);
    }

    @Test
    public void testLinearLayerWithBiasNative() throws Exception {
        TosaCompiler compiler = new TosaCompiler(true);

        var method = TosaCompilerTest.class.getMethod("linearLayerWithBias", Tensor.class, Tensor.class, Tensor.class);
        CompiledFunction fn = compiler.compile(method, 3); // rank=3

        // Input: [1, 1, 3]
        Tensor<Float> x = Tensor.ofShape(new long[]{1, 1, 3},
            1.0f, 2.0f, 3.0f);

        // Weights: [1, 3, 2]
        Tensor<Float> w = Tensor.ofShape(new long[]{1, 3, 2},
            0.1f, 0.2f,
            0.3f, 0.4f,
            0.5f, 0.6f);

        // Bias: [1, 1, 2] (broadcast-able to output shape)
        Tensor<Float> b = Tensor.ofShape(new long[]{1, 1, 2},
            0.5f, -0.5f);

        System.out.println("\n=== Linear Layer with Bias Test (Native) ===");
        System.out.println("  X [1x1x3] = " + x);
        System.out.println("  W [1x3x2] = " + w);
        System.out.println("  B [1x1x2] = " + b);

        // Native result
        Tensor<Float> nativeResult = fn.invoke(x, w, b);
        System.out.println("  native Y [1x1x2] = " + nativeResult);

        // Java result
        Tensor<Float> javaResult = linearLayerWithBias(x, w, b);
        System.out.println("  java Y [1x1x2] = " + javaResult);

        // Expected: [2.2 + 0.5, 2.8 + (-0.5)] = [2.7, 2.3]
        float expected0 = 2.2f + 0.5f;  // = 2.7
        float expected1 = 2.8f - 0.5f;  // = 2.3

        float native0 = nativeResult.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 0);
        float native1 = nativeResult.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 1);

        assertEquals(expected0, native0, 0.001f, "Native output[0] mismatch");
        assertEquals(expected1, native1, 0.001f, "Native output[1] mismatch");

        System.out.println("  Expected: [" + expected0 + ", " + expected1 + "]");
        System.out.println("=== Linear Layer with Bias: Native matches Java! ===");
    }

    /**
     * Test linear layer with ReLU: Y = ReLU(X @ W + B)
     * This is the pattern used in MNIST's FC layers.
     */
    @Reflect
    public static Tensor<Float> linearLayerWithRelu(Tensor<Float> x, Tensor<Float> w, Tensor<Float> b) {
        return x.matmul(w).add(b).relu();
    }

    @Test
    public void testLinearLayerWithReluNative() throws Exception {
        TosaCompiler compiler = new TosaCompiler(true);

        var method = TosaCompilerTest.class.getMethod("linearLayerWithRelu", Tensor.class, Tensor.class, Tensor.class);
        CompiledFunction fn = compiler.compile(method, 3); // rank=3

        // Input: [1, 1, 3]
        Tensor<Float> x = Tensor.ofShape(new long[]{1, 1, 3},
            1.0f, 2.0f, 3.0f);

        // Weights: [1, 3, 2]
        Tensor<Float> w = Tensor.ofShape(new long[]{1, 3, 2},
            0.1f, 0.2f,
            0.3f, 0.4f,
            0.5f, 0.6f);

        // Bias: [1, 1, 2] - one positive, one negative to test ReLU
        Tensor<Float> b = Tensor.ofShape(new long[]{1, 1, 2},
            0.5f, -3.0f);  // Output before ReLU: [2.7, -0.2], after ReLU: [2.7, 0.0]

        System.out.println("\n=== Linear Layer with ReLU Test (Native) ===");
        System.out.println("  X [1x1x3] = " + x);
        System.out.println("  W [1x3x2] = " + w);
        System.out.println("  B [1x1x2] = " + b);

        // Native result
        Tensor<Float> nativeResult = fn.invoke(x, w, b);
        System.out.println("  native Y [1x1x2] = " + nativeResult);

        // Java result
        Tensor<Float> javaResult = linearLayerWithRelu(x, w, b);
        System.out.println("  java Y [1x1x2] = " + javaResult);

        // Before ReLU: [2.2 + 0.5, 2.8 + (-3.0)] = [2.7, -0.2]
        // After ReLU:  [2.7, 0.0]
        float expected0 = 2.7f;   // positive, unchanged by ReLU
        float expected1 = 0.0f;   // negative becomes 0

        float native0 = nativeResult.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 0);
        float native1 = nativeResult.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 1);
        float java0 = javaResult.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 0);
        float java1 = javaResult.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 1);

        assertEquals(expected0, native0, 0.001f, "Native output[0] mismatch");
        assertEquals(expected1, native1, 0.001f, "Native output[1] mismatch");
        assertEquals(java0, native0, 0.001f, "Java/Native output[0] mismatch");
        assertEquals(java1, native1, 0.001f, "Java/Native output[1] mismatch");

        System.out.println("  Before ReLU: [2.7, -0.2]");
        System.out.println("  After ReLU (expected): [" + expected0 + ", " + expected1 + "]");
        System.out.println("=== Linear Layer with ReLU: Native matches Java! ===");
    }

    /**
     * Test a simple two-layer MLP: Y = ReLU(X @ W1 + B1) @ W2 + B2
     * This simulates MNIST's stacked FC layers.
     */
    @Reflect
    public static Tensor<Float> twoLayerMLP(Tensor<Float> x, Tensor<Float> w1, Tensor<Float> b1,
                                            Tensor<Float> w2, Tensor<Float> b2) {
        // Layer 1: hidden = ReLU(x @ w1 + b1)
        Tensor<Float> hidden = x.matmul(w1).add(b1).relu();
        // Layer 2: output = hidden @ w2 + b2
        return hidden.matmul(w2).add(b2);
    }

    @Test
    public void testTwoLayerMLPNative() throws Exception {
        TosaCompiler compiler = new TosaCompiler(true);

        var method = TosaCompilerTest.class.getMethod("twoLayerMLP",
            Tensor.class, Tensor.class, Tensor.class, Tensor.class, Tensor.class);
        CompiledFunction fn = compiler.compile(method, 3); // rank=3

        // Input: [1, 1, 2] - batch=1, 1 sample, 2 features
        Tensor<Float> x = Tensor.ofShape(new long[]{1, 1, 2},
            1.0f, 2.0f);

        // Layer 1 weights: [1, 2, 3] - 2 input features -> 3 hidden units
        Tensor<Float> w1 = Tensor.ofShape(new long[]{1, 2, 3},
            0.1f, 0.2f, -0.3f,   // feature 0 weights
            0.4f, -0.5f, 0.6f);  // feature 1 weights

        // Layer 1 bias: [1, 1, 3]
        Tensor<Float> b1 = Tensor.ofShape(new long[]{1, 1, 3},
            0.1f, 0.1f, 0.1f);

        // Layer 2 weights: [1, 3, 2] - 3 hidden units -> 2 output units
        Tensor<Float> w2 = Tensor.ofShape(new long[]{1, 3, 2},
            1.0f, 0.5f,
            0.5f, 1.0f,
            0.2f, 0.3f);

        // Layer 2 bias: [1, 1, 2]
        Tensor<Float> b2 = Tensor.ofShape(new long[]{1, 1, 2},
            0.0f, 0.0f);

        System.out.println("\n=== Two-Layer MLP Test (Native) ===");
        System.out.println("  X [1x1x2] = " + x);
        System.out.println("  W1 [1x2x3] = " + w1);
        System.out.println("  B1 [1x1x3] = " + b1);
        System.out.println("  W2 [1x3x2] = " + w2);
        System.out.println("  B2 [1x1x2] = " + b2);

        // Native result
        Tensor<Float> nativeResult = fn.invoke(x, w1, b1, w2, b2);
        System.out.println("  native Y [1x1x2] = " + nativeResult);

        // Java result
        Tensor<Float> javaResult = twoLayerMLP(x, w1, b1, w2, b2);
        System.out.println("  java Y [1x1x2] = " + javaResult);

        // Manual computation:
        // Layer 1: x @ w1 = [1, 2] @ [[0.1, 0.2, -0.3], [0.4, -0.5, 0.6]]
        //        = [1*0.1+2*0.4, 1*0.2+2*(-0.5), 1*(-0.3)+2*0.6]
        //        = [0.9, -0.8, 0.9]
        // + b1   = [1.0, -0.7, 1.0]
        // ReLU   = [1.0, 0.0, 1.0]
        //
        // Layer 2: hidden @ w2 = [1.0, 0.0, 1.0] @ [[1.0, 0.5], [0.5, 1.0], [0.2, 0.3]]
        //        = [1.0*1.0+0.0*0.5+1.0*0.2, 1.0*0.5+0.0*1.0+1.0*0.3]
        //        = [1.2, 0.8]
        // + b2   = [1.2, 0.8]

        float native0 = nativeResult.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 0);
        float native1 = nativeResult.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 1);
        float java0 = javaResult.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 0);
        float java1 = javaResult.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 1);

        assertEquals(java0, native0, 0.001f, "Java/Native output[0] mismatch");
        assertEquals(java1, native1, 0.001f, "Java/Native output[1] mismatch");

        System.out.println("  Computation trace:");
        System.out.println("    Layer 1: x @ w1 + b1 = [1.0, -0.7, 1.0]");
        System.out.println("    After ReLU: [1.0, 0.0, 1.0]");
        System.out.println("    Layer 2: hidden @ w2 + b2 = [1.2, 0.8]");
        System.out.println("=== Two-Layer MLP: Native matches Java! ===");
    }

    // ========== Conv2D and MaxPool2D Tests ==========

    /**
     * Simple Conv2D test function.
     * Uses TosaOperators.Conv2D with default parameters.
     */
    @Reflect
    public static Tensor<Float> simpleConv2D(Tensor<Float> input, Tensor<Float> weight, Tensor<Float> bias) {
        // Conv2D with default parameters: no padding, stride 1, dilation 1
        return TosaOperators.Conv2D(input, weight, bias,
            new long[]{0, 0, 0, 0},  // pad: top, bottom, left, right
            new long[]{1, 1},         // stride: height, width
            new long[]{1, 1});        // dilation: height, width
    }

    @Test
    public void testConv2DJava() throws Exception {
        // Test Java-side Conv2D
        // Input: [N=1, H=4, W=4, C=1] - single 4x4 image with 1 channel
        Tensor<Float> input = Tensor.ofShape(new long[]{1, 4, 4, 1},
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f, 16.0f);

        // Weight: [OC=1, KH=2, KW=2, IC=1] - single 2x2 kernel
        Tensor<Float> weight = Tensor.ofShape(new long[]{1, 2, 2, 1},
            1.0f, 0.0f,
            0.0f, 1.0f);

        // Bias: [OC=1]
        Tensor<Float> bias = Tensor.ofShape(new long[]{1}, 0.0f);

        System.out.println("\n=== Conv2D Test (Java) ===");
        System.out.println("  Input [1x4x4x1]");
        System.out.println("  Weight [1x2x2x1] (identity-like kernel)");
        System.out.println("  Bias [1] = 0");

        Tensor<Float> result = simpleConv2D(input, weight, bias);
        System.out.println("  Result shape: " + java.util.Arrays.toString(result.shape()));
        System.out.println("  Result [1x3x3x1] = " + result);

        // With a 2x2 kernel on a 4x4 input with no padding and stride 1:
        // Output is [1, 3, 3, 1]
        // The kernel [1,0;0,1] sums the diagonal elements
        // result[0,0] = 1*1 + 2*0 + 5*0 + 6*1 = 7
        // result[0,1] = 2*1 + 3*0 + 6*0 + 7*1 = 9
        // etc.
        assertEquals(7.0f, result.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 0), 0.001f);
        assertEquals(9.0f, result.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 1), 0.001f);

        System.out.println("=== Conv2D Java test passed! ===");
    }

    /**
     * Simple MaxPool2D test function.
     */
    @Reflect
    public static Tensor<Float> simpleMaxPool2D(Tensor<Float> input) {
        // MaxPool2D with 2x2 kernel, stride 2, no padding
        return TosaOperators.MaxPool2D(input,
            new long[]{2, 2},         // kernel: height, width
            new long[]{2, 2},         // stride: height, width
            new long[]{0, 0, 0, 0});  // pad: top, bottom, left, right
    }

    @Test
    public void testMaxPool2DJava() throws Exception {
        // Test Java-side MaxPool2D
        // Input: [N=1, H=4, W=4, C=1]
        Tensor<Float> input = Tensor.ofShape(new long[]{1, 4, 4, 1},
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f, 16.0f);

        System.out.println("\n=== MaxPool2D Test (Java) ===");
        System.out.println("  Input [1x4x4x1]:");
        System.out.println("    1  2  3  4");
        System.out.println("    5  6  7  8");
        System.out.println("    9 10 11 12");
        System.out.println("   13 14 15 16");

        Tensor<Float> result = simpleMaxPool2D(input);
        System.out.println("  Result shape: " + java.util.Arrays.toString(result.shape()));
        System.out.println("  Result [1x2x2x1] = " + result);

        // With 2x2 kernel and stride 2 on 4x4 input:
        // Output is [1, 2, 2, 1]
        // result[0,0] = max(1,2,5,6) = 6
        // result[0,1] = max(3,4,7,8) = 8
        // result[1,0] = max(9,10,13,14) = 14
        // result[1,1] = max(11,12,15,16) = 16
        assertEquals(6.0f, result.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 0), 0.001f);
        assertEquals(8.0f, result.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 1), 0.001f);
        assertEquals(14.0f, result.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 2), 0.001f);
        assertEquals(16.0f, result.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 3), 0.001f);

        System.out.println("  Expected: [6, 8, 14, 16]");
        System.out.println("=== MaxPool2D Java test passed! ===");
    }

    /**
     * Test Conv2D with ReLU - a common pattern in CNNs.
     */
    @Reflect
    public static Tensor<Float> conv2DWithRelu(Tensor<Float> input, Tensor<Float> weight, Tensor<Float> bias) {
        return TosaOperators.Conv2D(input, weight, bias,
            new long[]{0, 0, 0, 0},
            new long[]{1, 1},
            new long[]{1, 1}).relu();
    }

    @Test
    public void testConv2DWithReluJava() throws Exception {
        // Input: [1, 3, 3, 1]
        Tensor<Float> input = Tensor.ofShape(new long[]{1, 3, 3, 1},
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
            7.0f, 8.0f, 9.0f);

        // Weight: [1, 2, 2, 1] - kernel that produces some negative outputs
        Tensor<Float> weight = Tensor.ofShape(new long[]{1, 2, 2, 1},
            1.0f, -1.0f,
            -1.0f, 1.0f);

        // Bias: [1] - negative bias to ensure some outputs are negative before ReLU
        Tensor<Float> bias = Tensor.ofShape(new long[]{1}, -5.0f);

        System.out.println("\n=== Conv2D + ReLU Test (Java) ===");
        Tensor<Float> result = conv2DWithRelu(input, weight, bias);
        System.out.println("  Result [1x2x2x1] = " + result);

        // The kernel [1,-1;-1,1] computes: top-left + bottom-right - top-right - bottom-left
        // For a 2x2 window, this is a kind of "diagonal difference"
        // result[0,0] = 1 - 2 - 4 + 5 - 5 = -5 -> ReLU -> 0
        // result[0,1] = 2 - 3 - 5 + 6 - 5 = -5 -> ReLU -> 0
        // result[1,0] = 4 - 5 - 7 + 8 - 5 = -5 -> ReLU -> 0
        // result[1,1] = 5 - 6 - 8 + 9 - 5 = -5 -> ReLU -> 0
        // All outputs should be 0 after ReLU
        for (int i = 0; i < 4; i++) {
            float val = result.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
            assertEquals(0.0f, val, 0.001f, "Conv2D+ReLU result[" + i + "] should be 0");
        }

        System.out.println("  All values are 0 after ReLU (as expected)");
        System.out.println("=== Conv2D + ReLU Java test passed! ===");
    }

    // ========== Native Compilation Tests for Conv2D and MaxPool2D ==========

    /**
     * Test Conv2D native compilation with static shapes.
     * Conv2D requires at least partially static shapes for the TOSA-to-Linalg lowering.
     */
    @Test
    public void testConv2DNative() throws Exception {
        TosaCompiler compiler = new TosaCompiler(true);

        var method = TosaCompilerTest.class.getMethod("simpleConv2D", Tensor.class, Tensor.class, Tensor.class);

        // Define static shapes for Conv2D parameters:
        // Input: [N=1, H=4, W=4, IC=1]
        // Weight: [OC=1, KH=2, KW=2, IC=1]
        // Bias: [OC=1]
        long[][] paramShapes = {
            {1, 4, 4, 1},   // input shape
            {1, 2, 2, 1},   // weight shape
            {1}             // bias shape (1D)
        };

        CompiledFunction fn = compiler.compile(method, paramShapes);

        // Input: [N=1, H=4, W=4, C=1]
        Tensor<Float> input = Tensor.ofShape(new long[]{1, 4, 4, 1},
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f, 16.0f);

        // Weight: [OC=1, KH=2, KW=2, IC=1]
        Tensor<Float> weight = Tensor.ofShape(new long[]{1, 2, 2, 1},
            1.0f, 0.0f,
            0.0f, 1.0f);

        // Bias: [OC=1]
        Tensor<Float> bias = Tensor.ofShape(new long[]{1}, 0.0f);

        System.out.println("\n=== Conv2D Test (Native) ===");
        System.out.println("  Input [1x4x4x1]");
        System.out.println("  Weight [1x2x2x1]");
        System.out.println("  Bias [1] = 0");

        // Native result
        Tensor<Float> nativeResult = fn.invoke(input, weight, bias);
        System.out.println("  native result = " + nativeResult);

        // Java result
        Tensor<Float> javaResult = simpleConv2D(input, weight, bias);
        System.out.println("  java result = " + javaResult);

        // Verify results match
        long numElements = 1;
        for (long dim : nativeResult.shape()) numElements *= dim;

        for (int i = 0; i < numElements; i++) {
            float nativeVal = nativeResult.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
            float javaVal = javaResult.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
            assertEquals(javaVal, nativeVal, 0.001f,
                "Conv2D Native/Java mismatch at index " + i);
        }

        System.out.println("=== Conv2D: Native matches Java! ===");
    }

    @Test
    public void testMaxPool2DNative() throws Exception {
        TosaCompiler compiler = new TosaCompiler(true);

        var method = TosaCompilerTest.class.getMethod("simpleMaxPool2D", Tensor.class);
        CompiledFunction fn = compiler.compile(method, 4); // rank=4 for NHWC tensors

        // Input: [N=1, H=4, W=4, C=1]
        Tensor<Float> input = Tensor.ofShape(new long[]{1, 4, 4, 1},
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f, 16.0f);

        System.out.println("\n=== MaxPool2D Test (Native) ===");
        System.out.println("  Input [1x4x4x1]:");
        System.out.println("    1  2  3  4");
        System.out.println("    5  6  7  8");
        System.out.println("    9 10 11 12");
        System.out.println("   13 14 15 16");

        // Native result
        Tensor<Float> nativeResult = fn.invoke(input);
        System.out.println("  native result = " + nativeResult);

        // Java result
        Tensor<Float> javaResult = simpleMaxPool2D(input);
        System.out.println("  java result = " + javaResult);

        // Verify results match: [6, 8, 14, 16]
        float[] expected = {6.0f, 8.0f, 14.0f, 16.0f};
        for (int i = 0; i < expected.length; i++) {
            float nativeVal = nativeResult.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
            float javaVal = javaResult.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
            assertEquals(expected[i], nativeVal, 0.001f,
                "MaxPool2D Native expected mismatch at index " + i);
            assertEquals(javaVal, nativeVal, 0.001f,
                "MaxPool2D Native/Java mismatch at index " + i);
        }

        System.out.println("  Expected: [6, 8, 14, 16]");
        System.out.println("=== MaxPool2D: Native matches Java! ===");
    }

    @Test
    public void testConv2DWithReluNative() throws Exception {
        TosaCompiler compiler = new TosaCompiler(true);

        var method = TosaCompilerTest.class.getMethod("conv2DWithRelu", Tensor.class, Tensor.class, Tensor.class);

        // Define static shapes for Conv2D+ReLU parameters
        long[][] paramShapes = {
            {1, 3, 3, 1},   // input shape
            {1, 2, 2, 1},   // weight shape
            {1}             // bias shape (1D)
        };

        CompiledFunction fn = compiler.compile(method, paramShapes);

        // Input: [1, 3, 3, 1]
        Tensor<Float> input = Tensor.ofShape(new long[]{1, 3, 3, 1},
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
            7.0f, 8.0f, 9.0f);

        // Weight: [1, 2, 2, 1]
        Tensor<Float> weight = Tensor.ofShape(new long[]{1, 2, 2, 1},
            1.0f, -1.0f,
            -1.0f, 1.0f);

        // Bias: [1]
        Tensor<Float> bias = Tensor.ofShape(new long[]{1}, -5.0f);

        System.out.println("\n=== Conv2D + ReLU Test (Native) ===");

        // Native result
        Tensor<Float> nativeResult = fn.invoke(input, weight, bias);
        System.out.println("  native result = " + nativeResult);

        // Java result
        Tensor<Float> javaResult = conv2DWithRelu(input, weight, bias);
        System.out.println("  java result = " + javaResult);

        // All outputs should be 0 after ReLU
        for (int i = 0; i < 4; i++) {
            float nativeVal = nativeResult.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
            float javaVal = javaResult.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
            assertEquals(0.0f, nativeVal, 0.001f, "Conv2D+ReLU Native result[" + i + "] should be 0");
            assertEquals(javaVal, nativeVal, 0.001f, "Conv2D+ReLU Native/Java mismatch at index " + i);
        }

        System.out.println("  All values are 0 after ReLU");
        System.out.println("=== Conv2D + ReLU: Native matches Java! ===");
    }

    // ========== MNIST-Style Tests ==========

    /**
     * MNIST-style CNN layer: Conv2D(5x5) -> ReLU -> MaxPool2D(2x2)
     * This is the pattern used in the ONNX MNIST model's first conv block.
     *
     * Note: ONNX uses NCHW format, TOSA uses NHWC format.
     * The operations are mathematically equivalent but with different data layouts.
     */
    @Reflect
    public static Tensor<Float> mnistConvBlock(Tensor<Float> input, Tensor<Float> weight, Tensor<Float> bias) {
        // Conv2D with 5x5 kernel, no padding, stride 1
        Tensor<Float> conv = TosaOperators.Conv2D(input, weight, bias,
            new long[]{0, 0, 0, 0},  // no padding
            new long[]{1, 1},         // stride 1
            new long[]{1, 1});        // dilation 1

        // ReLU activation
        Tensor<Float> relu = conv.relu();

        // MaxPool2D with 2x2 kernel, stride 2
        return TosaOperators.MaxPool2D(relu,
            new long[]{2, 2},         // kernel size
            new long[]{2, 2},         // stride
            new long[]{0, 0, 0, 0}); // no padding
    }

    // Helper to create Float array from pattern
    private static Float[] generatePattern(int size, java.util.function.IntFunction<Float> generator) {
        Float[] data = new Float[size];
        for (int i = 0; i < size; i++) {
            data[i] = generator.apply(i);
        }
        return data;
    }

    @Test
    public void testMnistConvBlockJava() throws Exception {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("MNIST-Style Conv Block Test (Java)");
        System.out.println("Pattern: Conv2D(5x5) -> ReLU -> MaxPool2D(2x2)");
        System.out.println("=".repeat(70));

        // Input: [N=1, H=10, W=10, C=1] - small image for testing
        Float[] inputData = generatePattern(100, i -> (float)(i % 10));
        Tensor<Float> input = Tensor.ofShape(new long[]{1, 10, 10, 1}, inputData);

        // Weight: [OC=2, KH=5, KW=5, IC=1] - two 5x5 filters
        Float[] weightData = new Float[50];
        // Filter 1: detects edges (Sobel-like)
        for (int i = 0; i < 25; i++) {
            weightData[i] = (i < 12) ? -1.0f : 1.0f;
        }
        // Filter 2: simple averaging
        for (int i = 25; i < 50; i++) {
            weightData[i] = 1.0f / 25.0f;
        }
        Tensor<Float> weight = Tensor.ofShape(new long[]{2, 5, 5, 1}, weightData);

        // Bias: [OC=2]
        Tensor<Float> bias = Tensor.ofShape(new long[]{2}, 0.0f, 0.0f);

        System.out.println("\nInput [1x10x10x1]:");
        System.out.println("  0 1 2 3 4 5 6 7 8 9");
        System.out.println("  0 1 2 3 4 5 6 7 8 9");
        System.out.println("  ... (repeating pattern)");

        System.out.println("\nFilters:");
        System.out.println("  Filter 1: Edge detector (5x5)");
        System.out.println("  Filter 2: Averaging filter (5x5)");

        Tensor<Float> result = mnistConvBlock(input, weight, bias);

        System.out.println("\nResult shape: " + java.util.Arrays.toString(result.shape()));
        // Conv2D(10x10, 5x5 kernel, no padding) -> 6x6
        // MaxPool2D(6x6, 2x2 kernel, stride 2) -> 3x3
        // Output: [1, 3, 3, 2]

        // Print output values
        System.out.println("\nOutput values [1x3x3x2]:");
        for (int h = 0; h < 3; h++) {
            System.out.print("  Row " + h + ": ");
            for (int w = 0; w < 3; w++) {
                for (int c = 0; c < 2; c++) {
                    int idx = h * 3 * 2 + w * 2 + c;
                    float val = result.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, idx);
                    System.out.printf("[%.2f] ", val);
                }
                System.out.print("| ");
            }
            System.out.println();
        }

        assertNotNull(result);
        assertArrayEquals(new long[]{1, 3, 3, 2}, result.shape());
        System.out.println("\n=== MNIST Conv Block Java test passed! ===");
    }

    @Test
    public void testMnistConvBlockNative() throws Exception {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("MNIST-Style Conv Block Test (Native Compilation)");
        System.out.println("Comparing TOSA JIT vs Java Reference Implementation");
        System.out.println("=".repeat(70));

        TosaCompiler compiler = new TosaCompiler(true);

        var method = TosaCompilerTest.class.getMethod("mnistConvBlock", Tensor.class, Tensor.class, Tensor.class);

        // Static shapes for compilation
        long[][] paramShapes = {
            {1, 10, 10, 1},  // input: [N, H, W, IC]
            {2, 5, 5, 1},    // weight: [OC, KH, KW, IC]
            {2}              // bias: [OC]
        };

        CompiledFunction fn = compiler.compile(method, paramShapes);

        // Create test data (same as Java test)
        Float[] inputData = generatePattern(100, i -> (float)(i % 10));
        Tensor<Float> input = Tensor.ofShape(new long[]{1, 10, 10, 1}, inputData);

        Float[] weightData = new Float[50];
        for (int i = 0; i < 25; i++) {
            weightData[i] = (i < 12) ? -1.0f : 1.0f;
        }
        for (int i = 25; i < 50; i++) {
            weightData[i] = 1.0f / 25.0f;
        }
        Tensor<Float> weight = Tensor.ofShape(new long[]{2, 5, 5, 1}, weightData);

        Tensor<Float> bias = Tensor.ofShape(new long[]{2}, 0.0f, 0.0f);

        // Run native
        Tensor<Float> nativeResult = fn.invoke(input, weight, bias);

        // Run Java
        Tensor<Float> javaResult = mnistConvBlock(input, weight, bias);

        System.out.println("\nResults Comparison:");
        System.out.println("  Native shape: " + java.util.Arrays.toString(nativeResult.shape()));
        System.out.println("  Java shape:   " + java.util.Arrays.toString(javaResult.shape()));

        // Compare all values
        long numElements = 1;
        for (long dim : nativeResult.shape()) numElements *= dim;

        System.out.println("\nValue-by-value comparison:");
        for (int i = 0; i < numElements; i++) {
            float nativeVal = nativeResult.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
            float javaVal = javaResult.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
            String match = Math.abs(nativeVal - javaVal) < 0.001f ? "MATCH" : "DIFFER";
            if (i < 6) { // Print first few
                System.out.printf("  [%d] Native=%.4f, Java=%.4f (%s)%n", i, nativeVal, javaVal, match);
            }
        }
        if (numElements > 6) {
            System.out.println("  ... (showing first 6 of " + numElements + " values)");
        }

        // Verify all match
        for (int i = 0; i < numElements; i++) {
            float nativeVal = nativeResult.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
            float javaVal = javaResult.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
            assertEquals(javaVal, nativeVal, 0.001f,
                "MNIST Conv Block Native/Java mismatch at index " + i);
        }

        System.out.println("\n" + "=".repeat(70));
        System.out.println("SUCCESS: Native TOSA compilation matches Java reference!");
        System.out.println("This demonstrates that TOSA operations (Conv2D, ReLU, MaxPool2D)");
        System.out.println("produce identical results when compiled to native x86_64 code.");
        System.out.println("=".repeat(70));
    }

    /**
     * Test a complete forward pass similar to MNIST's feature extraction.
     * This chains two conv blocks like the MNIST model.
     */
    @Reflect
    public static Tensor<Float> mnistFeatureExtractor(
            Tensor<Float> input,
            Tensor<Float> conv1Weight, Tensor<Float> conv1Bias,
            Tensor<Float> conv2Weight, Tensor<Float> conv2Bias) {

        // First conv block: Conv2D(3x3) -> ReLU -> MaxPool2D(2x2)
        Tensor<Float> conv1 = TosaOperators.Conv2D(input, conv1Weight, conv1Bias,
            new long[]{0, 0, 0, 0}, new long[]{1, 1}, new long[]{1, 1});
        Tensor<Float> relu1 = conv1.relu();
        Tensor<Float> pool1 = TosaOperators.MaxPool2D(relu1,
            new long[]{2, 2}, new long[]{2, 2}, new long[]{0, 0, 0, 0});

        // Second conv block: Conv2D(3x3) -> ReLU -> MaxPool2D(2x2)
        Tensor<Float> conv2 = TosaOperators.Conv2D(pool1, conv2Weight, conv2Bias,
            new long[]{0, 0, 0, 0}, new long[]{1, 1}, new long[]{1, 1});
        Tensor<Float> relu2 = conv2.relu();
        Tensor<Float> pool2 = TosaOperators.MaxPool2D(relu2,
            new long[]{2, 2}, new long[]{2, 2}, new long[]{0, 0, 0, 0});

        return pool2;
    }

    @Test
    public void testMnistFeatureExtractorJava() throws Exception {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("MNIST Feature Extractor Test (Two Conv Blocks)");
        System.out.println("=".repeat(70));

        // Input: [1, 16, 16, 1]
        Float[] inputData = generatePattern(256, i -> (float)(i % 16) / 16.0f);
        Tensor<Float> input = Tensor.ofShape(new long[]{1, 16, 16, 1}, inputData);

        // Conv1: [2, 3, 3, 1] - 2 filters, 3x3 kernel
        Float[] conv1Weight = new Float[18];
        for (int i = 0; i < 9; i++) conv1Weight[i] = 0.1f; // Filter 1
        for (int i = 9; i < 18; i++) conv1Weight[i] = -0.1f; // Filter 2
        Tensor<Float> conv1W = Tensor.ofShape(new long[]{2, 3, 3, 1}, conv1Weight);
        Tensor<Float> conv1B = Tensor.ofShape(new long[]{2}, 0.0f, 0.0f);

        // Conv2: [4, 3, 3, 2] - 4 filters, 3x3 kernel, 2 input channels
        Float[] conv2Weight = new Float[72];
        for (int i = 0; i < 72; i++) conv2Weight[i] = (i % 2 == 0) ? 0.1f : -0.1f;
        Tensor<Float> conv2W = Tensor.ofShape(new long[]{4, 3, 3, 2}, conv2Weight);
        Tensor<Float> conv2B = Tensor.ofShape(new long[]{4}, 0.0f, 0.0f, 0.0f, 0.0f);

        // Forward pass:
        // Input: [1, 16, 16, 1]
        // Conv1(3x3): [1, 14, 14, 2]
        // Pool1(2x2): [1, 7, 7, 2]
        // Conv2(3x3): [1, 5, 5, 4]
        // Pool2(2x2): [1, 2, 2, 4]
        Tensor<Float> result = mnistFeatureExtractor(input, conv1W, conv1B, conv2W, conv2B);

        System.out.println("\nForward pass trace:");
        System.out.println("  Input:      [1, 16, 16, 1]");
        System.out.println("  After Conv1: [1, 14, 14, 2]  (3x3 kernel, 2 filters)");
        System.out.println("  After Pool1: [1, 7, 7, 2]    (2x2 max pool, stride 2)");
        System.out.println("  After Conv2: [1, 5, 5, 4]    (3x3 kernel, 4 filters)");
        System.out.println("  After Pool2: [1, 2, 2, 4]    (2x2 max pool, stride 2)");
        System.out.println("\nResult shape: " + java.util.Arrays.toString(result.shape()));
        System.out.println("Result: " + result);

        assertArrayEquals(new long[]{1, 2, 2, 4}, result.shape());
        System.out.println("\n=== MNIST Feature Extractor Java test passed! ===");
    }
}
