package mlir.tosa;

import org.junit.jupiter.api.Test;
import java.lang.invoke.MethodHandles;
import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;

import static mlir.tosa.TosaOperators.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Simple tests for TOSA operations via code reflection.
 *
 * These tests demonstrate the proof of concept for defining tensor operations
 * in Java and transforming them to MLIR TOSA operations.
 */
public class TosaSimpleTest {

    /**
     * Test TOSA Add operation with 1D float tensors
     */
    @Test
    public void testAdd1D() {
        var a = Tensor.ofFlat(1.0f, 2.0f, 3.0f);
        var b = Tensor.ofFlat(4.0f, 5.0f, 6.0f);

        var result = Add(a, b);

        var expected = Tensor.ofFlat(5.0f, 7.0f, 9.0f);
        assertEquals(expected, result);
    }

    /**
     * Test TOSA Mul operation with 1D float tensors
     */
    @Test
    public void testMul1D() {
        var a = Tensor.ofFlat(2.0f, 3.0f, 4.0f);
        var b = Tensor.ofFlat(5.0f, 6.0f, 7.0f);

        var result = Mul(a, b);

        var expected = Tensor.ofFlat(10.0f, 18.0f, 28.0f);
        assertEquals(expected, result);
    }

    /**
     * Test TOSA Add with scalar tensors
     */
    @Test
    public void testAddScalar() {
        var a = Tensor.ofScalar(3.5f);
        var b = Tensor.ofScalar(2.5f);

        var result = Add(a, b);

        var expected = Tensor.ofScalar(6.0f);
        assertEquals(expected, result);
    }

    /**
     * Test TOSA Mul with scalar tensors
     */
    @Test
    public void testMulScalar() {
        var a = Tensor.ofScalar(3.0f);
        var b = Tensor.ofScalar(4.0f);

        var result = Mul(a, b);

        var expected = Tensor.ofScalar(12.0f);
        assertEquals(expected, result);
    }

    /**
     * Test TOSA Add with 2D tensors
     */
    @Test
    public void testAdd2D() {
        // 2x3 tensor
        var a = Tensor.ofShape(new long[]{2, 3},
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f
        );
        var b = Tensor.ofShape(new long[]{2, 3},
            0.5f, 1.0f, 1.5f,
            2.0f, 2.5f, 3.0f
        );

        var result = Add(a, b);

        var expected = Tensor.ofShape(new long[]{2, 3},
            1.5f, 3.0f, 4.5f,
            6.0f, 7.5f, 9.0f
        );
        assertEquals(expected, result);
    }

    /**
     * Test TOSA Mul with 2D tensors
     */
    @Test
    public void testMul2D() {
        // 2x2 tensor
        var a = Tensor.ofShape(new long[]{2, 2},
            2.0f, 3.0f,
            4.0f, 5.0f
        );
        var b = Tensor.ofShape(new long[]{2, 2},
            1.0f, 2.0f,
            3.0f, 4.0f
        );

        var result = Mul(a, b);

        var expected = Tensor.ofShape(new long[]{2, 2},
            2.0f, 6.0f,
            12.0f, 20.0f
        );
        assertEquals(expected, result);
    }

    /**
     * Test with integer tensors
     */
    @Test
    public void testAddInt() {
        var a = Tensor.ofFlat(1, 2, 3);
        var b = Tensor.ofFlat(4, 5, 6);

        var result = Add(a, b);

        var expected = Tensor.ofFlat(5, 7, 9);
        assertEquals(expected, result);
    }

    /**
     * Test with double tensors
     */
    @Test
    public void testMulDouble() {
        var a = Tensor.ofFlat(1.5, 2.5, 3.5);
        var b = Tensor.ofFlat(2.0, 3.0, 4.0);

        var result = Mul(a, b);

        var expected = Tensor.ofFlat(3.0, 7.5, 14.0);
        assertEquals(expected, result);
    }

    /**
     * Test combined operations
     */
    @Test
    public void testCombinedOperations() {
        var a = Tensor.ofFlat(1.0f, 2.0f, 3.0f);
        var b = Tensor.ofFlat(2.0f, 3.0f, 4.0f);
        var c = Tensor.ofFlat(0.5f, 1.0f, 1.5f);

        // Compute (a + b) * c
        var sum = Add(a, b);       // [3.0, 5.0, 7.0]
        var result = Mul(sum, c);  // [1.5, 5.0, 10.5]

        var expected = Tensor.ofFlat(1.5f, 5.0f, 10.5f);
        assertEquals(expected, result);
    }

    /**
     * Test fluent API - instance methods
     */
    @Test
    public void testFluentAPI() {
        var a = Tensor.ofFlat(1.0f, 2.0f, 3.0f);
        var b = Tensor.ofFlat(2.0f, 3.0f, 4.0f);

        // Using fluent API: a.add(b) instead of Add(a, b)
        var sum = a.add(b);
        var expected = Tensor.ofFlat(3.0f, 5.0f, 7.0f);
        assertEquals(expected, sum);

        // Using fluent API: a.mul(b)
        var product = a.mul(b);
        expected = Tensor.ofFlat(2.0f, 6.0f, 12.0f);
        assertEquals(expected, product);
    }

    /**
     * Test fluent API with method chaining
     */
    @Test
    public void testFluentAPIChaining() {
        var a = Tensor.ofFlat(1.0f, 2.0f, 3.0f);
        var b = Tensor.ofFlat(2.0f, 3.0f, 4.0f);
        var c = Tensor.ofFlat(0.5f, 1.0f, 1.5f);

        // Compute (a + b) * c using fluent API
        var result = a.add(b).mul(c);  // Chain operations

        var expected = Tensor.ofFlat(1.5f, 5.0f, 10.5f);
        assertEquals(expected, result);
    }

    /**
     * Test shape validation (should fail on mismatched shapes)
     */
    @Test
    public void testShapeMismatch() {
        var a = Tensor.ofFlat(1.0f, 2.0f, 3.0f);
        var b = Tensor.ofFlat(4.0f, 5.0f);  // Different shape

        assertThrows(IllegalArgumentException.class, () -> Add(a, b));
    }

    /**
     * Test type promotion: adding a FLOAT32 tensor and an INT32 tensor should succeed
     * by implicitly promoting INT32 → FLOAT32 before the operation.
     */
    @Test
    @SuppressWarnings({"rawtypes", "unchecked"})
    public void testTypePromotion() {
        Tensor a = Tensor.ofFlat(1.0f, 2.0f, 3.0f);  // FLOAT32
        Tensor b = Tensor.ofFlat(4, 5, 6);            // INT32 → promoted to FLOAT32

        Tensor result = Add(a, b);

        assertEquals(Tensor.ElementType.FLOAT32, result.elementType());
        assertEquals(Tensor.ofFlat(5.0f, 7.0f, 9.0f), result);
    }

    /**
     * Test with @Reflect annotation using code reflection
     */
    @Reflect
    public Tensor<Float> computeExpression(Tensor<Float> a, Tensor<Float> b, Tensor<Float> c) {
        // This code is captured via code reflection
        return Add(Mul(a, b), c);
    }

    @Test
    @Reflect
    public void testCodeReflection() {
        var a = Tensor.ofFlat(1.0f, 2.0f);
        var b = Tensor.ofFlat(3.0f, 4.0f);
        var c = Tensor.ofFlat(5.0f, 6.0f);

        // Execute via TosaTransformer which captures the lambda
        var result = TosaTransformer.execute(
            MethodHandles.lookup(),
            () -> computeExpression(a, b, c)
        );

        var expected = Tensor.ofFlat(8.0f, 14.0f);  // (1*3)+5, (2*4)+6
        assertEquals(expected, result);
    }

    /**
     * Test code reflection with fluent API
     */
    @Reflect
    public Tensor<Float> fluentExpression(Tensor<Float> a, Tensor<Float> b, Tensor<Float> c) {
        // Using fluent API with code reflection
        return a.mul(b).add(c);
    }

    @Test
    @Reflect
    public void testCodeReflectionFluent() {
        var a = Tensor.ofFlat(2.0f, 3.0f);
        var b = Tensor.ofFlat(4.0f, 5.0f);
        var c = Tensor.ofFlat(1.0f, 2.0f);

        var result = TosaTransformer.execute(
            MethodHandles.lookup(),
            () -> fluentExpression(a, b, c)
        );

        var expected = Tensor.ofFlat(9.0f, 17.0f);  // (2*4)+1, (3*5)+2
        assertEquals(expected, result);
    }

    /**
     * Test analyzing quoted code
     *
     * NOTE: Disabled for now - inline lambdas like () -> a.add(b) cannot be captured
     * unless the method being called is marked with @Reflect.
     */
    // @Test
    // public void testAnalyzeQuoted() {
    //     var a = Tensor.ofFlat(1.0f, 2.0f);
    //     var b = Tensor.ofFlat(3.0f, 4.0f);
    //     var c = Tensor.ofFlat(1.0f, 1.0f);
    //
    //     // Call a @Reflect method instead
    //     var quoted = Op.ofQuotable(() -> computeExpression(a, b, c)).orElseThrow(() ->
    //         new IllegalArgumentException("Lambda is not quotable")
    //     );
    //
    //     // Analyze the captured operation tree
    //     String analysis = TosaTransformer.analyzeQuoted(quoted);
    //
    //     assertNotNull(analysis);
    //     assertTrue(analysis.contains("Operation tree analysis"));
    // }
}
