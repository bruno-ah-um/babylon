package mlir.tosa;

import jdk.incubator.code.Reflect;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test comparing TOSA operations with expected ONNX-equivalent results.
 *
 * Key difference: ONNX uses NCHW format, TOSA uses NHWC format.
 * This test verifies that given equivalent inputs (with transposed data),
 * the mathematical results are identical.
 */
public class TosaOnnxComparisonTest {

    /**
     * Transpose a 4D tensor from NCHW to NHWC format.
     * NCHW: [batch, channels, height, width]
     * NHWC: [batch, height, width, channels]
     */
    private static Float[] nchwToNhwc(Float[] nchw, int n, int c, int h, int w) {
        Float[] nhwc = new Float[nchw.length];
        for (int batch = 0; batch < n; batch++) {
            for (int channel = 0; channel < c; channel++) {
                for (int row = 0; row < h; row++) {
                    for (int col = 0; col < w; col++) {
                        int nchwIdx = batch * c * h * w + channel * h * w + row * w + col;
                        int nhwcIdx = batch * h * w * c + row * w * c + col * c + channel;
                        nhwc[nhwcIdx] = nchw[nchwIdx];
                    }
                }
            }
        }
        return nhwc;
    }

    /**
     * Transpose Conv2D weights from OIHW (ONNX) to OHWI (TOSA) format.
     * OIHW: [out_channels, in_channels, kernel_h, kernel_w]
     * OHWI: [out_channels, kernel_h, kernel_w, in_channels]
     */
    private static Float[] oihwToOhwi(Float[] oihw, int oc, int ic, int kh, int kw) {
        Float[] ohwi = new Float[oihw.length];
        for (int o = 0; o < oc; o++) {
            for (int i = 0; i < ic; i++) {
                for (int h = 0; h < kh; h++) {
                    for (int w = 0; w < kw; w++) {
                        int oihwIdx = o * ic * kh * kw + i * kh * kw + h * kw + w;
                        int ohwiIdx = o * kh * kw * ic + h * kw * ic + w * ic + i;
                        ohwi[ohwiIdx] = oihw[oihwIdx];
                    }
                }
            }
        }
        return ohwi;
    }

    /**
     * Transpose output from NHWC back to NCHW for comparison.
     */
    private static Float[] nhwcToNchw(float[] nhwc, int n, int h, int w, int c) {
        Float[] nchw = new Float[nhwc.length];
        for (int batch = 0; batch < n; batch++) {
            for (int channel = 0; channel < c; channel++) {
                for (int row = 0; row < h; row++) {
                    for (int col = 0; col < w; col++) {
                        int nhwcIdx = batch * h * w * c + row * w * c + col * c + channel;
                        int nchwIdx = batch * c * h * w + channel * h * w + row * w + col;
                        nchw[nchwIdx] = nhwc[nhwcIdx];
                    }
                }
            }
        }
        return nchw;
    }

    /**
     * Simple Conv2D operation for testing.
     */
    @Reflect
    public static Tensor<Float> conv2d(Tensor<Float> input, Tensor<Float> weight, Tensor<Float> bias) {
        return TosaOperators.Conv2D(input, weight, bias,
            new long[]{0, 0, 0, 0},  // no padding
            new long[]{1, 1},         // stride 1
            new long[]{1, 1});        // dilation 1
    }

    @Test
    public void testConv2DMatchesOnnxReference() throws Exception {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("TOSA vs ONNX Conv2D Comparison Test");
        System.out.println("=".repeat(70));

        // Define input in ONNX's NCHW format: [1, 1, 4, 4]
        // A simple 4x4 single-channel image
        Float[] inputNchw = new Float[] {
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f, 16.0f
        };

        // Define weights in ONNX's OIHW format: [1, 1, 2, 2]
        // A simple 2x2 kernel that sums diagonals
        Float[] weightOihw = new Float[] {
            1.0f, 0.0f,
            0.0f, 1.0f
        };

        // Bias: [1]
        Float[] bias = new Float[] { 0.0f };

        // Expected ONNX output in NCHW format: [1, 1, 3, 3]
        // Computed manually: kernel [1,0;0,1] sums diagonal elements
        // out[0,0] = 1*1 + 2*0 + 5*0 + 6*1 = 7
        // out[0,1] = 2*1 + 3*0 + 6*0 + 7*1 = 9
        // out[0,2] = 3*1 + 4*0 + 7*0 + 8*1 = 11
        // out[1,0] = 5*1 + 6*0 + 9*0 + 10*1 = 15
        // out[1,1] = 6*1 + 7*0 + 10*0 + 11*1 = 17
        // out[1,2] = 7*1 + 8*0 + 11*0 + 12*1 = 19
        // out[2,0] = 9*1 + 10*0 + 13*0 + 14*1 = 23
        // out[2,1] = 10*1 + 11*0 + 14*0 + 15*1 = 25
        // out[2,2] = 11*1 + 12*0 + 15*0 + 16*1 = 27
        float[] expectedNchw = {7, 9, 11, 15, 17, 19, 23, 25, 27};

        // Convert to TOSA's NHWC format
        Float[] inputNhwc = nchwToNhwc(inputNchw, 1, 1, 4, 4);
        Float[] weightOhwi = oihwToOhwi(weightOihw, 1, 1, 2, 2);

        System.out.println("\nInput (NCHW [1,1,4,4]):");
        System.out.println("  1  2  3  4");
        System.out.println("  5  6  7  8");
        System.out.println("  9 10 11 12");
        System.out.println(" 13 14 15 16");

        System.out.println("\nWeight (OIHW [1,1,2,2]):");
        System.out.println("  1 0");
        System.out.println("  0 1");

        // Create TOSA tensors
        Tensor<Float> input = Tensor.ofShape(new long[]{1, 4, 4, 1}, inputNhwc);
        Tensor<Float> weight = Tensor.ofShape(new long[]{1, 2, 2, 1}, weightOhwi);
        Tensor<Float> biasTensor = Tensor.ofShape(new long[]{1}, bias);

        // Run TOSA Conv2D
        Tensor<Float> result = conv2d(input, weight, biasTensor);

        System.out.println("\nTOSA output shape: " + java.util.Arrays.toString(result.shape()));

        // Extract and transpose result back to NCHW for comparison
        float[] resultNhwcRaw = new float[9];
        for (int i = 0; i < 9; i++) {
            resultNhwcRaw[i] = result.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
        }
        Float[] resultNchw = nhwcToNchw(resultNhwcRaw, 1, 3, 3, 1);

        System.out.println("\nExpected output (NCHW [1,1,3,3]):");
        System.out.println("   7  9 11");
        System.out.println("  15 17 19");
        System.out.println("  23 25 27");

        System.out.println("\nTOSA output (converted to NCHW):");
        for (int row = 0; row < 3; row++) {
            System.out.print("  ");
            for (int col = 0; col < 3; col++) {
                System.out.printf("%2.0f ", resultNchw[row * 3 + col]);
            }
            System.out.println();
        }

        // Verify results match
        System.out.println("\nComparison:");
        boolean allMatch = true;
        for (int i = 0; i < 9; i++) {
            float expected = expectedNchw[i];
            float actual = resultNchw[i];
            String status = Math.abs(expected - actual) < 0.001f ? "MATCH" : "DIFFER";
            if (!status.equals("MATCH")) allMatch = false;
            System.out.printf("  [%d] Expected=%.1f, TOSA=%.1f (%s)%n", i, expected, actual, status);
        }

        assertTrue(allMatch, "TOSA Conv2D should match ONNX reference");

        System.out.println("\n" + "=".repeat(70));
        System.out.println("SUCCESS: TOSA Conv2D produces same results as ONNX reference!");
        System.out.println("The data format transposition (NCHW<->NHWC) is handled correctly.");
        System.out.println("=".repeat(70));
    }

    @Test
    public void testConv2DNativeMatchesOnnxReference() throws Exception {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("TOSA Native vs ONNX Conv2D Comparison Test");
        System.out.println("=".repeat(70));

        TosaCompiler compiler = new TosaCompiler(false);
        var method = TosaOnnxComparisonTest.class.getMethod("conv2d", Tensor.class, Tensor.class, Tensor.class);

        long[][] paramShapes = {
            {1, 4, 4, 1},  // input NHWC
            {1, 2, 2, 1},  // weight OHWI
            {1}            // bias
        };

        CompiledFunction fn = compiler.compile(method, paramShapes);

        // Same test data as above
        Float[] inputNchw = new Float[] {
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f, 16.0f
        };
        Float[] weightOihw = new Float[] { 1.0f, 0.0f, 0.0f, 1.0f };
        Float[] bias = new Float[] { 0.0f };
        float[] expectedNchw = {7, 9, 11, 15, 17, 19, 23, 25, 27};

        // Convert to TOSA format
        Float[] inputNhwc = nchwToNhwc(inputNchw, 1, 1, 4, 4);
        Float[] weightOhwi = oihwToOhwi(weightOihw, 1, 1, 2, 2);

        Tensor<Float> input = Tensor.ofShape(new long[]{1, 4, 4, 1}, inputNhwc);
        Tensor<Float> weight = Tensor.ofShape(new long[]{1, 2, 2, 1}, weightOhwi);
        Tensor<Float> biasTensor = Tensor.ofShape(new long[]{1}, bias);

        // Run TOSA Native
        Tensor<Float> nativeResult = fn.invoke(input, weight, biasTensor);

        // Run TOSA Java
        Tensor<Float> javaResult = conv2d(input, weight, biasTensor);

        // Extract results
        float[] nativeNhwc = new float[9];
        float[] javaNhwc = new float[9];
        for (int i = 0; i < 9; i++) {
            nativeNhwc[i] = nativeResult.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
            javaNhwc[i] = javaResult.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
        }

        Float[] nativeNchw = nhwcToNchw(nativeNhwc, 1, 3, 3, 1);
        Float[] javaNchw = nhwcToNchw(javaNhwc, 1, 3, 3, 1);

        System.out.println("\nThree-way comparison: ONNX Reference vs TOSA Java vs TOSA Native");
        System.out.println("-".repeat(60));

        boolean allMatch = true;
        for (int i = 0; i < 9; i++) {
            float expected = expectedNchw[i];
            float java = javaNchw[i];
            float native_ = nativeNchw[i];

            boolean match = Math.abs(expected - java) < 0.001f && Math.abs(expected - native_) < 0.001f;
            if (!match) allMatch = false;

            System.out.printf("  [%d] ONNX=%.1f, Java=%.1f, Native=%.1f [%s]%n",
                i, expected, java, native_, match ? "ALL MATCH" : "DIFFER");
        }

        assertTrue(allMatch, "All three implementations should produce same results");

        System.out.println("\n" + "=".repeat(70));
        System.out.println("SUCCESS: ONNX Reference == TOSA Java == TOSA Native");
        System.out.println("All three produce identical results for Conv2D operation.");
        System.out.println("=".repeat(70));
    }

    @Test
    public void testMaxPool2DMatchesOnnxReference() throws Exception {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("TOSA vs ONNX MaxPool2D Comparison Test");
        System.out.println("=".repeat(70));

        // Input in NCHW: [1, 1, 4, 4]
        Float[] inputNchw = new Float[] {
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f, 16.0f
        };

        // Expected MaxPool2D output with 2x2 kernel, stride 2: [1, 1, 2, 2]
        // Window [0:2, 0:2] -> max(1,2,5,6) = 6
        // Window [0:2, 2:4] -> max(3,4,7,8) = 8
        // Window [2:4, 0:2] -> max(9,10,13,14) = 14
        // Window [2:4, 2:4] -> max(11,12,15,16) = 16
        float[] expectedNchw = {6, 8, 14, 16};

        // Convert to NHWC
        Float[] inputNhwc = nchwToNhwc(inputNchw, 1, 1, 4, 4);

        System.out.println("\nInput (NCHW [1,1,4,4]):");
        System.out.println("  1  2  3  4");
        System.out.println("  5  6  7  8");
        System.out.println("  9 10 11 12");
        System.out.println(" 13 14 15 16");

        Tensor<Float> input = Tensor.ofShape(new long[]{1, 4, 4, 1}, inputNhwc);

        // Run TOSA MaxPool2D
        Tensor<Float> result = TosaOperators.MaxPool2D(input,
            new long[]{2, 2},         // kernel
            new long[]{2, 2},         // stride
            new long[]{0, 0, 0, 0});  // pad

        // Extract and convert to NCHW
        float[] resultNhwc = new float[4];
        for (int i = 0; i < 4; i++) {
            resultNhwc[i] = result.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
        }
        Float[] resultNchw = nhwcToNchw(resultNhwc, 1, 2, 2, 1);

        System.out.println("\nExpected output (NCHW [1,1,2,2]):");
        System.out.println("   6  8");
        System.out.println("  14 16");

        System.out.println("\nTOSA output (converted to NCHW):");
        System.out.printf("  %2.0f %2.0f%n", resultNchw[0], resultNchw[1]);
        System.out.printf("  %2.0f %2.0f%n", resultNchw[2], resultNchw[3]);

        // Verify
        for (int i = 0; i < 4; i++) {
            assertEquals(expectedNchw[i], resultNchw[i], 0.001f,
                "MaxPool2D result mismatch at index " + i);
        }

        System.out.println("\n" + "=".repeat(70));
        System.out.println("SUCCESS: TOSA MaxPool2D produces same results as ONNX reference!");
        System.out.println("=".repeat(70));
    }
}
