package mlir.tosa;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.ValueLayout;

import static mlir.tosa.TosaOperators.*;

/**
 * Debug test to check intermediate convolution results.
 */
public class DebugConv {

    public static final int IMAGE_SIZE = 28;

    private static Tensor<Float> load(String resource, long... shape) {
        try (var in = DebugConv.class.getResourceAsStream(resource)) {
            if (in == null) {
                throw new IllegalStateException("Resource not found: " + resource);
            }
            return Tensor.ofBytes(shape, in.readAllBytes());
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * Transpose conv weights from OIHW (ONNX) to OHWI (TOSA).
     */
    private static Tensor<Float> transposeConvWeights(Tensor<Float> weights, long oc, long ic, long kh, long kw) {
        float[] srcData = new float[(int)(oc * ic * kh * kw)];
        float[] dstData = new float[(int)(oc * ic * kh * kw)];

        for (int i = 0; i < srcData.length; i++) {
            srcData[i] = weights.data().getAtIndex(ValueLayout.JAVA_FLOAT, i);
        }

        // Transpose from [OC, IC, KH, KW] to [OC, KH, KW, IC]
        for (int o = 0; o < oc; o++) {
            for (int i = 0; i < ic; i++) {
                for (int h = 0; h < kh; h++) {
                    for (int w = 0; w < kw; w++) {
                        int srcIdx = (int)(o * ic * kh * kw + i * kh * kw + h * kw + w);
                        int dstIdx = (int)(o * kh * kw * ic + h * kw * ic + w * ic + i);
                        dstData[dstIdx] = srcData[srcIdx];
                    }
                }
            }
        }

        return Tensor.ofFloats(new long[]{oc, kh, kw, ic}, dstData);
    }

    public static void main(String[] args) {
        System.out.println("=".repeat(80));
        System.out.println("Debug Conv - Step by step");
        System.out.println("=".repeat(80));

        // Load conv1 weights (OIHW -> OHWI)
        Tensor<Float> conv1WeightsRaw = load("mnist/conv1-weight-float-le", 6, 1, 5, 5);
        Tensor<Float> conv1Weights = transposeConvWeights(conv1WeightsRaw, 6, 1, 5, 5);
        Tensor<Float> conv1Bias = load("mnist/conv1-bias-float-le", 6);

        System.out.println("\nConv1 weights shape (transposed): " + java.util.Arrays.toString(conv1Weights.shape()));
        System.out.println("Conv1 bias shape: " + java.util.Arrays.toString(conv1Bias.shape()));

        // Print some conv1 weights
        System.out.println("\nConv1 weights (first filter, first 9 values in OHWI order):");
        for (int i = 0; i < 9; i++) {
            float val = conv1Weights.data().getAtIndex(ValueLayout.JAVA_FLOAT, i);
            System.out.printf("  [%d] = %.6f%n", i, val);
        }

        // Create a simple test input: 3x3 patch of 255s in center
        float[] inputData = new float[IMAGE_SIZE * IMAGE_SIZE];
        for (int y = 12; y < 16; y++) {
            for (int x = 12; x < 16; x++) {
                inputData[y * IMAGE_SIZE + x] = 255.0f;
            }
        }

        // Create NHWC tensor [1, 28, 28, 1]
        Tensor<Float> input = Tensor.ofFloats(new long[]{1, IMAGE_SIZE, IMAGE_SIZE, 1}, inputData);
        System.out.println("\nInput shape: " + java.util.Arrays.toString(input.shape()));

        // Scale to 0-1
        Tensor<Float> scaled = input.mul(Constant(1.0f / 255.0f));
        System.out.println("Scaled input (center region, should be ~1.0):");
        for (int y = 12; y < 16; y++) {
            for (int x = 12; x < 16; x++) {
                long idx = y * IMAGE_SIZE * 1 + x * 1 + 0;  // NHWC indexing
                float val = scaled.data().getAtIndex(ValueLayout.JAVA_FLOAT, idx);
                System.out.printf("  [%d,%d] = %.6f%n", y, x, val);
            }
        }

        // Run Conv2D
        Tensor<Float> conv1Out = Conv2D(scaled, conv1Weights, conv1Bias,
                new long[]{0, 0, 0, 0},  // no padding
                new long[]{1, 1},         // stride 1
                new long[]{1, 1});        // dilation 1

        System.out.println("\nConv1 output shape: " + java.util.Arrays.toString(conv1Out.shape()));

        // Print some conv1 output values at center
        System.out.println("Conv1 output at center region (y=10, x=10, all 6 channels):");
        for (int c = 0; c < 6; c++) {
            // NHWC: index = n*H*W*C + h*W*C + w*C + c
            long idx = 0 * 24 * 24 * 6 + 10 * 24 * 6 + 10 * 6 + c;
            float val = conv1Out.data().getAtIndex(ValueLayout.JAVA_FLOAT, idx);
            System.out.printf("  Channel %d: %.6f%n", c, val);
        }

        // Run ReLU
        Tensor<Float> relu1 = conv1Out.relu();
        System.out.println("\nAfter ReLU (same location):");
        for (int c = 0; c < 6; c++) {
            long idx = 0 * 24 * 24 * 6 + 10 * 24 * 6 + 10 * 6 + c;
            float val = relu1.data().getAtIndex(ValueLayout.JAVA_FLOAT, idx);
            System.out.printf("  Channel %d: %.6f%n", c, val);
        }

        // Sum of all conv1 output values
        long numElems = 1 * 24 * 24 * 6;
        float sum = 0, min = Float.MAX_VALUE, max = Float.MIN_VALUE;
        int numPositive = 0, numNegative = 0, numZero = 0;
        for (long i = 0; i < numElems; i++) {
            float val = conv1Out.data().getAtIndex(ValueLayout.JAVA_FLOAT, i);
            sum += val;
            min = Math.min(min, val);
            max = Math.max(max, val);
            if (val > 0) numPositive++;
            else if (val < 0) numNegative++;
            else numZero++;
        }
        System.out.printf("\nConv1 output stats: sum=%.2f, min=%.4f, max=%.4f, +ve=%d, -ve=%d, zero=%d%n",
                sum, min, max, numPositive, numNegative, numZero);
    }
}
