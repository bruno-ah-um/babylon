package oracle.code.onnx.mnist;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;
import java.nio.channels.FileChannel;
import java.util.function.Supplier;

import jdk.incubator.code.Reflect;
import oracle.code.onnx.OnnxRuntime;
import oracle.code.onnx.Tensor;

import static java.util.Optional.empty;
import static java.util.Optional.of;
import static oracle.code.onnx.OnnxOperators.*;

/**
 * Debug test to check intermediate convolution results in ONNX.
 */
public class DebugConv {

    static final int IMAGE_SIZE = 28;

    public static void main(String[] args) throws IOException {
        System.out.println("=".repeat(80));
        System.out.println("ONNX Debug Conv - Step by step");
        System.out.println("=".repeat(80));

        try (Arena arena = Arena.ofConfined()) {
            // Load conv1 weights (ONNX format: OIHW)
            var conv1Weight = floatTensor(arena, "conv1-weight-float-le", 6, 1, 5, 5);
            var conv1Bias = floatTensor(arena, "conv1-bias-float-le", 6);

            System.out.println("\nConv1 weights shape (OIHW): [6, 1, 5, 5]");
            System.out.println("Conv1 weights (first 9 values):");
            for (int i = 0; i < 9; i++) {
                float val = conv1Weight.data().getAtIndex(ValueLayout.JAVA_FLOAT, i);
                System.out.printf("  [%d] = %.6f%n", i, val);
            }

            // Create a simple test input: 4x4 patch of 255s in center
            float[] inputData = new float[IMAGE_SIZE * IMAGE_SIZE];
            for (int y = 12; y < 16; y++) {
                for (int x = 12; x < 16; x++) {
                    inputData[y * IMAGE_SIZE + x] = 255.0f;
                }
            }

            // Create NCHW tensor [1, 1, 28, 28]
            var input = Tensor.ofShape(arena, new long[]{1, 1, IMAGE_SIZE, IMAGE_SIZE}, inputData);
            System.out.println("\nInput shape: [1, 1, 28, 28] (NCHW)");

            // Check input values at center
            System.out.println("Input values at center (should be 255):");
            for (int y = 12; y < 14; y++) {
                for (int x = 12; x < 14; x++) {
                    // NCHW indexing: n*C*H*W + c*H*W + y*W + x
                    long idx = 0 * 1 * 28 * 28 + 0 * 28 * 28 + y * 28 + x;
                    float val = input.data().getAtIndex(ValueLayout.JAVA_FLOAT, idx);
                    System.out.printf("  [%d,%d] = %.1f%n", y, x, val);
                }
            }

            // Run just conv1 to see intermediate result
            var conv1Out = OnnxRuntime.execute(arena, MethodHandles.lookup(),
                    (@Reflect Supplier<Tensor<Float>>) () -> justConv1(input, conv1Weight, conv1Bias));

            System.out.println("\nConv1 output shape: " + java.util.Arrays.toString(conv1Out.shape()));

            // Print conv1 output at center (y=10, x=10) for all 6 channels
            // NCHW: index = n*C*H*W + c*H*W + h*W + w
            // Output is [1, 6, 24, 24]
            System.out.println("Conv1 output at center (y=10, x=10, all 6 channels):");
            for (int c = 0; c < 6; c++) {
                long idx = 0 * 6 * 24 * 24 + c * 24 * 24 + 10 * 24 + 10;
                float val = conv1Out.data().getAtIndex(ValueLayout.JAVA_FLOAT, idx);
                System.out.printf("  Channel %d: %.6f%n", c, val);
            }

            // Stats
            long numElems = 1 * 6 * 24 * 24;
            float sum = 0, min = Float.MAX_VALUE, max = Float.MIN_VALUE;
            int numPositive = 0, numNegative = 0;
            for (long i = 0; i < numElems; i++) {
                float val = conv1Out.data().getAtIndex(ValueLayout.JAVA_FLOAT, i);
                sum += val;
                min = Math.min(min, val);
                max = Math.max(max, val);
                if (val > 0) numPositive++;
                else if (val < 0) numNegative++;
            }
            System.out.printf("\nConv1 output stats: sum=%.2f, min=%.4f, max=%.4f, +ve=%d, -ve=%d%n",
                    sum, min, max, numPositive, numNegative);
        }
    }

    @Reflect
    public static Tensor<Float> justConv1(Tensor<Float> input, Tensor<Float> conv1Weight, Tensor<Float> conv1Bias) {
        // Scale to 0-1
        var scaled = Div(input, Constant(255f));
        // Conv with no padding, stride 1, dilation 1
        var conv1 = Conv(scaled, conv1Weight, of(conv1Bias), of(new long[4]),
                of(new long[]{1,1}), empty(), of(new long[]{1, 1, 1, 1}),
                of(1L), of(new long[]{5,5}));
        return conv1;
    }

    private static Tensor<Float> floatTensor(Arena arena, String resource, long... shape) throws IOException {
        try (var file = new RandomAccessFile(DebugConv.class.getResource(resource).getPath(), "r")) {
            return new Tensor(arena, file.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, file.length(), arena), Tensor.ElementType.FLOAT, shape);
        }
    }
}
