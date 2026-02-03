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
 * Debug test to check each layer output in ONNX.
 */
public class DebugFull {

    static final int IMAGE_SIZE = 28;

    private static void printStats(String name, Tensor<Float> tensor) {
        long[] shape = tensor.shape();
        long numElems = 1;
        for (long s : shape) numElems *= s;

        float sum = 0, min = Float.MAX_VALUE, max = Float.MIN_VALUE;
        for (long i = 0; i < numElems; i++) {
            float val = tensor.data().getAtIndex(ValueLayout.JAVA_FLOAT, i);
            sum += val;
            min = Math.min(min, val);
            max = Math.max(max, val);
        }
        System.out.printf("%s: shape=%s, sum=%.4f, min=%.4f, max=%.4f%n",
                name, java.util.Arrays.toString(shape), sum, min, max);
    }

    private static void printFirst10(String name, Tensor<Float> tensor) {
        System.out.printf("%s first 10: ", name);
        for (int i = 0; i < 10; i++) {
            float val = tensor.data().getAtIndex(ValueLayout.JAVA_FLOAT, i);
            System.out.printf("%.4f ", val);
        }
        System.out.println();
    }

    public static void main(String[] args) throws IOException {
        System.out.println("=".repeat(80));
        System.out.println("ONNX Debug Full Pipeline");
        System.out.println("=".repeat(80));

        try (Arena arena = Arena.ofConfined()) {
            var conv1Weight = floatTensor(arena, "conv1-weight-float-le", 6, 1, 5, 5);
            var conv1Bias = floatTensor(arena, "conv1-bias-float-le", 6);
            var conv2Weight = floatTensor(arena, "conv2-weight-float-le", 16, 6, 5, 5);
            var conv2Bias = floatTensor(arena, "conv2-bias-float-le", 16);
            var fc1Weight = floatTensor(arena, "fc1-weight-float-le", 120, 256);
            var fc1Bias = floatTensor(arena, "fc1-bias-float-le", 120);
            var fc2Weight = floatTensor(arena, "fc2-weight-float-le", 84, 120);
            var fc2Bias = floatTensor(arena, "fc2-bias-float-le", 84);
            var fc3Weight = floatTensor(arena, "fc3-weight-float-le", 10, 84);
            var fc3Bias = floatTensor(arena, "fc3-bias-float-le", 10);

            // Create test input: circle pattern (same as TOSA)
            float[] inputData = new float[IMAGE_SIZE * IMAGE_SIZE];
            for (int y = 0; y < IMAGE_SIZE; y++) {
                for (int x = 0; x < IMAGE_SIZE; x++) {
                    float dx = x - 14;
                    float dy = y - 14;
                    float dist = (float) Math.sqrt(dx * dx + dy * dy);
                    if (dist > 8 && dist < 12) {
                        inputData[y * IMAGE_SIZE + x] = 255.0f;
                    }
                }
            }

            // NCHW format [1, 1, 28, 28]
            var input = Tensor.ofShape(arena, new long[]{1, 1, IMAGE_SIZE, IMAGE_SIZE}, inputData);
            System.out.println("Input: shape=[1, 1, 28, 28], sum=61200.0000");

            // Step through each layer
            System.out.println("\n--- Layer by layer ---");

            // Scaled
            var scaled = OnnxRuntime.execute(arena, MethodHandles.lookup(),
                    (@Reflect Supplier<Tensor<Float>>) () -> Div(input, Constant(255f)));
            printStats("Scaled", scaled);

            // Conv1
            var conv1 = OnnxRuntime.execute(arena, MethodHandles.lookup(),
                    (@Reflect Supplier<Tensor<Float>>) () ->
                        Conv(Div(input, Constant(255f)), conv1Weight, of(conv1Bias), of(new long[4]),
                                of(new long[]{1,1}), empty(), of(new long[]{1, 1, 1, 1}),
                                of(1L), of(new long[]{5,5})));
            printStats("Conv1", conv1);

            // ReLU1
            var relu1 = OnnxRuntime.execute(arena, MethodHandles.lookup(),
                    (@Reflect Supplier<Tensor<Float>>) () ->
                        Relu(Conv(Div(input, Constant(255f)), conv1Weight, of(conv1Bias), of(new long[4]),
                                of(new long[]{1,1}), empty(), of(new long[]{1, 1, 1, 1}),
                                of(1L), of(new long[]{5,5}))));
            printStats("ReLU1", relu1);

            // Pool1 + subsequent layers
            var pool1 = OnnxRuntime.execute(arena, MethodHandles.lookup(),
                    (@Reflect Supplier<Tensor<Float>>) () ->
                        MaxPool(Relu(Conv(Div(input, Constant(255f)), conv1Weight, of(conv1Bias), of(new long[4]),
                                of(new long[]{1,1}), empty(), of(new long[]{1, 1, 1, 1}),
                                of(1L), of(new long[]{5,5}))),
                            of(new long[4]), of(new long[]{1,1}), empty(),
                            of(0L), empty(), of(new long[]{2, 2}), new long[]{2, 2}).Y());
            printStats("Pool1", pool1);

            // Full forward pass to get flatten output
            var flatten = OnnxRuntime.execute(arena, MethodHandles.lookup(),
                    (@Reflect Supplier<Tensor<Float>>) () -> {
                        var s = Div(input, Constant(255f));
                        var c1 = Conv(s, conv1Weight, of(conv1Bias), of(new long[4]),
                                of(new long[]{1,1}), empty(), of(new long[]{1, 1, 1, 1}),
                                of(1L), of(new long[]{5,5}));
                        var r1 = Relu(c1);
                        var p1 = MaxPool(r1, of(new long[4]), of(new long[]{1,1}), empty(),
                                of(0L), empty(), of(new long[]{2, 2}), new long[]{2, 2});
                        var c2 = Conv(p1.Y(), conv2Weight, of(conv2Bias), of(new long[4]),
                                of(new long[]{1,1}), empty(), of(new long[]{1, 1, 1, 1}),
                                of(1L), of(new long[]{5,5}));
                        var r2 = Relu(c2);
                        var p2 = MaxPool(r2, of(new long[4]), of(new long[]{1,1}), empty(),
                                of(0L), empty(), of(new long[]{2, 2}), new long[]{2, 2});
                        return Flatten(p2.Y(), of(1L));
                    });
            printStats("Flatten", flatten);
            printFirst10("Flatten", flatten);

            // Full CNN for logits
            var logits = OnnxRuntime.execute(arena, MethodHandles.lookup(),
                    (@Reflect Supplier<Tensor<Float>>) () -> {
                        var s = Div(input, Constant(255f));
                        var c1 = Conv(s, conv1Weight, of(conv1Bias), of(new long[4]),
                                of(new long[]{1,1}), empty(), of(new long[]{1, 1, 1, 1}),
                                of(1L), of(new long[]{5,5}));
                        var r1 = Relu(c1);
                        var p1 = MaxPool(r1, of(new long[4]), of(new long[]{1,1}), empty(),
                                of(0L), empty(), of(new long[]{2, 2}), new long[]{2, 2});
                        var c2 = Conv(p1.Y(), conv2Weight, of(conv2Bias), of(new long[4]),
                                of(new long[]{1,1}), empty(), of(new long[]{1, 1, 1, 1}),
                                of(1L), of(new long[]{5,5}));
                        var r2 = Relu(c2);
                        var p2 = MaxPool(r2, of(new long[4]), of(new long[]{1,1}), empty(),
                                of(0L), empty(), of(new long[]{2, 2}), new long[]{2, 2});
                        var f = Flatten(p2.Y(), of(1L));
                        var fc1 = Gemm(f, fc1Weight, of(fc1Bias), of(1f), of(1L), of(1f), empty());
                        var r3 = Relu(fc1);
                        var fc2 = Gemm(r3, fc2Weight, of(fc2Bias), of(1f), of(1L), of(1f), empty());
                        var r4 = Relu(fc2);
                        return Gemm(r4, fc3Weight, of(fc3Bias), of(1f), of(1L), of(1f), empty());
                    });
            printStats("FC3 (logits)", logits);
            printFirst10("FC3 (logits)", logits);

            // Full CNN with softmax
            var probs = OnnxRuntime.execute(arena, MethodHandles.lookup(),
                    (@Reflect Supplier<Tensor<Float>>) () -> {
                        var s = Div(input, Constant(255f));
                        var c1 = Conv(s, conv1Weight, of(conv1Bias), of(new long[4]),
                                of(new long[]{1,1}), empty(), of(new long[]{1, 1, 1, 1}),
                                of(1L), of(new long[]{5,5}));
                        var r1 = Relu(c1);
                        var p1 = MaxPool(r1, of(new long[4]), of(new long[]{1,1}), empty(),
                                of(0L), empty(), of(new long[]{2, 2}), new long[]{2, 2});
                        var c2 = Conv(p1.Y(), conv2Weight, of(conv2Bias), of(new long[4]),
                                of(new long[]{1,1}), empty(), of(new long[]{1, 1, 1, 1}),
                                of(1L), of(new long[]{5,5}));
                        var r2 = Relu(c2);
                        var p2 = MaxPool(r2, of(new long[4]), of(new long[]{1,1}), empty(),
                                of(0L), empty(), of(new long[]{2, 2}), new long[]{2, 2});
                        var f = Flatten(p2.Y(), of(1L));
                        var fc1 = Gemm(f, fc1Weight, of(fc1Bias), of(1f), of(1L), of(1f), empty());
                        var r3 = Relu(fc1);
                        var fc2 = Gemm(r3, fc2Weight, of(fc2Bias), of(1f), of(1L), of(1f), empty());
                        var r4 = Relu(fc2);
                        var fc3 = Gemm(r4, fc3Weight, of(fc3Bias), of(1f), of(1L), of(1f), empty());
                        return Softmax(fc3, of(1L));
                    });
            printStats("Softmax", probs);
            System.out.println("Final probabilities:");
            for (int i = 0; i < 10; i++) {
                float p = probs.data().getAtIndex(ValueLayout.JAVA_FLOAT, i);
                System.out.printf("  Digit %d: %.6f%n", i, p);
            }
        }
    }

    private static Tensor<Float> floatTensor(Arena arena, String resource, long... shape) throws IOException {
        try (var file = new RandomAccessFile(DebugFull.class.getResource(resource).getPath(), "r")) {
            return new Tensor(arena, file.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, file.length(), arena), Tensor.ElementType.FLOAT, shape);
        }
    }
}
