package oracle.code.onnx.mnist;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.function.Supplier;

import jdk.incubator.code.Reflect;
import oracle.code.onnx.OnnxRuntime;
import oracle.code.onnx.Tensor;

import static java.util.Optional.empty;
import static java.util.Optional.of;
import static oracle.code.onnx.OnnxOperators.*;

/**
 * Debug test to verify ONNX model behavior with simple inputs.
 */
public class WeightDebugTest {

    static final int IMAGE_SIZE = 28;

    public static void main(String[] args) throws IOException {
        try (Arena arena = Arena.ofConfined()) {
            // Load the weights
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

            System.out.println("Conv1 weight shape: " + Arrays.toString(conv1Weight.shape()));
            System.out.println("First 10 conv1 weight values:");
            for (int i = 0; i < 10; i++) {
                float val = conv1Weight.data().getAtIndex(ValueLayout.JAVA_FLOAT, i);
                System.out.printf("  [%d] = %.6f%n", i, val);
            }

            System.out.println("\nFC3 weight shape: " + Arrays.toString(fc3Weight.shape()));
            System.out.println("First 10 fc3 weight values:");
            for (int i = 0; i < 10; i++) {
                float val = fc3Weight.data().getAtIndex(ValueLayout.JAVA_FLOAT, i);
                System.out.printf("  [%d] = %.6f%n", i, val);
            }

            // Test with zero input
            System.out.println("\n--- Testing with zero input ---");
            float[] zeroInput = new float[IMAGE_SIZE * IMAGE_SIZE];
            float[] probs = runModel(arena, zeroInput,
                    conv1Weight, conv1Bias, conv2Weight, conv2Bias,
                    fc1Weight, fc1Bias, fc2Weight, fc2Bias, fc3Weight, fc3Bias);
            System.out.println("Predictions for zero input:");
            for (int i = 0; i < 10; i++) {
                System.out.printf("  Digit %d: %.6f%n", i, probs[i]);
            }

            // Test with all-white input
            System.out.println("\n--- Testing with all-white input ---");
            float[] whiteInput = new float[IMAGE_SIZE * IMAGE_SIZE];
            Arrays.fill(whiteInput, 255.0f);
            probs = runModel(arena, whiteInput,
                    conv1Weight, conv1Bias, conv2Weight, conv2Bias,
                    fc1Weight, fc1Bias, fc2Weight, fc2Bias, fc3Weight, fc3Bias);
            System.out.println("Predictions for all-white input:");
            for (int i = 0; i < 10; i++) {
                System.out.printf("  Digit %d: %.6f%n", i, probs[i]);
            }

            // Test with single center pixel
            System.out.println("\n--- Testing with single center pixel ---");
            float[] centerPixel = new float[IMAGE_SIZE * IMAGE_SIZE];
            centerPixel[14 * 28 + 14] = 255.0f;
            probs = runModel(arena, centerPixel,
                    conv1Weight, conv1Bias, conv2Weight, conv2Bias,
                    fc1Weight, fc1Bias, fc2Weight, fc2Bias, fc3Weight, fc3Bias);
            System.out.println("Predictions for center pixel:");
            for (int i = 0; i < 10; i++) {
                System.out.printf("  Digit %d: %.6f%n", i, probs[i]);
            }
        }
    }

    private static float[] runModel(Arena arena, float[] imageData,
            Tensor<Float> conv1Weight, Tensor<Float> conv1Bias,
            Tensor<Float> conv2Weight, Tensor<Float> conv2Bias,
            Tensor<Float> fc1Weight, Tensor<Float> fc1Bias,
            Tensor<Float> fc2Weight, Tensor<Float> fc2Bias,
            Tensor<Float> fc3Weight, Tensor<Float> fc3Bias) {

        var imageTensor = Tensor.ofShape(arena, new long[]{1, 1, IMAGE_SIZE, IMAGE_SIZE}, imageData);

        var predictionTensor = OnnxRuntime.execute(arena, MethodHandles.lookup(),
                (@Reflect Supplier<Tensor<Float>>) () -> cnn(
                    conv1Weight, conv1Bias,
                    conv2Weight, conv2Bias,
                    fc1Weight, fc1Bias,
                    fc2Weight, fc2Bias,
                    fc3Weight, fc3Bias,
                    imageTensor));

        return predictionTensor.data().toArray(ValueLayout.JAVA_FLOAT);
    }

    @Reflect
    public static Tensor<Float> cnn(
            Tensor<Float> conv1Weights,
            Tensor<Float> conv1Biases,
            Tensor<Float> conv2Weights,
            Tensor<Float> conv2Biases,
            Tensor<Float> fc1Weights,
            Tensor<Float> fc1Biases,
            Tensor<Float> fc2Weights,
            Tensor<Float> fc2Biases,
            Tensor<Float> fc3Weights,
            Tensor<Float> fc3Biases,
            Tensor<Float> inputImage) {

        var scaledInput = Div(inputImage, Constant(255f));
        var conv1 = Conv(scaledInput, conv1Weights, of(conv1Biases), of(new long[4]),
                of(new long[]{1,1}), empty(), of(new long[]{1, 1, 1, 1}),
                of(1L), of(new long[]{5,5}));
        var relu1 = Relu(conv1);
        var pool1 = MaxPool(relu1, of(new long[4]), of(new long[]{1,1}), empty(),
                of(0L), empty(), of(new long[]{2, 2}), new long[]{2, 2});
        var conv2 = Conv(pool1.Y(), conv2Weights, of(conv2Biases), of(new long[4]),
                of(new long[]{1,1}), empty(), of(new long[]{1, 1, 1, 1}),
                of(1L), of(new long[]{5,5}));
        var relu2 = Relu(conv2);
        var pool2 = MaxPool(relu2, of(new long[4]), of(new long[]{1,1}), empty(),
                of(0L), empty(), of(new long[]{2, 2}), new long[]{2, 2});
        var flatten = Flatten(pool2.Y(), of(1L));
        var fc1 = Gemm(flatten, fc1Weights, of(fc1Biases), of(1f), of(1L), of(1f), empty());
        var relu3 = Relu(fc1);
        var fc2 = Gemm(relu3, fc2Weights, of(fc2Biases), of(1f), of(1L), of(1f), empty());
        var relu4 = Relu(fc2);
        var fc3 = Gemm(relu4, fc3Weights, of(fc3Biases), of(1f), of(1L), of(1f), empty());
        var prediction = Softmax(fc3, of(1L));

        return prediction;
    }

    private static Tensor<Float> floatTensor(Arena arena, String resource, long... shape) throws IOException {
        try (var file = new RandomAccessFile(WeightDebugTest.class.getResource(resource).getPath(), "r")) {
            return new Tensor(arena, file.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, file.length(), arena), Tensor.ElementType.FLOAT, shape);
        }
    }
}
