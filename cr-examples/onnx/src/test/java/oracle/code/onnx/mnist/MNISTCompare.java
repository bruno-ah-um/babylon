package oracle.code.onnx.mnist;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
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
 * Simple test to run ONNX MNIST and print results for comparison with TOSA.
 * Uses the same test patterns as the TOSA MNISTComparisonTest.
 */
public class MNISTCompare {

    static final int IMAGE_SIZE = 28;

    public static void main(String[] args) throws IOException {
        System.out.println("=".repeat(80));
        System.out.println("ONNX MNIST Model Results");
        System.out.println("=".repeat(80));

        // Create test patterns (same as TOSA test)
        float[][] testImages = createTestImages();
        String[] testNames = {"Zero (circle)", "One (vertical line)", "Diagonal line", "Random noise"};

        try (Arena arena = Arena.ofConfined()) {
            // Load the weights (use simple name since getResource is relative to this class)
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

            System.out.println("ONNX model loaded successfully!\n");

            for (int t = 0; t < testImages.length; t++) {
                System.out.println("-".repeat(60));
                System.out.println("Test " + (t + 1) + ": " + testNames[t]);
                System.out.println("-".repeat(60));

                float[] imageData = testImages[t];

                // Create input tensor with NCHW format [1, 1, 28, 28]
                var imageTensor = Tensor.ofShape(arena, new long[]{1, 1, IMAGE_SIZE, IMAGE_SIZE}, imageData);

                // Run ONNX model using code reflection pattern
                var predictionTensor = OnnxRuntime.execute(arena, MethodHandles.lookup(),
                        (@Reflect Supplier<Tensor<Float>>) () -> cnn(
                            conv1Weight, conv1Bias,
                            conv2Weight, conv2Bias,
                            fc1Weight, fc1Bias,
                            fc2Weight, fc2Bias,
                            fc3Weight, fc3Bias,
                            imageTensor));

                // Convert to probabilities
                float[] onnxProbs = predictionTensor.data().toArray(ValueLayout.JAVA_FLOAT);

                // Print results
                System.out.println("\nONNX Predictions:");
                int onnxMaxIdx = printProbabilities(onnxProbs);
                System.out.printf("ONNX predicted digit: %d (confidence: %.2f%%)%n",
                    onnxMaxIdx, onnxProbs[onnxMaxIdx] * 100);

                // Verify probabilities sum to ~1
                float sum = 0;
                for (float p : onnxProbs) sum += p;
                System.out.printf("Probability sum: %.6f (should be ~1.0)%n", sum);
                System.out.println();
            }
        }
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

        // Scaling to 0-1
        var scaledInput = Div(inputImage, Constant(255f));

        // First conv layer
        var conv1 = Conv(scaledInput, conv1Weights, of(conv1Biases), of(new long[4]),
                of(new long[]{1,1}), empty(), of(new long[]{1, 1, 1, 1}),
                of(1L), of(new long[]{5,5}));
        var relu1 = Relu(conv1);

        // First pooling layer
        var pool1 = MaxPool(relu1, of(new long[4]), of(new long[]{1,1}), empty(),
                of(0L), empty(), of(new long[]{2, 2}), new long[]{2, 2});

        // Second conv layer
        var conv2 = Conv(pool1.Y(), conv2Weights, of(conv2Biases), of(new long[4]),
                of(new long[]{1,1}), empty(), of(new long[]{1, 1, 1, 1}),
                of(1L), of(new long[]{5,5}));
        var relu2 = Relu(conv2);

        // Second pooling layer
        var pool2 = MaxPool(relu2, of(new long[4]), of(new long[]{1,1}), empty(),
                of(0L), empty(), of(new long[]{2, 2}), new long[]{2, 2});

        // Flatten inputs
        var flatten = Flatten(pool2.Y(), of(1L));

        // First fully connected layer
        var fc1 = Gemm(flatten, fc1Weights, of(fc1Biases), of(1f), of(1L), of(1f), empty());
        var relu3 = Relu(fc1);

        // Second fully connected layer
        var fc2 = Gemm(relu3, fc2Weights, of(fc2Biases), of(1f), of(1L), of(1f), empty());
        var relu4 = Relu(fc2);

        // Softmax layer
        var fc3 = Gemm(relu4, fc3Weights, of(fc3Biases), of(1f), of(1L), of(1f), empty());
        var prediction = Softmax(fc3, of(1L));

        return prediction;
    }

    private static Tensor<Float> floatTensor(Arena arena, String resource, long... shape) throws IOException {
        try (var file = new RandomAccessFile(MNISTCompare.class.getResource(resource).getPath(), "r")) {
            return new Tensor(arena, file.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, file.length(), arena), Tensor.ElementType.FLOAT, shape);
        }
    }

    private static int printProbabilities(float[] probs) {
        int maxIdx = 0;
        float maxProb = 0;
        for (int i = 0; i < probs.length; i++) {
            System.out.printf("  Digit %d: %.6f%n", i, probs[i]);
            if (probs[i] > maxProb) {
                maxProb = probs[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    private static float[][] createTestImages() {
        float[][] images = new float[4][IMAGE_SIZE * IMAGE_SIZE];

        // Test 1: Circle pattern (digit 0)
        for (int y = 0; y < IMAGE_SIZE; y++) {
            for (int x = 0; x < IMAGE_SIZE; x++) {
                float dx = x - 14;
                float dy = y - 14;
                float dist = (float) Math.sqrt(dx * dx + dy * dy);
                if (dist > 8 && dist < 12) {
                    images[0][y * IMAGE_SIZE + x] = 255.0f;
                }
            }
        }

        // Test 2: Vertical line (digit 1)
        for (int y = 4; y < 24; y++) {
            images[1][y * IMAGE_SIZE + 14] = 255.0f;
            images[1][y * IMAGE_SIZE + 13] = 128.0f;
            images[1][y * IMAGE_SIZE + 15] = 128.0f;
        }

        // Test 3: Diagonal line
        for (int i = 4; i < 24; i++) {
            images[2][i * IMAGE_SIZE + i] = 255.0f;
            if (i > 0) images[2][i * IMAGE_SIZE + i - 1] = 128.0f;
            if (i < IMAGE_SIZE - 1) images[2][i * IMAGE_SIZE + i + 1] = 128.0f;
        }

        // Test 4: Random noise (for consistency check)
        java.util.Random rand = new java.util.Random(42); // Fixed seed for reproducibility
        for (int i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
            images[3][i] = rand.nextFloat() * 255.0f;
        }

        return images;
    }
}
