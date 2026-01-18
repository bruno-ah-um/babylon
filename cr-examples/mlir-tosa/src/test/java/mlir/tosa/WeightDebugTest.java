package mlir.tosa;

import java.io.IOException;
import java.io.UncheckedIOException;

/**
 * Debug test to verify weight loading and transposition.
 */
public class WeightDebugTest {

    private static Tensor<Float> load(String resource, long... shape) {
        try (var in = WeightDebugTest.class.getResourceAsStream(resource)) {
            if (in == null) {
                throw new IllegalStateException("Resource not found: " + resource);
            }
            return Tensor.ofBytes(shape, in.readAllBytes());
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    public static void main(String[] args) {
        // Load raw weights (ONNX format: OIHW)
        Tensor<Float> conv1Raw = load("mnist/conv1-weight-float-le", 6, 1, 5, 5);

        System.out.println("Conv1 weights shape: " + java.util.Arrays.toString(conv1Raw.shape()));
        System.out.println("First 10 raw values (OIHW format):");
        for (int i = 0; i < 10; i++) {
            float val = conv1Raw.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
            System.out.printf("  [%d] = %.6f%n", i, val);
        }

        // Compare with expected fc3 weights shape
        Tensor<Float> fc3Raw = load("mnist/fc3-weight-float-le", 10, 84);
        System.out.println("\nFC3 weights shape: " + java.util.Arrays.toString(fc3Raw.shape()));
        System.out.println("First 10 fc3 values:");
        for (int i = 0; i < 10; i++) {
            float val = fc3Raw.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
            System.out.printf("  [%d] = %.6f%n", i, val);
        }

        // Now test the model with a simple input (all zeros)
        System.out.println("\n--- Testing with zero input ---");
        MNISTModel model = new MNISTModel();

        float[] zeroInput = new float[28 * 28];  // all zeros
        float[] probs = model.classify(zeroInput);
        System.out.println("Predictions for zero input:");
        for (int i = 0; i < 10; i++) {
            System.out.printf("  Digit %d: %.6f%n", i, probs[i]);
        }

        // Test with all-white input
        System.out.println("\n--- Testing with all-white input ---");
        float[] whiteInput = new float[28 * 28];
        java.util.Arrays.fill(whiteInput, 255.0f);
        probs = model.classify(whiteInput);
        System.out.println("Predictions for all-white input:");
        for (int i = 0; i < 10; i++) {
            System.out.printf("  Digit %d: %.6f%n", i, probs[i]);
        }

        // Test with a simple single pixel at center
        System.out.println("\n--- Testing with single center pixel ---");
        float[] centerPixel = new float[28 * 28];
        centerPixel[14 * 28 + 14] = 255.0f;  // center pixel
        probs = model.classify(centerPixel);
        System.out.println("Predictions for center pixel:");
        for (int i = 0; i < 10; i++) {
            System.out.printf("  Digit %d: %.6f%n", i, probs[i]);
        }
    }
}
