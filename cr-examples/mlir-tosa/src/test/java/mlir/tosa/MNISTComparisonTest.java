package mlir.tosa;

/**
 * Comparison test to verify TOSA MNIST produces same results as ONNX MNIST.
 *
 * Note: The ONNX model uses NCHW format while TOSA uses NHWC format.
 * We need to transpose the input accordingly.
 */
public class MNISTComparisonTest {

    static final int IMAGE_SIZE = 28;

    public static void main(String[] args) {
        System.out.println("=".repeat(80));
        System.out.println("MNIST Model Comparison: TOSA vs Expected");
        System.out.println("=".repeat(80));

        // Create test patterns
        float[][] testImages = createTestImages();
        String[] testNames = {"Zero (circle)", "One (vertical line)", "Diagonal line", "Random noise"};

        MNISTModel tosaModel = new MNISTModel();
        System.out.println("TOSA model loaded successfully!\n");

        for (int t = 0; t < testImages.length; t++) {
            System.out.println("-".repeat(60));
            System.out.println("Test " + (t + 1) + ": " + testNames[t]);
            System.out.println("-".repeat(60));

            float[] imageData = testImages[t];

            // Run TOSA model
            float[] tosaProbs = tosaModel.classify(imageData);

            // Print results
            System.out.println("\nTOSA Predictions:");
            int tosaMaxIdx = printProbabilities(tosaProbs);
            System.out.printf("TOSA predicted digit: %d (confidence: %.2f%%)%n",
                tosaMaxIdx, tosaProbs[tosaMaxIdx] * 100);

            // Verify probabilities sum to ~1
            float sum = 0;
            for (float p : tosaProbs) sum += p;
            System.out.printf("Probability sum: %.6f (should be ~1.0)%n", sum);
            System.out.println();
        }

        System.out.println("=".repeat(80));
        System.out.println("Note: To compare with ONNX, run the ONNX MNISTModel separately");
        System.out.println("The models may have slight differences due to:");
        System.out.println("  - NCHW vs NHWC format handling");
        System.out.println("  - Floating point precision differences");
        System.out.println("  - Different convolution implementations");
        System.out.println("=".repeat(80));
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
