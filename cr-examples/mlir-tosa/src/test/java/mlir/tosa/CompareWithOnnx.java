package mlir.tosa;

import java.util.Arrays;

/**
 * Compare TOSA results with expected ONNX results.
 * The expected values are taken from running the ONNX model.
 */
public class CompareWithOnnx {

    public static final int IMAGE_SIZE = 28;

    public static void main(String[] args) {
        // Expected ONNX results (from MNISTCompare.java output)
        float[][] onnxExpected = {
            // Test 1: Zero (circle)
            {0.999931f, 0.000000f, 0.000066f, 0.000000f, 0.000000f,
             0.000000f, 0.000001f, 0.000000f, 0.000000f, 0.000001f},
            // Test 2: One (vertical line)
            {0.000004f, 0.998863f, 0.000017f, 0.000039f, 0.000444f,
             0.000012f, 0.000045f, 0.000493f, 0.000075f, 0.000008f},
            // Test 3: Diagonal line
            {0.001154f, 0.032275f, 0.098464f, 0.136055f, 0.012030f,
             0.382878f, 0.008647f, 0.012008f, 0.314415f, 0.002075f},
            // Test 4: Random noise
            {0.045115f, 0.041530f, 0.075040f, 0.200970f, 0.061815f,
             0.171202f, 0.022458f, 0.110587f, 0.166919f, 0.104364f}
        };

        String[] testNames = {"Zero (circle)", "One (vertical line)", "Diagonal line", "Random noise"};

        System.out.println("=".repeat(80));
        System.out.println("TOSA vs ONNX Comparison");
        System.out.println("=".repeat(80));

        MNISTModel model = new MNISTModel();
        float[][] testImages = createTestImages();

        for (int t = 0; t < testImages.length; t++) {
            System.out.println("\n" + "-".repeat(60));
            System.out.println("Test " + (t + 1) + ": " + testNames[t]);
            System.out.println("-".repeat(60));

            float[] tosaProbs = model.classify(testImages[t]);

            System.out.println("\nDigit | ONNX     | TOSA     | Diff");
            System.out.println("------|----------|----------|----------");
            for (int i = 0; i < 10; i++) {
                float diff = Math.abs(onnxExpected[t][i] - tosaProbs[i]);
                System.out.printf("  %d   | %.6f | %.6f | %.6f%n",
                        i, onnxExpected[t][i], tosaProbs[i], diff);
            }

            // Find predictions
            int onnxPred = argmax(onnxExpected[t]);
            int tosaPred = argmax(tosaProbs);
            System.out.printf("\nONNX predicts: %d (%.2f%%), TOSA predicts: %d (%.2f%%)%n",
                    onnxPred, onnxExpected[t][onnxPred] * 100,
                    tosaPred, tosaProbs[tosaPred] * 100);
            System.out.println("Match: " + (onnxPred == tosaPred ? "YES" : "NO"));
        }
    }

    private static int argmax(float[] arr) {
        int maxIdx = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[maxIdx]) maxIdx = i;
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
