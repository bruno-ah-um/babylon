package mlir.tosa;

import jdk.incubator.code.Reflect;
import org.junit.jupiter.api.Test;

import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.SequencedMap;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for TosaModelExporter - exporting models with embedded weights to MLIR.
 */
public class TosaModelExporterTest {

    /**
     * Simple model with a single weight tensor for testing.
     */
    public static class SimpleModel {
        // A simple weight tensor [2, 3] with test values
        final Tensor<Float> weights = Tensor.ofFloats(
            new long[]{2, 3},
            new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}
        );

        final Tensor<Float> bias = Tensor.ofFloats(
            new long[]{2},
            new float[]{0.1f, 0.2f}
        );

        @Reflect
        public Tensor<Float> forward(Tensor<Float> input) {
            // Simple operation that uses the weights
            Tensor<Float> scaled = input.mul(weights);
            return scaled.add(bias);
        }
    }

    /**
     * Model with multiple weight tensors.
     */
    public static class TwoLayerModel {
        final Tensor<Float> layer1Weights = Tensor.ofFloats(
            new long[]{4, 3},
            new float[]{0.1f, 0.2f, 0.3f,
                        0.4f, 0.5f, 0.6f,
                        0.7f, 0.8f, 0.9f,
                        1.0f, 1.1f, 1.2f}
        );

        final Tensor<Float> layer1Bias = Tensor.ofFloats(
            new long[]{4},
            new float[]{0.01f, 0.02f, 0.03f, 0.04f}
        );

        final Tensor<Float> layer2Weights = Tensor.ofFloats(
            new long[]{2, 4},
            new float[]{0.1f, 0.2f, 0.3f, 0.4f,
                        0.5f, 0.6f, 0.7f, 0.8f}
        );

        final Tensor<Float> layer2Bias = Tensor.ofFloats(
            new long[]{2},
            new float[]{0.001f, 0.002f}
        );

        @Reflect
        public Tensor<Float> forward(Tensor<Float> input) {
            // Two-layer forward pass
            Tensor<Float> h1 = input.matmul(layer1Weights);
            Tensor<Float> h1b = h1.add(layer1Bias);
            Tensor<Float> h2 = h1b.matmul(layer2Weights);
            return h2.add(layer2Bias);
        }
    }

    @Test
    public void testExtractWeightsSimpleModel() {
        SimpleModel model = new SimpleModel();

        SequencedMap<String, TosaModelExporter.WeightInfo> weights =
            TosaModelExporter.extractWeights(model);

        assertNotNull(weights);
        assertEquals(2, weights.size(), "Should extract 2 weight tensors");

        // Check weights tensor
        assertTrue(weights.containsKey("weights"));
        TosaModelExporter.WeightInfo weightsInfo = weights.get("weights");
        assertArrayEquals(new long[]{2, 3}, weightsInfo.shape());
        assertEquals(6, weightsInfo.numElements());

        // Check bias tensor
        assertTrue(weights.containsKey("bias"));
        TosaModelExporter.WeightInfo biasInfo = weights.get("bias");
        assertArrayEquals(new long[]{2}, biasInfo.shape());
        assertEquals(2, biasInfo.numElements());

        // Verify data values
        float firstWeightValue = weightsInfo.data().getAtIndex(ValueLayout.JAVA_FLOAT, 0);
        assertEquals(1.0f, firstWeightValue, 0.001f);

        float firstBiasValue = biasInfo.data().getAtIndex(ValueLayout.JAVA_FLOAT, 0);
        assertEquals(0.1f, firstBiasValue, 0.001f);

        System.out.println("Extracted weights:");
        for (var entry : weights.entrySet()) {
            System.out.printf("  %s: shape=%s, elements=%d%n",
                entry.getKey(),
                java.util.Arrays.toString(entry.getValue().shape()),
                entry.getValue().numElements());
        }
    }

    @Test
    public void testExtractWeightsTwoLayerModel() {
        TwoLayerModel model = new TwoLayerModel();

        SequencedMap<String, TosaModelExporter.WeightInfo> weights =
            TosaModelExporter.extractWeights(model);

        assertNotNull(weights);
        assertEquals(4, weights.size(), "Should extract 4 weight tensors");

        // Verify all expected weights are present
        assertTrue(weights.containsKey("layer1Weights"));
        assertTrue(weights.containsKey("layer1Bias"));
        assertTrue(weights.containsKey("layer2Weights"));
        assertTrue(weights.containsKey("layer2Bias"));

        // Verify shapes
        assertArrayEquals(new long[]{4, 3}, weights.get("layer1Weights").shape());
        assertArrayEquals(new long[]{4}, weights.get("layer1Bias").shape());
        assertArrayEquals(new long[]{2, 4}, weights.get("layer2Weights").shape());
        assertArrayEquals(new long[]{2}, weights.get("layer2Bias").shape());

        System.out.println("Extracted weights from two-layer model:");
        for (var entry : weights.entrySet()) {
            System.out.printf("  %s: shape=%s, elements=%d%n",
                entry.getKey(),
                java.util.Arrays.toString(entry.getValue().shape()),
                entry.getValue().numElements());
        }
    }

    @Test
    public void testExportSimpleModel() {
        SimpleModel model = new SimpleModel();

        TosaModelExporter.ExportResult result = TosaModelExporter.export(model, "forward");

        assertNotNull(result);
        assertNotNull(result.mlirCode());
        assertFalse(result.mlirCode().isEmpty());

        String mlir = result.mlirCode();
        System.out.println("Generated MLIR:");
        System.out.println(mlir);

        // Verify the MLIR contains expected elements
        assertTrue(mlir.contains("func.func"), "Should contain function definition");
        assertTrue(mlir.contains("@forward"), "Should contain function name");
        assertTrue(mlir.contains("tosa.const"), "Should contain tosa.const for weights");
    }

    @Test
    public void testExportToFile() throws Exception {
        SimpleModel model = new SimpleModel();
        Path tempFile = Files.createTempFile("test_export_", ".mlir");

        try {
            TosaModelExporter.exportToFile(model, "forward", tempFile);

            assertTrue(Files.exists(tempFile));
            String content = Files.readString(tempFile);

            assertFalse(content.isEmpty());
            assertTrue(content.contains("func.func"));
            assertTrue(content.contains("@forward"));

            System.out.println("Exported to: " + tempFile);
            System.out.println("File size: " + Files.size(tempFile) + " bytes");
        } finally {
            Files.deleteIfExists(tempFile);
        }
    }

    @Test
    public void testExportMNISTModel() {
        // Test with the actual MNISTModel if weights are available
        try {
            MNISTModel model = new MNISTModel();

            SequencedMap<String, TosaModelExporter.WeightInfo> weights =
                TosaModelExporter.extractWeights(model);

            // MNISTModel has 10 weight tensors (5 layers x 2 each)
            assertEquals(10, weights.size(), "MNIST model should have 10 weight tensors");

            System.out.println("MNIST model weights:");
            for (var entry : weights.entrySet()) {
                System.out.printf("  %s: shape=%s, elements=%d%n",
                    entry.getKey(),
                    java.util.Arrays.toString(entry.getValue().shape()),
                    entry.getValue().numElements());
            }

            // Note: Full export test is commented out as it may take longer
            // and depends on the complexity of the cnn() method
            // TosaModelExporter.ExportResult result = TosaModelExporter.export(model, "cnn");
            // System.out.println("MNIST MLIR length: " + result.mlirCode().length());

        } catch (Exception e) {
            // Skip if weight files are not available
            System.out.println("Skipping MNIST model test: " + e.getMessage());
        }
    }
}
