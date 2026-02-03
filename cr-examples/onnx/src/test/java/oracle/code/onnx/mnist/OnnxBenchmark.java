/*
 * Performance benchmark for ONNX Runtime execution.
 *
 * Uses the same model architecture and methodology as TosaBenchmark.
 */
package oracle.code.onnx.mnist;

import jdk.incubator.code.Reflect;
import oracle.code.onnx.OnnxRuntime;
import oracle.code.onnx.Tensor;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.invoke.MethodHandles;
import java.util.Random;
import java.util.function.Supplier;

import static java.util.Optional.empty;
import static java.util.Optional.of;
import static oracle.code.onnx.OnnxOperators.*;

/**
 * Benchmark comparing ONNX Runtime performance with the same model architecture
 * as the TOSA benchmarks for fair comparison.
 *
 * Run with: mvn test -Dtest=OnnxBenchmark#runOnnxBenchmark
 */
public class OnnxBenchmark {

    private static final int WARMUP_ITERATIONS = 20_000;
    private static final int BENCHMARK_ITERATIONS = 100_000;
    private static final int NUM_RANDOM_INPUTS = 100;

    /**
     * Feature extractor model - same architecture as TosaBenchmark.
     * Takes weights as parameters (required for ONNX code reflection pattern).
     */
    @Reflect
    static Tensor<Float> featureExtract(
            Tensor<Float> conv1Weights, Tensor<Float> conv1Bias,
            Tensor<Float> conv2Weights, Tensor<Float> conv2Bias,
            Tensor<Float> inputImage) {

        // First conv layer: [1, 1, 28, 28] -> [1, 6, 24, 24]
        var conv1 = Conv(inputImage, conv1Weights, of(conv1Bias), of(new long[4]),
                of(new long[]{1,1}), empty(), of(new long[]{1, 1}),
                of(1L), of(new long[]{5, 5}));
        var relu1 = Relu(conv1);

        // First pooling: [1, 6, 24, 24] -> [1, 6, 12, 12]
        var pool1 = MaxPool(relu1, of(new long[4]), of(new long[]{1,1}), empty(),
                of(0L), empty(), of(new long[]{2, 2}), new long[]{2, 2});

        // Second conv layer: [1, 6, 12, 12] -> [1, 16, 8, 8]
        var conv2 = Conv(pool1.Y(), conv2Weights, of(conv2Bias), of(new long[4]),
                of(new long[]{1,1}), empty(), of(new long[]{1, 1}),
                of(1L), of(new long[]{5, 5}));
        var relu2 = Relu(conv2);

        // Second pooling: [1, 16, 8, 8] -> [1, 16, 4, 4]
        var pool2 = MaxPool(relu2, of(new long[4]), of(new long[]{1,1}), empty(),
                of(0L), empty(), of(new long[]{2, 2}), new long[]{2, 2});

        return pool2.Y();
    }

    private static Tensor<Float> load(Arena arena, String resource, long... shape) {
        try (var in = MNISTModel.class.getResourceAsStream(resource)) {
            if (in == null) {
                throw new IllegalStateException("Resource not found: " + resource);
            }
            return Tensor.ofShape(arena, shape, in.readAllBytes(), Tensor.ElementType.FLOAT);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Test
    public void runOnnxBenchmark() throws Exception {
        System.out.println("=".repeat(80));
        System.out.println("ONNX Runtime Performance Benchmark");
        System.out.println("=".repeat(80));
        System.out.printf("Warmup iterations: %,d%n", WARMUP_ITERATIONS);
        System.out.printf("Benchmark iterations: %,d%n", BENCHMARK_ITERATIONS);
        System.out.println();

        // Create random inputs for benchmarking
        System.out.println("Generating random inputs...");
        Random random = new Random(42);  // Same seed as TosaBenchmark
        float[][] randomInputs = new float[NUM_RANDOM_INPUTS][28 * 28];
        for (int i = 0; i < NUM_RANDOM_INPUTS; i++) {
            for (int j = 0; j < 28 * 28; j++) {
                randomInputs[i][j] = random.nextFloat();
            }
        }

        // ===== ONNX RUNTIME BENCHMARK =====
        System.out.println("-".repeat(60));
        System.out.println("ONNX Runtime Benchmark (Feature Extractor - 2 conv layers)");
        System.out.println("-".repeat(60));

        // Load weights once outside the arena (they persist across iterations)
        Tensor<Float> conv1Weights = load(Arena.ofAuto(), "conv1-weight-float-le", 6, 1, 5, 5);
        Tensor<Float> conv1Bias = load(Arena.ofAuto(), "conv1-bias-float-le", 6);
        Tensor<Float> conv2Weights = load(Arena.ofAuto(), "conv2-weight-float-le", 16, 6, 5, 5);
        Tensor<Float> conv2Bias = load(Arena.ofAuto(), "conv2-bias-float-le", 16);

        // Warmup - also triggers ONNX model compilation/caching
        System.out.print("Warming up ONNX Runtime (" + WARMUP_ITERATIONS + " iterations)...");
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            try (Arena arena = Arena.ofConfined()) {
                var inputTensor = Tensor.ofShape(arena, new long[]{1, 1, 28, 28}, randomInputs[i % NUM_RANDOM_INPUTS]);
                OnnxRuntime.execute(arena, MethodHandles.lookup(),
                        (@Reflect Supplier<Tensor<Float>>) () -> featureExtract(
                                conv1Weights, conv1Bias, conv2Weights, conv2Bias, inputTensor));
            }
        }
        System.out.println(" done");

        // Benchmark
        System.out.print("Running ONNX Runtime benchmark (" + BENCHMARK_ITERATIONS + " iterations)...");
        long onnxStart = System.nanoTime();
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            try (Arena arena = Arena.ofConfined()) {
                var inputTensor = Tensor.ofShape(arena, new long[]{1, 1, 28, 28}, randomInputs[i % NUM_RANDOM_INPUTS]);
                OnnxRuntime.execute(arena, MethodHandles.lookup(),
                        (@Reflect Supplier<Tensor<Float>>) () -> featureExtract(
                                conv1Weights, conv1Bias, conv2Weights, conv2Bias, inputTensor));
            }
        }
        long onnxEnd = System.nanoTime();
        double onnxTimeMs = (onnxEnd - onnxStart) / 1_000_000.0;
        double onnxPerIterUs = (onnxTimeMs * 1000) / BENCHMARK_ITERATIONS;
        double onnxOpsPerSec = BENCHMARK_ITERATIONS / (onnxTimeMs / 1000.0);
        System.out.println(" done");
        System.out.printf("Total time: %.2f ms%n", onnxTimeMs);
        System.out.printf("Per iteration: %.2f µs%n", onnxPerIterUs);
        System.out.printf("Throughput: %,.0f inferences/sec%n", onnxOpsPerSec);

        // ===== RESULTS SUMMARY =====
        System.out.println();
        System.out.println("=".repeat(80));
        System.out.println("ONNX BENCHMARK RESULTS SUMMARY");
        System.out.println("=".repeat(80));
        System.out.println();
        System.out.printf("%-20s %15s %15s %18s%n", "Implementation", "Total (ms)", "Per iter (µs)", "Throughput (ops/s)");
        System.out.println("-".repeat(70));
        System.out.printf("%-20s %15.2f %15.2f %18.0f%n", "ONNX Runtime", onnxTimeMs, onnxPerIterUs, onnxOpsPerSec);
        System.out.println("-".repeat(70));
        System.out.println();
        System.out.println("Compare with TosaBenchmark results for full comparison:");
        System.out.println("  TOSA Java: ~1,335 µs/iter, ~749 ops/sec");
        System.out.println("  TOSA Native: ~215 µs/iter, ~4,656 ops/sec");
    }
}
