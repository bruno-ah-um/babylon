package mlir.tosa;

import jdk.incubator.code.Reflect;

/**
 * Benchmark comparing TOSA Native vs Java for a complete MNIST CNN model.
 *
 * The model architecture (LeNet-style):
 * - Input: [1, 28, 28, 1] (NHWC format)
 * - Conv1: 5x5 kernel, 6 output channels -> [1, 24, 24, 6]
 * - ReLU
 * - MaxPool1: 2x2, stride 2 -> [1, 12, 12, 6]
 * - Conv2: 5x5 kernel, 16 output channels -> [1, 8, 8, 16]
 * - ReLU
 * - MaxPool2: 2x2, stride 2 -> [1, 4, 4, 16] = [1, 256]
 * - Reshape (Flatten): [1, 1, 256]
 * - FC1: 256 -> 120 with ReLU
 * - FC2: 120 -> 84 with ReLU
 * - FC3: 84 -> 10 (output logits)
 *
 * Run with:
 * export JAVA_HOME=/home/microdoc/git/babylon/build/linux-x86_64-server-release/jdk
 * export LD_LIBRARY_PATH=/home/microdoc/git/babylon/cr-examples/mlir-tosa/lib/build
 * $JAVA_HOME/bin/java --enable-preview --add-modules=jdk.incubator.code --enable-native-access=ALL-UNNAMED \
 *   -Djava.library.path=$LD_LIBRARY_PATH -cp target/classes:target/test-classes mlir.tosa.TosaMnistBenchmark
 */
public class TosaMnistBenchmark {

    // Warmup iterations to let Java JIT optimize
    private static final int WARMUP_ITERATIONS = 500;
    // Measurement iterations
    private static final int MEASUREMENT_ITERATIONS = 2000;

    /**
     * Complete MNIST CNN model in TOSA.
     *
     * Architecture:
     * - Conv1 (5x5, 6 filters) -> ReLU -> MaxPool (2x2)
     * - Conv2 (5x5, 16 filters) -> ReLU -> MaxPool (2x2)
     * - Flatten
     * - FC1 (256->120) -> ReLU
     * - FC2 (120->84) -> ReLU
     * - FC3 (84->10) -> output logits
     */
    @Reflect
    public static Tensor<Float> mnistModel(
            Tensor<Float> input,
            // Conv layer weights
            Tensor<Float> conv1Weight, Tensor<Float> conv1Bias,
            Tensor<Float> conv2Weight, Tensor<Float> conv2Bias,
            // FC layer weights (3D for batched matmul)
            Tensor<Float> fc1Weight, Tensor<Float> fc1Bias,
            Tensor<Float> fc2Weight, Tensor<Float> fc2Bias,
            Tensor<Float> fc3Weight, Tensor<Float> fc3Bias) {

        // Conv block 1: Conv2D(5x5) -> ReLU -> MaxPool(2x2)
        // Input: [1, 28, 28, 1] -> Conv -> [1, 24, 24, 6] -> Pool -> [1, 12, 12, 6]
        Tensor<Float> conv1 = TosaOperators.Conv2D(input, conv1Weight, conv1Bias,
            new long[]{0, 0, 0, 0}, new long[]{1, 1}, new long[]{1, 1});
        Tensor<Float> relu1 = conv1.relu();
        Tensor<Float> pool1 = TosaOperators.MaxPool2D(relu1,
            new long[]{2, 2}, new long[]{2, 2}, new long[]{0, 0, 0, 0});

        // Conv block 2: Conv2D(5x5) -> ReLU -> MaxPool(2x2)
        // Input: [1, 12, 12, 6] -> Conv -> [1, 8, 8, 16] -> Pool -> [1, 4, 4, 16]
        Tensor<Float> conv2 = TosaOperators.Conv2D(pool1, conv2Weight, conv2Bias,
            new long[]{0, 0, 0, 0}, new long[]{1, 1}, new long[]{1, 1});
        Tensor<Float> relu2 = conv2.relu();
        Tensor<Float> pool2 = TosaOperators.MaxPool2D(relu2,
            new long[]{2, 2}, new long[]{2, 2}, new long[]{0, 0, 0, 0});

        // Flatten: [1, 4, 4, 16] -> [1, 1, 256]
        Tensor<Float> flat = TosaOperators.Reshape(pool2, new long[]{1, 1, 256});

        // FC1: [1, 1, 256] @ [1, 256, 120] -> [1, 1, 120]
        Tensor<Float> fc1 = flat.matmul(fc1Weight).add(fc1Bias).relu();

        // FC2: [1, 1, 120] @ [1, 120, 84] -> [1, 1, 84]
        Tensor<Float> fc2 = fc1.matmul(fc2Weight).add(fc2Bias).relu();

        // FC3: [1, 1, 84] @ [1, 84, 10] -> [1, 1, 10]
        Tensor<Float> fc3 = fc2.matmul(fc3Weight).add(fc3Bias);

        return fc3;
    }

    /**
     * Just the convolutional feature extraction part (conv blocks only).
     */
    @Reflect
    public static Tensor<Float> mnistConvLayers(
            Tensor<Float> input,
            Tensor<Float> conv1Weight, Tensor<Float> conv1Bias,
            Tensor<Float> conv2Weight, Tensor<Float> conv2Bias) {

        // Conv block 1
        Tensor<Float> conv1 = TosaOperators.Conv2D(input, conv1Weight, conv1Bias,
            new long[]{0, 0, 0, 0}, new long[]{1, 1}, new long[]{1, 1});
        Tensor<Float> relu1 = conv1.relu();
        Tensor<Float> pool1 = TosaOperators.MaxPool2D(relu1,
            new long[]{2, 2}, new long[]{2, 2}, new long[]{0, 0, 0, 0});

        // Conv block 2
        Tensor<Float> conv2 = TosaOperators.Conv2D(pool1, conv2Weight, conv2Bias,
            new long[]{0, 0, 0, 0}, new long[]{1, 1}, new long[]{1, 1});
        Tensor<Float> relu2 = conv2.relu();
        Tensor<Float> pool2 = TosaOperators.MaxPool2D(relu2,
            new long[]{2, 2}, new long[]{2, 2}, new long[]{0, 0, 0, 0});

        return pool2;
    }

    /**
     * Just the fully connected layers.
     */
    @Reflect
    public static Tensor<Float> mnistFcLayers(
            Tensor<Float> flat,
            Tensor<Float> fc1Weight, Tensor<Float> fc1Bias,
            Tensor<Float> fc2Weight, Tensor<Float> fc2Bias,
            Tensor<Float> fc3Weight, Tensor<Float> fc3Bias) {

        Tensor<Float> fc1 = flat.matmul(fc1Weight).add(fc1Bias).relu();
        Tensor<Float> fc2 = fc1.matmul(fc2Weight).add(fc2Bias).relu();
        Tensor<Float> fc3 = fc2.matmul(fc3Weight).add(fc3Bias);

        return fc3;
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=".repeat(70));
        System.out.println("TOSA MNIST Complete Model Benchmark: Native vs Java");
        System.out.println("=".repeat(70));
        System.out.println();
        System.out.println("Model architecture (LeNet-style CNN):");
        System.out.println("  Input:    [1, 28, 28, 1]   (MNIST image, NHWC format)");
        System.out.println("  Conv1:    5x5 kernel, 6 filters -> [1, 24, 24, 6]");
        System.out.println("  ReLU + MaxPool (2x2) -> [1, 12, 12, 6]");
        System.out.println("  Conv2:    5x5 kernel, 16 filters -> [1, 8, 8, 16]");
        System.out.println("  ReLU + MaxPool (2x2) -> [1, 4, 4, 16]");
        System.out.println("  Flatten:  [1, 1, 256]");
        System.out.println("  FC1:      256 -> 120 + ReLU");
        System.out.println("  FC2:      120 -> 84 + ReLU");
        System.out.println("  FC3:      84 -> 10 (output logits)");
        System.out.println();
        System.out.println("Configuration:");
        System.out.println("  Warmup iterations:     " + WARMUP_ITERATIONS);
        System.out.println("  Measurement iterations: " + MEASUREMENT_ITERATIONS);
        System.out.println();

        // ==================== Setup Test Data ====================

        System.out.println("Setting up model weights and compiling native functions...");

        // Input: [1, 28, 28, 1]
        Float[] inputData = new Float[784];
        for (int i = 0; i < 784; i++) {
            inputData[i] = (float)(i % 28) / 255.0f;  // Simulated pixel values
        }
        Tensor<Float> input = Tensor.ofShape(new long[]{1, 28, 28, 1}, inputData);

        // Conv1 weights: [6, 5, 5, 1] = 150 params
        Float[] conv1WData = new Float[150];
        for (int i = 0; i < 150; i++) {
            conv1WData[i] = (float)((i % 7) - 3) * 0.1f;
        }
        Tensor<Float> conv1Weight = Tensor.ofShape(new long[]{6, 5, 5, 1}, conv1WData);
        Tensor<Float> conv1Bias = Tensor.ofShape(new long[]{6}, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

        // Conv2 weights: [16, 5, 5, 6] = 2400 params
        Float[] conv2WData = new Float[2400];
        for (int i = 0; i < 2400; i++) {
            conv2WData[i] = (float)((i % 5) - 2) * 0.05f;
        }
        Tensor<Float> conv2Weight = Tensor.ofShape(new long[]{16, 5, 5, 6}, conv2WData);
        Float[] conv2BData = new Float[16];
        for (int i = 0; i < 16; i++) conv2BData[i] = 0.0f;
        Tensor<Float> conv2Bias = Tensor.ofShape(new long[]{16}, conv2BData);

        // FC1 weights: [1, 256, 120] = 30720 params
        Float[] fc1WData = new Float[30720];
        for (int i = 0; i < 30720; i++) {
            fc1WData[i] = (float)((i % 11) - 5) * 0.01f;
        }
        Tensor<Float> fc1Weight = Tensor.ofShape(new long[]{1, 256, 120}, fc1WData);
        Float[] fc1BData = new Float[120];
        for (int i = 0; i < 120; i++) fc1BData[i] = 0.0f;
        Tensor<Float> fc1Bias = Tensor.ofShape(new long[]{1, 1, 120}, fc1BData);

        // FC2 weights: [1, 120, 84] = 10080 params
        Float[] fc2WData = new Float[10080];
        for (int i = 0; i < 10080; i++) {
            fc2WData[i] = (float)((i % 9) - 4) * 0.02f;
        }
        Tensor<Float> fc2Weight = Tensor.ofShape(new long[]{1, 120, 84}, fc2WData);
        Float[] fc2BData = new Float[84];
        for (int i = 0; i < 84; i++) fc2BData[i] = 0.0f;
        Tensor<Float> fc2Bias = Tensor.ofShape(new long[]{1, 1, 84}, fc2BData);

        // FC3 weights: [1, 84, 10] = 840 params
        Float[] fc3WData = new Float[840];
        for (int i = 0; i < 840; i++) {
            fc3WData[i] = (float)((i % 7) - 3) * 0.03f;
        }
        Tensor<Float> fc3Weight = Tensor.ofShape(new long[]{1, 84, 10}, fc3WData);
        Float[] fc3BData = new Float[10];
        for (int i = 0; i < 10; i++) fc3BData[i] = 0.0f;
        Tensor<Float> fc3Bias = Tensor.ofShape(new long[]{1, 1, 10}, fc3BData);

        // Pre-computed flattened input for FC-only benchmark
        Float[] flatData = new Float[256];
        for (int i = 0; i < 256; i++) flatData[i] = (float)(i % 16) * 0.1f;
        Tensor<Float> flatInput = Tensor.ofShape(new long[]{1, 1, 256}, flatData);

        System.out.println("  Total model parameters: ~44,000");
        System.out.println();

        // ==================== Compile Native Functions ====================

        System.out.println("Compiling native functions (this may take a moment)...");
        TosaCompiler compiler = new TosaCompiler(false);

        // Compile conv layers only
        var convMethod = TosaMnistBenchmark.class.getMethod("mnistConvLayers",
            Tensor.class, Tensor.class, Tensor.class, Tensor.class, Tensor.class);
        long[][] convParamShapes = {
            {1, 28, 28, 1},   // input
            {6, 5, 5, 1},     // conv1 weight
            {6},              // conv1 bias
            {16, 5, 5, 6},    // conv2 weight
            {16}              // conv2 bias
        };
        CompiledFunction nativeConvLayers = compiler.compile(convMethod, convParamShapes);
        System.out.println("  Conv layers compiled");

        // Compile FC layers only
        var fcMethod = TosaMnistBenchmark.class.getMethod("mnistFcLayers",
            Tensor.class, Tensor.class, Tensor.class, Tensor.class, Tensor.class, Tensor.class, Tensor.class);
        long[][] fcParamShapes = {
            {1, 1, 256},      // flat input
            {1, 256, 120},    // fc1 weight
            {1, 1, 120},      // fc1 bias
            {1, 120, 84},     // fc2 weight
            {1, 1, 84},       // fc2 bias
            {1, 84, 10},      // fc3 weight
            {1, 1, 10}        // fc3 bias
        };
        CompiledFunction nativeFcLayers = compiler.compile(fcMethod, fcParamShapes);
        System.out.println("  FC layers compiled");

        System.out.println("Setup complete!\n");

        // ==================== Benchmark 1: Conv Layers Only ====================

        System.out.println("-".repeat(70));
        System.out.println("Benchmark 1: Convolutional Layers Only");
        System.out.println("  [1,28,28,1] -> Conv1 -> Pool1 -> Conv2 -> Pool2 -> [1,4,4,16]");
        System.out.println("-".repeat(70));

        // Warmup Java
        System.out.print("Warming up Java (" + WARMUP_ITERATIONS + " iterations)...");
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            Tensor<Float> result = mnistConvLayers(input, conv1Weight, conv1Bias, conv2Weight, conv2Bias);
            if (i == 0) System.out.print(" output: " + java.util.Arrays.toString(result.shape()));
        }
        System.out.println(" done");

        // Warmup Native
        System.out.print("Warming up Native (" + WARMUP_ITERATIONS + " iterations)...");
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            Tensor<Float> result = nativeConvLayers.invoke(input, conv1Weight, conv1Bias, conv2Weight, conv2Bias);
        }
        System.out.println(" done");

        // Measure Java
        System.out.print("Measuring Java (" + MEASUREMENT_ITERATIONS + " iterations)...");
        long javaConvStart = System.nanoTime();
        for (int i = 0; i < MEASUREMENT_ITERATIONS; i++) {
            Tensor<Float> result = mnistConvLayers(input, conv1Weight, conv1Bias, conv2Weight, conv2Bias);
        }
        long javaConvTime = System.nanoTime() - javaConvStart;
        double javaConvAvgUs = (javaConvTime / 1000.0) / MEASUREMENT_ITERATIONS;
        System.out.println(" done");

        // Measure Native
        System.out.print("Measuring Native (" + MEASUREMENT_ITERATIONS + " iterations)...");
        long nativeConvStart = System.nanoTime();
        for (int i = 0; i < MEASUREMENT_ITERATIONS; i++) {
            Tensor<Float> result = nativeConvLayers.invoke(input, conv1Weight, conv1Bias, conv2Weight, conv2Bias);
        }
        long nativeConvTime = System.nanoTime() - nativeConvStart;
        double nativeConvAvgUs = (nativeConvTime / 1000.0) / MEASUREMENT_ITERATIONS;
        System.out.println(" done\n");

        System.out.printf("  Java avg:   %10.2f us/op%n", javaConvAvgUs);
        System.out.printf("  Native avg: %10.2f us/op%n", nativeConvAvgUs);
        System.out.printf("  Speedup:    %10.2fx%n", javaConvAvgUs / nativeConvAvgUs);
        System.out.println();

        // ==================== Benchmark 2: FC Layers Only ====================

        System.out.println("-".repeat(70));
        System.out.println("Benchmark 2: Fully Connected Layers Only");
        System.out.println("  [1,1,256] -> FC1 -> FC2 -> FC3 -> [1,1,10]");
        System.out.println("-".repeat(70));

        // Warmup Java
        System.out.print("Warming up Java (" + WARMUP_ITERATIONS + " iterations)...");
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            Tensor<Float> result = mnistFcLayers(flatInput, fc1Weight, fc1Bias, fc2Weight, fc2Bias, fc3Weight, fc3Bias);
            if (i == 0) System.out.print(" output: " + java.util.Arrays.toString(result.shape()));
        }
        System.out.println(" done");

        // Warmup Native
        System.out.print("Warming up Native (" + WARMUP_ITERATIONS + " iterations)...");
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            Tensor<Float> result = nativeFcLayers.invoke(flatInput, fc1Weight, fc1Bias, fc2Weight, fc2Bias, fc3Weight, fc3Bias);
        }
        System.out.println(" done");

        // Measure Java
        System.out.print("Measuring Java (" + MEASUREMENT_ITERATIONS + " iterations)...");
        long javaFcStart = System.nanoTime();
        for (int i = 0; i < MEASUREMENT_ITERATIONS; i++) {
            Tensor<Float> result = mnistFcLayers(flatInput, fc1Weight, fc1Bias, fc2Weight, fc2Bias, fc3Weight, fc3Bias);
        }
        long javaFcTime = System.nanoTime() - javaFcStart;
        double javaFcAvgUs = (javaFcTime / 1000.0) / MEASUREMENT_ITERATIONS;
        System.out.println(" done");

        // Measure Native
        System.out.print("Measuring Native (" + MEASUREMENT_ITERATIONS + " iterations)...");
        long nativeFcStart = System.nanoTime();
        for (int i = 0; i < MEASUREMENT_ITERATIONS; i++) {
            Tensor<Float> result = nativeFcLayers.invoke(flatInput, fc1Weight, fc1Bias, fc2Weight, fc2Bias, fc3Weight, fc3Bias);
        }
        long nativeFcTime = System.nanoTime() - nativeFcStart;
        double nativeFcAvgUs = (nativeFcTime / 1000.0) / MEASUREMENT_ITERATIONS;
        System.out.println(" done\n");

        System.out.printf("  Java avg:   %10.2f us/op%n", javaFcAvgUs);
        System.out.printf("  Native avg: %10.2f us/op%n", nativeFcAvgUs);
        System.out.printf("  Speedup:    %10.2fx%n", javaFcAvgUs / nativeFcAvgUs);
        System.out.println();

        // ==================== Benchmark 3: Full Model (Conv + FC combined) ====================

        System.out.println("-".repeat(70));
        System.out.println("Benchmark 3: Full MNIST Model (End-to-End)");
        System.out.println("  [1,28,28,1] -> Conv layers -> FC layers -> [1,1,10]");
        System.out.println("-".repeat(70));

        // For full model, run conv + fc sequentially
        // Warmup Java
        System.out.print("Warming up Java (" + WARMUP_ITERATIONS + " iterations)...");
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            Tensor<Float> convOut = mnistConvLayers(input, conv1Weight, conv1Bias, conv2Weight, conv2Bias);
            Tensor<Float> flat = TosaOperators.Reshape(convOut, new long[]{1, 1, 256});
            Tensor<Float> result = mnistFcLayers(flat, fc1Weight, fc1Bias, fc2Weight, fc2Bias, fc3Weight, fc3Bias);
            if (i == 0) System.out.print(" output: " + java.util.Arrays.toString(result.shape()));
        }
        System.out.println(" done");

        // Warmup Native
        System.out.print("Warming up Native (" + WARMUP_ITERATIONS + " iterations)...");
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            Tensor<Float> convOut = nativeConvLayers.invoke(input, conv1Weight, conv1Bias, conv2Weight, conv2Bias);
            Tensor<Float> flat = TosaOperators.Reshape(convOut, new long[]{1, 1, 256});
            Tensor<Float> result = nativeFcLayers.invoke(flat, fc1Weight, fc1Bias, fc2Weight, fc2Bias, fc3Weight, fc3Bias);
        }
        System.out.println(" done");

        // Measure Java
        System.out.print("Measuring Java (" + MEASUREMENT_ITERATIONS + " iterations)...");
        long javaFullStart = System.nanoTime();
        for (int i = 0; i < MEASUREMENT_ITERATIONS; i++) {
            Tensor<Float> convOut = mnistConvLayers(input, conv1Weight, conv1Bias, conv2Weight, conv2Bias);
            Tensor<Float> flat = TosaOperators.Reshape(convOut, new long[]{1, 1, 256});
            Tensor<Float> result = mnistFcLayers(flat, fc1Weight, fc1Bias, fc2Weight, fc2Bias, fc3Weight, fc3Bias);
        }
        long javaFullTime = System.nanoTime() - javaFullStart;
        double javaFullAvgUs = (javaFullTime / 1000.0) / MEASUREMENT_ITERATIONS;
        System.out.println(" done");

        // Measure Native
        System.out.print("Measuring Native (" + MEASUREMENT_ITERATIONS + " iterations)...");
        long nativeFullStart = System.nanoTime();
        for (int i = 0; i < MEASUREMENT_ITERATIONS; i++) {
            Tensor<Float> convOut = nativeConvLayers.invoke(input, conv1Weight, conv1Bias, conv2Weight, conv2Bias);
            Tensor<Float> flat = TosaOperators.Reshape(convOut, new long[]{1, 1, 256});
            Tensor<Float> result = nativeFcLayers.invoke(flat, fc1Weight, fc1Bias, fc2Weight, fc2Bias, fc3Weight, fc3Bias);
        }
        long nativeFullTime = System.nanoTime() - nativeFullStart;
        double nativeFullAvgUs = (nativeFullTime / 1000.0) / MEASUREMENT_ITERATIONS;
        System.out.println(" done\n");

        System.out.printf("  Java avg:   %10.2f us/op%n", javaFullAvgUs);
        System.out.printf("  Native avg: %10.2f us/op%n", nativeFullAvgUs);
        System.out.printf("  Speedup:    %10.2fx%n", javaFullAvgUs / nativeFullAvgUs);
        System.out.println();

        // ==================== Summary ====================

        System.out.println("=".repeat(70));
        System.out.println("SUMMARY: MNIST Complete Model Benchmark");
        System.out.println("=".repeat(70));
        System.out.println();
        System.out.println("| Layer               | Java (us)   | Native (us)  | Speedup  |");
        System.out.println("|---------------------|-------------|--------------|----------|");
        System.out.printf("| Conv Layers         | %11.2f | %12.2f | %7.2fx |%n",
            javaConvAvgUs, nativeConvAvgUs, javaConvAvgUs / nativeConvAvgUs);
        System.out.printf("| FC Layers           | %11.2f | %12.2f | %7.2fx |%n",
            javaFcAvgUs, nativeFcAvgUs, javaFcAvgUs / nativeFcAvgUs);
        System.out.printf("| Full Model (e2e)    | %11.2f | %12.2f | %7.2fx |%n",
            javaFullAvgUs, nativeFullAvgUs, javaFullAvgUs / nativeFullAvgUs);
        System.out.println();

        // Throughput calculation
        double javaThroughput = 1_000_000.0 / javaFullAvgUs;   // images/sec
        double nativeThroughput = 1_000_000.0 / nativeFullAvgUs;
        System.out.printf("Throughput (images/sec):%n");
        System.out.printf("  Java:   %,.0f images/sec%n", javaThroughput);
        System.out.printf("  Native: %,.0f images/sec%n", nativeThroughput);
        System.out.println();

        if (nativeFullAvgUs < javaFullAvgUs) {
            System.out.println("Native TOSA compilation provides " +
                String.format("%.1fx", javaFullAvgUs / nativeFullAvgUs) +
                " speedup for complete MNIST inference!");
        } else {
            System.out.println("Java JIT is faster for this model configuration.");
        }
        System.out.println();
        System.out.println("Note: This benchmark uses synthetic weights. Real MNIST weights");
        System.out.println("from training would produce similar performance characteristics.");
    }
}
