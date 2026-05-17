package mlir.tosa;

import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.dialect.core.CoreOp;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.SequencedMap;

import static mlir.tosa.TosaOperators.*;

/**
 * Benchmark comparing TOSA Native, TOSA Java, and ONNX Runtime performance.
 *
 * Run with: mvn test -Dtest=TosaBenchmark#runFullBenchmark
 */
public class TosaBenchmark {

    private static final int WARMUP_ITERATIONS = 20_000;
    private static final int BENCHMARK_ITERATIONS = 100_000;
    private static final int NUM_RANDOM_INPUTS = 100;

    /**
     * MNIST Feature Extractor - simplified CNN for benchmarking.
     *
     * Architecture:
     * - Conv1 (1->6 channels, 5x5) -> ReLU -> MaxPool (2x2)
     * - Conv2 (6->16 channels, 5x5) -> ReLU -> MaxPool (2x2)
     *
     * Input: [1, 28, 28, 1] (NHWC)
     * Output: [1, 4, 4, 16] (NHWC)
     */
    public static class MNISTFeatureExtractor {

        final Tensor<Float> conv1Weights;
        final Tensor<Float> conv1Bias;
        final Tensor<Float> conv2Weights;
        final Tensor<Float> conv2Bias;

        public MNISTFeatureExtractor() {
            Tensor<Float> conv1WeightsRaw = load("mnist/conv1-weight-float-le", 6, 1, 5, 5);
            conv1Weights = transposeConvWeights(conv1WeightsRaw, 6, 1, 5, 5);
            conv1Bias = load("mnist/conv1-bias-float-le", 6);

            Tensor<Float> conv2WeightsRaw = load("mnist/conv2-weight-float-le", 16, 6, 5, 5);
            conv2Weights = transposeConvWeights(conv2WeightsRaw, 16, 6, 5, 5);
            conv2Bias = load("mnist/conv2-bias-float-le", 16);
        }

        private static Tensor<Float> load(String resource, long... shape) {
            try (var in = MNISTFeatureExtractor.class.getResourceAsStream(resource)) {
                if (in == null) {
                    throw new IllegalStateException("Resource not found: " + resource);
                }
                return Tensor.ofBytes(shape, in.readAllBytes());
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }

        private static Tensor<Float> transposeConvWeights(Tensor<Float> weights, long oc, long ic, long kh, long kw) {
            float[] srcData = new float[(int)(oc * ic * kh * kw)];
            float[] dstData = new float[(int)(oc * ic * kh * kw)];

            for (int i = 0; i < srcData.length; i++) {
                srcData[i] = weights.data().getAtIndex(ValueLayout.JAVA_FLOAT, i);
            }

            for (int o = 0; o < oc; o++) {
                for (int i = 0; i < ic; i++) {
                    for (int h = 0; h < kh; h++) {
                        for (int w = 0; w < kw; w++) {
                            int srcIdx = (int)(o * ic * kh * kw + i * kh * kw + h * kw + w);
                            int dstIdx = (int)(o * kh * kw * ic + h * kw * ic + w * ic + i);
                            dstData[dstIdx] = srcData[srcIdx];
                        }
                    }
                }
            }

            return Tensor.ofFloats(new long[]{oc, kh, kw, ic}, dstData);
        }

        @Reflect
        public Tensor<Float> extract(Tensor<Float> inputImage) {
            Tensor<Float> conv1 = Conv2D(inputImage, conv1Weights, conv1Bias,
                    new long[]{0, 0, 0, 0}, new long[]{1, 1}, new long[]{1, 1});
            Tensor<Float> relu1 = conv1.relu();
            Tensor<Float> pool1 = MaxPool2D(relu1, new long[]{2, 2}, new long[]{2, 2}, new long[]{0, 0, 0, 0});

            Tensor<Float> conv2 = Conv2D(pool1, conv2Weights, conv2Bias,
                    new long[]{0, 0, 0, 0}, new long[]{1, 1}, new long[]{1, 1});
            Tensor<Float> relu2 = conv2.relu();

            return MaxPool2D(relu2, new long[]{2, 2}, new long[]{2, 2}, new long[]{0, 0, 0, 0});
        }
    }

    @Test
    public void runFullBenchmark() throws Exception {
        System.out.println("=".repeat(80));
        System.out.println("TOSA Performance Benchmark: Native vs Java vs ONNX Runtime");
        System.out.println("=".repeat(80));
        System.out.printf("Warmup iterations: %,d%n", WARMUP_ITERATIONS);
        System.out.printf("Benchmark iterations: %,d%n", BENCHMARK_ITERATIONS);
        System.out.println();

        // Create model
        MNISTFeatureExtractor model = new MNISTFeatureExtractor();

        // Create random inputs for benchmarking
        System.out.println("Generating random inputs...");
        Random random = new Random(42);
        float[][] randomInputs = new float[NUM_RANDOM_INPUTS][28 * 28];
        for (int i = 0; i < NUM_RANDOM_INPUTS; i++) {
            for (int j = 0; j < 28 * 28; j++) {
                randomInputs[i][j] = random.nextFloat();
            }
        }

        // Pre-create tensors for TOSA benchmarks
        @SuppressWarnings("unchecked")
        Tensor<Float>[] inputTensors = (Tensor<Float>[]) new Tensor[NUM_RANDOM_INPUTS];
        for (int i = 0; i < NUM_RANDOM_INPUTS; i++) {
            inputTensors[i] = Tensor.ofFloats(new long[]{1, 28, 28, 1}, randomInputs[i]);
        }

        // Compile native function
        System.out.println("Compiling native function...");
        NativeFunction nativeFunc = compileNativeFunction(model);
        System.out.println("Native compilation complete.\n");

        // ===== TOSA JAVA BENCHMARK =====
        System.out.println("-".repeat(60));
        System.out.println("TOSA Java Benchmark");
        System.out.println("-".repeat(60));

        // Warmup
        System.out.print("Warming up TOSA Java (" + WARMUP_ITERATIONS + " iterations)...");
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            model.extract(inputTensors[i % NUM_RANDOM_INPUTS]);
        }
        System.out.println(" done");

        // Benchmark
        System.out.print("Running TOSA Java benchmark (" + BENCHMARK_ITERATIONS + " iterations)...");
        long javaStart = System.nanoTime();
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            model.extract(inputTensors[i % NUM_RANDOM_INPUTS]);
        }
        long javaEnd = System.nanoTime();
        double javaTimeMs = (javaEnd - javaStart) / 1_000_000.0;
        double javaPerIterUs = (javaTimeMs * 1000) / BENCHMARK_ITERATIONS;
        double javaOpsPerSec = BENCHMARK_ITERATIONS / (javaTimeMs / 1000.0);
        System.out.println(" done");
        System.out.printf("Total time: %.2f ms%n", javaTimeMs);
        System.out.printf("Per iteration: %.2f µs%n", javaPerIterUs);
        System.out.printf("Throughput: %,.0f inferences/sec%n", javaOpsPerSec);
        System.out.println();

        // ===== TOSA NATIVE BENCHMARK =====
        System.out.println("-".repeat(60));
        System.out.println("TOSA Native Benchmark");
        System.out.println("-".repeat(60));

        // Warmup
        System.out.print("Warming up TOSA Native (" + WARMUP_ITERATIONS + " iterations)...");
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            nativeFunc.invoke(inputTensors[i % NUM_RANDOM_INPUTS]);
        }
        System.out.println(" done");

        // Benchmark
        System.out.print("Running TOSA Native benchmark (" + BENCHMARK_ITERATIONS + " iterations)...");
        long nativeStart = System.nanoTime();
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            nativeFunc.invoke(inputTensors[i % NUM_RANDOM_INPUTS]);
        }
        long nativeEnd = System.nanoTime();
        double nativeTimeMs = (nativeEnd - nativeStart) / 1_000_000.0;
        double nativePerIterUs = (nativeTimeMs * 1000) / BENCHMARK_ITERATIONS;
        double nativeOpsPerSec = BENCHMARK_ITERATIONS / (nativeTimeMs / 1000.0);
        System.out.println(" done");
        System.out.printf("Total time: %.2f ms%n", nativeTimeMs);
        System.out.printf("Per iteration: %.2f µs%n", nativePerIterUs);
        System.out.printf("Throughput: %,.0f inferences/sec%n", nativeOpsPerSec);
        System.out.println();

        // ===== RESULTS SUMMARY =====
        System.out.println("=".repeat(80));
        System.out.println("BENCHMARK RESULTS SUMMARY");
        System.out.println("=".repeat(80));
        System.out.println();
        System.out.printf("%-20s %15s %15s %18s%n", "Implementation", "Total (ms)", "Per iter (µs)", "Throughput (ops/s)");
        System.out.println("-".repeat(70));
        System.out.printf("%-20s %15.2f %15.2f %18.0f%n", "TOSA Java", javaTimeMs, javaPerIterUs, javaOpsPerSec);
        System.out.printf("%-20s %15.2f %15.2f %18.0f%n", "TOSA Native", nativeTimeMs, nativePerIterUs, nativeOpsPerSec);
        System.out.println("-".repeat(70));

        double speedup = javaTimeMs / nativeTimeMs;
        if (speedup > 1) {
            System.out.printf("%nNative is %.2fx FASTER than Java%n", speedup);
        } else {
            System.out.printf("%nJava is %.2fx FASTER than Native%n", 1.0 / speedup);
        }

        // Verify correctness
        System.out.println("\n" + "-".repeat(60));
        System.out.println("Correctness Verification");
        System.out.println("-".repeat(60));
        Tensor<Float> javaOutput = model.extract(inputTensors[0]);
        float[] nativeOutput = nativeFunc.invoke(inputTensors[0]);

        boolean allMatch = true;
        int mismatches = 0;
        long outputSize = javaOutput.shape()[0] * javaOutput.shape()[1] * javaOutput.shape()[2] * javaOutput.shape()[3];
        for (int i = 0; i < outputSize; i++) {
            float javaVal = javaOutput.data().getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float nativeVal = nativeOutput[i];
            float diff = Math.abs(javaVal - nativeVal);
            if (diff > 0.0001f) {
                allMatch = false;
                mismatches++;
                if (mismatches <= 5) {
                    System.out.printf("MISMATCH at [%d]: Java=%.6f, Native=%.6f, diff=%.6f%n", i, javaVal, nativeVal, diff);
                }
            }
        }
        if (allMatch) {
            System.out.println("✓ All " + outputSize + " outputs match between Java and Native implementations");
        } else {
            System.out.printf("✗ %d mismatches out of %d outputs%n", mismatches, outputSize);
        }
    }

    /**
     * Compile the model to native code with embedded weights.
     */
    private NativeFunction compileNativeFunction(MNISTFeatureExtractor model) throws Exception {
        var method = MNISTFeatureExtractor.class.getMethod("extract", Tensor.class);
        CoreOp.FuncOp funcOp = Op.ofMethod(method).orElseThrow();

        SequencedMap<String, TosaModelExporter.WeightInfo> weights =
            TosaModelExporter.extractWeights(model);

        long[] inputShape = {1, 28, 28, 1};
        String mlir = TosaCodeGenerator.generateTosaWithWeights(
            funcOp, "extract", weights, model, inputShape);

        // Write MLIR to file
        Path tempDir = Files.createTempDirectory("tosa_benchmark_");
        Path mlirFile = tempDir.resolve("extract.mlir");
        Path llvmDialectFile = tempDir.resolve("extract_llvm.mlir");
        Path llvmIrFile = tempDir.resolve("extract.ll");
        Path soFile = tempDir.resolve("libextract.so");

        Files.writeString(mlirFile, mlir);

        String mlirOpt = "/usr/lib/llvm-19/bin/mlir-opt";
        String mlirTranslate = "/usr/lib/llvm-19/bin/mlir-translate";

        // Lower TOSA to LLVM with optimization passes
        ProcessBuilder pb1 = new ProcessBuilder(
            mlirOpt, mlirFile.toString(),
            "--pass-pipeline=builtin.module(" +
                "func.func(tosa-to-linalg-named)," +
                "func.func(tosa-to-linalg)," +
                "func.func(tosa-to-arith)," +
                "func.func(tosa-to-tensor)," +
                "func.func(linalg-fuse-elementwise-ops)," +
                "one-shot-bufferize{bufferize-function-boundaries}," +
                "func.func(convert-linalg-to-loops)," +
                "func.func(lower-affine)," +
                "func.func(arith-expand)," +
                "func.func(convert-scf-to-cf)," +
                "convert-arith-to-llvm," +
                "convert-func-to-llvm," +
                "convert-cf-to-llvm," +
                "finalize-memref-to-llvm," +
                "reconcile-unrealized-casts" +
            ")",
            "-o", llvmDialectFile.toString()
        );
        pb1.redirectErrorStream(true);
        Process p1 = pb1.start();
        String out1 = new String(p1.getInputStream().readAllBytes());
        if (p1.waitFor() != 0) throw new RuntimeException("mlir-opt failed: " + out1);

        // Translate to LLVM IR
        ProcessBuilder pb2 = new ProcessBuilder(
            mlirTranslate, "--mlir-to-llvmir", llvmDialectFile.toString(), "-o", llvmIrFile.toString());
        pb2.redirectErrorStream(true);
        Process p2 = pb2.start();
        String out2 = new String(p2.getInputStream().readAllBytes());
        if (p2.waitFor() != 0) throw new RuntimeException("mlir-translate failed: " + out2);

        // Compile to .so with optimizations
        // Note: -march=native enables AVX/AVX2/AVX-512 but vectorization is limited
        // because MLIR's convert-linalg-to-loops generates scalar loops.
        // For better performance, would need MLIR vectorization passes or optimized libraries.
        ProcessBuilder pb3 = new ProcessBuilder(
            "clang", "-shared", "-fPIC",
            "-O3",
            "-march=native",      // Use native CPU features (AVX, AVX2, FMA, etc.)
            "-ffast-math",        // Fast floating-point
            "-o", soFile.toString(), llvmIrFile.toString());
        pb3.redirectErrorStream(true);
        Process p3 = pb3.start();
        String out3 = new String(p3.getInputStream().readAllBytes());
        if (p3.waitFor() != 0) throw new RuntimeException("clang failed: " + out3);

        return new NativeFunction(soFile);
    }

    /**
     * Wrapper for native function invocation.
     */
    private static class NativeFunction {
        private final MethodHandle handle;
        private final Arena libraryArena;

        NativeFunction(Path soFile) {
            this.libraryArena = Arena.ofAuto();
            SymbolLookup lookup = SymbolLookup.libraryLookup(soFile, libraryArena);

            MemorySegment funcAddr = lookup.find("extract").orElseThrow(
                () -> new RuntimeException("Function 'extract' not found"));

            int rank = 4;
            List<MemoryLayout> paramLayouts = new ArrayList<>();
            paramLayouts.add(ValueLayout.ADDRESS);
            paramLayouts.add(ValueLayout.ADDRESS);
            paramLayouts.add(ValueLayout.JAVA_LONG);
            for (int d = 0; d < rank; d++) paramLayouts.add(ValueLayout.JAVA_LONG);
            for (int d = 0; d < rank; d++) paramLayouts.add(ValueLayout.JAVA_LONG);

            List<MemoryLayout> returnMembers = new ArrayList<>();
            returnMembers.add(ValueLayout.ADDRESS);
            returnMembers.add(ValueLayout.ADDRESS);
            returnMembers.add(ValueLayout.JAVA_LONG);
            for (int d = 0; d < rank; d++) returnMembers.add(ValueLayout.JAVA_LONG);
            for (int d = 0; d < rank; d++) returnMembers.add(ValueLayout.JAVA_LONG);
            MemoryLayout returnLayout = MemoryLayout.structLayout(returnMembers.toArray(new MemoryLayout[0]));

            FunctionDescriptor descriptor = FunctionDescriptor.of(returnLayout,
                paramLayouts.toArray(new MemoryLayout[0]));

            Linker linker = Linker.nativeLinker();
            this.handle = linker.downcallHandle(funcAddr, descriptor);
        }

        float[] invoke(Tensor<Float> input) {
            try (Arena invokeArena = Arena.ofConfined()) {
                Object[] args = new Object[12];
                args[0] = (SegmentAllocator) invokeArena;
                args[1] = input.data();
                args[2] = input.data();
                args[3] = 0L;
                args[4] = 1L;
                args[5] = 28L;
                args[6] = 28L;
                args[7] = 1L;
                args[8] = 784L;
                args[9] = 28L;
                args[10] = 1L;
                args[11] = 1L;

                MemorySegment resultStruct = (MemorySegment) handle.invokeWithArguments(args);

                long offset = ValueLayout.ADDRESS.byteSize() * 2 + ValueLayout.JAVA_LONG.byteSize();
                long[] resultShape = new long[4];
                for (int d = 0; d < 4; d++) {
                    resultShape[d] = resultStruct.get(ValueLayout.JAVA_LONG, offset + d * ValueLayout.JAVA_LONG.byteSize());
                }

                long resultElements = resultShape[0] * resultShape[1] * resultShape[2] * resultShape[3];
                MemorySegment alignedPtr = resultStruct.get(ValueLayout.ADDRESS, ValueLayout.ADDRESS.byteSize());
                MemorySegment nativeData = alignedPtr.reinterpret(resultElements * 4);

                float[] output = new float[(int) resultElements];
                for (int i = 0; i < resultElements; i++) {
                    output[i] = nativeData.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                }
                return output;
            } catch (Throwable t) {
                throw new RuntimeException("Native invocation failed", t);
            }
        }
    }

    // Legacy benchmark methods from original file
    @Reflect
    public static Tensor<Float> convBlock(Tensor<Float> input, Tensor<Float> weight, Tensor<Float> bias) {
        Tensor<Float> conv = TosaOperators.Conv2D(input, weight, bias,
            new long[]{0, 0, 0, 0}, new long[]{1, 1}, new long[]{1, 1});
        Tensor<Float> relu = conv.relu();
        return TosaOperators.MaxPool2D(relu,
            new long[]{2, 2}, new long[]{2, 2}, new long[]{0, 0, 0, 0});
    }

    @Reflect
    public static Tensor<Float> mnistConvBlock(Tensor<Float> input, Tensor<Float> weight, Tensor<Float> bias) {
        Tensor<Float> conv = TosaOperators.Conv2D(input, weight, bias,
            new long[]{0, 0, 0, 0}, new long[]{1, 1}, new long[]{1, 1});
        Tensor<Float> relu = conv.relu();
        return TosaOperators.MaxPool2D(relu,
            new long[]{2, 2}, new long[]{2, 2}, new long[]{0, 0, 0, 0});
    }
}
