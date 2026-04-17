package mlir.tosa;

import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Compiles TOSA operations to native x86_64 code using MLIR.
 *
 * This compiler takes Java methods annotated with @Reflect, generates TOSA MLIR,
 * lowers it through the MLIR pipeline to LLVM IR, compiles to a shared library,
 * and provides a way to invoke the native code from Java.
 *
 * Pipeline: TOSA -> Linalg -> Loops -> LLVM Dialect -> LLVM IR -> x86_64 .so
 */
public final class TosaCompiler {

    /**
     * Intermediate compilation stages, for inspection via {@link #generateAtStage}.
     */
    public enum Stage {
        /** Input TOSA MLIR as generated from the code model. */
        TOSA,
        /** After lowering TOSA to Linalg + Arith + Tensor dialects. */
        LINALG,
        /** After full lowering to LLVM dialect, ready for mlir-translate. */
        LLVM_DIALECT,
        /** LLVM IR text produced by mlir-translate (the .ll file). */
        LLVM_IR
    }

    private static final String LINALG_PIPELINE = "builtin.module(" +
        "func.func(tosa-to-linalg-named)," +
        "func.func(tosa-to-linalg)," +
        "func.func(tosa-to-arith)," +
        "func.func(tosa-to-tensor)," +
        "func.func(linalg-fuse-elementwise-ops)" +
        ")";

    private static final String MLIR_OPT = "/usr/lib/llvm-19/bin/mlir-opt";
    private static final String MLIR_TRANSLATE = "/usr/lib/llvm-19/bin/mlir-translate";
    private static final String CLANG = "clang";

    private final boolean verbose;

    public TosaCompiler() {
        this(false);
    }

    public TosaCompiler(boolean verbose) {
        this.verbose = verbose;
    }

    /**
     * Return the MLIR text at the given lowering stage, using dynamic tensor shapes.
     *
     * @param method The @Reflect annotated method
     * @param stage  The lowering stage to inspect
     * @return MLIR text at that stage
     */
    public String generateAtStage(Method method, Stage stage) {
        CoreOp.FuncOp funcOp = Op.ofMethod(method).orElseThrow(
            () -> new IllegalArgumentException("Method is not reflectable: " + method.getName())
        );
        return generateAtStage(funcOp, method.getName(), stage, null);
    }

    /**
     * Return the MLIR text at the given lowering stage, using static tensor shapes.
     *
     * @param method      The @Reflect annotated method
     * @param stage       The lowering stage to inspect
     * @param paramShapes Static shapes for each parameter (null entries use dynamic shapes)
     * @return MLIR text at that stage
     */
    public String generateAtStage(Method method, Stage stage, long[][] paramShapes) {
        CoreOp.FuncOp funcOp = Op.ofMethod(method).orElseThrow(
            () -> new IllegalArgumentException("Method is not reflectable: " + method.getName())
        );
        return generateAtStage(funcOp, method.getName(), stage, paramShapes);
    }

    private String generateAtStage(CoreOp.FuncOp funcOp, String funcName, Stage stage, long[][] paramShapes) {
        try {
            String tosaMlir = (paramShapes != null)
                ? TosaCodeGenerator.generateTosa(funcOp, funcName, paramShapes)
                : TosaCodeGenerator.generateTosa(funcOp, funcName);

            if (stage == Stage.TOSA) {
                return tosaMlir;
            }

            Path tempDir = Files.createTempDirectory("tosa_stage_");
            Path inputFile = tempDir.resolve(funcName + ".mlir");
            Path outputFile = tempDir.resolve(funcName + "_out.mlir");
            Files.writeString(inputFile, tosaMlir);

            if (stage == Stage.LINALG) {
                runMlirOptPipeline(inputFile, outputFile, LINALG_PIPELINE);
            } else if (stage == Stage.LLVM_DIALECT) {
                runMlirOpt(inputFile, outputFile, funcName);
            } else {
                // LLVM_IR: full mlir-opt pipeline, then mlir-translate to .ll
                Path llvmDialectFile = tempDir.resolve(funcName + "_llvm.mlir");
                runMlirOpt(inputFile, llvmDialectFile, funcName);
                runMlirTranslate(llvmDialectFile, outputFile);
            }

            return Files.readString(outputFile);
        } catch (IOException | InterruptedException e) {
            throw new RuntimeException("Stage generation failed", e);
        }
    }

    private void runMlirOptPipeline(Path inputFile, Path outputFile, String pipeline)
            throws IOException, InterruptedException {
        List<String> command = new ArrayList<>();
        command.add(MLIR_OPT);
        command.add(inputFile.toString());
        command.add("--pass-pipeline=" + pipeline);
        command.add("-o");
        command.add(outputFile.toString());
        runCommand(command, "mlir-opt");
    }

    /**
     * Compile a @Reflect annotated method to native code.
     * Defaults to 1D tensors.
     *
     * @param method The method to compile
     * @return A CompiledFunction that can be invoked with tensors
     */
    public CompiledFunction compile(Method method) {
        return compile(method, 1);
    }

    /**
     * Compile a @Reflect annotated method to native code with specified tensor rank.
     *
     * @param method The method to compile
     * @param tensorRank The rank of tensors (1=1D, 2=2D, etc.)
     * @return A CompiledFunction that can be invoked with tensors
     */
    public CompiledFunction compile(Method method, int tensorRank) {
        CoreOp.FuncOp funcOp = Op.ofMethod(method).orElseThrow(
            () -> new IllegalArgumentException("Method is not reflectable: " + method.getName())
        );
        return compile(funcOp, method.getName(), tensorRank);
    }

    /**
     * Compile a @Reflect annotated method to native code with static shapes.
     * This enables native compilation of operations like Conv2D that require
     * at least partially static shapes for lowering.
     *
     * @param method The method to compile
     * @param paramShapes Array of shapes for each parameter (null entries use dynamic shapes)
     * @return A CompiledFunction that can be invoked with tensors
     */
    public CompiledFunction compile(Method method, long[][] paramShapes) {
        CoreOp.FuncOp funcOp = Op.ofMethod(method).orElseThrow(
            () -> new IllegalArgumentException("Method is not reflectable: " + method.getName())
        );
        return compile(funcOp, method.getName(), paramShapes);
    }

    /**
     * Compile a FuncOp to native code.
     * Defaults to 1D tensors.
     *
     * @param funcOp The function operation from code reflection
     * @param funcName The name to use for the generated function
     * @return A CompiledFunction that can be invoked with tensors
     */
    public CompiledFunction compile(CoreOp.FuncOp funcOp, String funcName) {
        return compile(funcOp, funcName, 1);
    }

    /**
     * Compile a FuncOp to native code with specified tensor rank.
     *
     * @param funcOp The function operation from code reflection
     * @param funcName The name to use for the generated function
     * @param tensorRank The rank of tensors (1=1D, 2=2D, etc.)
     * @return A CompiledFunction that can be invoked with tensors
     */
    public CompiledFunction compile(CoreOp.FuncOp funcOp, String funcName, int tensorRank) {
        try {
            Path tempDir = Files.createTempDirectory("tosa_compile_");

            Path tosaFile = tempDir.resolve(funcName + ".mlir");
            final Path llvmDialectFile = tempDir.resolve(funcName + "_llvm.mlir");
            final Path llvmIrFile = tempDir.resolve(funcName + ".ll");
            final Path soFile = tempDir.resolve("lib" + funcName + ".so");

            // Step 1: Generate TOSA MLIR with the specified tensor rank
            if (verbose) {
                System.out.println("[TosaCompiler] Generating TOSA MLIR (rank=" + tensorRank + ")...");
            }
            String tosaMlir = TosaCodeGenerator.generateTosa(funcOp, funcName, tensorRank);
            Files.writeString(tosaFile, tosaMlir);
            if (verbose) {
                System.out.println("[TosaCompiler] TOSA MLIR:\n" + tosaMlir);
            }

            // Step 2: Lower TOSA to LLVM dialect via mlir-opt
            if (verbose) {
                System.out.println("[TosaCompiler] Lowering TOSA -> LLVM dialect...");
            }
            runMlirOpt(tosaFile, llvmDialectFile, funcName);

            // Step 3: Translate LLVM dialect to LLVM IR
            if (verbose) {
                System.out.println("[TosaCompiler] Translating to LLVM IR...");
            }
            runMlirTranslate(llvmDialectFile, llvmIrFile);

            // Step 4: Compile LLVM IR to shared library
            if (verbose) {
                System.out.println("[TosaCompiler] Compiling to shared library...");
            }
            runClang(llvmIrFile, soFile);

            // Step 5: Load the shared library and create CompiledFunction
            if (verbose) {
                System.out.println("[TosaCompiler] Loading shared library: " + soFile);
            }
            return loadCompiledFunction(soFile, funcName, funcOp, tensorRank);

        } catch (IOException | InterruptedException e) {
            throw new RuntimeException("Compilation failed", e);
        }
    }

    /**
     * Compile a FuncOp to native code with static shapes.
     * This enables native compilation of operations like Conv2D that require
     * at least partially static shapes for lowering.
     *
     * @param funcOp The function operation from code reflection
     * @param funcName The name to use for the generated function
     * @param paramShapes Array of shapes for each parameter (null entries use dynamic shapes)
     * @return A CompiledFunction that can be invoked with tensors
     */
    public CompiledFunction compile(CoreOp.FuncOp funcOp, String funcName, long[][] paramShapes) {
        try {
            Path tempDir = Files.createTempDirectory("tosa_compile_");

            Path tosaFile = tempDir.resolve(funcName + ".mlir");
            final Path llvmDialectFile = tempDir.resolve(funcName + "_llvm.mlir");
            final Path llvmIrFile = tempDir.resolve(funcName + ".ll");
            final Path soFile = tempDir.resolve("lib" + funcName + ".so");

            // Determine tensor rank from the first non-null shape
            int tensorRank = 4; // Default to 4D for Conv2D
            for (long[] shape : paramShapes) {
                if (shape != null) {
                    tensorRank = shape.length;
                    break;
                }
            }

            // Step 1: Generate TOSA MLIR with static shapes
            if (verbose) {
                System.out.println("[TosaCompiler] Generating TOSA MLIR with static shapes...");
            }
            String tosaMlir = TosaCodeGenerator.generateTosa(funcOp, funcName, paramShapes);
            Files.writeString(tosaFile, tosaMlir);
            if (verbose) {
                System.out.println("[TosaCompiler] TOSA MLIR:\n" + tosaMlir);
            }

            // Step 2: Lower TOSA to LLVM dialect via mlir-opt
            if (verbose) {
                System.out.println("[TosaCompiler] Lowering TOSA -> LLVM dialect...");
            }
            runMlirOpt(tosaFile, llvmDialectFile, funcName);

            // Step 3: Translate LLVM dialect to LLVM IR
            if (verbose) {
                System.out.println("[TosaCompiler] Translating to LLVM IR...");
            }
            runMlirTranslate(llvmDialectFile, llvmIrFile);

            // Step 4: Compile LLVM IR to shared library
            if (verbose) {
                System.out.println("[TosaCompiler] Compiling to shared library...");
            }
            runClang(llvmIrFile, soFile);

            // Step 5: Load the shared library and create CompiledFunction
            if (verbose) {
                System.out.println("[TosaCompiler] Loading shared library: " + soFile);
            }

            // Extract per-parameter ranks from shapes
            int[] paramRanks = new int[paramShapes.length];
            for (int i = 0; i < paramShapes.length; i++) {
                paramRanks[i] = (paramShapes[i] != null) ? paramShapes[i].length : tensorRank;
            }

            return loadCompiledFunction(soFile, funcName, funcOp, tensorRank, paramRanks);

        } catch (IOException | InterruptedException e) {
            throw new RuntimeException("Compilation failed", e);
        }
    }

    private void runMlirOpt(Path inputFile, Path outputFile, String funcName) throws IOException, InterruptedException {
        // MLIR lowering pipeline: TOSA -> Linalg -> Bufferization -> LLVM
        List<String> command = new ArrayList<>();
        command.add(MLIR_OPT);
        command.add(inputFile.toString());

        // Pipeline passes
        // Note: tosa-to-arith converts tosa.const -> arith.constant for bufferization
        command.add("--pass-pipeline=builtin.module(" +
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
            ")");

        command.add("-o");
        command.add(outputFile.toString());

        runCommand(command, "mlir-opt");
    }

    private void runMlirTranslate(Path inputFile, Path outputFile) throws IOException, InterruptedException {
        List<String> command = List.of(
            MLIR_TRANSLATE,
            "--mlir-to-llvmir",
            inputFile.toString(),
            "-o",
            outputFile.toString()
        );
        runCommand(command, "mlir-translate");
    }

    private void runClang(Path inputFile, Path outputFile) throws IOException, InterruptedException {
        List<String> command = List.of(
            CLANG,
            "-shared",
            "-fPIC",
            "-O2",
            "-o",
            outputFile.toString(),
            inputFile.toString()
        );
        runCommand(command, "clang");
    }

    private void runCommand(List<String> command, String toolName) throws IOException, InterruptedException {
        if (verbose) {
            System.out.println("[TosaCompiler] Running: " + String.join(" ", command));
        }

        ProcessBuilder pb = new ProcessBuilder(command);
        pb.redirectErrorStream(true);
        Process process = pb.start();

        String output = new String(process.getInputStream().readAllBytes());
        int exitCode = process.waitFor();

        if (exitCode != 0) {
            throw new RuntimeException(toolName + " failed with exit code " + exitCode + ":\n" + output);
        }

        if (verbose && !output.isEmpty()) {
            System.out.println("[TosaCompiler] " + toolName + " output:\n" + output);
        }
    }

    private CompiledFunction loadCompiledFunction(Path soFile, String funcName, CoreOp.FuncOp funcOp, int tensorRank) {
        // Load the shared library
        Arena arena = Arena.ofAuto();
        SymbolLookup lookup = SymbolLookup.libraryLookup(soFile, arena);

        // Find the function symbol
        MemorySegment funcAddr = lookup.find(funcName).orElseThrow(
            () -> new RuntimeException("Function symbol not found: " + funcName)
        );

        // Determine the function signature from the FuncOp
        int numParams = funcOp.body().entryBlock().parameters().size();

        // For MLIR memref-based functions, the signature is more complex:
        // Each tensor parameter becomes: (ptr, ptr, offset, size0, size1, ..., stride0, stride1, ...)
        // For a 2D tensor: (allocated_ptr, aligned_ptr, offset, size0, size1, stride0, stride1)
        // That's 7 values per 2D tensor

        // Build the function descriptor with the specified tensor rank
        FunctionDescriptor descriptor = buildFunctionDescriptor(numParams, tensorRank);

        Linker linker = Linker.nativeLinker();
        MethodHandle handle = linker.downcallHandle(funcAddr, descriptor);

        return new CompiledFunction(handle, funcName, numParams, soFile, arena);
    }

    private CompiledFunction loadCompiledFunction(Path soFile, String funcName, CoreOp.FuncOp funcOp,
                                                   int outputRank, int[] paramRanks) {
        // Load the shared library
        Arena arena = Arena.ofAuto();
        SymbolLookup lookup = SymbolLookup.libraryLookup(soFile, arena);

        // Find the function symbol
        MemorySegment funcAddr = lookup.find(funcName).orElseThrow(
            () -> new RuntimeException("Function symbol not found: " + funcName)
        );

        // Determine the function signature from the FuncOp
        int numParams = funcOp.body().entryBlock().parameters().size();

        // Build the function descriptor with per-parameter ranks
        FunctionDescriptor descriptor = buildFunctionDescriptor(paramRanks, outputRank);

        Linker linker = Linker.nativeLinker();
        MethodHandle handle = linker.downcallHandle(funcAddr, descriptor);

        return new CompiledFunction(handle, funcName, numParams, soFile, arena);
    }

    /**
     * Build function descriptor with per-parameter ranks.
     * This is needed for operations like Conv2D where different parameters have different ranks
     * (e.g., 4D input/weight, 1D bias).
     *
     * @param paramRanks Rank for each parameter
     * @param outputRank Rank of the output tensor
     * @return Function descriptor for FFM
     */
    private FunctionDescriptor buildFunctionDescriptor(int[] paramRanks, int outputRank) {
        List<MemoryLayout> paramLayouts = new ArrayList<>();

        for (int rank : paramRanks) {
            paramLayouts.add(ValueLayout.ADDRESS);   // allocated ptr
            paramLayouts.add(ValueLayout.ADDRESS);   // aligned ptr
            paramLayouts.add(ValueLayout.JAVA_LONG); // offset

            // Add size and stride for each dimension
            for (int d = 0; d < rank; d++) {
                paramLayouts.add(ValueLayout.JAVA_LONG); // size[d]
            }
            for (int d = 0; d < rank; d++) {
                paramLayouts.add(ValueLayout.JAVA_LONG); // stride[d]
            }
        }

        // Build return type struct layout using outputRank
        List<MemoryLayout> returnMembers = new ArrayList<>();
        returnMembers.add(ValueLayout.ADDRESS);    // allocated ptr
        returnMembers.add(ValueLayout.ADDRESS);    // aligned ptr
        returnMembers.add(ValueLayout.JAVA_LONG);  // offset
        for (int d = 0; d < outputRank; d++) {
            returnMembers.add(ValueLayout.JAVA_LONG);  // size[d]
        }
        for (int d = 0; d < outputRank; d++) {
            returnMembers.add(ValueLayout.JAVA_LONG);  // stride[d]
        }

        MemoryLayout returnLayout = MemoryLayout.structLayout(returnMembers.toArray(new MemoryLayout[0]));

        return FunctionDescriptor.of(returnLayout, paramLayouts.toArray(new MemoryLayout[0]));
    }

    /**
     * Build function descriptor for the specified tensor rank.
     *
     * For MLIR's memref ABI, each tensor parameter becomes:
     * - rank 0 (scalar): 3 values (ptr, ptr, offset)
     * - rank 1 (1D):     5 values (ptr, ptr, offset, size[0], stride[0])
     * - rank 2 (2D):     7 values (ptr, ptr, offset, size[0], size[1], stride[0], stride[1])
     *
     * Return type:
     * - rank 0: { ptr, ptr, i64 } = 24 bytes, returned in registers (RAX, RDX)
     * - rank 1: { ptr, ptr, i64, [1 x i64], [1 x i64] } = 40 bytes, returned via sret
     * - rank 2: { ptr, ptr, i64, [2 x i64], [2 x i64] } = 56 bytes, returned via sret
     *
     * @param numParams Number of tensor parameters
     * @param rank Tensor rank (0=scalar, 1=1D, 2=2D)
     * @return Function descriptor for FFM
     */
    private FunctionDescriptor buildFunctionDescriptor(int numParams, int rank) {
        List<MemoryLayout> paramLayouts = new ArrayList<>();

        for (int i = 0; i < numParams; i++) {
            paramLayouts.add(ValueLayout.ADDRESS);   // allocated ptr
            paramLayouts.add(ValueLayout.ADDRESS);   // aligned ptr
            paramLayouts.add(ValueLayout.JAVA_LONG); // offset

            // Add size and stride for each dimension
            for (int d = 0; d < rank; d++) {
                paramLayouts.add(ValueLayout.JAVA_LONG); // size[d]
            }
            for (int d = 0; d < rank; d++) {
                paramLayouts.add(ValueLayout.JAVA_LONG); // stride[d]
            }
        }

        // Build return type struct layout
        // { ptr, ptr, i64, [rank x i64], [rank x i64] }
        List<MemoryLayout> returnMembers = new ArrayList<>();
        returnMembers.add(ValueLayout.ADDRESS);    // allocated ptr
        returnMembers.add(ValueLayout.ADDRESS);    // aligned ptr
        returnMembers.add(ValueLayout.JAVA_LONG);  // offset
        for (int d = 0; d < rank; d++) {
            returnMembers.add(ValueLayout.JAVA_LONG);  // size[d]
        }
        for (int d = 0; d < rank; d++) {
            returnMembers.add(ValueLayout.JAVA_LONG);  // stride[d]
        }

        MemoryLayout returnLayout = MemoryLayout.structLayout(returnMembers.toArray(new MemoryLayout[0]));

        return FunctionDescriptor.of(returnLayout, paramLayouts.toArray(new MemoryLayout[0]));
    }
}
