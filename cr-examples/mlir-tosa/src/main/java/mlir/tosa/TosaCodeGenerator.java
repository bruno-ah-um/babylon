package mlir.tosa;

import jdk.incubator.code.Block;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.ArrayType;
import jdk.incubator.code.dialect.java.JavaOp;
import mlir.tosa.bindings.mlir_tosa_c_api_h;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SequencedMap;

/**
 * Generates TOSA/MLIR code from Java code reflection models using native MLIR bindings.
 *
 * This class transforms the Java code model captured via @Reflect annotations
 * into TOSA (Tensor Operator Set Architecture) MLIR representation using the
 * native mlir_tosa_c library for proper MLIR IR construction and validation.
 */
public final class TosaCodeGenerator {

    // Ensure the native library is extracted from the JAR (if bundled) before
    // mlir_tosa_c_api_h's static initializer tries to look it up.
    static { TosaNativeLoader.ensureLoaded(); }

    private TosaCodeGenerator() {
        // Utility class
    }

    /**
     * Generate TOSA MLIR code from a @Reflect annotated method.
     *
     * @param method The method to transform
     * @return TOSA MLIR code as a string
     */
    public static String generateTosa(Method method) {
        CoreOp.FuncOp funcOp = Op.ofMethod(method).orElseThrow(
            () -> new IllegalArgumentException("Method is not reflectable: " + method.getName())
        );
        return generateTosa(funcOp, method.getName());
    }

    /**
     * Generate TOSA MLIR code from a FuncOp.
     * Defaults to 1D dynamic tensors.
     *
     * @param funcOp The function operation from code reflection
     * @param funcName The name to use for the generated function
     * @return TOSA MLIR code as a string
     */
    public static String generateTosa(CoreOp.FuncOp funcOp, String funcName) {
        return generateTosa(funcOp, funcName, 1);
    }

    /**
     * Generate TOSA MLIR code from a FuncOp with specified tensor rank.
     *
     * @param funcOp The function operation from code reflection
     * @param funcName The name to use for the generated function
     * @param tensorRank The rank of tensors (1=1D, 2=2D, etc.)
     * @return TOSA MLIR code as a string
     */
    public static String generateTosa(CoreOp.FuncOp funcOp, String funcName, int tensorRank) {
        return generateTosa(funcOp, funcName, tensorRank, null);
    }

    /**
     * Generate TOSA MLIR code from a FuncOp with static shapes.
     * This enables native compilation of operations like Conv2D that require
     * at least partially static shapes for lowering.
     *
     * @param funcOp The function operation from code reflection
     * @param funcName The name to use for the generated function
     * @param paramShapes Array of shapes for each parameter (null entries use dynamic shapes)
     * @return TOSA MLIR code as a string
     */
    public static String generateTosa(CoreOp.FuncOp funcOp, String funcName, long[][] paramShapes) {
        // Determine tensor rank from the first non-null shape
        int tensorRank = 4; // Default to 4D
        for (long[] shape : paramShapes) {
            if (shape != null) {
                tensorRank = shape.length;
                break;
            }
        }
        return generateTosa(funcOp, funcName, tensorRank, paramShapes);
    }

    /**
     * Generate TOSA MLIR code from a FuncOp with embedded weights.
     * This method is used when exporting trained models with weights.
     *
     * @param funcOp The function operation from code reflection
     * @param funcName The name to use for the generated function
     * @param weights Map of field name to WeightInfo for embedding constants
     * @param modelInstance The model instance for resolving field accesses
     * @return TOSA MLIR code as a string with embedded weight constants
     */
    public static String generateTosaWithWeights(CoreOp.FuncOp funcOp, String funcName,
                                                   SequencedMap<String, TosaModelExporter.WeightInfo> weights,
                                                   Object modelInstance) {
        try (Arena arena = Arena.ofConfined()) {
            // Create MLIR context and module
            MemorySegment context = mlir_tosa_c_api_h.mlir_context_create();
            if (context.equals(MemorySegment.NULL)) {
                throw new RuntimeException("Failed to create MLIR context: " + getLastError());
            }

            MemorySegment module = mlir_tosa_c_api_h.mlir_module_create(context);
            if (module.equals(MemorySegment.NULL)) {
                mlir_tosa_c_api_h.mlir_context_destroy(context);
                throw new RuntimeException("Failed to create MLIR module: " + getLastError());
            }

            try {
                // Default to 4D tensors for CNN models
                int tensorRank = 4;
                GeneratorContext ctx = new GeneratorContext(arena, context, tensorRank, null, weights, modelInstance);

                // Get function parameters - filter out weight parameters (they become constants)
                List<Block.Parameter> params = funcOp.body().entryBlock().parameters();

                // Create input types for Tensor parameters only
                // Skip non-Tensor parameters (like 'this' for instance methods)
                List<MemorySegment> inputTypes = new ArrayList<>();
                List<Block.Parameter> inputParams = new ArrayList<>();
                for (Block.Parameter param : params) {
                    String typeStr = param.type().toString();
                    // Skip non-Tensor parameters
                    if (!typeStr.contains("Tensor")) {
                        continue;
                    }
                    MemorySegment tosaType = javaTypeToNativeTosaType(ctx, param.type(), ctx.tensorRank);
                    inputTypes.add(tosaType);
                    inputParams.add(param);
                }

                // Create output type
                MemorySegment outputType = javaTypeToNativeTosaType(
                    ctx, funcOp.invokableType().returnType(), tensorRank);

                // Allocate arrays for input/output types
                MemorySegment inputTypesArray = arena.allocate(ValueLayout.ADDRESS, inputTypes.size());
                for (int i = 0; i < inputTypes.size(); i++) {
                    inputTypesArray.setAtIndex(ValueLayout.ADDRESS, i, inputTypes.get(i));
                }

                MemorySegment outputTypesArray = arena.allocate(ValueLayout.ADDRESS, 1);
                outputTypesArray.setAtIndex(ValueLayout.ADDRESS, 0, outputType);

                // Create function name as native string
                MemorySegment funcNameNative = arena.allocateFrom(funcName);

                // Create the function
                MemorySegment function = mlir_tosa_c_api_h.mlir_function_create(
                    module,
                    funcNameNative,
                    inputTypesArray,
                    inputTypes.size(),
                    outputTypesArray,
                    1
                );

                if (function.equals(MemorySegment.NULL)) {
                    throw new RuntimeException("Failed to create MLIR function: " + getLastError());
                }

                // Map function arguments to values
                for (int i = 0; i < inputParams.size(); i++) {
                    MemorySegment arg = mlir_tosa_c_api_h.mlir_function_get_argument(function, i);
                    ctx.valueHandles.put(inputParams.get(i), arg);
                }

                // Process the function body
                Block entryBlock = funcOp.body().entryBlock();
                for (Op op : entryBlock.ops()) {
                    processOpWithWeights(op, ctx, function);
                }

                // Verify the module
                int verifyResult = mlir_tosa_c_api_h.mlir_module_verify(module);
                if (verifyResult != 0) {
                    throw new RuntimeException("MLIR module verification failed: " + getLastError());
                }

                // Convert to string
                MemorySegment strPtr = mlir_tosa_c_api_h.mlir_module_to_string(module);
                if (strPtr.equals(MemorySegment.NULL)) {
                    throw new RuntimeException("Failed to convert module to string: " + getLastError());
                }

                String result = strPtr.reinterpret(Long.MAX_VALUE).getString(0);
                mlir_tosa_c_api_h.mlir_string_destroy(strPtr);

                return result;
            } finally {
                mlir_tosa_c_api_h.mlir_module_destroy(module);
                mlir_tosa_c_api_h.mlir_context_destroy(context);
            }
        }
    }

    /**
     * Generate TOSA MLIR code from a FuncOp with embedded weights and static shapes.
     * This version supports native compilation by using static tensor shapes.
     *
     * @param funcOp The function operation from code reflection
     * @param funcName The name to use for the generated function
     * @param weights Map of field name to WeightInfo for embedding constants
     * @param modelInstance The model instance for resolving field accesses
     * @param inputShape Static shape for the input tensor (e.g., [1, 28, 28, 1] for MNIST)
     * @return TOSA MLIR code as a string with embedded weight constants and static shapes
     */
    public static String generateTosaWithWeights(CoreOp.FuncOp funcOp, String funcName,
                                                   SequencedMap<String, TosaModelExporter.WeightInfo> weights,
                                                   Object modelInstance,
                                                   long[] inputShape) {
        try (Arena arena = Arena.ofConfined()) {
            // Create MLIR context and module
            MemorySegment context = mlir_tosa_c_api_h.mlir_context_create();
            if (context.equals(MemorySegment.NULL)) {
                throw new RuntimeException("Failed to create MLIR context: " + getLastError());
            }

            MemorySegment module = mlir_tosa_c_api_h.mlir_module_create(context);
            if (module.equals(MemorySegment.NULL)) {
                mlir_tosa_c_api_h.mlir_context_destroy(context);
                throw new RuntimeException("Failed to create MLIR module: " + getLastError());
            }

            try {
                int tensorRank = inputShape.length;
                GeneratorContext ctx = new GeneratorContext(arena, context, tensorRank, null, weights, modelInstance);

                // Get function parameters - filter to only Tensor parameters
                List<Block.Parameter> params = funcOp.body().entryBlock().parameters();

                // Create input types with static shapes (only for Tensor parameters)
                List<MemorySegment> inputTypes = new ArrayList<>();
                List<Block.Parameter> inputParams = new ArrayList<>();
                int tensorParamIndex = 0;
                for (Block.Parameter param : params) {
                    String typeStr = param.type().toString();
                    // Skip non-Tensor parameters (like 'this' for instance methods)
                    if (!typeStr.contains("Tensor")) {
                        continue;
                    }

                    MemorySegment tosaType;
                    // Apply static shape to the first tensor parameter (the input)
                    if (tensorParamIndex == 0 && inputShape != null) {
                        tosaType = javaTypeToNativeTosaTypeStatic(ctx, param.type(), inputShape);
                    } else {
                        tosaType = javaTypeToNativeTosaType(ctx, param.type(), tensorRank);
                    }
                    inputTypes.add(tosaType);
                    inputParams.add(param);
                    tensorParamIndex++;
                }

                // Create output type (dynamic - MLIR will infer it)
                MemorySegment outputType = javaTypeToNativeTosaType(
                    ctx, funcOp.invokableType().returnType(), tensorRank);

                // Allocate arrays for input/output types
                MemorySegment inputTypesArray = arena.allocate(ValueLayout.ADDRESS, inputTypes.size());
                for (int i = 0; i < inputTypes.size(); i++) {
                    inputTypesArray.setAtIndex(ValueLayout.ADDRESS, i, inputTypes.get(i));
                }

                MemorySegment outputTypesArray = arena.allocate(ValueLayout.ADDRESS, 1);
                outputTypesArray.setAtIndex(ValueLayout.ADDRESS, 0, outputType);

                // Create function name as native string
                MemorySegment funcNameNative = arena.allocateFrom(funcName);

                // Create the function
                MemorySegment function = mlir_tosa_c_api_h.mlir_function_create(
                    module,
                    funcNameNative,
                    inputTypesArray,
                    inputTypes.size(),
                    outputTypesArray,
                    1
                );

                if (function.equals(MemorySegment.NULL)) {
                    throw new RuntimeException("Failed to create MLIR function: " + getLastError());
                }

                // Map function arguments to values
                for (int i = 0; i < inputParams.size(); i++) {
                    MemorySegment arg = mlir_tosa_c_api_h.mlir_function_get_argument(function, i);
                    ctx.valueHandles.put(inputParams.get(i), arg);
                }

                // Process the function body
                Block entryBlock = funcOp.body().entryBlock();
                for (Op op : entryBlock.ops()) {
                    processOpWithWeights(op, ctx, function);
                }

                // Verify the module
                int verifyResult = mlir_tosa_c_api_h.mlir_module_verify(module);
                if (verifyResult != 0) {
                    throw new RuntimeException("MLIR module verification failed: " + getLastError());
                }

                // Convert to string
                MemorySegment strPtr = mlir_tosa_c_api_h.mlir_module_to_string(module);
                if (strPtr.equals(MemorySegment.NULL)) {
                    throw new RuntimeException("Failed to convert module to string: " + getLastError());
                }

                String result = strPtr.reinterpret(Long.MAX_VALUE).getString(0);
                mlir_tosa_c_api_h.mlir_string_destroy(strPtr);

                return result;
            } finally {
                mlir_tosa_c_api_h.mlir_module_destroy(module);
                mlir_tosa_c_api_h.mlir_context_destroy(context);
            }
        }
    }

    /**
     * Generate TOSA MLIR code from a captured lambda (LambdaOp).
     *
     * This is the entry point for the TosaTransformer path: a Quotable lambda is
     * captured via code reflection and transformed to TOSA MLIR.  Lambda parameters
     * must all be {@code Tensor<?>} values; the output type is inferred as a dynamic
     * tensor of the same element type as the first parameter.
     *
     * @param lambdaOp  The lambda operation from {@code Op.ofLambda()}
     * @param funcName  Name for the generated MLIR function
     * @param tensorRank Rank of input/output tensors (1 = 1-D dynamic, 2 = 2-D dynamic, …)
     * @return TOSA MLIR text representation
     */
    public static String generateTosaFromLambda(JavaOp.LambdaOp lambdaOp, String funcName, int tensorRank) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment context = mlir_tosa_c_api_h.mlir_context_create();
            if (context.equals(MemorySegment.NULL)) {
                throw new RuntimeException("Failed to create MLIR context: " + getLastError());
            }

            MemorySegment module = mlir_tosa_c_api_h.mlir_module_create(context);
            if (module.equals(MemorySegment.NULL)) {
                mlir_tosa_c_api_h.mlir_context_destroy(context);
                throw new RuntimeException("Failed to create MLIR module: " + getLastError());
            }

            try {
                GeneratorContext ctx = new GeneratorContext(arena, context, tensorRank, null);

                Block entryBlock = lambdaOp.body().entryBlock();
                List<Block.Parameter> params = entryBlock.parameters();

                // Build input types from lambda parameters
                List<MemorySegment> inputTypes = new ArrayList<>();
                List<Block.Parameter> tensorParams = new ArrayList<>();
                for (Block.Parameter param : params) {
                    if (!param.type().toString().contains("Tensor")) {
                        continue; // skip captured non-tensor values
                    }
                    inputTypes.add(javaTypeToNativeTosaType(ctx, param.type(), tensorRank));
                    tensorParams.add(param);
                }

                // Output type: dynamic tensor, element type inferred from first tensor param
                MemorySegment elemType = tensorParams.isEmpty()
                    ? mlir_tosa_c_api_h.mlir_type_create_f32(context)
                    : inferElementType(ctx, tensorParams.get(0).type());
                MemorySegment outputType = mlir_tosa_c_api_h.mlir_type_create_tensor_dynamic(
                    context, tensorRank, elemType);

                MemorySegment inputTypesArray = arena.allocate(ValueLayout.ADDRESS, inputTypes.size());
                for (int i = 0; i < inputTypes.size(); i++) {
                    inputTypesArray.setAtIndex(ValueLayout.ADDRESS, i, inputTypes.get(i));
                }
                MemorySegment outputTypesArray = arena.allocate(ValueLayout.ADDRESS, 1);
                outputTypesArray.setAtIndex(ValueLayout.ADDRESS, 0, outputType);

                MemorySegment funcNameNative = arena.allocateFrom(funcName);
                MemorySegment function = mlir_tosa_c_api_h.mlir_function_create(
                    module, funcNameNative,
                    inputTypesArray, inputTypes.size(),
                    outputTypesArray, 1);

                if (function.equals(MemorySegment.NULL)) {
                    throw new RuntimeException("Failed to create MLIR function: " + getLastError());
                }

                // Map tensor params to function arguments
                for (int i = 0; i < tensorParams.size(); i++) {
                    MemorySegment arg = mlir_tosa_c_api_h.mlir_function_get_argument(function, i);
                    ctx.valueHandles.put(tensorParams.get(i), arg);
                }

                // Walk all ops in the lambda body
                for (Op op : entryBlock.ops()) {
                    processOp(op, ctx, function);
                }

                int verifyResult = mlir_tosa_c_api_h.mlir_module_verify(module);
                if (verifyResult != 0) {
                    throw new RuntimeException("MLIR module verification failed: " + getLastError());
                }

                MemorySegment strPtr = mlir_tosa_c_api_h.mlir_module_to_string(module);
                if (strPtr.equals(MemorySegment.NULL)) {
                    throw new RuntimeException("Failed to convert module to string: " + getLastError());
                }

                String result = strPtr.reinterpret(Long.MAX_VALUE).getString(0);
                mlir_tosa_c_api_h.mlir_string_destroy(strPtr);
                return result;

            } finally {
                mlir_tosa_c_api_h.mlir_module_destroy(module);
                mlir_tosa_c_api_h.mlir_context_destroy(context);
            }
        }
    }

    /**
     * Resolve element type from a Java {@code Tensor<T>} type string.
     */
    private static MemorySegment inferElementType(GeneratorContext ctx, TypeElement type) {
        String typeStr = type.toString();
        if (typeStr.contains("Tensor<java.lang.Double>") || typeStr.contains("Tensor<Double>")) {
            return mlir_tosa_c_api_h.mlir_type_create_f64(ctx.context);
        } else if (typeStr.contains("Tensor<java.lang.Integer>") || typeStr.contains("Tensor<Integer>")) {
            return mlir_tosa_c_api_h.mlir_type_create_i32(ctx.context);
        } else if (typeStr.contains("Tensor<java.lang.Long>") || typeStr.contains("Tensor<Long>")) {
            return mlir_tosa_c_api_h.mlir_type_create_i64(ctx.context);
        }
        return mlir_tosa_c_api_h.mlir_type_create_f32(ctx.context); // default float32
    }

    /**
     * Generate TOSA MLIR code from a FuncOp with specified tensor rank and optional static shapes.
     *
     * @param funcOp The function operation from code reflection
     * @param funcName The name to use for the generated function
     * @param tensorRank The rank of tensors (1=1D, 2=2D, etc.)
     * @param paramShapes Optional array of shapes for each parameter (null for dynamic shapes)
     * @return TOSA MLIR code as a string
     */
    private static String generateTosa(CoreOp.FuncOp funcOp, String funcName, int tensorRank, long[][] paramShapes) {
        try (Arena arena = Arena.ofConfined()) {
            // Create MLIR context and module
            MemorySegment context = mlir_tosa_c_api_h.mlir_context_create();
            if (context.equals(MemorySegment.NULL)) {
                throw new RuntimeException("Failed to create MLIR context: " + getLastError());
            }

            MemorySegment module = mlir_tosa_c_api_h.mlir_module_create(context);
            if (module.equals(MemorySegment.NULL)) {
                mlir_tosa_c_api_h.mlir_context_destroy(context);
                throw new RuntimeException("Failed to create MLIR module: " + getLastError());
            }

            try {
                GeneratorContext ctx = new GeneratorContext(arena, context, tensorRank, paramShapes);

                // Get function parameters
                List<Block.Parameter> params = funcOp.body().entryBlock().parameters();

                // Create input types - use static shapes if provided, otherwise dynamic
                List<MemorySegment> inputTypes = new ArrayList<>();
                for (int i = 0; i < params.size(); i++) {
                    Block.Parameter param = params.get(i);
                    long[] shape = (paramShapes != null && i < paramShapes.length) ? paramShapes[i] : null;
                    MemorySegment tosaType;
                    if (shape != null) {
                        tosaType = javaTypeToNativeTosaTypeStatic(ctx, param.type(), shape);
                    } else {
                        tosaType = javaTypeToNativeTosaType(ctx, param.type(), ctx.tensorRank);
                    }
                    inputTypes.add(tosaType);
                }

                // Create output type - infer from input shapes if possible, otherwise use dynamic
                // For Conv2D: output shape depends on input, kernel, padding, stride
                MemorySegment outputType = inferOutputType(ctx, funcOp, paramShapes);

                // Allocate arrays for input/output types
                MemorySegment inputTypesArray = arena.allocate(ValueLayout.ADDRESS, inputTypes.size());
                for (int i = 0; i < inputTypes.size(); i++) {
                    inputTypesArray.setAtIndex(ValueLayout.ADDRESS, i, inputTypes.get(i));
                }

                MemorySegment outputTypesArray = arena.allocate(ValueLayout.ADDRESS, 1);
                outputTypesArray.setAtIndex(ValueLayout.ADDRESS, 0, outputType);

                // Create function name as native string
                MemorySegment funcNameNative = arena.allocateFrom(funcName);

                // Create the function
                MemorySegment function = mlir_tosa_c_api_h.mlir_function_create(
                    module,
                    funcNameNative,
                    inputTypesArray,
                    inputTypes.size(),
                    outputTypesArray,
                    1
                );

                if (function.equals(MemorySegment.NULL)) {
                    throw new RuntimeException("Failed to create MLIR function: " + getLastError());
                }

                // Map function arguments to values
                for (int i = 0; i < params.size(); i++) {
                    MemorySegment arg = mlir_tosa_c_api_h.mlir_function_get_argument(function, i);
                    ctx.valueHandles.put(params.get(i), arg);
                }

                // Process the function body
                Block entryBlock = funcOp.body().entryBlock();
                for (Op op : entryBlock.ops()) {
                    processOp(op, ctx, function);
                }

                // Verify the module
                int verifyResult = mlir_tosa_c_api_h.mlir_module_verify(module);
                if (verifyResult != 0) {
                    throw new RuntimeException("MLIR module verification failed: " + getLastError());
                }

                // Convert to string
                MemorySegment strPtr = mlir_tosa_c_api_h.mlir_module_to_string(module);
                if (strPtr.equals(MemorySegment.NULL)) {
                    throw new RuntimeException("Failed to convert module to string: " + getLastError());
                }

                String result = strPtr.reinterpret(Long.MAX_VALUE).getString(0);
                mlir_tosa_c_api_h.mlir_string_destroy(strPtr);

                return result;
            } finally {
                mlir_tosa_c_api_h.mlir_module_destroy(module);
                mlir_tosa_c_api_h.mlir_context_destroy(context);
            }
        }
    }

    /**
     * Write TOSA MLIR to a text file.
     *
     * @param funcOp The function operation from code reflection
     * @param funcName The name to use for the generated function
     * @param filename The output file path
     */
    public static void writeTosaToFile(CoreOp.FuncOp funcOp, String funcName, String filename) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment context = mlir_tosa_c_api_h.mlir_context_create();
            if (context.equals(MemorySegment.NULL)) {
                throw new RuntimeException("Failed to create MLIR context: " + getLastError());
            }

            MemorySegment module = mlir_tosa_c_api_h.mlir_module_create(context);
            if (module.equals(MemorySegment.NULL)) {
                mlir_tosa_c_api_h.mlir_context_destroy(context);
                throw new RuntimeException("Failed to create MLIR module: " + getLastError());
            }

            try {
                GeneratorContext ctx = new GeneratorContext(arena, context, 1); // Default to 1D tensors
                buildFunction(funcOp, funcName, module, ctx, arena);

                // Write to file
                MemorySegment filenameNative = arena.allocateFrom(filename);
                int writeResult = mlir_tosa_c_api_h.mlir_module_write_text(module, filenameNative);
                if (writeResult != 0) {
                    throw new RuntimeException("Failed to write MLIR to file: " + getLastError());
                }
            } finally {
                mlir_tosa_c_api_h.mlir_module_destroy(module);
                mlir_tosa_c_api_h.mlir_context_destroy(context);
            }
        }
    }

    /**
     * Write TOSA MLIR bytecode to a file.
     *
     * @param funcOp The function operation from code reflection
     * @param funcName The name to use for the generated function
     * @param filename The output file path
     */
    public static void writeTosaBytecode(CoreOp.FuncOp funcOp, String funcName, String filename) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment context = mlir_tosa_c_api_h.mlir_context_create();
            if (context.equals(MemorySegment.NULL)) {
                throw new RuntimeException("Failed to create MLIR context: " + getLastError());
            }

            MemorySegment module = mlir_tosa_c_api_h.mlir_module_create(context);
            if (module.equals(MemorySegment.NULL)) {
                mlir_tosa_c_api_h.mlir_context_destroy(context);
                throw new RuntimeException("Failed to create MLIR module: " + getLastError());
            }

            try {
                GeneratorContext ctx = new GeneratorContext(arena, context, 1); // Default to 1D tensors
                buildFunction(funcOp, funcName, module, ctx, arena);

                // Write bytecode to file
                MemorySegment filenameNative = arena.allocateFrom(filename);
                int writeResult = mlir_tosa_c_api_h.mlir_module_write_bytecode(module, filenameNative);
                if (writeResult != 0) {
                    throw new RuntimeException("Failed to write MLIR bytecode to file: " + getLastError());
                }
            } finally {
                mlir_tosa_c_api_h.mlir_module_destroy(module);
                mlir_tosa_c_api_h.mlir_context_destroy(context);
            }
        }
    }

    /**
     * Build a function in the given module.
     */
    private static void buildFunction(CoreOp.FuncOp funcOp, String funcName,
                                       MemorySegment module, GeneratorContext ctx, Arena arena) {
        List<Block.Parameter> params = funcOp.body().entryBlock().parameters();

        // Create input types with the correct tensor rank
        List<MemorySegment> inputTypes = new ArrayList<>();
        for (Block.Parameter param : params) {
            MemorySegment tosaType = javaTypeToNativeTosaType(ctx, param.type(), ctx.tensorRank);
            inputTypes.add(tosaType);
        }

        // Create output type with the correct tensor rank
        MemorySegment outputType = javaTypeToNativeTosaType(ctx, funcOp.invokableType().returnType(), ctx.tensorRank);

        // Allocate arrays for input/output types
        MemorySegment inputTypesArray = arena.allocate(ValueLayout.ADDRESS, inputTypes.size());
        for (int i = 0; i < inputTypes.size(); i++) {
            inputTypesArray.setAtIndex(ValueLayout.ADDRESS, i, inputTypes.get(i));
        }

        MemorySegment outputTypesArray = arena.allocate(ValueLayout.ADDRESS, 1);
        outputTypesArray.setAtIndex(ValueLayout.ADDRESS, 0, outputType);

        // Create function name as native string
        MemorySegment funcNameNative = arena.allocateFrom(funcName);

        // Create the function
        MemorySegment function = mlir_tosa_c_api_h.mlir_function_create(
            module,
            funcNameNative,
            inputTypesArray,
            inputTypes.size(),
            outputTypesArray,
            1
        );

        if (function.equals(MemorySegment.NULL)) {
            throw new RuntimeException("Failed to create MLIR function: " + getLastError());
        }

        // Map function arguments to values
        for (int i = 0; i < params.size(); i++) {
            MemorySegment arg = mlir_tosa_c_api_h.mlir_function_get_argument(function, i);
            ctx.valueHandles.put(params.get(i), arg);
        }

        // Process the function body
        Block entryBlock = funcOp.body().entryBlock();
        for (Op op : entryBlock.ops()) {
            processOp(op, ctx, function);
        }

        // Verify the module
        int verifyResult = mlir_tosa_c_api_h.mlir_module_verify(module);
        if (verifyResult != 0) {
            throw new RuntimeException("MLIR module verification failed: " + getLastError());
        }
    }

    /**
     * Process a single operation from the code model.
     */
    private static MemorySegment processOp(Op op, GeneratorContext ctx, MemorySegment function) {
        return switch (op) {
            case CoreOp.VarOp varOp -> {
                // Variable declarations - map the var to the input value
                Value inputValue = varOp.operands().getFirst();
                MemorySegment handle = ctx.valueHandles.get(inputValue);
                if (handle != null) {
                    ctx.varToHandle.put(varOp.result(), handle);
                }
                yield MemorySegment.NULL;
            }
            case CoreOp.VarAccessOp.VarLoadOp loadOp -> {
                // Variable loads - look up the actual value
                Value varValue = loadOp.operands().getFirst();
                MemorySegment handle = ctx.varToHandle.get(varValue);
                if (handle != null) {
                    ctx.valueHandles.put(loadOp.result(), handle);
                }
                yield MemorySegment.NULL;
            }
            case JavaOp.InvokeOp invokeOp -> {
                yield processInvokeOp(invokeOp, ctx, function);
            }
            case CoreOp.ReturnOp returnOp -> {
                Value returnValue = returnOp.operands().getFirst();
                MemorySegment valueHandle = ctx.valueHandles.get(returnValue);
                if (valueHandle != null && !valueHandle.equals(MemorySegment.NULL)) {
                    // Create array with single return value
                    MemorySegment valuesArray = ctx.arena.allocate(ValueLayout.ADDRESS, 1);
                    valuesArray.setAtIndex(ValueLayout.ADDRESS, 0, valueHandle);
                    mlir_tosa_c_api_h.mlir_function_add_return(function, valuesArray, 1);
                }
                yield MemorySegment.NULL;
            }
            default -> MemorySegment.NULL;
        };
    }

    /**
     * Process a single operation from the code model, with support for weight injection.
     * This handles FieldLoadOp to inject constant tensors for model weights.
     */
    private static MemorySegment processOpWithWeights(Op op, GeneratorContext ctx, MemorySegment function) {
        return switch (op) {
            case CoreOp.VarOp varOp -> {
                // Variable declarations - map the var to the input value
                Value inputValue = varOp.operands().getFirst();
                MemorySegment handle = ctx.valueHandles.get(inputValue);
                if (handle != null) {
                    ctx.varToHandle.put(varOp.result(), handle);
                }
                yield MemorySegment.NULL;
            }
            case CoreOp.VarAccessOp.VarLoadOp loadOp -> {
                // Variable loads - look up the actual value
                Value varValue = loadOp.operands().getFirst();
                MemorySegment handle = ctx.varToHandle.get(varValue);
                if (handle != null) {
                    ctx.valueHandles.put(loadOp.result(), handle);
                }
                yield MemorySegment.NULL;
            }
            case JavaOp.FieldAccessOp.FieldLoadOp flo -> {
                // Field loads - check if this is a weight tensor field
                String fieldName = flo.fieldReference().name();
                if (ctx.weights != null && ctx.weights.containsKey(fieldName)) {
                    TosaModelExporter.WeightInfo weight = ctx.weights.get(fieldName);
                    MemorySegment constValue = createConstantTensor(function, weight, ctx);
                    if (constValue != null && !constValue.equals(MemorySegment.NULL)) {
                        ctx.valueHandles.put(flo.result(), constValue);
                    }
                }
                yield MemorySegment.NULL;
            }
            case JavaOp.InvokeOp invokeOp -> {
                yield processInvokeOp(invokeOp, ctx, function);
            }
            case CoreOp.ReturnOp returnOp -> {
                Value returnValue = returnOp.operands().getFirst();
                MemorySegment valueHandle = ctx.valueHandles.get(returnValue);
                if (valueHandle != null && !valueHandle.equals(MemorySegment.NULL)) {
                    // Create array with single return value
                    MemorySegment valuesArray = ctx.arena.allocate(ValueLayout.ADDRESS, 1);
                    valuesArray.setAtIndex(ValueLayout.ADDRESS, 0, valueHandle);
                    mlir_tosa_c_api_h.mlir_function_add_return(function, valuesArray, 1);
                }
                yield MemorySegment.NULL;
            }
            default -> MemorySegment.NULL;
        };
    }

    /**
     * Create a constant tensor from weight data using the C API.
     *
     * @param function The MLIR function handle
     * @param weight The weight information containing shape and data
     * @param ctx The generator context
     * @return MLIR value handle for the constant tensor
     */
    private static MemorySegment createConstantTensor(MemorySegment function,
                                                       TosaModelExporter.WeightInfo weight,
                                                       GeneratorContext ctx) {
        long[] shape = weight.shape();
        long numElements = weight.numElements();

        // Allocate shape array
        MemorySegment shapeArray = ctx.arena.allocate(ValueLayout.JAVA_LONG, shape.length);
        for (int i = 0; i < shape.length; i++) {
            shapeArray.setAtIndex(ValueLayout.JAVA_LONG, i, shape[i]);
        }

        // Create tensor type
        MemorySegment elementType = mlir_tosa_c_api_h.mlir_type_create_f32(ctx.context);
        MemorySegment tensorType = mlir_tosa_c_api_h.mlir_type_create_tensor_ranked(
            ctx.context, shapeArray, shape.length, elementType);

        // Copy weight data to native memory
        MemorySegment dataArray = ctx.arena.allocate(ValueLayout.JAVA_FLOAT, numElements);
        MemorySegment.copy(weight.data(), ValueLayout.JAVA_FLOAT, 0,
                           dataArray, ValueLayout.JAVA_FLOAT, 0, numElements);

        // Create constant via C API
        return mlir_tosa_c_api_h.mlir_tosa_const_f32(
            function, dataArray, numElements, shapeArray, shape.length, tensorType);
    }

    /**
     * Process a method invocation (TOSA operation).
     */
    private static MemorySegment processInvokeOp(JavaOp.InvokeOp invokeOp, GeneratorContext ctx,
                                                  MemorySegment function) {
        String methodRef = invokeOp.invokeReference().toString();
        String methodName = invokeOp.invokeReference().name();

        // Check if this is a TOSA operation
        if (!methodRef.contains("TosaOperators::") && !methodRef.contains("Tensor::")) {
            return MemorySegment.NULL;
        }

        // Get operand handles for tensor operands only
        // Some operations (Conv2D, MaxPool2D) have non-tensor operands (long[] arrays)
        // that won't be in valueHandles - we extract those separately
        List<Value> operands = invokeOp.operands();
        List<MemorySegment> operandHandles = new ArrayList<>();
        boolean hasArrayOperands = methodName.equals("Conv2D") || methodName.equals("MaxPool2D")
            || methodName.equals("AvgPool2D") || methodName.equals("Reshape");
        for (Value operand : operands) {
            MemorySegment handle = ctx.valueHandles.get(operand);
            if (handle == null) {
                if (!hasArrayOperands) {
                    return MemorySegment.NULL; // Skip if operand not found (for ops without array args)
                }
                // For ops with array operands, stop collecting at first non-tensor operand
                break;
            }
            operandHandles.add(handle);
        }

        // Get result type with the correct tensor rank
        MemorySegment resultType = javaTypeToNativeTosaType(ctx, invokeOp.result().type(), ctx.tensorRank);

        // Generate the appropriate TOSA operation
        MemorySegment result = switch (methodName) {
            // Binary operations
            case "Add", "add" -> {
                if (operandHandles.size() >= 2) {
                    yield mlir_tosa_c_api_h.mlir_tosa_add(function,
                        operandHandles.get(0), operandHandles.get(1), resultType);
                }
                yield MemorySegment.NULL;
            }
            case "Sub", "sub" -> {
                if (operandHandles.size() >= 2) {
                    yield mlir_tosa_c_api_h.mlir_tosa_sub(function,
                        operandHandles.get(0), operandHandles.get(1), resultType);
                }
                yield MemorySegment.NULL;
            }
            case "Mul", "mul" -> {
                if (operandHandles.size() >= 2) {
                    yield mlir_tosa_c_api_h.mlir_tosa_mul(function,
                        operandHandles.get(0), operandHandles.get(1), resultType);
                }
                yield MemorySegment.NULL;
            }
            case "MatMul", "matmul" -> {
                if (operandHandles.size() >= 2) {
                    yield mlir_tosa_c_api_h.mlir_tosa_matmul(function,
                        operandHandles.get(0), operandHandles.get(1), resultType);
                }
                yield MemorySegment.NULL;
            }

            // Unary operations
            case "Negate", "negate" -> {
                if (!operandHandles.isEmpty()) {
                    yield mlir_tosa_c_api_h.mlir_tosa_negate(function,
                        operandHandles.get(0), resultType);
                }
                yield MemorySegment.NULL;
            }
            case "Reciprocal", "reciprocal" -> {
                if (!operandHandles.isEmpty()) {
                    yield mlir_tosa_c_api_h.mlir_tosa_reciprocal(function,
                        operandHandles.get(0), resultType);
                }
                yield MemorySegment.NULL;
            }
            case "Exp", "exp" -> {
                if (!operandHandles.isEmpty()) {
                    yield mlir_tosa_c_api_h.mlir_tosa_exp(function,
                        operandHandles.get(0), resultType);
                }
                yield MemorySegment.NULL;
            }
            case "Log", "log" -> {
                if (!operandHandles.isEmpty()) {
                    yield mlir_tosa_c_api_h.mlir_tosa_log(function,
                        operandHandles.get(0), resultType);
                }
                yield MemorySegment.NULL;
            }

            // Activation functions
            case "Relu", "relu" -> {
                if (!operandHandles.isEmpty()) {
                    yield mlir_tosa_c_api_h.mlir_tosa_relu(function,
                        operandHandles.get(0), resultType);
                }
                yield MemorySegment.NULL;
            }
            case "Sigmoid", "sigmoid" -> {
                if (!operandHandles.isEmpty()) {
                    yield mlir_tosa_c_api_h.mlir_tosa_sigmoid(function,
                        operandHandles.get(0), resultType);
                }
                yield MemorySegment.NULL;
            }
            case "Tanh", "tanh" -> {
                if (!operandHandles.isEmpty()) {
                    yield mlir_tosa_c_api_h.mlir_tosa_tanh(function,
                        operandHandles.get(0), resultType);
                }
                yield MemorySegment.NULL;
            }

            // Division uses reciprocal + mul
            case "Div", "div" -> {
                if (operandHandles.size() >= 2) {
                    MemorySegment reciprocal = mlir_tosa_c_api_h.mlir_tosa_reciprocal(function,
                        operandHandles.get(1), resultType);
                    yield mlir_tosa_c_api_h.mlir_tosa_mul(function,
                        operandHandles.get(0), reciprocal, resultType);
                }
                yield MemorySegment.NULL;
            }

            // Conv2D operation
            // Operands: input, weight, bias (tensors) + pad, stride, dilation (arrays)
            // For now we extract constant arrays from the invocation
            case "Conv2D" -> {
                if (operandHandles.size() >= 3) {
                    // Extract constant array values from the invoke operands
                    long[] pad = extractLongArrayFromOperand(invokeOp, 3);
                    long[] stride = extractLongArrayFromOperand(invokeOp, 4);
                    long[] dilation = extractLongArrayFromOperand(invokeOp, 5);

                    if (pad == null) {
                        pad = new long[]{0, 0, 0, 0};
                    }
                    if (stride == null) {
                        stride = new long[]{1, 1};
                    }
                    if (dilation == null) {
                        dilation = new long[]{1, 1};
                    }

                    MemorySegment padArray = ctx.arena.allocate(ValueLayout.JAVA_LONG, 4);
                    MemorySegment strideArray = ctx.arena.allocate(ValueLayout.JAVA_LONG, 2);
                    MemorySegment dilationArray = ctx.arena.allocate(ValueLayout.JAVA_LONG, 2);

                    for (int i = 0; i < 4; i++) {
                        padArray.setAtIndex(ValueLayout.JAVA_LONG, i, pad[i]);
                    }
                    for (int i = 0; i < 2; i++) {
                        strideArray.setAtIndex(ValueLayout.JAVA_LONG, i, stride[i]);
                    }
                    for (int i = 0; i < 2; i++) {
                        dilationArray.setAtIndex(ValueLayout.JAVA_LONG, i, dilation[i]);
                    }

                    yield mlir_tosa_c_api_h.mlir_tosa_conv2d(function,
                        operandHandles.get(0), operandHandles.get(1), operandHandles.get(2),
                        padArray, strideArray, dilationArray, resultType);
                }
                yield MemorySegment.NULL;
            }

            // MaxPool2D operation
            // Operands: input (tensor) + kernel, stride, pad (arrays)
            case "MaxPool2D" -> {
                if (!operandHandles.isEmpty()) {
                    // Extract constant array values from the invoke operands
                    long[] kernel = extractLongArrayFromOperand(invokeOp, 1);
                    long[] stride = extractLongArrayFromOperand(invokeOp, 2);
                    long[] pad = extractLongArrayFromOperand(invokeOp, 3);

                    if (kernel == null) {
                        kernel = new long[]{2, 2};
                    }
                    if (stride == null) {
                        stride = new long[]{2, 2};
                    }
                    if (pad == null) {
                        pad = new long[]{0, 0, 0, 0};
                    }

                    MemorySegment kernelArray = ctx.arena.allocate(ValueLayout.JAVA_LONG, 2);
                    MemorySegment strideArray = ctx.arena.allocate(ValueLayout.JAVA_LONG, 2);
                    MemorySegment padArray = ctx.arena.allocate(ValueLayout.JAVA_LONG, 4);

                    for (int i = 0; i < 2; i++) {
                        kernelArray.setAtIndex(ValueLayout.JAVA_LONG, i, kernel[i]);
                    }
                    for (int i = 0; i < 2; i++) {
                        strideArray.setAtIndex(ValueLayout.JAVA_LONG, i, stride[i]);
                    }
                    for (int i = 0; i < 4; i++) {
                        padArray.setAtIndex(ValueLayout.JAVA_LONG, i, pad[i]);
                    }

                    yield mlir_tosa_c_api_h.mlir_tosa_max_pool2d(function,
                        operandHandles.get(0),
                        kernelArray, strideArray, padArray, resultType);
                }
                yield MemorySegment.NULL;
            }

            // AvgPool2D operation
            case "AvgPool2D" -> {
                if (!operandHandles.isEmpty()) {
                    long[] kernel = extractLongArrayFromOperand(invokeOp, 1);
                    long[] stride = extractLongArrayFromOperand(invokeOp, 2);
                    long[] pad = extractLongArrayFromOperand(invokeOp, 3);

                    if (kernel == null) {
                        kernel = new long[]{2, 2};
                    }
                    if (stride == null) {
                        stride = new long[]{2, 2};
                    }
                    if (pad == null) {
                        pad = new long[]{0, 0, 0, 0};
                    }

                    MemorySegment kernelArray = ctx.arena.allocate(ValueLayout.JAVA_LONG, 2);
                    MemorySegment strideArray = ctx.arena.allocate(ValueLayout.JAVA_LONG, 2);
                    MemorySegment padArray = ctx.arena.allocate(ValueLayout.JAVA_LONG, 4);

                    for (int i = 0; i < 2; i++) {
                        kernelArray.setAtIndex(ValueLayout.JAVA_LONG, i, kernel[i]);
                    }
                    for (int i = 0; i < 2; i++) {
                        strideArray.setAtIndex(ValueLayout.JAVA_LONG, i, stride[i]);
                    }
                    for (int i = 0; i < 4; i++) {
                        padArray.setAtIndex(ValueLayout.JAVA_LONG, i, pad[i]);
                    }

                    yield mlir_tosa_c_api_h.mlir_tosa_avg_pool2d(function,
                        operandHandles.get(0),
                        kernelArray, strideArray, padArray, resultType);
                }
                yield MemorySegment.NULL;
            }

            // Reshape operation
            // Operands: input (tensor) + newShape (long[] array)
            case "Reshape" -> {
                if (!operandHandles.isEmpty()) {
                    // Extract new shape from the operand
                    long[] newShape = extractLongArrayFromOperand(invokeOp, 1);

                    if (newShape == null) {
                        // If we can't extract the shape, try to use a default flatten
                        newShape = new long[]{-1};
                    }

                    MemorySegment shapeArray = ctx.arena.allocate(ValueLayout.JAVA_LONG, newShape.length);
                    for (int i = 0; i < newShape.length; i++) {
                        shapeArray.setAtIndex(ValueLayout.JAVA_LONG, i, newShape[i]);
                    }

                    yield mlir_tosa_c_api_h.mlir_tosa_reshape(function,
                        operandHandles.get(0),
                        shapeArray, newShape.length, resultType);
                }
                yield MemorySegment.NULL;
            }

            // ReduceSum operation
            // Operands: input (tensor) + axis (int) + keepDims (boolean, ignored at MLIR level)
            case "ReduceSum" -> {
                if (!operandHandles.isEmpty()) {
                    Integer axis = extractIntFromOperand(invokeOp, 1);
                    yield mlir_tosa_c_api_h.mlir_tosa_reduce_sum(function,
                        operandHandles.get(0), axis != null ? axis : 0, resultType);
                }
                yield MemorySegment.NULL;
            }

            // ReduceMax operation
            case "ReduceMax" -> {
                if (!operandHandles.isEmpty()) {
                    Integer axis = extractIntFromOperand(invokeOp, 1);
                    yield mlir_tosa_c_api_h.mlir_tosa_reduce_max(function,
                        operandHandles.get(0), axis != null ? axis : 0, resultType);
                }
                yield MemorySegment.NULL;
            }

            default -> MemorySegment.NULL;
        };

        // Store the result handle
        if (result != null && !result.equals(MemorySegment.NULL)) {
            ctx.valueHandles.put(invokeOp.result(), result);
        }

        return result;
    }

    /**
     * Convert Java type to native TOSA tensor type with static shape.
     *
     * @param ctx Generator context
     * @param type The Java TypeElement
     * @param shape The static shape dimensions
     * @return MLIR tensor type handle with static shape
     */
    private static MemorySegment javaTypeToNativeTosaTypeStatic(GeneratorContext ctx, TypeElement type, long[] shape) {
        String typeStr = type.toString();

        // Determine element type
        MemorySegment elementType;
        if (typeStr.contains("Tensor<java.lang.Float>") || typeStr.contains("Tensor<Float>")) {
            elementType = mlir_tosa_c_api_h.mlir_type_create_f32(ctx.context);
        } else if (typeStr.contains("Tensor<java.lang.Double>") || typeStr.contains("Tensor<Double>")) {
            elementType = mlir_tosa_c_api_h.mlir_type_create_f64(ctx.context);
        } else if (typeStr.contains("Tensor<java.lang.Integer>") || typeStr.contains("Tensor<Integer>")) {
            elementType = mlir_tosa_c_api_h.mlir_type_create_i32(ctx.context);
        } else if (typeStr.contains("Tensor<java.lang.Long>") || typeStr.contains("Tensor<Long>")) {
            elementType = mlir_tosa_c_api_h.mlir_type_create_i64(ctx.context);
        } else {
            // Default to float32
            elementType = mlir_tosa_c_api_h.mlir_type_create_f32(ctx.context);
        }

        // Create static tensor type with the specified shape
        MemorySegment shapeArray = ctx.arena.allocate(ValueLayout.JAVA_LONG, shape.length);
        for (int i = 0; i < shape.length; i++) {
            shapeArray.setAtIndex(ValueLayout.JAVA_LONG, i, shape[i]);
        }
        return mlir_tosa_c_api_h.mlir_type_create_tensor_ranked(ctx.context, shapeArray, shape.length, elementType);
    }

    /**
     * Infer output type from function and input shapes.
     * For operations like Conv2D, the output shape depends on input shapes and operation parameters.
     */
    private static MemorySegment inferOutputType(GeneratorContext ctx, CoreOp.FuncOp funcOp, long[][] paramShapes) {
        TypeElement returnType = funcOp.invokableType().returnType();

        // If no static shapes provided, use dynamic output
        if (paramShapes == null) {
            return javaTypeToNativeTosaType(ctx, returnType, ctx.tensorRank);
        }

        // For now, use the first parameter's shape as a template for the output
        // This works for element-wise ops and gives a reasonable default
        // More sophisticated shape inference would be needed for different ops
        for (long[] shape : paramShapes) {
            if (shape != null) {
                // For Conv2D: output shape is [N, OH, OW, OC] where OC comes from weight shape
                // For simplicity, use dynamic output type which MLIR can infer
                return javaTypeToNativeTosaType(ctx, returnType, shape.length);
            }
        }

        return javaTypeToNativeTosaType(ctx, returnType, ctx.tensorRank);
    }

    /**
     * Convert Java type to native TOSA tensor type with specified rank.
     *
     * @param ctx Generator context
     * @param type The Java TypeElement
     * @param rank The tensor rank (0=scalar, 1=1D vector, 2=2D matrix, etc.)
     * @return MLIR tensor type handle
     */
    private static MemorySegment javaTypeToNativeTosaType(GeneratorContext ctx, TypeElement type, int rank) {
        String typeStr = type.toString();

        // Determine element type
        MemorySegment elementType;
        if (typeStr.contains("Tensor<java.lang.Float>") || typeStr.contains("Tensor<Float>")) {
            elementType = mlir_tosa_c_api_h.mlir_type_create_f32(ctx.context);
        } else if (typeStr.contains("Tensor<java.lang.Double>") || typeStr.contains("Tensor<Double>")) {
            elementType = mlir_tosa_c_api_h.mlir_type_create_f64(ctx.context);
        } else if (typeStr.contains("Tensor<java.lang.Integer>") || typeStr.contains("Tensor<Integer>")) {
            elementType = mlir_tosa_c_api_h.mlir_type_create_i32(ctx.context);
        } else if (typeStr.contains("Tensor<java.lang.Long>") || typeStr.contains("Tensor<Long>")) {
            elementType = mlir_tosa_c_api_h.mlir_type_create_i64(ctx.context);
        } else {
            // Default to float32
            elementType = mlir_tosa_c_api_h.mlir_type_create_f32(ctx.context);
        }

        // Create dynamic tensor type with the specified rank
        // rank=0 creates scalar tensor<f32>
        // rank=1 creates 1D tensor<?xf32>
        // rank=2 creates 2D tensor<?x?xf32>
        return mlir_tosa_c_api_h.mlir_type_create_tensor_dynamic(ctx.context, rank, elementType);
    }

    /**
     * Get the last error message from the native library.
     */
    private static String getLastError() {
        MemorySegment errorPtr = mlir_tosa_c_api_h.mlir_get_last_error();
        if (errorPtr.equals(MemorySegment.NULL)) {
            return "Unknown error";
        }
        return errorPtr.reinterpret(Long.MAX_VALUE).getString(0);
    }

    /**
     * Extract a long[] constant from an invoke operand by scanning preceding ops in the block.
     *
     * Handles the common patterns produced by the Java compiler for array literals:
     * <pre>
     *   new long[]{0, 0, 0, 0}   →  NewOp + 4× ArrayStoreOp
     *   long[] v = ...; f(v)     →  VarOp + VarLoadOp
     * </pre>
     * Type conversions (int → long) via {@code ConvOp} are also folded.
     *
     * @param invokeOp     The invoke operation whose operand to extract
     * @param operandIndex 0-based index of the array operand within the invocation
     * @return The extracted long[] value, or {@code null} if the value is not a
     *         statically-known constant (falls back to caller's default)
     */
    private static long[] extractLongArrayFromOperand(
            JavaOp.InvokeOp invokeOp, int operandIndex) {
        List<Value> operands = invokeOp.operands();
        if (operandIndex >= operands.size()) {
            return null;
        }
        Object result = evaluateConstant(operands.get(operandIndex), invokeOp.parent());
        return result instanceof long[] arr ? arr.clone() : null;
    }

    /**
     * Extract a scalar int constant from an invoke operand.
     *
     * @param invokeOp     The invoke operation
     * @param operandIndex 0-based index of the int operand
     * @return The int value, or {@code null} if not statically known
     */
    private static Integer extractIntFromOperand(JavaOp.InvokeOp invokeOp, int operandIndex) {
        List<Value> operands = invokeOp.operands();
        if (operandIndex >= operands.size()) {
            return null;
        }
        Object result = evaluateConstant(operands.get(operandIndex), invokeOp.parent());
        if (result instanceof Integer i) return i;
        if (result instanceof Long l) return l.intValue();
        return null;
    }

    /**
     * Walk ops in {@code block} up to (not including) {@code target}'s defining op,
     * building a constant-folded value map, then return the value mapped to {@code target}.
     *
     * <p>Supported op patterns:
     * <ul>
     *   <li>{@code CoreOp.ConstantOp} – primitive/string literals</li>
     *   <li>{@code JavaOp.ConvOp} – numeric widening (int → long, etc.)</li>
     *   <li>{@code CoreOp.VarOp / VarLoadOp / VarStoreOp} – local variables</li>
     *   <li>{@code JavaOp.NewOp} with {@code ArrayType} result – {@code new long[n]}</li>
     *   <li>{@code JavaOp.ArrayAccessOp.ArrayStoreOp} – array element initialization</li>
     * </ul>
     */
    private static Object evaluateConstant(Value target, Block block) {
        // valueMap: SSA Value → constant Java object
        // For VarOp results we store a one-element Object[] as a mutable box.
        Map<Value, Object> valueMap = new HashMap<>();

        for (Op op : block.ops()) {
            switch (op) {
                case CoreOp.ConstantOp co -> valueMap.put(co.result(), co.value());

                case JavaOp.ConvOp co -> {
                    // Numeric type widening / narrowing (e.g. int literal stored in long[])
                    Object v = valueMap.get(co.operands().getFirst());
                    if (v instanceof Number n) {
                        valueMap.put(co.result(), n.longValue());
                    }
                }

                case CoreOp.VarOp vo -> {
                    Object init = vo.isUninitialized() || vo.operands().isEmpty()
                        ? null
                        : valueMap.get(vo.operands().getFirst());
                    valueMap.put(vo.result(), new Object[]{init}); // mutable box
                }

                case CoreOp.VarAccessOp.VarLoadOp lo -> {
                    Object box = valueMap.get(lo.operands().getFirst());
                    if (box instanceof Object[] b && b[0] != null) {
                        valueMap.put(lo.result(), b[0]);
                    }
                }

                case CoreOp.VarAccessOp.VarStoreOp so -> {
                    Object box = valueMap.get(so.operands().get(0));
                    if (box instanceof Object[] b) {
                        b[0] = valueMap.get(so.operands().get(1));
                    }
                }

                case JavaOp.NewOp no when no.resultType() instanceof ArrayType -> {
                    // new long[size]  →  allocate zero-filled long[]
                    Object sizeObj = valueMap.get(no.operands().getFirst());
                    if (sizeObj instanceof Integer size) {
                        valueMap.put(no.result(), new long[size]);
                    }
                }

                case JavaOp.ArrayAccessOp.ArrayStoreOp so -> {
                    Object arr = valueMap.get(so.operands().get(0));
                    Object idx = valueMap.get(so.operands().get(1));
                    Object val = valueMap.get(so.operands().get(2));
                    if (arr instanceof long[] a && idx instanceof Integer i && val instanceof Number n) {
                        a[i] = n.longValue();
                    }
                }

                default -> {}
            }
        }

        return valueMap.get(target);
    }

    /**
     * Context for tracking values during code generation.
     */
    private static class GeneratorContext {
        final Arena arena;
        final MemorySegment context;
        final int tensorRank;
        final long[][] paramShapes; // Optional static shapes for parameters
        final SequencedMap<String, TosaModelExporter.WeightInfo> weights; // Optional weights for embedding
        final Object modelInstance; // Optional model instance for field access resolution
        final Map<Value, MemorySegment> valueHandles = new HashMap<>();
        final Map<Value, MemorySegment> varToHandle = new HashMap<>();

        GeneratorContext(Arena arena, MemorySegment context, int tensorRank) {
            this(arena, context, tensorRank, null, null, null);
        }

        GeneratorContext(Arena arena, MemorySegment context, int tensorRank, long[][] paramShapes) {
            this(arena, context, tensorRank, paramShapes, null, null);
        }

        GeneratorContext(Arena arena, MemorySegment context, int tensorRank, long[][] paramShapes,
                         SequencedMap<String, TosaModelExporter.WeightInfo> weights, Object modelInstance) {
            this.arena = arena;
            this.context = context;
            this.tensorRank = tensorRank;
            this.paramShapes = paramShapes;
            this.weights = weights;
            this.modelInstance = modelInstance;
        }
    }
}
