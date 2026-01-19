package mlir.tosa;

import jdk.incubator.code.Block;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
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

/**
 * Generates TOSA/MLIR code from Java code reflection models using native MLIR bindings.
 *
 * This class transforms the Java code model captured via @Reflect annotations
 * into TOSA (Tensor Operator Set Architecture) MLIR representation using the
 * native mlir_tosa_c library for proper MLIR IR construction and validation.
 */
public final class TosaCodeGenerator {

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
     *
     * @param funcOp The function operation from code reflection
     * @param funcName The name to use for the generated function
     * @return TOSA MLIR code as a string
     */
    public static String generateTosa(CoreOp.FuncOp funcOp, String funcName) {
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
                GeneratorContext ctx = new GeneratorContext(arena, context);

                // Get function parameters
                List<Block.Parameter> params = funcOp.body().entryBlock().parameters();

                // Create input types
                List<MemorySegment> inputTypes = new ArrayList<>();
                for (Block.Parameter param : params) {
                    MemorySegment tosaType = javaTypeToNativeTosaType(ctx, param.type());
                    inputTypes.add(tosaType);
                }

                // Create output type
                MemorySegment outputType = javaTypeToNativeTosaType(ctx, funcOp.invokableType().returnType());

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
                GeneratorContext ctx = new GeneratorContext(arena, context);
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
                GeneratorContext ctx = new GeneratorContext(arena, context);
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

        // Create input types
        List<MemorySegment> inputTypes = new ArrayList<>();
        for (Block.Parameter param : params) {
            MemorySegment tosaType = javaTypeToNativeTosaType(ctx, param.type());
            inputTypes.add(tosaType);
        }

        // Create output type
        MemorySegment outputType = javaTypeToNativeTosaType(ctx, funcOp.invokableType().returnType());

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
     * Process a method invocation (TOSA operation).
     */
    private static MemorySegment processInvokeOp(JavaOp.InvokeOp invokeOp, GeneratorContext ctx,
                                                  MemorySegment function) {
        String methodRef = invokeOp.invokeDescriptor().toString();
        String methodName = invokeOp.invokeDescriptor().name();

        // Check if this is a TOSA operation
        if (!methodRef.contains("TosaOperators::") && !methodRef.contains("Tensor::")) {
            return MemorySegment.NULL;
        }

        // Get operand handles
        List<Value> operands = invokeOp.operands();
        List<MemorySegment> operandHandles = new ArrayList<>();
        for (Value operand : operands) {
            MemorySegment handle = ctx.valueHandles.get(operand);
            if (handle == null) {
                return MemorySegment.NULL; // Skip if operand not found
            }
            operandHandles.add(handle);
        }

        // Get result type
        MemorySegment resultType = javaTypeToNativeTosaType(ctx, invokeOp.result().type());

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

            default -> MemorySegment.NULL;
        };

        // Store the result handle
        if (result != null && !result.equals(MemorySegment.NULL)) {
            ctx.valueHandles.put(invokeOp.result(), result);
        }

        return result;
    }

    /**
     * Convert Java type to native TOSA tensor type.
     */
    private static MemorySegment javaTypeToNativeTosaType(GeneratorContext ctx, TypeElement type) {
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

        // Create dynamic tensor type (unknown shape)
        // Using rank=0 for unranked tensor (tensor<*xf32>)
        return mlir_tosa_c_api_h.mlir_type_create_tensor_dynamic(ctx.context, 0, elementType);
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
     * Context for tracking values during code generation.
     */
    private static class GeneratorContext {
        final Arena arena;
        final MemorySegment context;
        final Map<Value, MemorySegment> valueHandles = new HashMap<>();
        final Map<Value, MemorySegment> varToHandle = new HashMap<>();

        GeneratorContext(Arena arena, MemorySegment context) {
            this.arena = arena;
            this.context = context;
        }
    }
}
