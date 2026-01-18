package mlir.tosa;

import jdk.incubator.code.Block;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;

import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Generates TOSA/MLIR code from Java code reflection models.
 *
 * This class transforms the Java code model captured via @Reflect annotations
 * into TOSA (Tensor Operator Set Architecture) MLIR representation.
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
        StringBuilder sb = new StringBuilder();
        GeneratorContext ctx = new GeneratorContext();

        // Get function parameters
        List<Block.Parameter> params = funcOp.body().entryBlock().parameters();

        // Build function signature
        sb.append("func.func @").append(funcName).append("(");

        // Add parameters with TOSA tensor types
        for (int i = 0; i < params.size(); i++) {
            if (i > 0) sb.append(", ");
            Block.Parameter param = params.get(i);
            String paramName = getParamName(funcOp, i);
            String tosaType = javaTypeToTosaType(param.type());
            ctx.valueNames.put(param, "%" + paramName);
            sb.append("%").append(paramName).append(": ").append(tosaType);
        }

        // Determine return type from function
        String returnType = javaTypeToTosaType(funcOp.invokableType().returnType());
        sb.append(") -> ").append(returnType).append(" {\n");

        // Process the function body
        Block entryBlock = funcOp.body().entryBlock();
        for (Op op : entryBlock.ops()) {
            String opCode = generateOpCode(op, ctx);
            if (opCode != null && !opCode.isEmpty()) {
                sb.append(opCode);
            }
        }

        sb.append("}\n");
        return sb.toString();
    }

    /**
     * Get parameter name from var declarations in the function body.
     */
    private static String getParamName(CoreOp.FuncOp funcOp, int paramIndex) {
        Block entryBlock = funcOp.body().entryBlock();
        int varIndex = 0;
        for (Op op : entryBlock.ops()) {
            if (op instanceof CoreOp.VarOp varOp) {
                if (varIndex == paramIndex) {
                    return varOp.varName();
                }
                varIndex++;
            }
        }
        return "arg" + paramIndex;
    }

    /**
     * Generate code for a single operation.
     */
    private static String generateOpCode(Op op, GeneratorContext ctx) {
        return switch (op) {
            case CoreOp.VarOp varOp -> {
                // Variable declarations - map the var to the input value
                Value inputValue = varOp.operands().getFirst();
                String inputName = ctx.valueNames.get(inputValue);
                if (inputName != null) {
                    ctx.varToValue.put(varOp.result(), inputName);
                }
                yield null; // No code generated for var declarations
            }
            case CoreOp.VarAccessOp.VarLoadOp loadOp -> {
                // Variable loads - look up the actual value
                Value varValue = loadOp.operands().getFirst();
                String varName = ctx.varToValue.get(varValue);
                if (varName != null) {
                    ctx.valueNames.put(loadOp.result(), varName);
                }
                yield null; // No code generated for var loads
            }
            case JavaOp.InvokeOp invokeOp -> {
                yield generateInvokeOp(invokeOp, ctx);
            }
            case CoreOp.ReturnOp returnOp -> {
                Value returnValue = returnOp.operands().getFirst();
                String valueName = ctx.valueNames.get(returnValue);
                yield "    return " + valueName + " : " + javaTypeToTosaType(returnValue.type()) + "\n";
            }
            default -> null;
        };
    }

    /**
     * Generate code for method invocations (TOSA operations).
     */
    private static String generateInvokeOp(JavaOp.InvokeOp invokeOp, GeneratorContext ctx) {
        String methodRef = invokeOp.invokeDescriptor().toString();
        String methodName = invokeOp.invokeDescriptor().name();

        // Check if this is a TOSA operation
        if (methodRef.contains("TosaOperators::") || methodRef.contains("Tensor::")) {
            String tosaOp = mapToTosaOp(methodName);
            if (tosaOp != null) {
                return generateTosaOp(tosaOp, invokeOp, ctx);
            }
        }

        return null;
    }

    /**
     * Map Java method names to TOSA operation names.
     */
    private static String mapToTosaOp(String methodName) {
        return switch (methodName) {
            // Arithmetic operations
            case "Add", "add" -> "tosa.add";
            case "Mul", "mul" -> "tosa.mul";
            case "Sub", "sub" -> "tosa.sub";
            case "Div", "div" -> "tosa.reciprocal+mul"; // TOSA uses reciprocal+mul for div
            case "Negate", "negate" -> "tosa.negate";
            case "Reciprocal", "reciprocal" -> "tosa.reciprocal";

            // Matrix operations
            case "MatMul", "matmul" -> "tosa.matmul";

            // Convolution and pooling
            case "Conv2D", "conv2d" -> "tosa.conv2d";
            case "MaxPool2D", "maxPool2d" -> "tosa.max_pool2d";
            case "AvgPool2D", "avgPool2d" -> "tosa.avg_pool2d";

            // Activations
            case "Clamp", "clamp" -> "tosa.clamp";
            case "Relu", "relu" -> "tosa.clamp"; // ReLU is clamp(0, max)

            // Shape operations
            case "Reshape", "reshape" -> "tosa.reshape";
            case "Flatten", "flatten" -> "tosa.reshape"; // Flatten uses reshape

            // Reduction operations
            case "ReduceSum", "reduceSum" -> "tosa.reduce_sum";
            case "ReduceMax", "reduceMax" -> "tosa.reduce_max";

            // Element-wise operations
            case "Exp", "exp" -> "tosa.exp";

            // Softmax (composed operation)
            case "Softmax", "softmax" -> "tosa.softmax"; // Note: TOSA doesn't have native softmax

            default -> null;
        };
    }

    /**
     * Generate TOSA operation code.
     */
    private static String generateTosaOp(String tosaOp, JavaOp.InvokeOp invokeOp, GeneratorContext ctx) {
        StringBuilder sb = new StringBuilder();

        // Get operand names
        List<Value> operands = invokeOp.operands();
        String[] operandNames = new String[operands.size()];
        String[] operandTypes = new String[operands.size()];

        for (int i = 0; i < operands.size(); i++) {
            operandNames[i] = ctx.valueNames.get(operands.get(i));
            operandTypes[i] = javaTypeToTosaType(operands.get(i).type());
        }

        // Generate result name
        String resultName = "%" + ctx.nextResultIndex++;
        ctx.valueNames.put(invokeOp.result(), resultName);

        // Get result type
        String resultType = javaTypeToTosaType(invokeOp.result().type());

        // Format: %result = tosa.op %operand1, %operand2 : (type1, type2) -> result_type
        sb.append("    ").append(resultName).append(" = ").append(tosaOp).append(" ");

        // Add operands
        for (int i = 0; i < operandNames.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(operandNames[i]);
        }

        // Add type signature
        sb.append(" : (");
        for (int i = 0; i < operandTypes.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(operandTypes[i]);
        }
        sb.append(") -> ").append(resultType);

        sb.append("\n");
        return sb.toString();
    }

    /**
     * Convert Java type to TOSA tensor type.
     * For simplicity, we use generic tensor types. In a full implementation,
     * we would track actual tensor shapes.
     */
    private static String javaTypeToTosaType(TypeElement type) {
        String typeStr = type.toString();

        // Extract element type from Tensor<T>
        if (typeStr.contains("Tensor<java.lang.Float>") || typeStr.contains("Tensor<Float>")) {
            return "tensor<*xf32>";  // Unknown shape, float32
        } else if (typeStr.contains("Tensor<java.lang.Double>") || typeStr.contains("Tensor<Double>")) {
            return "tensor<*xf64>";  // Unknown shape, float64
        } else if (typeStr.contains("Tensor<java.lang.Integer>") || typeStr.contains("Tensor<Integer>")) {
            return "tensor<*xi32>";  // Unknown shape, int32
        } else if (typeStr.contains("Tensor<java.lang.Long>") || typeStr.contains("Tensor<Long>")) {
            return "tensor<*xi64>";  // Unknown shape, int64
        } else if (typeStr.contains("Tensor")) {
            return "tensor<*xf32>";  // Default to float32
        }

        return "tensor<*xf32>";  // Default
    }

    /**
     * Context for tracking values during code generation.
     */
    private static class GeneratorContext {
        Map<Value, String> valueNames = new HashMap<>();
        Map<Value, String> varToValue = new HashMap<>();
        int nextResultIndex = 0;
    }
}
