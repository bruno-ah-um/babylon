package mlir.tosa;

import jdk.incubator.code.Op;
import jdk.incubator.code.Quoted;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.function.Supplier;

import static mlir.tosa.TosaOperators.*;

/**
 * Demo application showcasing code reflection for a classical neural network linear layer.
 *
 * This demonstrates how Java code for Y = XW + b (MatMul + Add) can be captured
 * using code reflection and displayed as a TOSA-like operation graph.
 *
 * Run with: mvn exec:java -Dexec.mainClass=mlir.tosa.LinearLayerDemo
 */
public class LinearLayerDemo {

    /**
     * Classical linear layer: Y = X @ W + b
     *
     * This method is marked with @Reflect so it can be captured via code reflection.
     * Uses the fluent API: x.matmul(w).add(b)
     *
     * @param x Input tensor [batch_size, input_features]
     * @param w Weight matrix [input_features, output_features]
     * @param b Bias vector [1, output_features] (broadcast over batch)
     * @return Output tensor [batch_size, output_features]
     */
    @Reflect
    public static Tensor<Float> linearLayer(Tensor<Float> x, Tensor<Float> w, Tensor<Float> b) {
        // Y = X @ W + b (fluent API)
        return x.matmul(w).add(b);
    }

    /**
     * Two stacked linear layers (MLP) using fluent API
     */
    @Reflect
    public static Tensor<Float> twoLayerMLP(Tensor<Float> x,
                                             Tensor<Float> w1, Tensor<Float> b1,
                                             Tensor<Float> w2, Tensor<Float> b2) {
        // Layer 1: hidden = x @ w1 + b1
        // Layer 2: output = hidden @ w2 + b2
        return x.matmul(w1).add(b1).matmul(w2).add(b2);
    }

    @Reflect
    public static void main(String[] args) {
        System.out.println("=".repeat(80));
        System.out.println("TOSA Linear Layer Demo - Code Reflection Showcase");
        System.out.println("=".repeat(80));
        System.out.println();

        // Create sample tensors for a simple linear layer
        // Input: batch_size=2, input_features=3
        // Output: output_features=2
        // X: [2, 3], W: [3, 2], b: [1, 2]

        // Input tensor X [2, 3]
        Tensor<Float> x = Tensor.ofShape(new long[]{2, 3},
            1.0f, 2.0f, 3.0f,   // batch 0
            4.0f, 5.0f, 6.0f    // batch 1
        );

        // Weight matrix W [3, 2]
        Tensor<Float> w = Tensor.ofShape(new long[]{3, 2},
            0.1f, 0.2f,   // row 0
            0.3f, 0.4f,   // row 1
            0.5f, 0.6f    // row 2
        );

        // Bias vector b [1, 2] (will be broadcast)
        Tensor<Float> b = Tensor.ofShape(new long[]{1, 2},
            0.1f, 0.2f
        );

        System.out.println("Input tensors:");
        System.out.println("  X (input)  : " + x);
        System.out.println("  W (weights): " + w);
        System.out.println("  b (bias)   : " + b);
        System.out.println();

        // ============================================================
        // 1. Execute the linear layer and show result
        // ============================================================
        System.out.println("-".repeat(80));
        System.out.println("1. EXECUTING LINEAR LAYER: Y = X @ W + b");
        System.out.println("-".repeat(80));

        Tensor<Float> y = linearLayer(x, w, b);
        System.out.println("  Result Y: " + y);

        // Manual verification:
        // X @ W = [[1*0.1+2*0.3+3*0.5, 1*0.2+2*0.4+3*0.6], [4*0.1+5*0.3+6*0.5, 4*0.2+5*0.4+6*0.6]]
        //       = [[0.1+0.6+1.5, 0.2+0.8+1.8], [0.4+1.5+3.0, 0.8+2.0+3.6]]
        //       = [[2.2, 2.8], [4.9, 6.4]]
        // + b  = [[2.3, 3.0], [5.0, 6.6]]
        System.out.println("  Expected: [2.3, 3.0, 5.0, 6.6]");
        System.out.println();

        // ============================================================
        // 2. Capture and display the code model using reflection
        // ============================================================
        System.out.println("-".repeat(80));
        System.out.println("2. CODE REFLECTION - Method Body (linearLayer)");
        System.out.println("-".repeat(80));

        // Get the code model of the @Reflect method directly
        try {
            Method linearLayerMethod = LinearLayerDemo.class.getMethod(
                "linearLayer", Tensor.class, Tensor.class, Tensor.class);
            CoreOp.FuncOp funcOp = Op.ofMethod(linearLayerMethod).orElseThrow(
                () -> new IllegalStateException("Method not reflectable"));

            System.out.println("  Method: linearLayer(x, w, b)");
            System.out.println("  Java code: return x.matmul(w).add(b);");
            System.out.println();
            System.out.println("  Code Model (showing MatMul and Add operations):");
            System.out.println("  " + "-".repeat(50));
            for (String line : funcOp.toText().split("\n")) {
                System.out.println("  " + line);
            }
        } catch (NoSuchMethodException e) {
            System.out.println("  Error: " + e.getMessage());
        }
        System.out.println();

        // ============================================================
        // 3. GENERATED TOSA CODE from Code Reflection
        // ============================================================
        System.out.println("-".repeat(80));
        System.out.println("3. GENERATED TOSA CODE (from Code Reflection)");
        System.out.println("-".repeat(80));

        try {
            Method linearLayerMethodForTosa = LinearLayerDemo.class.getMethod(
                "linearLayer", Tensor.class, Tensor.class, Tensor.class);
            String tosaCode = TosaCodeGenerator.generateTosa(linearLayerMethodForTosa);
            System.out.println("  Generated TOSA/MLIR code:");
            System.out.println();
            for (String line : tosaCode.split("\n")) {
                System.out.println("  " + line);
            }
        } catch (NoSuchMethodException e) {
            System.out.println("  Error: " + e.getMessage());
        }
        System.out.println();

        // ============================================================
        // 4. Two-layer MLP example
        // ============================================================
        System.out.println("-".repeat(80));
        System.out.println("4. TWO-LAYER MLP EXAMPLE");
        System.out.println("-".repeat(80));

        // Create second layer weights
        // Layer 1: [3] -> [4], Layer 2: [4] -> [2]
        Tensor<Float> x2 = Tensor.ofShape(new long[]{1, 3}, 1.0f, 2.0f, 3.0f);
        Tensor<Float> w1 = Tensor.ofShape(new long[]{3, 4},
            0.1f, 0.2f, 0.3f, 0.4f,
            0.1f, 0.2f, 0.3f, 0.4f,
            0.1f, 0.2f, 0.3f, 0.4f
        );
        Tensor<Float> b1 = Tensor.ofShape(new long[]{1, 4}, 0.0f, 0.0f, 0.0f, 0.0f);
        Tensor<Float> w2 = Tensor.ofShape(new long[]{4, 2},
            0.5f, 0.5f,
            0.5f, 0.5f,
            0.5f, 0.5f,
            0.5f, 0.5f
        );
        Tensor<Float> b2 = Tensor.ofShape(new long[]{1, 2}, 0.1f, 0.1f);

        Tensor<Float> mlpResult = twoLayerMLP(x2, w1, b1, w2, b2);
        System.out.println("  Input: " + x2);
        System.out.println("  MLP Output: " + mlpResult);

        // Generate TOSA code for MLP
        try {
            Method mlpMethod = LinearLayerDemo.class.getMethod(
                "twoLayerMLP", Tensor.class, Tensor.class, Tensor.class, Tensor.class, Tensor.class);

            System.out.println();
            System.out.println("  Java code:");
            System.out.println("    return x.matmul(w1).add(b1).matmul(w2).add(b2);");
            System.out.println();
            System.out.println("  Generated TOSA code:");
            String mlpTosa = TosaCodeGenerator.generateTosa(mlpMethod);
            for (String line : mlpTosa.split("\n")) {
                System.out.println("  " + line);
            }
        } catch (NoSuchMethodException e) {
            System.out.println("  Error: " + e.getMessage());
        }

        System.out.println();
        System.out.println("=".repeat(80));
        System.out.println("Demo complete! The operation trees above show how Java code is captured");
        System.out.println("via code reflection and can be transformed to TOSA/MLIR operations.");
        System.out.println("=".repeat(80));
    }
}
