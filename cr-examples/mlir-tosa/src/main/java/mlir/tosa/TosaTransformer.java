package mlir.tosa;

import java.lang.invoke.MethodHandles;
import java.util.function.Supplier;
import jdk.incubator.code.Op;
import jdk.incubator.code.Quoted;
import jdk.incubator.code.dialect.java.JavaOp;

/**
 * Transformer for converting Java code to MLIR TOSA operations via code reflection.
 *
 * This class uses the Code Reflection API (jdk.incubator.code) to capture
 * Java methods annotated with @Reflect and transform them to TOSA operations.
 */
public final class TosaTransformer {

    private TosaTransformer() {
        // Utility class
    }

    /**
     * Execute a quotable lambda containing TOSA operations.
     *
     * This method captures the lambda using code reflection and executes it.
     * Future enhancements will transform the captured code to MLIR TOSA IR.
     *
     * @param lookup Method handles lookup for reflection
     * @param quotableLambda The lambda containing TOSA operations (must be Quotable)
     * @param <T> Return type
     * @return Result of executing the TOSA operations
     */
    public static <T> T execute(MethodHandles.Lookup lookup, Supplier<T> quotableLambda) {
        // Capture the lambda as a Quoted object
        Quoted<JavaOp.LambdaOp> quoted = Op.ofLambda(quotableLambda).orElseThrow(() ->
            new IllegalArgumentException("Lambda is not quotable - ensure @Reflect annotation is present")
        );

        // Log the captured operation tree (for debugging)
        if (Boolean.getBoolean("tosa.debug")) {
            System.out.println("Captured operation tree:");
            System.out.println(quoted.op());
        }

        // For now, execute the lambda directly via Java runtime.
        // When tosa.debug is set, also generate and print the MLIR representation.
        if (Boolean.getBoolean("tosa.debug")) {
            try {
                String mlir = transformToMlirTosa(quoted);
                System.out.println("Generated MLIR TOSA:");
                System.out.println(mlir);
            } catch (Exception e) {
                System.out.println("MLIR generation skipped: " + e.getMessage());
            }
        }
        return quotableLambda.get();
    }

    /**
     * Transform a Quoted operation tree to MLIR TOSA representation.
     *
     * This method will analyze the operation tree and identify TOSA operations
     * to be transformed into MLIR.
     *
     * @param quoted The quoted code representation
     * @return Information about the transformation (to be defined)
     */
    public static String analyzeQuoted(Quoted<?> quoted) {
        Op op = quoted.op();

        StringBuilder analysis = new StringBuilder();
        analysis.append("Operation tree analysis:\n");
        analysis.append("Op type: ").append(op.getClass().getSimpleName()).append("\n");
        analysis.append("Op: ").append(op).append("\n");

        // Analyze if this is a lambda
        if (op instanceof JavaOp.LambdaOp lambdaOp) {
            analysis.append("Lambda body: ").append(lambdaOp.body()).append("\n");
        }

        return analysis.toString();
    }

    /**
     * Transform a Quoted lambda to MLIR TOSA text representation.
     *
     * <p>Pipeline:
     * <ol>
     *   <li>Validate that the quoted operation is a {@code LambdaOp}.</li>
     *   <li>Walk the lambda body, mapping {@code TosaOperators} call sites to native
     *       MLIR TOSA operations via the C API.</li>
     *   <li>Serialize the resulting MLIR module to a string.</li>
     * </ol>
     *
     * @param quoted Quoted lambda captured via {@code Op.ofLambda()}
     * @param tensorRank Rank of tensors used in the lambda (1 = 1-D, 2 = 2-D, …)
     * @return TOSA MLIR text for the lambda body
     * @throws IllegalArgumentException if the quoted operation is not a lambda
     */
    public static String transformToMlirTosa(Quoted<?> quoted, int tensorRank) {
        Op op = quoted.op();
        if (!(op instanceof JavaOp.LambdaOp lambdaOp)) {
            throw new IllegalArgumentException(
                "Expected a lambda expression, got: " + op.getClass().getSimpleName());
        }
        return TosaCodeGenerator.generateTosaFromLambda(lambdaOp, "lambda_func", tensorRank);
    }

    /**
     * Transform a Quoted lambda to MLIR TOSA text representation using 1-D dynamic tensors.
     *
     * @param quoted Quoted lambda captured via {@code Op.ofLambda()}
     * @return TOSA MLIR text for the lambda body
     */
    public static String transformToMlirTosa(Quoted<?> quoted) {
        return transformToMlirTosa(quoted, 1);
    }
}
