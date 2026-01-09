package mlir.tosa;

/**
 * Interpreter for TOSA operations.
 *
 * This class intercepts TOSA operator calls and delegates to the TosaRuntime
 * for actual execution via MLIR.
 */
public final class TosaInterpreter {

    private TosaInterpreter() {
        // Utility class
    }

    /**
     * Interpret a TOSA operation call.
     *
     * This method is called when a TOSA operator (like Add or Mul) is invoked.
     *
     * @param opName The TOSA operation name (e.g., "Add", "Mul")
     * @param inputs The input tensors
     * @param <T> The tensor element type
     * @return The result tensor
     */
    @SuppressWarnings("unchecked")
    public static <T> Tensor<T> interpret(String opName, Tensor<T>... inputs) {
        // Delegate to the runtime for execution
        return (Tensor<T>) TosaRuntime.executeOperation(opName, inputs);
    }
}
