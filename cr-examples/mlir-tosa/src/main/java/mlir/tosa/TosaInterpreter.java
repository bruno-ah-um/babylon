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
     * Interpret a TOSA operation call (binary operations).
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

    /**
     * Interpret Conv2D operation.
     */
    public static <T> Tensor<T> interpretConv2D(String opName, Tensor<T> input, Tensor<T> weight,
                                                  Tensor<T> bias, long[] pad, long[] stride, long[] dilation) {
        return TosaRuntime.executeConv2D(input, weight, bias, pad, stride, dilation);
    }

    /**
     * Interpret pooling operations (MaxPool2D, AvgPool2D).
     */
    public static <T> Tensor<T> interpretPooling(String opName, Tensor<T> input,
                                                   long[] kernel, long[] stride, long[] pad) {
        return TosaRuntime.executePooling(opName, input, kernel, stride, pad);
    }

    /**
     * Interpret clamp operation.
     */
    public static <T> Tensor<T> interpretClamp(String opName, Tensor<T> input, float minVal, float maxVal) {
        return TosaRuntime.executeClamp(input, minVal, maxVal);
    }

    /**
     * Interpret reshape operation.
     */
    public static <T> Tensor<T> interpretReshape(String opName, Tensor<T> input, long[] newShape) {
        return TosaRuntime.executeReshape(input, newShape);
    }

    /**
     * Interpret unary operations (Reciprocal, Exp, etc.).
     */
    public static <T> Tensor<T> interpretUnary(String opName, Tensor<T> input) {
        return TosaRuntime.executeUnary(opName, input);
    }

    /**
     * Interpret reduce operations (ReduceSum, ReduceMax, etc.).
     */
    public static <T> Tensor<T> interpretReduce(String opName, Tensor<T> input, int axis, boolean keepDims) {
        return TosaRuntime.executeReduce(opName, input, axis, keepDims);
    }
}
