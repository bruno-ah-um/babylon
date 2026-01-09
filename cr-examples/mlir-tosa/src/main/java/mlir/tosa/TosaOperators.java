package mlir.tosa;

/**
 * TOSA (Tensor Operator Set Architecture) operators.
 *
 * These operators provide element-wise operations on tensors and can be
 * captured via code reflection for transformation to MLIR TOSA.
 */
public final class TosaOperators {

    private TosaOperators() {
        // Utility class
    }

    /**
     * TOSA Add operation - element-wise addition of two tensors.
     *
     * Performs element-wise addition: output = input1 + input2
     *
     * @param input1 First input tensor
     * @param input2 Second input tensor
     * @param <T> Element type (must match for both inputs)
     * @return Result tensor with same shape and type as inputs
     */
    public static <T> Tensor<T> Add(Tensor<T> input1, Tensor<T> input2) {
        return TosaInterpreter.interpret("Add", input1, input2);
    }

    /**
     * TOSA Mul operation - element-wise multiplication of two tensors.
     *
     * Performs element-wise multiplication: output = input1 * input2
     *
     * @param input1 First input tensor
     * @param input2 Second input tensor
     * @param <T> Element type (must match for both inputs)
     * @return Result tensor with same shape and type as inputs
     */
    public static <T> Tensor<T> Mul(Tensor<T> input1, Tensor<T> input2) {
        return TosaInterpreter.interpret("Mul", input1, input2);
    }

    /**
     * TOSA MatMul operation - matrix multiplication of two tensors.
     *
     * Performs matrix multiplication: output = input1 @ input2
     *
     * For 2D tensors with shapes [M, K] and [K, N], produces output of shape [M, N].
     * For batched 3D tensors with shapes [B, M, K] and [B, K, N], produces [B, M, N].
     *
     * @param input1 First input tensor (left matrix)
     * @param input2 Second input tensor (right matrix)
     * @param <T> Element type (must match for both inputs)
     * @return Result tensor from matrix multiplication
     */
    public static <T> Tensor<T> MatMul(Tensor<T> input1, Tensor<T> input2) {
        return TosaInterpreter.interpret("MatMul", input1, input2);
    }

    // Future TOSA operations can be added here following the same pattern:
    // - Sub (subtraction)
    // - Negate
    // - Reciprocal
    // - Exp, Log
    // - Relu, Sigmoid, Tanh
    // etc.
}
