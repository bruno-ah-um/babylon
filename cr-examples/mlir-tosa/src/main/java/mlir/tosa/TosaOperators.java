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

    // ========== Convolution Operations ==========

    /**
     * TOSA Conv2D operation - 2D convolution.
     *
     * Input shape: [N, IH, IW, IC] (batch, height, width, input channels)
     * Weight shape: [OC, KH, KW, IC] (output channels, kernel height, kernel width, input channels)
     * Bias shape: [OC] (output channels)
     * Output shape: [N, OH, OW, OC]
     *
     * @param input Input tensor [N, IH, IW, IC]
     * @param weight Weight tensor [OC, KH, KW, IC]
     * @param bias Bias tensor [OC]
     * @param pad Padding [top, bottom, left, right]
     * @param stride Stride [stride_h, stride_w]
     * @param dilation Dilation [dilation_h, dilation_w]
     * @param <T> Element type
     * @return Output tensor [N, OH, OW, OC]
     */
    public static <T> Tensor<T> Conv2D(Tensor<T> input, Tensor<T> weight, Tensor<T> bias,
                                        long[] pad, long[] stride, long[] dilation) {
        return TosaInterpreter.interpretConv2D("Conv2D", input, weight, bias, pad, stride, dilation);
    }

    // ========== Pooling Operations ==========

    /**
     * TOSA MaxPool2D operation - 2D max pooling.
     *
     * Input shape: [N, IH, IW, C]
     * Output shape: [N, OH, OW, C]
     *
     * @param input Input tensor [N, IH, IW, C]
     * @param kernel Kernel size [kernel_h, kernel_w]
     * @param stride Stride [stride_h, stride_w]
     * @param pad Padding [top, bottom, left, right]
     * @param <T> Element type
     * @return Output tensor [N, OH, OW, C]
     */
    public static <T> Tensor<T> MaxPool2D(Tensor<T> input, long[] kernel, long[] stride, long[] pad) {
        return TosaInterpreter.interpretPooling("MaxPool2D", input, kernel, stride, pad);
    }

    /**
     * TOSA AvgPool2D operation - 2D average pooling.
     *
     * Input shape: [N, IH, IW, C]
     * Output shape: [N, OH, OW, C]
     *
     * @param input Input tensor [N, IH, IW, C]
     * @param kernel Kernel size [kernel_h, kernel_w]
     * @param stride Stride [stride_h, stride_w]
     * @param pad Padding [top, bottom, left, right]
     * @param <T> Element type
     * @return Output tensor [N, OH, OW, C]
     */
    public static <T> Tensor<T> AvgPool2D(Tensor<T> input, long[] kernel, long[] stride, long[] pad) {
        return TosaInterpreter.interpretPooling("AvgPool2D", input, kernel, stride, pad);
    }

    // ========== Activation Operations ==========

    /**
     * TOSA Clamp operation - clamps values to a range.
     *
     * output = min(max(input, min_val), max_val)
     *
     * Use for ReLU: Clamp(input, 0, Float.MAX_VALUE)
     * Use for ReLU6: Clamp(input, 0, 6)
     *
     * @param input Input tensor
     * @param minVal Minimum value
     * @param maxVal Maximum value
     * @param <T> Element type
     * @return Clamped tensor
     */
    public static <T> Tensor<T> Clamp(Tensor<T> input, float minVal, float maxVal) {
        return TosaInterpreter.interpretClamp("Clamp", input, minVal, maxVal);
    }

    /**
     * ReLU activation using TOSA Clamp.
     *
     * output = max(0, input)
     *
     * @param input Input tensor
     * @param <T> Element type
     * @return ReLU activated tensor
     */
    public static <T> Tensor<T> Relu(Tensor<T> input) {
        return Clamp(input, 0.0f, Float.MAX_VALUE);
    }

    /**
     * TOSA Sigmoid activation - element-wise sigmoid function.
     *
     * output = 1 / (1 + exp(-x))
     *
     * @param input Input tensor
     * @param <T> Element type
     * @return Sigmoid activated tensor
     */
    public static <T> Tensor<T> Sigmoid(Tensor<T> input) {
        return TosaInterpreter.interpretUnary("Sigmoid", input);
    }

    /**
     * TOSA Tanh activation - element-wise hyperbolic tangent.
     *
     * output = tanh(x)
     *
     * @param input Input tensor
     * @param <T> Element type
     * @return Tanh activated tensor
     */
    public static <T> Tensor<T> Tanh(Tensor<T> input) {
        return TosaInterpreter.interpretUnary("Tanh", input);
    }

    // ========== Shape Operations ==========

    /**
     * TOSA Reshape operation - reshapes tensor to new shape.
     *
     * @param input Input tensor
     * @param newShape New shape (total elements must match)
     * @param <T> Element type
     * @return Reshaped tensor
     */
    public static <T> Tensor<T> Reshape(Tensor<T> input, long[] newShape) {
        return TosaInterpreter.interpretReshape("Reshape", input, newShape);
    }

    /**
     * Flatten operation using TOSA Reshape.
     *
     * Flattens dimensions from axis onwards into a single dimension.
     *
     * @param input Input tensor
     * @param axis Axis from which to flatten (default 1 keeps batch dimension)
     * @param <T> Element type
     * @return Flattened tensor
     */
    public static <T> Tensor<T> Flatten(Tensor<T> input, int axis) {
        long[] shape = input.shape();

        // Calculate new shape: [shape[0:axis], product(shape[axis:])]
        long batchDims = 1;
        for (int i = 0; i < axis; i++) {
            batchDims *= shape[i];
        }

        long flatDims = 1;
        for (int i = axis; i < shape.length; i++) {
            flatDims *= shape[i];
        }

        return Reshape(input, new long[]{batchDims, flatDims});
    }

    // ========== Arithmetic Operations ==========

    /**
     * TOSA Reciprocal operation - element-wise 1/x.
     *
     * @param input Input tensor
     * @param <T> Element type
     * @return Tensor with reciprocal values
     */
    public static <T> Tensor<T> Reciprocal(Tensor<T> input) {
        return TosaInterpreter.interpretUnary("Reciprocal", input);
    }

    /**
     * TOSA Exp operation - element-wise exponential.
     *
     * @param input Input tensor
     * @param <T> Element type
     * @return Tensor with exp(x) values
     */
    public static <T> Tensor<T> Exp(Tensor<T> input) {
        return TosaInterpreter.interpretUnary("Exp", input);
    }

    /**
     * TOSA ReduceSum operation - sum reduction along an axis.
     *
     * @param input Input tensor
     * @param axis Axis to reduce
     * @param keepDims Whether to keep reduced dimensions
     * @param <T> Element type
     * @return Reduced tensor
     */
    public static <T> Tensor<T> ReduceSum(Tensor<T> input, int axis, boolean keepDims) {
        return TosaInterpreter.interpretReduce("ReduceSum", input, axis, keepDims);
    }

    /**
     * TOSA ReduceMax operation - max reduction along an axis.
     *
     * @param input Input tensor
     * @param axis Axis to reduce
     * @param keepDims Whether to keep reduced dimensions
     * @param <T> Element type
     * @return Reduced tensor
     */
    public static <T> Tensor<T> ReduceMax(Tensor<T> input, int axis, boolean keepDims) {
        return TosaInterpreter.interpretReduce("ReduceMax", input, axis, keepDims);
    }

    /**
     * TOSA Sub operation - element-wise subtraction.
     *
     * @param input1 First input tensor
     * @param input2 Second input tensor
     * @param <T> Element type
     * @return Result of input1 - input2
     */
    public static <T> Tensor<T> Sub(Tensor<T> input1, Tensor<T> input2) {
        return TosaInterpreter.interpret("Sub", input1, input2);
    }

    /**
     * Softmax operation composed from TOSA primitives.
     *
     * softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
     *
     * @param input Input tensor
     * @param axis Axis along which to compute softmax
     * @param <T> Element type
     * @return Softmax probabilities
     */
    public static <T> Tensor<T> Softmax(Tensor<T> input, int axis) {
        // For numerical stability: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
        Tensor<T> maxVal = ReduceMax(input, axis, true);
        Tensor<T> shifted = Sub(input, maxVal);
        Tensor<T> expVal = Exp(shifted);
        Tensor<T> sumExp = ReduceSum(expVal, axis, true);
        Tensor<T> recipSum = Reciprocal(sumExp);
        return Mul(expVal, recipSum);
    }

    /**
     * Constant operation - creates a scalar constant tensor.
     *
     * @param value Scalar value
     * @return Scalar tensor
     */
    public static Tensor<Float> Constant(float value) {
        return Tensor.ofScalar(value);
    }

    /**
     * Division using reciprocal and multiply.
     *
     * div(a, b) = a * reciprocal(b)
     *
     * @param input1 Numerator tensor
     * @param input2 Denominator tensor
     * @param <T> Element type
     * @return Result of input1 / input2
     */
    public static <T> Tensor<T> Div(Tensor<T> input1, Tensor<T> input2) {
        return Mul(input1, Reciprocal(input2));
    }
}
