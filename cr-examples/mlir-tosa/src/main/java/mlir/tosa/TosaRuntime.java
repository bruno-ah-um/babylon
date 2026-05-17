package mlir.tosa;

// TODO: Uncomment when jextract bindings are generated
// import mlir.tosa.bindings.*;
// import static mlir.tosa.bindings.mlir_tosa_c_api_h.*;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Runtime for executing TOSA operations via MLIR C API.
 *
 * This class provides the bridge between Java tensor operations and the
 * native MLIR TOSA library compiled from C++.
 */
public final class TosaRuntime {

    private static final TosaRuntime INSTANCE = new TosaRuntime();

    @SuppressWarnings("unused") // Keeps the auto-arena alive for the lifetime of the singleton
    private final Arena arena;
    @SuppressWarnings("unused") // TODO: use when jextract bindings are available
    private final MemorySegment context;
    @SuppressWarnings("unused") // TODO: use when jextract bindings are available
    private final MemorySegment module;

    private TosaRuntime() {
        this.arena = Arena.ofAuto();
        // TODO: Initialize MLIR context when jextract bindings are available
        this.context = null; // mlir_context_create();
        this.module = null; // mlir_module_create(context);
    }

    public static TosaRuntime getInstance() {
        return INSTANCE;
    }

    /**
     * Execute a TOSA operation on input tensors.
     *
     * @param opName The TOSA operation name (e.g., "Add", "Mul")
     * @param inputs Input tensors
     * @param <T> Tensor element type
     * @return Result tensor
     */
    @SuppressWarnings("unchecked")
    public static <T> Tensor<T> executeOperation(String opName, Tensor<T>... inputs) {
        TosaRuntime runtime = getInstance();

        return switch (opName) {
            case "Add" -> runtime.executeBinaryOpWithBroadcast(inputs[0], inputs[1], "add");
            case "Mul" -> runtime.executeBinaryOpWithBroadcast(inputs[0], inputs[1], "mul");
            case "Sub" -> runtime.executeBinaryOpWithBroadcast(inputs[0], inputs[1], "sub");
            case "MatMul" -> runtime.executeMatMul(inputs[0], inputs[1]);
            default -> throw new UnsupportedOperationException("Unsupported TOSA operation: " + opName);
        };
    }

    /**
     * Execute Conv2D operation.
     * Input: [N, IH, IW, IC], Weight: [OC, KH, KW, IC], Bias: [OC]
     * Output: [N, OH, OW, OC]
     */
    public static <T> Tensor<T> executeConv2D(Tensor<T> input, Tensor<T> weight, Tensor<T> bias,
                                               long[] pad, long[] stride, long[] dilation) {
        if (input.rank() != 4) {
            throw new IllegalArgumentException(
                "Conv2D input must be rank 4 [N, IH, IW, IC], got rank " + input.rank());
        }
        if (weight.rank() != 4) {
            throw new IllegalArgumentException(
                "Conv2D weight must be rank 4 [OC, KH, KW, IC], got rank " + weight.rank());
        }
        if (bias != null && bias.rank() != 1) {
            throw new IllegalArgumentException(
                "Conv2D bias must be rank 1 [OC], got rank " + bias.rank());
        }

        long[] inputShape = input.shape();   // [N, IH, IW, IC]
        long[] weightShape = weight.shape(); // [OC, KH, KW, IC]

        long IC = inputShape[3];
        long weightIC = weightShape[3];
        if (IC != weightIC) {
            throw new IllegalArgumentException(
                "Conv2D input channels (" + IC + ") must match weight input channels (" + weightIC + ")");
        }
        if (bias != null && bias.shape()[0] != weightShape[0]) {
            throw new IllegalArgumentException(
                "Conv2D bias length (" + bias.shape()[0] + ") must match output channels (" + weightShape[0] + ")");
        }

        long N = inputShape[0];
        long IH = inputShape[1];
        long IW = inputShape[2];

        long OC = weightShape[0];
        long KH = weightShape[1];
        long KW = weightShape[2];

        long padTop = pad[0];
        long padBottom = pad[1];
        long padLeft = pad[2];
        long padRight = pad[3];
        long strideH = stride[0];
        long strideW = stride[1];
        long dilationH = dilation[0];
        long dilationW = dilation[1];

        // Calculate output dimensions
        long OH = (IH + padTop + padBottom - dilationH * (KH - 1) - 1) / strideH + 1;
        long OW = (IW + padLeft + padRight - dilationW * (KW - 1) - 1) / strideW + 1;

        Arena resultArena = Arena.ofAuto();
        Tensor.ElementType type = input.elementType();
        long[] outputShape = new long[]{N, OH, OW, OC};
        long numElements = N * OH * OW * OC;
        MemorySegment resultData = resultArena.allocate(type.valueLayout(), numElements);

        // Perform convolution
        for (long n = 0; n < N; n++) {
            for (long oh = 0; oh < OH; oh++) {
                for (long ow = 0; ow < OW; ow++) {
                    for (long oc = 0; oc < OC; oc++) {
                        float sum = 0.0f;

                        // Get bias
                        if (bias != null) {
                            sum = bias.data().getAtIndex(ValueLayout.JAVA_FLOAT, oc);
                        }

                        // Convolve
                        for (long kh = 0; kh < KH; kh++) {
                            for (long kw = 0; kw < KW; kw++) {
                                long ih = oh * strideH + kh * dilationH - padTop;
                                long iw = ow * strideW + kw * dilationW - padLeft;

                                if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                                    for (long ic = 0; ic < IC; ic++) {
                                        long inputIdx = n * IH * IW * IC + ih * IW * IC + iw * IC + ic;
                                        long weightIdx = oc * KH * KW * IC + kh * KW * IC + kw * IC + ic;

                                        float inputVal = input.data().getAtIndex(ValueLayout.JAVA_FLOAT, inputIdx);
                                        float weightVal = weight.data().getAtIndex(ValueLayout.JAVA_FLOAT, weightIdx);
                                        sum += inputVal * weightVal;
                                    }
                                }
                            }
                        }

                        long outputIdx = n * OH * OW * OC + oh * OW * OC + ow * OC + oc;
                        resultData.setAtIndex(ValueLayout.JAVA_FLOAT, outputIdx, sum);
                    }
                }
            }
        }

        return Tensor.ofRaw(resultArena, outputShape, type, resultData);
    }

    /**
     * Execute pooling operation (MaxPool2D or AvgPool2D).
     * Input: [N, IH, IW, C], Output: [N, OH, OW, C]
     */
    public static <T> Tensor<T> executePooling(String opName, Tensor<T> input,
                                                long[] kernel, long[] stride, long[] pad) {
        long[] inputShape = input.shape(); // [N, IH, IW, C]

        long N = inputShape[0];
        long IH = inputShape[1];
        long IW = inputShape[2];
        long C = inputShape[3];

        long KH = kernel[0];
        long KW = kernel[1];
        long strideH = stride[0];
        long strideW = stride[1];
        long padTop = pad[0];
        long padBottom = pad[1];
        long padLeft = pad[2];
        long padRight = pad[3];

        // Calculate output dimensions
        long OH = (IH + padTop + padBottom - KH) / strideH + 1;
        long OW = (IW + padLeft + padRight - KW) / strideW + 1;

        Arena resultArena = Arena.ofAuto();
        Tensor.ElementType type = input.elementType();
        long[] outputShape = new long[]{N, OH, OW, C};
        long numElements = N * OH * OW * C;
        MemorySegment resultData = resultArena.allocate(type.valueLayout(), numElements);

        boolean isMax = opName.equals("MaxPool2D");

        for (long n = 0; n < N; n++) {
            for (long oh = 0; oh < OH; oh++) {
                for (long ow = 0; ow < OW; ow++) {
                    for (long c = 0; c < C; c++) {
                        float result = isMax ? Float.NEGATIVE_INFINITY : 0.0f;
                        int count = 0;

                        for (long kh = 0; kh < KH; kh++) {
                            for (long kw = 0; kw < KW; kw++) {
                                long ih = oh * strideH + kh - padTop;
                                long iw = ow * strideW + kw - padLeft;

                                if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                                    long inputIdx = n * IH * IW * C + ih * IW * C + iw * C + c;
                                    float val = input.data().getAtIndex(ValueLayout.JAVA_FLOAT, inputIdx);

                                    if (isMax) {
                                        result = Math.max(result, val);
                                    } else {
                                        result += val;
                                        count++;
                                    }
                                }
                            }
                        }

                        if (!isMax && count > 0) {
                            result /= count;
                        }

                        long outputIdx = n * OH * OW * C + oh * OW * C + ow * C + c;
                        resultData.setAtIndex(ValueLayout.JAVA_FLOAT, outputIdx, result);
                    }
                }
            }
        }

        return Tensor.ofRaw(resultArena, outputShape, type, resultData);
    }

    /**
     * Execute clamp operation.
     * output = min(max(input, minVal), maxVal)
     */
    public static <T> Tensor<T> executeClamp(Tensor<T> input, float minVal, float maxVal) {
        Arena resultArena = Arena.ofAuto();
        long numElements = input.numElements();
        Tensor.ElementType type = input.elementType();
        MemorySegment resultData = resultArena.allocate(type.valueLayout(), numElements);

        for (long i = 0; i < numElements; i++) {
            float val = input.data().getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float clamped = Math.min(Math.max(val, minVal), maxVal);
            resultData.setAtIndex(ValueLayout.JAVA_FLOAT, i, clamped);
        }

        return Tensor.ofRaw(resultArena, input.shape(), type, resultData);
    }

    /**
     * Execute reshape operation.
     */
    public static <T> Tensor<T> executeReshape(Tensor<T> input, long[] newShape) {
        long inputElements = input.numElements();
        long outputElements = 1;
        for (long dim : newShape) {
            outputElements *= dim;
        }

        if (inputElements != outputElements) {
            throw new IllegalArgumentException(
                "Cannot reshape: input has " + inputElements + " elements, " +
                "but new shape requires " + outputElements
            );
        }

        // Reshape is just a view change - data stays the same
        return Tensor.ofRaw(input.arena(), newShape, input.elementType(), input.data());
    }

    /**
     * Execute unary operations (Reciprocal, Exp, etc.).
     */
    public static <T> Tensor<T> executeUnary(String opName, Tensor<T> input) {
        Arena resultArena = Arena.ofAuto();
        long numElements = input.numElements();
        Tensor.ElementType type = input.elementType();
        MemorySegment resultData = resultArena.allocate(type.valueLayout(), numElements);

        for (long i = 0; i < numElements; i++) {
            float val = input.data().getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float result = switch (opName) {
                case "Reciprocal" -> 1.0f / val;
                case "Exp" -> (float) Math.exp(val);
                case "Log" -> (float) Math.log(val);
                case "Neg" -> -val;
                case "Abs" -> Math.abs(val);
                case "Sqrt" -> (float) Math.sqrt(val);
                case "Sigmoid" -> 1.0f / (1.0f + (float) Math.exp(-val));
                case "Tanh" -> (float) Math.tanh(val);
                default -> throw new UnsupportedOperationException("Unknown unary op: " + opName);
            };
            resultData.setAtIndex(ValueLayout.JAVA_FLOAT, i, result);
        }

        return Tensor.ofRaw(resultArena, input.shape(), type, resultData);
    }

    /**
     * Execute reduce operations (ReduceSum, ReduceMax, etc.).
     */
    public static <T> Tensor<T> executeReduce(String opName, Tensor<T> input, int axis, boolean keepDims) {
        long[] inputShape = input.shape();
        int rank = inputShape.length;

        // Handle negative axis
        if (axis < 0) {
            axis += rank;
        }

        // Calculate output shape
        long[] outputShape;
        if (keepDims) {
            outputShape = inputShape.clone();
            outputShape[axis] = 1;
        } else {
            outputShape = new long[rank - 1];
            for (int i = 0, j = 0; i < rank; i++) {
                if (i != axis) {
                    outputShape[j++] = inputShape[i];
                }
            }
            if (outputShape.length == 0) {
                outputShape = new long[]{1}; // scalar case
            }
        }

        Arena resultArena = Arena.ofAuto();
        Tensor.ElementType type = input.elementType();
        long numOutputElements = 1;
        for (long dim : outputShape) {
            numOutputElements *= dim;
        }
        MemorySegment resultData = resultArena.allocate(type.valueLayout(), numOutputElements);

        // Initialize output
        boolean isMax = opName.equals("ReduceMax");
        float initVal = isMax ? Float.NEGATIVE_INFINITY : 0.0f;
        for (long i = 0; i < numOutputElements; i++) {
            resultData.setAtIndex(ValueLayout.JAVA_FLOAT, i, initVal);
        }

        // Compute strides for input
        long[] strides = new long[rank];
        strides[rank - 1] = 1;
        for (int i = rank - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * inputShape[i + 1];
        }

        // Iterate over all input elements
        long numInputElements = input.numElements();
        for (long i = 0; i < numInputElements; i++) {
            // Convert flat index to multi-dimensional indices
            long[] indices = new long[rank];
            long remaining = i;
            for (int d = 0; d < rank; d++) {
                indices[d] = remaining / strides[d];
                remaining %= strides[d];
            }

            // Compute output index (skip the reduced axis)
            long outputIdx = 0;
            long outputStride = 1;
            for (int d = rank - 1; d >= 0; d--) {
                if (d != axis) {
                    if (keepDims) {
                        outputIdx += indices[d] * outputStride;
                        outputStride *= outputShape[d];
                    } else {
                        int outD = d > axis ? d - 1 : d;
                        if (outD >= 0 && outD < outputShape.length) {
                            long outStride = 1;
                            for (int k = outputShape.length - 1; k > outD; k--) {
                                outStride *= outputShape[k];
                            }
                            outputIdx += indices[d] * outStride;
                        }
                    }
                }
            }

            float inputVal = input.data().getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float currentVal = resultData.getAtIndex(ValueLayout.JAVA_FLOAT, outputIdx);

            float newVal = switch (opName) {
                case "ReduceSum" -> currentVal + inputVal;
                case "ReduceMax" -> Math.max(currentVal, inputVal);
                case "ReduceMin" -> Math.min(currentVal, inputVal);
                case "ReduceMean" -> currentVal + inputVal; // divide later
                default -> throw new UnsupportedOperationException("Unknown reduce op: " + opName);
            };

            resultData.setAtIndex(ValueLayout.JAVA_FLOAT, outputIdx, newVal);
        }

        // For ReduceMean, divide by axis size
        if (opName.equals("ReduceMean")) {
            long axisSize = inputShape[axis];
            for (long i = 0; i < numOutputElements; i++) {
                float val = resultData.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                resultData.setAtIndex(ValueLayout.JAVA_FLOAT, i, val / axisSize);
            }
        }

        return Tensor.ofRaw(resultArena, outputShape, type, resultData);
    }

    /**
     * Promotion hierarchy for binary operations on mixed element types.
     *
     * <pre>
     *   INT32  + INT64  → INT64
     *   INT32  + FLOAT32 → FLOAT32
     *   INT32  + FLOAT64 → FLOAT64
     *   INT64  + FLOAT32 → FLOAT32
     *   INT64  + FLOAT64 → FLOAT64
     *   FLOAT32 + FLOAT64 → FLOAT64
     * </pre>
     *
     * When the two types are the same the input is returned unchanged.
     */
    static Tensor.ElementType promoteTypes(Tensor.ElementType a, Tensor.ElementType b) {
        if (a == b) return a;
        if (a == Tensor.ElementType.FLOAT64 || b == Tensor.ElementType.FLOAT64) return Tensor.ElementType.FLOAT64;
        if (a == Tensor.ElementType.FLOAT32 || b == Tensor.ElementType.FLOAT32) return Tensor.ElementType.FLOAT32;
        return Tensor.ElementType.INT64; // INT32 + INT64
    }

    /**
     * Cast all elements of {@code input} to {@code targetType}.
     * Uses {@code double} as the lossless intermediate representation for float types;
     * integer types are widened or narrowed with standard Java semantics.
     */
    @SuppressWarnings("unchecked")
    static <T> Tensor<T> castTensor(Tensor<?> input, Tensor.ElementType targetType) {
        if (input.elementType() == targetType) {
            return (Tensor<T>) input;
        }
        Arena resultArena = Arena.ofAuto();
        long numElements = input.numElements();
        MemorySegment resultData = resultArena.allocate(targetType.valueLayout(), numElements);

        for (long i = 0; i < numElements; i++) {
            double val = switch (input.elementType()) {
                case FLOAT32 -> input.data().getAtIndex(ValueLayout.JAVA_FLOAT, i);
                case FLOAT64 -> input.data().getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                case INT32   -> input.data().getAtIndex(ValueLayout.JAVA_INT, i);
                case INT64   -> input.data().getAtIndex(ValueLayout.JAVA_LONG, i);
            };
            switch (targetType) {
                case FLOAT32 -> resultData.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) val);
                case FLOAT64 -> resultData.setAtIndex(ValueLayout.JAVA_DOUBLE, i, val);
                case INT32   -> resultData.setAtIndex(ValueLayout.JAVA_INT, i, (int) val);
                case INT64   -> resultData.setAtIndex(ValueLayout.JAVA_LONG, i, (long) val);
            }
        }
        return Tensor.ofRaw(resultArena, input.shape(), targetType, resultData);
    }

    /**
     * Execute a binary operation with full broadcasting support.
     * When the two input tensors have different element types, the narrower type is
     * promoted to the wider type before the operation (see {@link #promoteTypes}).
     */
    @SuppressWarnings("unchecked")
    private <T> Tensor<T> executeBinaryOpWithBroadcast(Tensor<T> input1, Tensor<T> input2, String opName) {
        if (input1.elementType() != input2.elementType()) {
            Tensor.ElementType promoted = promoteTypes(input1.elementType(), input2.elementType());
            return executeBinaryOpWithBroadcast(
                (Tensor<T>) castTensor(input1, promoted),
                (Tensor<T>) castTensor(input2, promoted),
                opName);
        }

        long[] shape1 = input1.shape();
        long[] shape2 = input2.shape();

        // Same shape - simple element-wise operation
        if (java.util.Arrays.equals(shape1, shape2)) {
            return computeElementWise(input1, input2, opName);
        }

        // Scalar broadcasting
        if (shape2.length == 0) {
            return computeWithScalar(input1, input2.data().getAtIndex(ValueLayout.JAVA_FLOAT, 0), opName);
        }
        if (shape1.length == 0) {
            return computeWithScalar(input2, input1.data().getAtIndex(ValueLayout.JAVA_FLOAT, 0), opName);
        }

        // General broadcasting for same rank - check if shapes are compatible
        if (shape1.length == shape2.length) {
            // Check if shapes are broadcastable (each dim must match or be 1)
            boolean broadcastable = true;
            for (int i = 0; i < shape1.length; i++) {
                if (shape1[i] != shape2[i] && shape1[i] != 1 && shape2[i] != 1) {
                    broadcastable = false;
                    break;
                }
            }
            if (broadcastable) {
                return computeWithBroadcast(input1, input2, opName);
            }
        }

        throw new IllegalArgumentException(
            "Shape mismatch and not broadcastable: " + java.util.Arrays.toString(shape1) +
            " vs " + java.util.Arrays.toString(shape2)
        );
    }

    /**
     * Compute element-wise operation (same shape tensors).
     */
    private <T> Tensor<T> computeElementWise(Tensor<T> input1, Tensor<T> input2, String opName) {
        Arena resultArena = Arena.ofAuto();
        long numElements = input1.numElements();
        Tensor.ElementType type = input1.elementType();
        MemorySegment resultData = resultArena.allocate(type.valueLayout(), numElements);

        for (long i = 0; i < numElements; i++) {
            switch (type) {
                case FLOAT32 -> {
                    float v1 = input1.data().getAtIndex(ValueLayout.JAVA_FLOAT, i);
                    float v2 = input2.data().getAtIndex(ValueLayout.JAVA_FLOAT, i);
                    resultData.setAtIndex(ValueLayout.JAVA_FLOAT, i, applyOp(v1, v2, opName));
                }
                case FLOAT64 -> {
                    double v1 = input1.data().getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                    double v2 = input2.data().getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                    resultData.setAtIndex(ValueLayout.JAVA_DOUBLE, i, applyOpDouble(v1, v2, opName));
                }
                case INT32 -> {
                    int v1 = input1.data().getAtIndex(ValueLayout.JAVA_INT, i);
                    int v2 = input2.data().getAtIndex(ValueLayout.JAVA_INT, i);
                    resultData.setAtIndex(ValueLayout.JAVA_INT, i, applyOpInt(v1, v2, opName));
                }
                case INT64 -> {
                    long v1 = input1.data().getAtIndex(ValueLayout.JAVA_LONG, i);
                    long v2 = input2.data().getAtIndex(ValueLayout.JAVA_LONG, i);
                    resultData.setAtIndex(ValueLayout.JAVA_LONG, i, applyOpLong(v1, v2, opName));
                }
                default -> throw new IllegalStateException("Unknown element type: " + type);
            }
        }

        return Tensor.ofRaw(resultArena, input1.shape(), type, resultData);
    }

    /**
     * Compute operation with scalar broadcast.
     */
    private <T> Tensor<T> computeWithScalar(Tensor<T> tensor, float scalar, String opName) {
        Arena resultArena = Arena.ofAuto();
        long numElements = tensor.numElements();
        Tensor.ElementType type = tensor.elementType();
        MemorySegment resultData = resultArena.allocate(type.valueLayout(), numElements);

        for (long i = 0; i < numElements; i++) {
            float v = tensor.data().getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float result = applyOp(v, scalar, opName);
            resultData.setAtIndex(ValueLayout.JAVA_FLOAT, i, result);
        }

        return Tensor.ofRaw(resultArena, tensor.shape(), type, resultData);
    }

    /**
     * Compute operation with broadcasting (same rank tensors).
     */
    private <T> Tensor<T> computeWithBroadcast(Tensor<T> input1, Tensor<T> input2, String opName) {
        long[] shape1 = input1.shape();
        long[] shape2 = input2.shape();
        int rank = shape1.length;

        // Compute output shape
        long[] outputShape = new long[rank];
        for (int i = 0; i < rank; i++) {
            outputShape[i] = Math.max(shape1[i], shape2[i]);
        }

        Arena resultArena = Arena.ofAuto();
        Tensor.ElementType type = input1.elementType();
        long numElements = 1;
        for (long dim : outputShape) {
            numElements *= dim;
        }
        MemorySegment resultData = resultArena.allocate(type.valueLayout(), numElements);

        // Compute strides for output
        long[] outStrides = new long[rank];
        outStrides[rank - 1] = 1;
        for (int i = rank - 2; i >= 0; i--) {
            outStrides[i] = outStrides[i + 1] * outputShape[i + 1];
        }

        // Compute strides for inputs (with broadcasting)
        long[] strides1 = new long[rank];
        long[] strides2 = new long[rank];
        strides1[rank - 1] = shape1[rank - 1] == 1 ? 0 : 1;
        strides2[rank - 1] = shape2[rank - 1] == 1 ? 0 : 1;
        for (int i = rank - 2; i >= 0; i--) {
            long s1 = shape1[i] == 1 ? 0 : 1;
            long s2 = shape2[i] == 1 ? 0 : 1;
            strides1[i] = s1 * (i + 1 < rank ? (shape1[i + 1] == 1 ? 1 : shape1[i + 1]) * strides1[i + 1] : 1);
            strides2[i] = s2 * (i + 1 < rank ? (shape2[i + 1] == 1 ? 1 : shape2[i + 1]) * strides2[i + 1] : 1);
        }

        // Iterate over output elements
        for (long outIdx = 0; outIdx < numElements; outIdx++) {
            // Convert flat index to multi-dim indices
            long[] indices = new long[rank];
            long remaining = outIdx;
            for (int d = 0; d < rank; d++) {
                indices[d] = remaining / outStrides[d];
                remaining %= outStrides[d];
            }

            // Compute input indices with broadcasting
            long idx1 = 0;
            long idx2 = 0;
            long stride1 = 1;
            long stride2 = 1;
            for (int d = rank - 1; d >= 0; d--) {
                long i1 = shape1[d] == 1 ? 0 : indices[d];
                long i2 = shape2[d] == 1 ? 0 : indices[d];
                idx1 += i1 * stride1;
                idx2 += i2 * stride2;
                stride1 *= shape1[d];
                stride2 *= shape2[d];
            }

            float v1 = input1.data().getAtIndex(ValueLayout.JAVA_FLOAT, idx1);
            float v2 = input2.data().getAtIndex(ValueLayout.JAVA_FLOAT, idx2);
            float result = applyOp(v1, v2, opName);
            resultData.setAtIndex(ValueLayout.JAVA_FLOAT, outIdx, result);
        }

        return Tensor.ofRaw(resultArena, outputShape, type, resultData);
    }

    /**
     * Apply binary operation (float).
     */
    private float applyOp(float v1, float v2, String opName) {
        return switch (opName) {
            case "add" -> v1 + v2;
            case "sub" -> v1 - v2;
            case "mul" -> v1 * v2;
            default -> throw new UnsupportedOperationException("Unknown op: " + opName);
        };
    }

    /**
     * Apply binary operation (double).
     */
    private double applyOpDouble(double v1, double v2, String opName) {
        return switch (opName) {
            case "add" -> v1 + v2;
            case "sub" -> v1 - v2;
            case "mul" -> v1 * v2;
            default -> throw new UnsupportedOperationException("Unknown op: " + opName);
        };
    }

    /**
     * Apply binary operation (int).
     */
    private int applyOpInt(int v1, int v2, String opName) {
        return switch (opName) {
            case "add" -> v1 + v2;
            case "sub" -> v1 - v2;
            case "mul" -> v1 * v2;
            default -> throw new UnsupportedOperationException("Unknown op: " + opName);
        };
    }

    /**
     * Apply binary operation (long).
     */
    private long applyOpLong(long v1, long v2, String opName) {
        return switch (opName) {
            case "add" -> v1 + v2;
            case "sub" -> v1 - v2;
            case "mul" -> v1 * v2;
            default -> throw new UnsupportedOperationException("Unknown op: " + opName);
        };
    }

    /**
     * Execute TOSA MatMul operation
     * For 2D tensors: [M, K] @ [K, N] -> [M, N]
     * For 3D tensors (batched): [B, M, K] @ [B, K, N] -> [B, M, N]
     */
    private <T> Tensor<T> executeMatMul(Tensor<T> input1, Tensor<T> input2) {
        // Validate shapes for matrix multiplication
        long[] shape1 = input1.shape();
        long[] shape2 = input2.shape();

        if (shape1.length != shape2.length) {
            throw new IllegalArgumentException(
                "MatMul requires tensors of the same rank, got shapes: " +
                java.util.Arrays.toString(shape1) + " and " + java.util.Arrays.toString(shape2)
            );
        }

        if (shape1.length == 2) {
            return executeMatMul2D(input1, input2);
        } else if (shape1.length == 3) {
            return executeMatMul3D(input1, input2);
        } else {
            throw new IllegalArgumentException(
                "MatMul requires 2D or 3D tensors, got shapes: " +
                java.util.Arrays.toString(shape1) + " and " + java.util.Arrays.toString(shape2)
            );
        }
    }

    /**
     * Execute 2D MatMul: [M, K] @ [K, N] -> [M, N]
     */
    private <T> Tensor<T> executeMatMul2D(Tensor<T> input1, Tensor<T> input2) {
        long[] shape1 = input1.shape();
        long[] shape2 = input2.shape();

        long M = shape1[0];
        long K1 = shape1[1];
        long K2 = shape2[0];
        long N = shape2[1];

        if (K1 != K2) {
            throw new IllegalArgumentException(
                "MatMul inner dimensions must match: [" + M + ", " + K1 + "] @ [" + K2 + ", " + N + "]"
            );
        }

        if (input1.elementType() != input2.elementType()) {
            throw new IllegalArgumentException(
                "Element type mismatch: " + input1.elementType() + " vs " + input2.elementType()
            );
        }

        Arena resultArena = Arena.ofAuto();
        Tensor.ElementType type = input1.elementType();
        long[] resultShape = new long[]{M, N};
        MemorySegment resultData = resultArena.allocate(type.valueLayout(), M * N);

        for (long m = 0; m < M; m++) {
            for (long n = 0; n < N; n++) {
                long resultIdx = m * N + n;
                switch (type) {
                    case FLOAT32 -> {
                        float sum = 0.0f;
                        for (long k = 0; k < K1; k++) {
                            float v1 = input1.data().getAtIndex(ValueLayout.JAVA_FLOAT, m * K1 + k);
                            float v2 = input2.data().getAtIndex(ValueLayout.JAVA_FLOAT, k * N + n);
                            sum += v1 * v2;
                        }
                        resultData.setAtIndex(ValueLayout.JAVA_FLOAT, resultIdx, sum);
                    }
                    case FLOAT64 -> {
                        double sum = 0.0;
                        for (long k = 0; k < K1; k++) {
                            double v1 = input1.data().getAtIndex(ValueLayout.JAVA_DOUBLE, m * K1 + k);
                            double v2 = input2.data().getAtIndex(ValueLayout.JAVA_DOUBLE, k * N + n);
                            sum += v1 * v2;
                        }
                        resultData.setAtIndex(ValueLayout.JAVA_DOUBLE, resultIdx, sum);
                    }
                    case INT32 -> {
                        int sum = 0;
                        for (long k = 0; k < K1; k++) {
                            int v1 = input1.data().getAtIndex(ValueLayout.JAVA_INT, (int) (m * K1 + k));
                            int v2 = input2.data().getAtIndex(ValueLayout.JAVA_INT, (int) (k * N + n));
                            sum += v1 * v2;
                        }
                        resultData.setAtIndex(ValueLayout.JAVA_INT, resultIdx, sum);
                    }
                    case INT64 -> {
                        long sum = 0;
                        for (long k = 0; k < K1; k++) {
                            long v1 = input1.data().getAtIndex(ValueLayout.JAVA_LONG, m * K1 + k);
                            long v2 = input2.data().getAtIndex(ValueLayout.JAVA_LONG, k * N + n);
                            sum += v1 * v2;
                        }
                        resultData.setAtIndex(ValueLayout.JAVA_LONG, resultIdx, sum);
                    }
                    default -> throw new IllegalStateException("Unknown element type: " + type);
                }
            }
        }

        return Tensor.ofRaw(resultArena, resultShape, type, resultData);
    }

    /**
     * Execute 3D batched MatMul: [B, M, K] @ [B, K, N] -> [B, M, N]
     */
    private <T> Tensor<T> executeMatMul3D(Tensor<T> input1, Tensor<T> input2) {
        long[] shape1 = input1.shape();
        long[] shape2 = input2.shape();

        long B1 = shape1[0];
        long M = shape1[1];
        long K1 = shape1[2];
        long B2 = shape2[0];
        long K2 = shape2[1];
        long N = shape2[2];

        if (B1 != B2) {
            throw new IllegalArgumentException(
                "MatMul batch dimensions must match: " + B1 + " vs " + B2
            );
        }

        if (K1 != K2) {
            throw new IllegalArgumentException(
                "MatMul inner dimensions must match: ["
                    + B1 + ", " + M + ", " + K1 + "] @ [" + B2 + ", " + K2 + ", " + N + "]"
            );
        }

        if (input1.elementType() != input2.elementType()) {
            throw new IllegalArgumentException(
                "Element type mismatch: " + input1.elementType() + " vs " + input2.elementType()
            );
        }

        Arena resultArena = Arena.ofAuto();
        Tensor.ElementType type = input1.elementType();
        long[] resultShape = new long[]{B1, M, N};
        MemorySegment resultData = resultArena.allocate(type.valueLayout(), B1 * M * N);

        for (long b = 0; b < B1; b++) {
            long input1BatchOffset = b * M * K1;
            long input2BatchOffset = b * K2 * N;
            long resultBatchOffset = b * M * N;

            for (long m = 0; m < M; m++) {
                for (long n = 0; n < N; n++) {
                    long resultIdx = resultBatchOffset + m * N + n;
                    switch (type) {
                        case FLOAT32 -> {
                            float sum = 0.0f;
                            for (long k = 0; k < K1; k++) {
                                float v1 = input1.data().getAtIndex(
                                    ValueLayout.JAVA_FLOAT, input1BatchOffset + m * K1 + k);
                                float v2 = input2.data().getAtIndex(
                                    ValueLayout.JAVA_FLOAT, input2BatchOffset + k * N + n);
                                sum += v1 * v2;
                            }
                            resultData.setAtIndex(ValueLayout.JAVA_FLOAT, resultIdx, sum);
                        }
                        case FLOAT64 -> {
                            double sum = 0.0;
                            for (long k = 0; k < K1; k++) {
                                double v1 = input1.data().getAtIndex(
                                    ValueLayout.JAVA_DOUBLE, input1BatchOffset + m * K1 + k);
                                double v2 = input2.data().getAtIndex(
                                    ValueLayout.JAVA_DOUBLE, input2BatchOffset + k * N + n);
                                sum += v1 * v2;
                            }
                            resultData.setAtIndex(ValueLayout.JAVA_DOUBLE, resultIdx, sum);
                        }
                        case INT32 -> {
                            int sum = 0;
                            for (long k = 0; k < K1; k++) {
                                int v1 = input1.data().getAtIndex(
                                    ValueLayout.JAVA_INT, (int) (input1BatchOffset + m * K1 + k));
                                int v2 = input2.data().getAtIndex(
                                    ValueLayout.JAVA_INT, (int) (input2BatchOffset + k * N + n));
                                sum += v1 * v2;
                            }
                            resultData.setAtIndex(ValueLayout.JAVA_INT, resultIdx, sum);
                        }
                        case INT64 -> {
                            long sum = 0;
                            for (long k = 0; k < K1; k++) {
                                long v1 = input1.data().getAtIndex(
                                    ValueLayout.JAVA_LONG, input1BatchOffset + m * K1 + k);
                                long v2 = input2.data().getAtIndex(
                                    ValueLayout.JAVA_LONG, input2BatchOffset + k * N + n);
                                sum += v1 * v2;
                            }
                            resultData.setAtIndex(ValueLayout.JAVA_LONG, resultIdx, sum);
                        }
                        default -> throw new IllegalStateException("Unknown element type: " + type);
                    }
                }
            }
        }

        return Tensor.ofRaw(resultArena, resultShape, type, resultData);
    }

    // TODO: Implement when jextract bindings are available
    // /**
    //  * Get or create MLIR element type
    //  */
    // private MemorySegment getOrCreateElementType(Tensor.ElementType elementType) { ... }
    //
    // /**
    //  * Create MLIR tensor type from shape and element type
    //  */
    // private MemorySegment createTensorType(long[] shape, MemorySegment elementType) { ... }

    /**
     * Dump the MLIR module (for debugging)
     */
    public void dumpModule() {
        System.out.println("MLIR module dumping not yet implemented (requires jextract bindings)");
    }

}
