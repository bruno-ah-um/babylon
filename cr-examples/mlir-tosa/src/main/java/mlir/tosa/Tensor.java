package mlir.tosa;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;

/**
 * A generic tensor abstraction for TOSA operations.
 * Supports multi-dimensional arrays with various element types.
 *
 * @param <T> The element type (Float, Double, Integer, Long)
 */
public final class Tensor<T> {

    /**
     * TOSA/MLIR supported element types
     */
    public enum ElementType {
        FLOAT32(Float.class, ValueLayout.JAVA_FLOAT, 4),
        FLOAT64(Double.class, ValueLayout.JAVA_DOUBLE, 8),
        INT32(Integer.class, ValueLayout.JAVA_INT, 4),
        INT64(Long.class, ValueLayout.JAVA_LONG, 8);

        private final Class<?> javaClass;
        private final ValueLayout valueLayout;
        private final int sizeInBytes;

        ElementType(Class<?> javaClass, ValueLayout valueLayout, int sizeInBytes) {
            this.javaClass = javaClass;
            this.valueLayout = valueLayout;
            this.sizeInBytes = sizeInBytes;
        }

        public Class<?> javaClass() {
            return javaClass;
        }

        public ValueLayout valueLayout() {
            return valueLayout;
        }

        public int sizeInBytes() {
            return sizeInBytes;
        }

        public static ElementType fromClass(Class<?> clazz) {
            for (ElementType type : values()) {
                if (type.javaClass.equals(clazz)) {
                    return type;
                }
            }
            throw new IllegalArgumentException("Unsupported element type: " + clazz);
        }
    }

    private final long[] shape;
    private final ElementType elementType;
    private final MemorySegment data;
    private final Arena arena;

    private Tensor(Arena arena, long[] shape, ElementType elementType, MemorySegment data) {
        this.arena = arena;
        this.shape = Arrays.copyOf(shape, shape.length);
        this.elementType = elementType;
        this.data = data;
    }

    // Factory methods for creating tensors

    /**
     * Create a scalar tensor (0-dimensional)
     */
    @SuppressWarnings("unchecked")
    public static <T> Tensor<T> ofScalar(T value) {
        Arena arena = Arena.ofAuto();
        ElementType type = ElementType.fromClass(value.getClass());
        MemorySegment segment = arena.allocate(type.valueLayout());

        switch (type) {
            case FLOAT32 -> segment.set(ValueLayout.JAVA_FLOAT, 0, (Float) value);
            case FLOAT64 -> segment.set(ValueLayout.JAVA_DOUBLE, 0, (Double) value);
            case INT32 -> segment.set(ValueLayout.JAVA_INT, 0, (Integer) value);
            case INT64 -> segment.set(ValueLayout.JAVA_LONG, 0, (Long) value);
            default -> throw new IllegalStateException("Unknown element type: " + type);
        }

        return new Tensor<>(arena, new long[0], type, segment);
    }

    /**
     * Create a 1D tensor from varargs
     */
    @SuppressWarnings("unchecked")
    public static <T> Tensor<T> ofFlat(T... values) {
        if (values.length == 0) {
            throw new IllegalArgumentException("Cannot create tensor from empty array");
        }

        Arena arena = Arena.ofAuto();
        ElementType type = ElementType.fromClass(values[0].getClass());
        long[] shape = {values.length};

        MemorySegment segment = arena.allocate(type.valueLayout(), values.length);

        for (int i = 0; i < values.length; i++) {
            switch (type) {
                case FLOAT32 -> segment.setAtIndex(ValueLayout.JAVA_FLOAT, i, (Float) values[i]);
                case FLOAT64 -> segment.setAtIndex(ValueLayout.JAVA_DOUBLE, i, (Double) values[i]);
                case INT32 -> segment.setAtIndex(ValueLayout.JAVA_INT, i, (Integer) values[i]);
                case INT64 -> segment.setAtIndex(ValueLayout.JAVA_LONG, i, (Long) values[i]);
                default -> throw new IllegalStateException("Unknown element type: " + type);
            }
        }

        return new Tensor<>(arena, shape, type, segment);
    }

    /**
     * Create a multi-dimensional tensor with given shape and flat data
     */
    @SuppressWarnings("unchecked")
    public static <T> Tensor<T> ofShape(long[] shape, T... values) {
        long totalElements = Arrays.stream(shape).reduce(1, (a, b) -> a * b);
        if (values.length != totalElements) {
            throw new IllegalArgumentException(
                "Data length " + values.length + " doesn't match shape " + Arrays.toString(shape)
            );
        }

        if (values.length == 0) {
            throw new IllegalArgumentException("Cannot create tensor from empty array");
        }

        Arena arena = Arena.ofAuto();
        ElementType type = ElementType.fromClass(values[0].getClass());

        MemorySegment segment = arena.allocate(type.valueLayout(), values.length);

        for (int i = 0; i < values.length; i++) {
            switch (type) {
                case FLOAT32 -> segment.setAtIndex(ValueLayout.JAVA_FLOAT, i, (Float) values[i]);
                case FLOAT64 -> segment.setAtIndex(ValueLayout.JAVA_DOUBLE, i, (Double) values[i]);
                case INT32 -> segment.setAtIndex(ValueLayout.JAVA_INT, i, (Integer) values[i]);
                case INT64 -> segment.setAtIndex(ValueLayout.JAVA_LONG, i, (Long) values[i]);
                default -> throw new IllegalStateException("Unknown element type: " + type);
            }
        }

        return new Tensor<>(arena, shape, type, segment);
    }

    /**
     * Create a tensor from raw memory segment
     */
    public static <T> Tensor<T> ofRaw(Arena arena, long[] shape, ElementType elementType, MemorySegment data) {
        return new Tensor<>(arena, shape, elementType, data);
    }

    /**
     * Create a tensor from raw bytes (little-endian float32).
     * Useful for loading pre-trained weights from binary files.
     *
     * @param shape Tensor shape
     * @param bytes Raw bytes (little-endian float32)
     * @return Float tensor with loaded data
     */
    @SuppressWarnings("unchecked")
    public static Tensor<Float> ofBytes(long[] shape, byte[] bytes) {
        Arena arena = Arena.ofAuto();
        long numElements = 1;
        for (long dim : shape) {
            numElements *= dim;
        }

        if (bytes.length != numElements * 4) {
            throw new IllegalArgumentException(
                "Byte array length " + bytes.length + " doesn't match shape " +
                java.util.Arrays.toString(shape) + " (expected " + (numElements * 4) + " bytes)"
            );
        }

        MemorySegment segment = arena.allocate(ValueLayout.JAVA_FLOAT, numElements);

        // Convert bytes to floats (little-endian)
        for (int i = 0; i < numElements; i++) {
            int offset = i * 4;
            int bits = (bytes[offset] & 0xFF) |
                       ((bytes[offset + 1] & 0xFF) << 8) |
                       ((bytes[offset + 2] & 0xFF) << 16) |
                       ((bytes[offset + 3] & 0xFF) << 24);
            float value = Float.intBitsToFloat(bits);
            segment.setAtIndex(ValueLayout.JAVA_FLOAT, i, value);
        }

        return new Tensor<>(arena, shape, ElementType.FLOAT32, segment);
    }

    /**
     * Create a tensor from float array with given shape.
     * Used for creating input tensors from image data.
     *
     * @param shape Tensor shape
     * @param data Float array data
     * @return Float tensor
     */
    public static Tensor<Float> ofFloats(long[] shape, float[] data) {
        Arena arena = Arena.ofAuto();
        long numElements = 1;
        for (long dim : shape) {
            numElements *= dim;
        }

        if (data.length != numElements) {
            throw new IllegalArgumentException(
                "Data length " + data.length + " doesn't match shape " +
                java.util.Arrays.toString(shape) + " (expected " + numElements + " elements)"
            );
        }

        MemorySegment segment = arena.allocate(ValueLayout.JAVA_FLOAT, numElements);
        for (int i = 0; i < data.length; i++) {
            segment.setAtIndex(ValueLayout.JAVA_FLOAT, i, data[i]);
        }

        return new Tensor<>(arena, shape, ElementType.FLOAT32, segment);
    }

    // Accessors

    public long[] shape() {
        return Arrays.copyOf(shape, shape.length);
    }

    public ElementType elementType() {
        return elementType;
    }

    public MemorySegment data() {
        return data;
    }

    public Arena arena() {
        return arena;
    }

    public int rank() {
        return shape.length;
    }

    public long numElements() {
        if (shape.length == 0) {
            return 1; // scalar
        }
        return Arrays.stream(shape).reduce(1, (a, b) -> a * b);
    }

    // Fluent API - Instance methods for tensor operations

    /**
     * Element-wise addition with another tensor.
     * Fluent API: a.add(b) instead of TosaOperators.Add(a, b)
     *
     * @param other The tensor to add
     * @return Result of this + other
     */
    public Tensor<T> add(Tensor<T> other) {
        return TosaOperators.Add(this, other);
    }

    /**
     * Element-wise multiplication with another tensor.
     * Fluent API: a.mul(b) instead of TosaOperators.Mul(a, b)
     *
     * @param other The tensor to multiply
     * @return Result of this * other
     */
    public Tensor<T> mul(Tensor<T> other) {
        return TosaOperators.Mul(this, other);
    }

    /**
     * Matrix multiplication with another tensor.
     * Fluent API: a.matmul(b) instead of TosaOperators.MatMul(a, b)
     *
     * For 2D tensors: [M, K] @ [K, N] -> [M, N]
     *
     * @param other The tensor to multiply (right matrix)
     * @return Result of this @ other
     */
    public Tensor<T> matmul(Tensor<T> other) {
        return TosaOperators.MatMul(this, other);
    }

    /**
     * Element-wise subtraction.
     * Fluent API: a.sub(b) instead of TosaOperators.Sub(a, b)
     *
     * @param other The tensor to subtract
     * @return Result of this - other
     */
    public Tensor<T> sub(Tensor<T> other) {
        return TosaOperators.Sub(this, other);
    }

    /**
     * ReLU activation (max(0, x)).
     * Fluent API: a.relu() instead of TosaOperators.Relu(a)
     *
     * @return ReLU activated tensor
     */
    public Tensor<T> relu() {
        return TosaOperators.Relu(this);
    }

    /**
     * Clamp values to a range.
     * Fluent API: a.clamp(min, max) instead of TosaOperators.Clamp(a, min, max)
     *
     * @param minVal Minimum value
     * @param maxVal Maximum value
     * @return Clamped tensor
     */
    public Tensor<T> clamp(float minVal, float maxVal) {
        return TosaOperators.Clamp(this, minVal, maxVal);
    }

    /**
     * Reshape tensor to new shape.
     * Fluent API: a.reshape(shape) instead of TosaOperators.Reshape(a, shape)
     *
     * @param newShape New shape (total elements must match)
     * @return Reshaped tensor
     */
    public Tensor<T> reshape(long[] newShape) {
        return TosaOperators.Reshape(this, newShape);
    }

    /**
     * Flatten tensor from given axis.
     * Fluent API: a.flatten(axis) instead of TosaOperators.Flatten(a, axis)
     *
     * @param axis Axis from which to flatten
     * @return Flattened tensor
     */
    public Tensor<T> flatten(int axis) {
        return TosaOperators.Flatten(this, axis);
    }

    /**
     * Element-wise exponential.
     * Fluent API: a.exp() instead of TosaOperators.Exp(a)
     *
     * @return Tensor with exp(x) values
     */
    public Tensor<T> exp() {
        return TosaOperators.Exp(this);
    }

    /**
     * Element-wise reciprocal (1/x).
     * Fluent API: a.reciprocal() instead of TosaOperators.Reciprocal(a)
     *
     * @return Tensor with 1/x values
     */
    public Tensor<T> reciprocal() {
        return TosaOperators.Reciprocal(this);
    }

    /**
     * Element-wise division.
     * Fluent API: a.div(b) instead of TosaOperators.Div(a, b)
     *
     * @param other The divisor tensor
     * @return Result of this / other
     */
    public Tensor<T> div(Tensor<T> other) {
        return TosaOperators.Div(this, other);
    }

    /**
     * Softmax along an axis.
     * Fluent API: a.softmax(axis) instead of TosaOperators.Softmax(a, axis)
     *
     * @param axis Axis along which to compute softmax
     * @return Softmax probabilities
     */
    public Tensor<T> softmax(int axis) {
        return TosaOperators.Softmax(this, axis);
    }

    // Utility methods

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Tensor{shape=").append(Arrays.toString(shape));
        sb.append(", type=").append(elementType);
        sb.append(", data=[");

        long numElements = numElements();
        long displayLimit = Math.min(numElements, 10);

        for (int i = 0; i < displayLimit; i++) {
            if (i > 0) {
                sb.append(", ");
            }
            switch (elementType) {
                case FLOAT32 -> sb.append(data.getAtIndex(ValueLayout.JAVA_FLOAT, i));
                case FLOAT64 -> sb.append(data.getAtIndex(ValueLayout.JAVA_DOUBLE, i));
                case INT32 -> sb.append(data.getAtIndex(ValueLayout.JAVA_INT, i));
                case INT64 -> sb.append(data.getAtIndex(ValueLayout.JAVA_LONG, i));
                default -> throw new IllegalStateException("Unknown element type: " + elementType);
            }
        }

        if (numElements > displayLimit) {
            sb.append(", ...");
        }

        sb.append("]}");
        return sb.toString();
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (!(obj instanceof Tensor<?> other)) {
            return false;
        }

        if (!Arrays.equals(shape, other.shape)) {
            return false;
        }
        if (elementType != other.elementType) {
            return false;
        }

        long numElements = numElements();
        for (int i = 0; i < numElements; i++) {
            boolean equal = switch (elementType) {
                case FLOAT32 -> Float.compare(
                    data.getAtIndex(ValueLayout.JAVA_FLOAT, i),
                    other.data.getAtIndex(ValueLayout.JAVA_FLOAT, i)
                ) == 0;
                case FLOAT64 -> Double.compare(
                    data.getAtIndex(ValueLayout.JAVA_DOUBLE, i),
                    other.data.getAtIndex(ValueLayout.JAVA_DOUBLE, i)
                ) == 0;
                case INT32 -> data.getAtIndex(ValueLayout.JAVA_INT, i) ==
                    other.data.getAtIndex(ValueLayout.JAVA_INT, i);
                case INT64 -> data.getAtIndex(ValueLayout.JAVA_LONG, i) ==
                    other.data.getAtIndex(ValueLayout.JAVA_LONG, i);
            };
            if (!equal) {
                return false;
            }
        }

        return true;
    }

    @Override
    public int hashCode() {
        int result = Arrays.hashCode(shape);
        result = 31 * result + elementType.hashCode();
        return result;
    }
}
