package mlir.tosa;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SegmentAllocator;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;

/**
 * Represents a compiled native function that can be invoked with Tensor arguments.
 *
 * This class wraps a native function compiled from TOSA/MLIR and provides
 * a convenient API for invoking it with Java Tensor objects.
 */
public final class CompiledFunction {

    private static final MethodHandle FREE_HANDLE;

    static {
        try {
            FREE_HANDLE = Linker.nativeLinker().downcallHandle(
                Linker.nativeLinker().defaultLookup().find("free").orElseThrow(),
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));
        } catch (Exception e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    private final MethodHandle handle;
    private final String name;
    private final int numParams;
    private final Path soFile;
    @SuppressWarnings("unused") // Keeps library loaded
    private final Arena libraryArena;

    CompiledFunction(MethodHandle handle, String name, int numParams, Path soFile, Arena libraryArena) {
        this.handle = handle;
        this.name = name;
        this.numParams = numParams;
        this.soFile = soFile;
        this.libraryArena = libraryArena; // Must keep reference to prevent library unloading
    }

    /**
     * Invoke the compiled function with tensor arguments.
     *
     * @param tensors Input tensors (must match the function's parameter count)
     * @return The result tensor
     */
    @SuppressWarnings("unchecked")
    public <T> Tensor<T> invoke(Tensor<?>... tensors) {
        if (tensors.length != numParams) {
            throw new IllegalArgumentException(
                "Expected " + numParams + " tensors, got " + tensors.length);
        }

        // Verify all tensors have the same element type
        Tensor.ElementType elementType = tensors[0].elementType();
        for (Tensor<?> t : tensors) {
            if (t.elementType() != elementType) {
                throw new IllegalArgumentException(
                    "All tensors must have the same element type");
            }
        }

        // Each tensor may have a different rank (e.g., Conv2D: 4D input, 4D weight, 1D bias)
        // Calculate total number of argument values needed
        int totalValues = 0;
        int maxRank = 0;
        for (Tensor<?> tensor : tensors) {
            int rank = tensor.rank();
            // Values per tensor: 3 (base) + rank (sizes) + rank (strides) = 3 + 2*rank
            totalValues += 3 + 2 * rank;
            maxRank = Math.max(maxRank, rank);
        }

        try (Arena arena = Arena.ofConfined()) {
            // For struct return (sret ABI), FFM expects the first argument to be a SegmentAllocator
            // The allocator provides space for the return struct
            // Arguments: [SegmentAllocator, tensor1_args..., tensor2_args...]
            Object[] args = new Object[1 + totalValues];

            // First arg: SegmentAllocator for the return struct
            args[0] = (SegmentAllocator) arena;

            // Add each tensor's memref descriptor components
            int argIndex = 1;
            for (Tensor<?> tensor : tensors) {
                MemorySegment data = tensor.data();
                long[] shape = tensor.shape();
                final int rank = tensor.rank();

                args[argIndex++] = data;  // allocated ptr
                args[argIndex++] = data;  // aligned ptr
                args[argIndex++] = 0L;    // offset

                // Add sizes for each dimension
                for (int d = 0; d < rank; d++) {
                    args[argIndex++] = shape[d];
                }

                // Add strides for each dimension (row-major order)
                // For a contiguous row-major tensor, stride[d] = product of shape[d+1:]
                for (int d = 0; d < rank; d++) {
                    long stride = 1;
                    for (int k = d + 1; k < rank; k++) {
                        stride *= shape[k];
                    }
                    args[argIndex++] = stride;
                }
            }

            // Use maxRank for the output tensor
            final int rank = maxRank;

            // Invoke the native function
            // Returns a MemorySegment representing the return struct: { ptr, ptr, i64, [rank x i64], [rank x i64] }
            MemorySegment resultStruct = (MemorySegment) handle.invokeWithArguments(args);

            // Extract fields from the result struct
            // Layout: allocated_ptr, aligned_ptr, offset, sizes[rank], strides[rank]
            long offset = 0;
            MemorySegment allocatedPtr = resultStruct.get(ValueLayout.ADDRESS, offset);
            offset += ValueLayout.ADDRESS.byteSize();
            final MemorySegment alignedPtr = resultStruct.get(ValueLayout.ADDRESS, offset);
            offset += ValueLayout.ADDRESS.byteSize();
            offset += ValueLayout.JAVA_LONG.byteSize();

            // For dynamic tensors, we use the output size from the struct
            // (which is the max of input sizes for broadcasting)
            long[] resultShape = new long[rank];
            for (int d = 0; d < rank; d++) {
                resultShape[d] = resultStruct.get(ValueLayout.JAVA_LONG, offset + d * ValueLayout.JAVA_LONG.byteSize());
            }

            long resultElements = 1;
            for (long dim : resultShape) {
                resultElements *= dim;
            }

            // Copy result data to a new tensor (the native function allocated memory via malloc)
            Arena resultArena = Arena.ofAuto();
            MemorySegment resultCopy = resultArena.allocate(
                elementType.valueLayout(), resultElements);

            // The aligned pointer points to the actual data
            resultCopy.copyFrom(alignedPtr.reinterpret(resultElements * elementType.sizeInBytes()));

            // Free the malloc'd memory returned by the native function
            FREE_HANDLE.invoke(allocatedPtr);

            return (Tensor<T>) Tensor.ofRaw(resultArena, resultShape, elementType, resultCopy);

        } catch (Throwable e) {
            throw new RuntimeException("Native invocation failed", e);
        }
    }

    /**
     * Get the function name.
     */
    public String name() {
        return name;
    }

    /**
     * Get the number of parameters this function expects.
     */
    public int numParams() {
        return numParams;
    }

    /**
     * Get the path to the compiled shared library.
     */
    public Path libraryPath() {
        return soFile;
    }

    @Override
    public String toString() {
        return "CompiledFunction{name='" + name + "', params=" + numParams + ", library=" + soFile + "}";
    }
}
