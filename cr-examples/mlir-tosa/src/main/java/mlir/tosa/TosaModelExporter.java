package mlir.tosa;

import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.dialect.core.CoreOp;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.SequencedMap;

/**
 * Exports TOSA models with embedded weights to MLIR.
 *
 * This class extracts Tensor fields from model instances via reflection
 * and generates MLIR with inline weight constants using tosa.const operations.
 *
 * <pre>{@code
 * // Example usage:
 * MNISTModel model = new MNISTModel();
 * String mlir = TosaModelExporter.export(model, "cnn");
 * TosaModelExporter.exportToFile(model, "cnn", "mnist.mlir");
 * }</pre>
 */
public final class TosaModelExporter {

    private TosaModelExporter() {
        // Utility class
    }

    /**
     * Information about an extracted weight tensor.
     */
    public record WeightInfo(
        String name,
        long[] shape,
        MemorySegment data,
        long numElements
    ) {
        /**
         * Create WeightInfo from a Tensor.
         */
        public static WeightInfo fromTensor(String name, Tensor<?> tensor) {
            long[] shape = tensor.shape();
            long numElements = 1;
            for (long dim : shape) {
                numElements *= dim;
            }
            return new WeightInfo(name, shape, tensor.data(), numElements);
        }
    }

    /**
     * Export result containing MLIR code and weight metadata.
     */
    public record ExportResult(
        String mlirCode,
        SequencedMap<String, WeightInfo> weights
    ) {}

    /**
     * Export a model to MLIR with embedded weights.
     *
     * @param modelInstance The model instance with Tensor fields
     * @param methodName The @Reflect annotated method to export
     * @return ExportResult with MLIR code and weight metadata
     */
    public static ExportResult export(Object modelInstance, String methodName) {
        Class<?> modelClass = modelInstance.getClass();

        // 1. Extract weights from Tensor fields
        SequencedMap<String, WeightInfo> weights = extractWeights(modelInstance);

        // 2. Get the @Reflect method
        Method method = findReflectMethod(modelClass, methodName);
        CoreOp.FuncOp funcOp = Op.ofMethod(method).orElseThrow(
            () -> new IllegalArgumentException("Method is not reflectable: " + methodName)
        );

        // 3. Generate MLIR with embedded weights
        String mlir = TosaCodeGenerator.generateTosaWithWeights(
            funcOp, methodName, weights, modelInstance);

        return new ExportResult(mlir, weights);
    }

    /**
     * Export a model to an MLIR file with embedded weights.
     *
     * @param modelInstance The model instance with Tensor fields
     * @param methodName The @Reflect annotated method to export
     * @param filename The output file path
     */
    public static void exportToFile(Object modelInstance, String methodName, String filename) {
        ExportResult result = export(modelInstance, methodName);
        try {
            Files.writeString(Path.of(filename), result.mlirCode());
        } catch (IOException e) {
            throw new RuntimeException("Failed to write MLIR to file: " + filename, e);
        }
    }

    /**
     * Export a model to an MLIR file with embedded weights.
     *
     * @param modelInstance The model instance with Tensor fields
     * @param methodName The @Reflect annotated method to export
     * @param outputPath The output file path
     */
    public static void exportToFile(Object modelInstance, String methodName, Path outputPath) {
        ExportResult result = export(modelInstance, methodName);
        try {
            Files.writeString(outputPath, result.mlirCode());
        } catch (IOException e) {
            throw new RuntimeException("Failed to write MLIR to file: " + outputPath, e);
        }
    }

    /**
     * Extract all Tensor fields from the model instance.
     *
     * @param modelInstance The model instance to extract weights from
     * @return Ordered map of field name to WeightInfo
     */
    public static SequencedMap<String, WeightInfo> extractWeights(Object modelInstance) {
        SequencedMap<String, WeightInfo> weights = new LinkedHashMap<>();
        Class<?> clazz = modelInstance.getClass();

        for (Field field : clazz.getDeclaredFields()) {
            if (Tensor.class.isAssignableFrom(field.getType())) {
                field.setAccessible(true);
                try {
                    Tensor<?> tensor = (Tensor<?>) field.get(modelInstance);
                    if (tensor != null) {
                        weights.put(field.getName(), WeightInfo.fromTensor(field.getName(), tensor));
                    }
                } catch (IllegalAccessException e) {
                    throw new RuntimeException("Cannot access field: " + field.getName(), e);
                }
            }
        }

        return weights;
    }

    /**
     * Find a @Reflect annotated method by name.
     *
     * @param clazz The class to search in
     * @param methodName The method name to find
     * @return The Method object
     * @throws IllegalArgumentException if method not found or not annotated
     */
    private static Method findReflectMethod(Class<?> clazz, String methodName) {
        for (Method m : clazz.getDeclaredMethods()) {
            if (m.getName().equals(methodName) && m.isAnnotationPresent(Reflect.class)) {
                return m;
            }
        }
        throw new IllegalArgumentException("No @Reflect method named '" + methodName + "' found in " + clazz.getName());
    }
}
