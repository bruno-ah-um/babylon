#ifndef MLIR_TOSA_C_API_H
#define MLIR_TOSA_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle types */
typedef struct MLIRContext_ *MLIRContextHandle;
typedef struct MLIRModule_ *MLIRModuleHandle;
typedef struct MLIRFunction_ *MLIRFunctionHandle;
typedef struct MLIRValue_ *MLIRValueHandle;
typedef struct MLIRType_ *MLIRTypeHandle;

/* Result codes */
typedef enum {
    MLIR_SUCCESS = 0,
    MLIR_ERROR_INVALID_PARAMETER = 1,
    MLIR_ERROR_VERIFICATION_FAILED = 2,
    MLIR_ERROR_FILE_IO = 3,
    MLIR_ERROR_INTERNAL = 4
} MLIRResultCode;

/* Context management */
MLIRContextHandle mlir_context_create(void);
void mlir_context_destroy(MLIRContextHandle context);

/* Module management */
MLIRModuleHandle mlir_module_create(MLIRContextHandle context);
void mlir_module_destroy(MLIRModuleHandle module);
MLIRResultCode mlir_module_verify(MLIRModuleHandle module);
MLIRResultCode mlir_module_dump(MLIRModuleHandle module);
MLIRResultCode mlir_module_write_text(MLIRModuleHandle module, const char *filename);
MLIRResultCode mlir_module_write_bytecode(MLIRModuleHandle module, const char *filename);
char *mlir_module_to_string(MLIRModuleHandle module);
void mlir_string_destroy(char *str);

/* Type creation */
MLIRTypeHandle mlir_type_create_f32(MLIRContextHandle context);
MLIRTypeHandle mlir_type_create_f64(MLIRContextHandle context);
MLIRTypeHandle mlir_type_create_i32(MLIRContextHandle context);
MLIRTypeHandle mlir_type_create_i64(MLIRContextHandle context);
MLIRTypeHandle mlir_type_create_tensor_ranked(MLIRContextHandle context, const int64_t *shape,
                                              size_t rank, MLIRTypeHandle elementType);
MLIRTypeHandle mlir_type_create_tensor_dynamic(MLIRContextHandle context, size_t rank,
                                               MLIRTypeHandle elementType);
void mlir_type_destroy(MLIRTypeHandle type);

/* Function creation */
MLIRFunctionHandle mlir_function_create(MLIRModuleHandle module, const char *name,
                                        MLIRTypeHandle *inputTypes, size_t numInputs,
                                        MLIRTypeHandle *outputTypes, size_t numOutputs);
void mlir_function_destroy(MLIRFunctionHandle function);
MLIRValueHandle mlir_function_get_argument(MLIRFunctionHandle function, size_t index);

/* TOSA operations */
MLIRValueHandle mlir_tosa_add(MLIRFunctionHandle function, MLIRValueHandle lhs, MLIRValueHandle rhs,
                              MLIRTypeHandle resultType);

MLIRValueHandle mlir_tosa_sub(MLIRFunctionHandle function, MLIRValueHandle lhs, MLIRValueHandle rhs,
                              MLIRTypeHandle resultType);

MLIRValueHandle mlir_tosa_mul(MLIRFunctionHandle function, MLIRValueHandle lhs, MLIRValueHandle rhs,
                              MLIRTypeHandle resultType);

MLIRValueHandle mlir_tosa_negate(MLIRFunctionHandle function, MLIRValueHandle input,
                                 MLIRTypeHandle resultType);

MLIRValueHandle mlir_tosa_reciprocal(MLIRFunctionHandle function, MLIRValueHandle input,
                                     MLIRTypeHandle resultType);

MLIRValueHandle mlir_tosa_exp(MLIRFunctionHandle function, MLIRValueHandle input,
                              MLIRTypeHandle resultType);

MLIRValueHandle mlir_tosa_log(MLIRFunctionHandle function, MLIRValueHandle input,
                              MLIRTypeHandle resultType);

MLIRValueHandle mlir_tosa_matmul(MLIRFunctionHandle function, MLIRValueHandle lhs,
                                 MLIRValueHandle rhs, MLIRTypeHandle resultType);

MLIRValueHandle mlir_tosa_relu(MLIRFunctionHandle function, MLIRValueHandle input,
                               MLIRTypeHandle resultType);

MLIRValueHandle mlir_tosa_sigmoid(MLIRFunctionHandle function, MLIRValueHandle input,
                                  MLIRTypeHandle resultType);

MLIRValueHandle mlir_tosa_tanh(MLIRFunctionHandle function, MLIRValueHandle input,
                               MLIRTypeHandle resultType);

/**
 * TOSA Conv2D operation.
 * Performs 2D convolution over the input tensor using the weight tensor.
 *
 * @param function Function handle
 * @param input Input tensor [N, H, W, C] (NHWC format)
 * @param weight Weight tensor [OC, KH, KW, IC] (OHWI format)
 * @param bias Bias tensor [OC]
 * @param pad Padding [top, bottom, left, right] (4 values)
 * @param stride Stride [height, width] (2 values)
 * @param dilation Dilation [height, width] (2 values)
 * @param resultType Result tensor type
 * @return Result value handle
 */
MLIRValueHandle mlir_tosa_conv2d(MLIRFunctionHandle function, MLIRValueHandle input,
                                 MLIRValueHandle weight, MLIRValueHandle bias, const int64_t *pad,
                                 const int64_t *stride, const int64_t *dilation,
                                 MLIRTypeHandle resultType);

/**
 * TOSA MaxPool2D operation.
 * Performs max pooling over the input tensor.
 *
 * @param function Function handle
 * @param input Input tensor [N, H, W, C] (NHWC format)
 * @param kernel Kernel size [height, width] (2 values)
 * @param stride Stride [height, width] (2 values)
 * @param pad Padding [top, bottom, left, right] (4 values)
 * @param resultType Result tensor type
 * @return Result value handle
 */
MLIRValueHandle mlir_tosa_max_pool2d(MLIRFunctionHandle function, MLIRValueHandle input,
                                     const int64_t *kernel, const int64_t *stride,
                                     const int64_t *pad, MLIRTypeHandle resultType);

/**
 * TOSA AvgPool2D operation.
 * Performs average pooling over the input tensor.
 *
 * @param function Function handle
 * @param input Input tensor [N, H, W, C] (NHWC format)
 * @param kernel Kernel size [height, width] (2 values)
 * @param stride Stride [height, width] (2 values)
 * @param pad Padding [top, bottom, left, right] (4 values)
 * @param resultType Result tensor type
 * @return Result value handle
 */
MLIRValueHandle mlir_tosa_avg_pool2d(MLIRFunctionHandle function, MLIRValueHandle input,
                                     const int64_t *kernel, const int64_t *stride,
                                     const int64_t *pad, MLIRTypeHandle resultType);

/**
 * TOSA Reshape operation.
 * Reshapes the input tensor to the specified shape.
 *
 * @param function Function handle
 * @param input Input tensor
 * @param newShape Array of new shape dimensions
 * @param numDims Number of dimensions in new shape
 * @param resultType Result tensor type
 * @return Result value handle
 */
MLIRValueHandle mlir_tosa_reshape(MLIRFunctionHandle function, MLIRValueHandle input,
                                  const int64_t *newShape, size_t numDims,
                                  MLIRTypeHandle resultType);

/**
 * TOSA Reduce Sum operation.
 * Reduces the input tensor along the specified axis using sum.
 *
 * @param function Function handle
 * @param input Input tensor
 * @param axis Axis to reduce along
 * @param resultType Result tensor type
 * @return Result value handle
 */
MLIRValueHandle mlir_tosa_reduce_sum(MLIRFunctionHandle function, MLIRValueHandle input,
                                     int64_t axis, MLIRTypeHandle resultType);

/**
 * TOSA Reduce Max operation.
 * Reduces the input tensor along the specified axis using max.
 *
 * @param function Function handle
 * @param input Input tensor
 * @param axis Axis to reduce along
 * @param resultType Result tensor type
 * @return Result value handle
 */
MLIRValueHandle mlir_tosa_reduce_max(MLIRFunctionHandle function, MLIRValueHandle input,
                                     int64_t axis, MLIRTypeHandle resultType);

/**
 * TOSA Const operation.
 * Creates a constant tensor with embedded float32 data.
 *
 * @param function Function handle
 * @param data Raw float32 data array
 * @param numElements Number of elements in data array
 * @param shape Tensor shape dimensions
 * @param numDims Number of dimensions in shape
 * @param resultType Result tensor type (must match shape and f32 element type)
 * @return Value handle for the constant tensor
 */
MLIRValueHandle mlir_tosa_const_f32(MLIRFunctionHandle function, const float *data,
                                    size_t numElements, const int64_t *shape, size_t numDims,
                                    MLIRTypeHandle resultType);

/* Function finalization */
MLIRResultCode mlir_function_add_return(MLIRFunctionHandle function, MLIRValueHandle *values,
                                        size_t numValues);

/* Value operations */
void mlir_value_destroy(MLIRValueHandle value);

/* Error handling */
const char *mlir_get_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* MLIR_TOSA_C_API_H */
