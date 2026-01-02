#ifndef MLIR_TOSA_C_API_H
#define MLIR_TOSA_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle types */
typedef struct MLIRContext_* MLIRContextHandle;
typedef struct MLIRModule_* MLIRModuleHandle;
typedef struct MLIRFunction_* MLIRFunctionHandle;
typedef struct MLIRValue_* MLIRValueHandle;
typedef struct MLIRType_* MLIRTypeHandle;

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
MLIRResultCode mlir_module_write_text(MLIRModuleHandle module, const char* filename);
MLIRResultCode mlir_module_write_bytecode(MLIRModuleHandle module, const char* filename);
char* mlir_module_to_string(MLIRModuleHandle module);
void mlir_string_destroy(char* str);

/* Type creation */
MLIRTypeHandle mlir_type_create_f32(MLIRContextHandle context);
MLIRTypeHandle mlir_type_create_f64(MLIRContextHandle context);
MLIRTypeHandle mlir_type_create_i32(MLIRContextHandle context);
MLIRTypeHandle mlir_type_create_i64(MLIRContextHandle context);
MLIRTypeHandle mlir_type_create_tensor_ranked(MLIRContextHandle context,
                                               const int64_t* shape,
                                               size_t rank,
                                               MLIRTypeHandle elementType);
MLIRTypeHandle mlir_type_create_tensor_dynamic(MLIRContextHandle context,
                                                size_t rank,
                                                MLIRTypeHandle elementType);
void mlir_type_destroy(MLIRTypeHandle type);

/* Function creation */
MLIRFunctionHandle mlir_function_create(MLIRModuleHandle module,
                                        const char* name,
                                        MLIRTypeHandle* inputTypes,
                                        size_t numInputs,
                                        MLIRTypeHandle* outputTypes,
                                        size_t numOutputs);
void mlir_function_destroy(MLIRFunctionHandle function);
MLIRValueHandle mlir_function_get_argument(MLIRFunctionHandle function, size_t index);

/* TOSA operations */
MLIRValueHandle mlir_tosa_add(MLIRFunctionHandle function,
                              MLIRValueHandle lhs,
                              MLIRValueHandle rhs,
                              MLIRTypeHandle resultType);

MLIRValueHandle mlir_tosa_sub(MLIRFunctionHandle function,
                              MLIRValueHandle lhs,
                              MLIRValueHandle rhs,
                              MLIRTypeHandle resultType);

MLIRValueHandle mlir_tosa_mul(MLIRFunctionHandle function,
                              MLIRValueHandle lhs,
                              MLIRValueHandle rhs,
                              MLIRTypeHandle resultType);

MLIRValueHandle mlir_tosa_negate(MLIRFunctionHandle function,
                                 MLIRValueHandle input,
                                 MLIRTypeHandle resultType);

MLIRValueHandle mlir_tosa_reciprocal(MLIRFunctionHandle function,
                                     MLIRValueHandle input,
                                     MLIRTypeHandle resultType);

MLIRValueHandle mlir_tosa_exp(MLIRFunctionHandle function,
                              MLIRValueHandle input,
                              MLIRTypeHandle resultType);

MLIRValueHandle mlir_tosa_log(MLIRFunctionHandle function,
                              MLIRValueHandle input,
                              MLIRTypeHandle resultType);

MLIRValueHandle mlir_tosa_matmul(MLIRFunctionHandle function,
                                 MLIRValueHandle lhs,
                                 MLIRValueHandle rhs,
                                 MLIRTypeHandle resultType);

MLIRValueHandle mlir_tosa_relu(MLIRFunctionHandle function,
                               MLIRValueHandle input,
                               MLIRTypeHandle resultType);

MLIRValueHandle mlir_tosa_sigmoid(MLIRFunctionHandle function,
                                  MLIRValueHandle input,
                                  MLIRTypeHandle resultType);

MLIRValueHandle mlir_tosa_tanh(MLIRFunctionHandle function,
                               MLIRValueHandle input,
                               MLIRTypeHandle resultType);

/* Function finalization */
MLIRResultCode mlir_function_add_return(MLIRFunctionHandle function,
                                        MLIRValueHandle* values,
                                        size_t numValues);

/* Value operations */
void mlir_value_destroy(MLIRValueHandle value);

/* Error handling */
const char* mlir_get_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* MLIR_TOSA_C_API_H */
