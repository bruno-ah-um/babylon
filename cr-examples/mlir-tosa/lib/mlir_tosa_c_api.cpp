#include "mlir_tosa_c_api.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/ArrayRef.h"
#include <cfloat>
#include <string>
#include <vector>
#include <memory>

using namespace mlir;

/* Internal structures */
struct MLIRContext_ {
    std::unique_ptr<MLIRContext> context;

    MLIRContext_() : context(std::make_unique<MLIRContext>()) {
        context->getOrLoadDialect<func::FuncDialect>();
        context->getOrLoadDialect<tosa::TosaDialect>();
    }
};

struct MLIRModule_ {
    MLIRContext* context;
    ModuleOp module;

    MLIRModule_(MLIRContext* ctx) : context(ctx) {
        auto loc = UnknownLoc::get(ctx);
        module = ModuleOp::create(loc);
    }
};

struct MLIRFunction_ {
    MLIRContext* context;
    func::FuncOp funcOp;
    OpBuilder builder;

    MLIRFunction_(MLIRContext* ctx, func::FuncOp op)
        : context(ctx), funcOp(op), builder(ctx) {
        Block* entryBlock = op.addEntryBlock();
        builder.setInsertionPointToStart(entryBlock);
    }
};

struct MLIRValue_ {
    Value value;

    explicit MLIRValue_(Value v) : value(v) {}
};

struct MLIRType_ {
    Type type;

    explicit MLIRType_(Type t) : type(t) {}
};

/* Thread-local error storage */
thread_local std::string lastError;

static void setError(const std::string& error) {
    lastError = error;
}

/* Context management */
MLIRContextHandle mlir_context_create(void) {
    try {
        return new struct MLIRContext_();
    } catch (const std::exception& e) {
        setError(std::string("Failed to create context: ") + e.what());
        return nullptr;
    }
}

void mlir_context_destroy(MLIRContextHandle context) {
    delete context;
}

/* Module management */
MLIRModuleHandle mlir_module_create(MLIRContextHandle context) {
    if (!context) {
        setError("Invalid context handle");
        return nullptr;
    }

    try {
        return new struct MLIRModule_(context->context.get());
    } catch (const std::exception& e) {
        setError(std::string("Failed to create module: ") + e.what());
        return nullptr;
    }
}

void mlir_module_destroy(MLIRModuleHandle module) {
    if (module) {
        module->module.erase();
    }
    delete module;
}

MLIRResultCode mlir_module_verify(MLIRModuleHandle module) {
    if (!module) {
        setError("Invalid module handle");
        return MLIR_ERROR_INVALID_PARAMETER;
    }

    if (failed(verify(module->module))) {
        setError("Module verification failed");
        return MLIR_ERROR_VERIFICATION_FAILED;
    }

    return MLIR_SUCCESS;
}

MLIRResultCode mlir_module_dump(MLIRModuleHandle module) {
    if (!module) {
        setError("Invalid module handle");
        return MLIR_ERROR_INVALID_PARAMETER;
    }

    module->module.dump();
    return MLIR_SUCCESS;
}

MLIRResultCode mlir_module_write_text(MLIRModuleHandle module, const char* filename) {
    if (!module || !filename) {
        setError("Invalid parameter");
        return MLIR_ERROR_INVALID_PARAMETER;
    }

    std::error_code ec;
    auto output = std::make_unique<llvm::ToolOutputFile>(filename, ec, llvm::sys::fs::OF_None);

    if (ec) {
        setError(std::string("Failed to create file: ") + ec.message());
        return MLIR_ERROR_FILE_IO;
    }

    module->module.print(output->os());
    output->keep();

    return MLIR_SUCCESS;
}

MLIRResultCode mlir_module_write_bytecode(MLIRModuleHandle module, const char* filename) {
    if (!module || !filename) {
        setError("Invalid parameter");
        return MLIR_ERROR_INVALID_PARAMETER;
    }

    std::error_code ec;
    auto output = std::make_unique<llvm::ToolOutputFile>(filename, ec, llvm::sys::fs::OF_None);

    if (ec) {
        setError(std::string("Failed to create file: ") + ec.message());
        return MLIR_ERROR_FILE_IO;
    }

    if (failed(writeBytecodeToFile(module->module, output->os()))) {
        setError("Failed to write bytecode");
        return MLIR_ERROR_INTERNAL;
    }

    output->keep();
    return MLIR_SUCCESS;
}

char* mlir_module_to_string(MLIRModuleHandle module) {
    if (!module) {
        setError("Invalid module handle");
        return nullptr;
    }

    std::string str;
    llvm::raw_string_ostream os(str);
    module->module.print(os);
    os.flush();

    char* result = static_cast<char*>(malloc(str.length() + 1));
    if (result) {
        strcpy(result, str.c_str());
    }
    return result;
}

void mlir_string_destroy(char* str) {
    free(str);
}

/* Type creation */
MLIRTypeHandle mlir_type_create_f32(MLIRContextHandle context) {
    if (!context) {
        setError("Invalid context handle");
        return nullptr;
    }

    OpBuilder builder(context->context.get());
    return new struct MLIRType_(builder.getF32Type());
}

MLIRTypeHandle mlir_type_create_f64(MLIRContextHandle context) {
    if (!context) {
        setError("Invalid context handle");
        return nullptr;
    }

    OpBuilder builder(context->context.get());
    return new struct MLIRType_(builder.getF64Type());
}

MLIRTypeHandle mlir_type_create_i32(MLIRContextHandle context) {
    if (!context) {
        setError("Invalid context handle");
        return nullptr;
    }

    OpBuilder builder(context->context.get());
    return new struct MLIRType_(builder.getI32Type());
}

MLIRTypeHandle mlir_type_create_i64(MLIRContextHandle context) {
    if (!context) {
        setError("Invalid context handle");
        return nullptr;
    }

    OpBuilder builder(context->context.get());
    return new struct MLIRType_(builder.getI64Type());
}

MLIRTypeHandle mlir_type_create_tensor_ranked(MLIRContextHandle context,
                                               const int64_t* shape,
                                               size_t rank,
                                               MLIRTypeHandle elementType) {
    if (!context || !elementType || (rank > 0 && !shape)) {
        setError("Invalid parameter");
        return nullptr;
    }

    std::vector<int64_t> shapeVec(shape, shape + rank);
    auto tensorType = RankedTensorType::get(shapeVec, elementType->type);
    return new struct MLIRType_(tensorType);
}

MLIRTypeHandle mlir_type_create_tensor_dynamic(MLIRContextHandle context,
                                                size_t rank,
                                                MLIRTypeHandle elementType) {
    if (!context || !elementType) {
        setError("Invalid parameter");
        return nullptr;
    }

    std::vector<int64_t> shape(rank, ShapedType::kDynamic);
    auto tensorType = RankedTensorType::get(shape, elementType->type);
    return new struct MLIRType_(tensorType);
}

void mlir_type_destroy(MLIRTypeHandle type) {
    delete type;
}

/* Function creation */
MLIRFunctionHandle mlir_function_create(MLIRModuleHandle module,
                                        const char* name,
                                        MLIRTypeHandle* inputTypes,
                                        size_t numInputs,
                                        MLIRTypeHandle* outputTypes,
                                        size_t numOutputs) {
    if (!module || !name) {
        setError("Invalid parameter");
        return nullptr;
    }

    OpBuilder builder(module->context);
    builder.setInsertionPointToEnd(module->module.getBody());

    std::vector<Type> inputs;
    for (size_t i = 0; i < numInputs; i++) {
        if (!inputTypes[i]) {
            setError("Invalid input type");
            return nullptr;
        }
        inputs.push_back(inputTypes[i]->type);
    }

    std::vector<Type> outputs;
    for (size_t i = 0; i < numOutputs; i++) {
        if (!outputTypes[i]) {
            setError("Invalid output type");
            return nullptr;
        }
        outputs.push_back(outputTypes[i]->type);
    }

    auto funcType = builder.getFunctionType(inputs, outputs);
    auto loc = UnknownLoc::get(module->context);
    auto funcOp = builder.create<func::FuncOp>(loc, name, funcType);

    return new struct MLIRFunction_(module->context, funcOp);
}

void mlir_function_destroy(MLIRFunctionHandle function) {
    delete function;
}

MLIRValueHandle mlir_function_get_argument(MLIRFunctionHandle function, size_t index) {
    if (!function) {
        setError("Invalid function handle");
        return nullptr;
    }

    Block* entryBlock = &function->funcOp.getBody().front();
    if (index >= entryBlock->getNumArguments()) {
        setError("Argument index out of range");
        return nullptr;
    }

    return new struct MLIRValue_(entryBlock->getArgument(index));
}

/* TOSA operations */
MLIRValueHandle mlir_tosa_add(MLIRFunctionHandle function,
                              MLIRValueHandle lhs,
                              MLIRValueHandle rhs,
                              MLIRTypeHandle resultType) {
    if (!function || !lhs || !rhs || !resultType) {
        setError("Invalid parameter");
        return nullptr;
    }

    auto loc = UnknownLoc::get(function->context);
    auto addOp = function->builder.create<tosa::AddOp>(loc, resultType->type, lhs->value, rhs->value);
    return new struct MLIRValue_(addOp.getResult());
}

MLIRValueHandle mlir_tosa_sub(MLIRFunctionHandle function,
                              MLIRValueHandle lhs,
                              MLIRValueHandle rhs,
                              MLIRTypeHandle resultType) {
    if (!function || !lhs || !rhs || !resultType) {
        setError("Invalid parameter");
        return nullptr;
    }

    auto loc = UnknownLoc::get(function->context);
    auto subOp = function->builder.create<tosa::SubOp>(loc, resultType->type, lhs->value, rhs->value);
    return new struct MLIRValue_(subOp.getResult());
}

MLIRValueHandle mlir_tosa_mul(MLIRFunctionHandle function,
                              MLIRValueHandle lhs,
                              MLIRValueHandle rhs,
                              MLIRTypeHandle resultType) {
    if (!function || !lhs || !rhs || !resultType) {
        setError("Invalid parameter");
        return nullptr;
    }

    auto loc = UnknownLoc::get(function->context);
    auto mulOp = function->builder.create<tosa::MulOp>(loc, resultType->type, lhs->value, rhs->value, 0);
    return new struct MLIRValue_(mulOp.getResult());
}

MLIRValueHandle mlir_tosa_negate(MLIRFunctionHandle function,
                                 MLIRValueHandle input,
                                 MLIRTypeHandle resultType) {
    if (!function || !input || !resultType) {
        setError("Invalid parameter");
        return nullptr;
    }

    auto loc = UnknownLoc::get(function->context);
    auto negateOp = function->builder.create<tosa::NegateOp>(loc, resultType->type, input->value);
    return new struct MLIRValue_(negateOp.getResult());
}

MLIRValueHandle mlir_tosa_reciprocal(MLIRFunctionHandle function,
                                     MLIRValueHandle input,
                                     MLIRTypeHandle resultType) {
    if (!function || !input || !resultType) {
        setError("Invalid parameter");
        return nullptr;
    }

    auto loc = UnknownLoc::get(function->context);
    auto reciprocalOp = function->builder.create<tosa::ReciprocalOp>(loc, resultType->type, input->value);
    return new struct MLIRValue_(reciprocalOp.getResult());
}

MLIRValueHandle mlir_tosa_exp(MLIRFunctionHandle function,
                              MLIRValueHandle input,
                              MLIRTypeHandle resultType) {
    if (!function || !input || !resultType) {
        setError("Invalid parameter");
        return nullptr;
    }

    auto loc = UnknownLoc::get(function->context);
    auto expOp = function->builder.create<tosa::ExpOp>(loc, resultType->type, input->value);
    return new struct MLIRValue_(expOp.getResult());
}

MLIRValueHandle mlir_tosa_log(MLIRFunctionHandle function,
                              MLIRValueHandle input,
                              MLIRTypeHandle resultType) {
    if (!function || !input || !resultType) {
        setError("Invalid parameter");
        return nullptr;
    }

    auto loc = UnknownLoc::get(function->context);
    auto logOp = function->builder.create<tosa::LogOp>(loc, resultType->type, input->value);
    return new struct MLIRValue_(logOp.getResult());
}

MLIRValueHandle mlir_tosa_matmul(MLIRFunctionHandle function,
                                 MLIRValueHandle lhs,
                                 MLIRValueHandle rhs,
                                 MLIRTypeHandle resultType) {
    if (!function || !lhs || !rhs || !resultType) {
        setError("Invalid parameter");
        return nullptr;
    }

    auto loc = UnknownLoc::get(function->context);
    auto matmulOp = function->builder.create<tosa::MatMulOp>(loc, resultType->type, lhs->value, rhs->value);
    return new struct MLIRValue_(matmulOp.getResult());
}

MLIRValueHandle mlir_tosa_relu(MLIRFunctionHandle function,
                               MLIRValueHandle input,
                               MLIRTypeHandle resultType) {
    if (!function || !input || !resultType) {
        setError("Invalid parameter");
        return nullptr;
    }

    auto loc = UnknownLoc::get(function->context);
    auto minFp = function->builder.getFloatAttr(function->builder.getF32Type(), 0.0f);
    auto maxFp = function->builder.getFloatAttr(function->builder.getF32Type(), FLT_MAX);
    auto clampOp = function->builder.create<tosa::ClampOp>(loc, resultType->type, input->value,
                                                           0, INT64_MAX, minFp, maxFp);
    return new struct MLIRValue_(clampOp.getResult());
}

MLIRValueHandle mlir_tosa_sigmoid(MLIRFunctionHandle function,
                                  MLIRValueHandle input,
                                  MLIRTypeHandle resultType) {
    if (!function || !input || !resultType) {
        setError("Invalid parameter");
        return nullptr;
    }

    auto loc = UnknownLoc::get(function->context);
    auto sigmoidOp = function->builder.create<tosa::SigmoidOp>(loc, resultType->type, input->value);
    return new struct MLIRValue_(sigmoidOp.getResult());
}

MLIRValueHandle mlir_tosa_tanh(MLIRFunctionHandle function,
                               MLIRValueHandle input,
                               MLIRTypeHandle resultType) {
    if (!function || !input || !resultType) {
        setError("Invalid parameter");
        return nullptr;
    }

    auto loc = UnknownLoc::get(function->context);
    auto tanhOp = function->builder.create<tosa::TanhOp>(loc, resultType->type, input->value);
    return new struct MLIRValue_(tanhOp.getResult());
}

MLIRValueHandle mlir_tosa_conv2d(MLIRFunctionHandle function,
                                  MLIRValueHandle input,
                                  MLIRValueHandle weight,
                                  MLIRValueHandle bias,
                                  const int64_t* pad,
                                  const int64_t* stride,
                                  const int64_t* dilation,
                                  MLIRTypeHandle resultType) {
    if (!function || !input || !weight || !bias || !pad || !stride || !dilation || !resultType) {
        setError("Invalid parameter");
        return nullptr;
    }

    auto loc = UnknownLoc::get(function->context);

    // Create dense arrays for pad, stride, dilation
    auto padAttr = function->builder.getDenseI64ArrayAttr({pad[0], pad[1], pad[2], pad[3]});
    auto strideAttr = function->builder.getDenseI64ArrayAttr({stride[0], stride[1]});
    auto dilationAttr = function->builder.getDenseI64ArrayAttr({dilation[0], dilation[1]});

    // Conv2D requires bias to be 1D tensor [OC]. If bias is multi-dimensional,
    // we need to reshape it to 1D first.
    mlir::Value biasValue = bias->value;
    auto biasType = mlir::dyn_cast<mlir::RankedTensorType>(biasValue.getType());
    if (biasType && biasType.getRank() > 1) {
        // Create 1D dynamic tensor type for bias
        auto elementType = biasType.getElementType();
        auto bias1DType = mlir::RankedTensorType::get({mlir::ShapedType::kDynamic}, elementType);

        // Create new_shape attribute for reshape: [-1] to flatten to 1D
        auto newShapeAttr = function->builder.getDenseI64ArrayAttr({-1});

        auto reshapeOp = function->builder.create<tosa::ReshapeOp>(
            loc, bias1DType, biasValue, newShapeAttr);
        biasValue = reshapeOp.getResult();
    }

    auto conv2dOp = function->builder.create<tosa::Conv2DOp>(
        loc, resultType->type,
        input->value, weight->value, biasValue,
        padAttr, strideAttr, dilationAttr);

    return new struct MLIRValue_(conv2dOp.getResult());
}

MLIRValueHandle mlir_tosa_max_pool2d(MLIRFunctionHandle function,
                                      MLIRValueHandle input,
                                      const int64_t* kernel,
                                      const int64_t* stride,
                                      const int64_t* pad,
                                      MLIRTypeHandle resultType) {
    if (!function || !input || !kernel || !stride || !pad || !resultType) {
        setError("Invalid parameter");
        return nullptr;
    }

    auto loc = UnknownLoc::get(function->context);

    // Create dense arrays for kernel, stride, pad
    auto kernelAttr = function->builder.getDenseI64ArrayAttr({kernel[0], kernel[1]});
    auto strideAttr = function->builder.getDenseI64ArrayAttr({stride[0], stride[1]});
    auto padAttr = function->builder.getDenseI64ArrayAttr({pad[0], pad[1], pad[2], pad[3]});

    auto maxPool2dOp = function->builder.create<tosa::MaxPool2dOp>(
        loc, resultType->type,
        input->value,
        kernelAttr, strideAttr, padAttr);

    return new struct MLIRValue_(maxPool2dOp.getResult());
}

MLIRValueHandle mlir_tosa_avg_pool2d(MLIRFunctionHandle function,
                                      MLIRValueHandle input,
                                      const int64_t* kernel,
                                      const int64_t* stride,
                                      const int64_t* pad,
                                      MLIRTypeHandle resultType) {
    if (!function || !input || !kernel || !stride || !pad || !resultType) {
        setError("Invalid parameter");
        return nullptr;
    }

    auto loc = UnknownLoc::get(function->context);

    // Create dense arrays for kernel, stride, pad
    auto kernelAttr = function->builder.getDenseI64ArrayAttr({kernel[0], kernel[1]});
    auto strideAttr = function->builder.getDenseI64ArrayAttr({stride[0], stride[1]});
    auto padAttr = function->builder.getDenseI64ArrayAttr({pad[0], pad[1], pad[2], pad[3]});

    // AvgPool2d requires acc_type attribute - use TypeAttr::get
    auto accType = TypeAttr::get(function->builder.getF32Type());

    auto avgPool2dOp = function->builder.create<tosa::AvgPool2dOp>(
        loc, resultType->type,
        input->value,
        kernelAttr, strideAttr, padAttr,
        accType);

    return new struct MLIRValue_(avgPool2dOp.getResult());
}

MLIRValueHandle mlir_tosa_reshape(MLIRFunctionHandle function,
                                   MLIRValueHandle input,
                                   const int64_t* newShape,
                                   size_t numDims,
                                   MLIRTypeHandle resultType) {
    if (!function || !input || !newShape || !resultType) {
        setError("Invalid parameter");
        return nullptr;
    }

    auto loc = UnknownLoc::get(function->context);

    // Create new_shape attribute
    std::vector<int64_t> shapeVec(newShape, newShape + numDims);
    auto newShapeAttr = function->builder.getDenseI64ArrayAttr(shapeVec);

    auto reshapeOp = function->builder.create<tosa::ReshapeOp>(
        loc, resultType->type,
        input->value,
        newShapeAttr);

    return new struct MLIRValue_(reshapeOp.getResult());
}

MLIRValueHandle mlir_tosa_reduce_sum(MLIRFunctionHandle function,
                                      MLIRValueHandle input,
                                      int64_t axis,
                                      MLIRTypeHandle resultType) {
    if (!function || !input || !resultType) {
        setError("Invalid parameter");
        return nullptr;
    }

    auto loc = UnknownLoc::get(function->context);

    auto axisAttr = function->builder.getI32IntegerAttr(axis);

    auto reduceSumOp = function->builder.create<tosa::ReduceSumOp>(
        loc, resultType->type,
        input->value,
        axisAttr);

    return new struct MLIRValue_(reduceSumOp.getResult());
}

MLIRValueHandle mlir_tosa_reduce_max(MLIRFunctionHandle function,
                                      MLIRValueHandle input,
                                      int64_t axis,
                                      MLIRTypeHandle resultType) {
    if (!function || !input || !resultType) {
        setError("Invalid parameter");
        return nullptr;
    }

    auto loc = UnknownLoc::get(function->context);

    auto axisAttr = function->builder.getI32IntegerAttr(axis);

    auto reduceMaxOp = function->builder.create<tosa::ReduceMaxOp>(
        loc, resultType->type,
        input->value,
        axisAttr);

    return new struct MLIRValue_(reduceMaxOp.getResult());
}

MLIRValueHandle mlir_tosa_const_f32(MLIRFunctionHandle function,
                                     const float* data,
                                     size_t numElements,
                                     const int64_t* shape,
                                     size_t numDims,
                                     MLIRTypeHandle resultType) {
    if (!function || !data || !resultType) {
        setError("Invalid parameter");
        return nullptr;
    }

    if (numDims > 0 && !shape) {
        setError("Shape is required for non-scalar tensors");
        return nullptr;
    }

    // Validate that numElements matches the product of shape dimensions
    size_t expectedElements = 1;
    for (size_t i = 0; i < numDims; i++) {
        expectedElements *= shape[i];
    }
    if (expectedElements != numElements) {
        setError("Number of elements does not match shape dimensions");
        return nullptr;
    }

    auto loc = UnknownLoc::get(function->context);

    // Get the tensor type from the provided result type
    auto tensorType = mlir::dyn_cast<RankedTensorType>(resultType->type);
    if (!tensorType) {
        setError("Result type must be a ranked tensor type");
        return nullptr;
    }

    // Create a vector from the input data
    std::vector<float> dataVec(data, data + numElements);

    // Create DenseElementsAttr from the float data
    auto denseAttr = DenseElementsAttr::get(tensorType, llvm::ArrayRef(dataVec));

    // Create tosa.const operation
    auto constOp = function->builder.create<tosa::ConstOp>(loc, tensorType, denseAttr);

    return new struct MLIRValue_(constOp.getResult());
}

/* Function finalization */
MLIRResultCode mlir_function_add_return(MLIRFunctionHandle function,
                                        MLIRValueHandle* values,
                                        size_t numValues) {
    if (!function) {
        setError("Invalid function handle");
        return MLIR_ERROR_INVALID_PARAMETER;
    }

    auto loc = UnknownLoc::get(function->context);

    if (numValues == 0) {
        function->builder.create<func::ReturnOp>(loc);
    } else {
        std::vector<Value> returnValues;
        for (size_t i = 0; i < numValues; i++) {
            if (!values[i]) {
                setError("Invalid return value");
                return MLIR_ERROR_INVALID_PARAMETER;
            }
            returnValues.push_back(values[i]->value);
        }
        function->builder.create<func::ReturnOp>(loc, returnValues);
    }

    return MLIR_SUCCESS;
}

/* Value operations */
void mlir_value_destroy(MLIRValueHandle value) {
    delete value;
}

/* Error handling */
const char* mlir_get_last_error(void) {
    return lastError.c_str();
}
