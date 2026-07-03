#ifndef JETSTREAM_DOMAINS_ML_ONNX_INFERENCE_ONNX_TYPES_HH
#define JETSTREAM_DOMAINS_ML_ONNX_INFERENCE_ONNX_TYPES_HH

#include <optional>
#include <string_view>

#include <onnxruntime_cxx_api.h>

#include "jetstream/memory/types.hh"

namespace Jetstream {

inline std::optional<DataType> OnnxTensorElementTypeToDataType(const ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:    return DataType::F32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:   return DataType::F64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:    return DataType::I16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:    return DataType::I32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:    return DataType::I64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:     return DataType::I8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:   return DataType::U16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:   return DataType::U32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:   return DataType::U64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:    return DataType::U8;
        default:                                     return std::nullopt;
    }
}

inline ONNXTensorElementDataType DataTypeToOnnxTensorElementType(const DataType type) {
    switch (type) {
        case DataType::F32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        case DataType::F64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
        case DataType::I16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
        case DataType::I32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
        case DataType::I64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        case DataType::I8:  return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
        case DataType::U16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
        case DataType::U32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
        case DataType::U64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
        case DataType::U8:  return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
        default:            return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
}

inline bool IsSupportedOnnxInferenceTensorType(const DataType type) {
    switch (type) {
        case DataType::F32:
        case DataType::F64:
        case DataType::I16:
        case DataType::I32:
        case DataType::I64:
        case DataType::I8:
        case DataType::U16:
        case DataType::U32:
        case DataType::U64:
        case DataType::U8:
            return true;
        default:
            return false;
    }
}

inline std::string_view OnnxTensorElementTypeName(const ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return "F32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:  return "F64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:   return "I16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   return "I32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   return "I64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:    return "I8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:  return "U16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:  return "U32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:  return "U64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   return "U8";
        default:                                    return "UNSUPPORTED";
    }
}

}  // namespace Jetstream

#endif  // JETSTREAM_DOMAINS_ML_ONNX_INFERENCE_ONNX_TYPES_HH
