#include "jetstream/memory/types.hh"

#include <complex>

namespace Jetstream {

const char* GetDeviceName(const DeviceType& device) {
    static const std::unordered_map<DeviceType, const char*> deviceNames = {
        {DeviceType::None,   "none"},
        {DeviceType::CPU,    "cpu"},
        {DeviceType::CUDA,   "cuda"},
        {DeviceType::Metal,  "metal"},
        {DeviceType::Vulkan, "vulkan"},
        {DeviceType::WebGPU, "webgpu"}
    };
    return deviceNames.at(device);
}

const char* GetDevicePrettyName(const DeviceType& device) {
    static const std::unordered_map<DeviceType, const char*> deviceNames = {
        {DeviceType::None,   "None"},
        {DeviceType::CPU,    "CPU"},
        {DeviceType::CUDA,   "CUDA"},
        {DeviceType::Metal,  "Metal"},
        {DeviceType::Vulkan, "Vulkan"},
        {DeviceType::WebGPU, "WebGPU"}
    };
    if (!deviceNames.contains(device)) {
        return "None";
    }
    return deviceNames.at(device);
}

DeviceType StringToDevice(const std::string& device) {
    std::string deviceL = device;
    std::transform(deviceL.begin(), deviceL.end(), deviceL.begin(), ::tolower);
    static const std::unordered_map<std::string, DeviceType> deviceNames = {
        {"none",   DeviceType::None},
        {"cpu",    DeviceType::CPU},
        {"cuda",   DeviceType::CUDA},
        {"metal",  DeviceType::Metal},
        {"vulkan", DeviceType::Vulkan},
        {"webgpu", DeviceType::WebGPU}
    };
    if (!deviceNames.contains(deviceL)) {
        return DeviceType::None;
    }
    return deviceNames.at(device);
}

std::string_view LocationName(const Location& loc) {
    static const std::unordered_map<Location, std::string_view> names = {
        {Location::None, "None"},
        {Location::Host, "Host"},
        {Location::Device, "Device"},
        {Location::Unified, "Unified"},
    };

    if (!names.contains(loc)) {
        return "None";
    }
    return names.at(loc);
}

std::size_t DataTypeSize(const DataType& type) {
    switch (type) {
        case DataType::F32:  return sizeof(F32);
        case DataType::F64:  return sizeof(F64);
        case DataType::I8:   return sizeof(I8);
        case DataType::I16:  return sizeof(I16);
        case DataType::I32:  return sizeof(I32);
        case DataType::I64:  return sizeof(I64);
        case DataType::U8:   return sizeof(U8);
        case DataType::U16:  return sizeof(U16);
        case DataType::U32:  return sizeof(U32);
        case DataType::U64:  return sizeof(U64);
        case DataType::CF32: return sizeof(CF32);
        case DataType::CF64: return sizeof(CF64);
        case DataType::CI8:  return sizeof(CI8);
        case DataType::CI16: return sizeof(CI16);
        case DataType::CI32: return sizeof(CI32);
        case DataType::CI64: return sizeof(CI64);
        case DataType::CU8:  return sizeof(CU8);
        case DataType::CU16: return sizeof(CU16);
        case DataType::CU32: return sizeof(CU32);
        case DataType::CU64: return sizeof(CU64);
        case DataType::None:
        default:
            return 0;
    }
}

bool IsDataTypeComplex(const DataType& type) {
    switch (type) {
        case DataType::CF32:
        case DataType::CF64:
        case DataType::CI8:
        case DataType::CI16:
        case DataType::CI32:
        case DataType::CI64:
        case DataType::CU8:
        case DataType::CU16:
        case DataType::CU32:
        case DataType::CU64:
            return true;
        default:
            return false;
    }
}

std::string_view DataTypeToName(const DataType& type) {
    static const std::unordered_map<DataType, std::string_view> names = {
        {DataType::None, "NONE"},
        {DataType::F32,  "F32"},
        {DataType::F64,  "F64"},
        {DataType::I8,   "I8"},
        {DataType::I16,  "I16"},
        {DataType::I32,  "I32"},
        {DataType::I64,  "I64"},
        {DataType::U8,   "U8"},
        {DataType::U16,  "U16"},
        {DataType::U32,  "U32"},
        {DataType::U64,  "U64"},
        {DataType::CF32, "CF32"},
        {DataType::CF64, "CF64"},
        {DataType::CI8,  "CI8"},
        {DataType::CI16, "CI16"},
        {DataType::CI32, "CI32"},
        {DataType::CI64, "CI64"},
        {DataType::CU8,  "CU8"},
        {DataType::CU16, "CU16"},
        {DataType::CU32, "CU32"},
        {DataType::CU64, "CU64"},
    };

    if (!names.contains(type)) {
        return "None";
    }
    return names.at(type);
}

DataType NameToDataType(const std::string_view& name) {
    static const std::unordered_map<std::string_view, DataType> names = {
        {"NONE", DataType::None},
        {"F32",  DataType::F32},
        {"F64",  DataType::F64},
        {"I8",   DataType::I8},
        {"I16",  DataType::I16},
        {"I32",  DataType::I32},
        {"I64",  DataType::I64},
        {"U8",   DataType::U8},
        {"U16",  DataType::U16},
        {"U32",  DataType::U32},
        {"U64",  DataType::U64},
        {"CF32", DataType::CF32},
        {"CF64", DataType::CF64},
        {"CI8",  DataType::CI8},
        {"CI16", DataType::CI16},
        {"CI32", DataType::CI32},
        {"CI64", DataType::CI64},
        {"CU8",  DataType::CU8},
        {"CU16", DataType::CU16},
        {"CU32", DataType::CU32},
        {"CU64", DataType::CU64},
    };

    if (!names.contains(name)) {
        return DataType::None;
    }
    return names.at(name);
}

}  // namespace Jetstream
