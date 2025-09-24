#ifndef JETSTREAM_MEMORY2_TYPES_HH
#define JETSTREAM_MEMORY2_TYPES_HH

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <ostream>

#include "jetstream/logger.hh"
#include "jetstream/memory/macros.hh"
#include "jetstream/memory/types.hh"

namespace Jetstream::mem2 {

using Device = Jetstream::Device;

enum class Location : uint8_t {
    None    = 0,
    Host    = 1 << 0,
    Device  = 1 << 1,
    Unified = Host | Device,
};

inline constexpr Location operator|(Location lhs, Location rhs) {
    return static_cast<Location>(static_cast<uint8_t>(lhs) | static_cast<uint8_t>(rhs));
}

inline constexpr Location operator&(Location lhs, Location rhs) {
    return static_cast<Location>(static_cast<uint8_t>(lhs) & static_cast<uint8_t>(rhs));
}

using Index = U64;
using Shape = std::vector<Index>;

enum class DataType : uint8_t {
    None = 0,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    CF32,
    CF64,
    CI8,
    CI16,
    CI32,
    CI64,
    CU8,
    CU16,
    CU32,
    CU64,
};

inline constexpr std::size_t DataTypeSize(const DataType type) {
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

inline constexpr bool IsComplex(const DataType type) {
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

inline std::string_view DataTypeName(const DataType type) {
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
        return "UNKNOWN";
    }
    return names.at(type);
}

inline std::string_view LocationName(const Location loc) {
    static const std::unordered_map<Location, std::string_view> names = {
        {Location::None, "None"},
        {Location::Host, "Host"},
        {Location::Device, "Device"},
        {Location::Unified, "Unified"},
    };

    if (!names.contains(loc)) {
        return "UNKNOWN";
    }
    return names.at(loc);
}

inline std::ostream& operator<<(std::ostream& os, const DataType& type) {
    os << DataTypeName(type);
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const Location& loc) {
    os << LocationName(loc);
    return os;
}

template<typename T>
inline constexpr DataType TypeToDataType() {
    return DataType::None; // Default case
}

template<> inline constexpr DataType TypeToDataType<F32>() { return DataType::F32; }
template<> inline constexpr DataType TypeToDataType<F64>() { return DataType::F64; }
template<> inline constexpr DataType TypeToDataType<I8>() { return DataType::I8; }
template<> inline constexpr DataType TypeToDataType<I16>() { return DataType::I16; }
template<> inline constexpr DataType TypeToDataType<I32>() { return DataType::I32; }
template<> inline constexpr DataType TypeToDataType<I64>() { return DataType::I64; }
template<> inline constexpr DataType TypeToDataType<U8>() { return DataType::U8; }
template<> inline constexpr DataType TypeToDataType<U16>() { return DataType::U16; }
template<> inline constexpr DataType TypeToDataType<U32>() { return DataType::U32; }
template<> inline constexpr DataType TypeToDataType<U64>() { return DataType::U64; }
template<> inline constexpr DataType TypeToDataType<CF32>() { return DataType::CF32; }
template<> inline constexpr DataType TypeToDataType<CF64>() { return DataType::CF64; }
template<> inline constexpr DataType TypeToDataType<CI8>() { return DataType::CI8; }
template<> inline constexpr DataType TypeToDataType<CI16>() { return DataType::CI16; }
template<> inline constexpr DataType TypeToDataType<CI32>() { return DataType::CI32; }
template<> inline constexpr DataType TypeToDataType<CI64>() { return DataType::CI64; }
template<> inline constexpr DataType TypeToDataType<CU8>() { return DataType::CU8; }
template<> inline constexpr DataType TypeToDataType<CU16>() { return DataType::CU16; }
template<> inline constexpr DataType TypeToDataType<CU32>() { return DataType::CU32; }
template<> inline constexpr DataType TypeToDataType<CU64>() { return DataType::CU64; }

template<DataType DT>
struct DataTypeToType {
    using type = void; // Default case
};

template<> struct DataTypeToType<DataType::F32> { using type = F32; };
template<> struct DataTypeToType<DataType::F64> { using type = F64; };
template<> struct DataTypeToType<DataType::I8> { using type = I8; };
template<> struct DataTypeToType<DataType::I16> { using type = I16; };
template<> struct DataTypeToType<DataType::I32> { using type = I32; };
template<> struct DataTypeToType<DataType::I64> { using type = I64; };
template<> struct DataTypeToType<DataType::U8> { using type = U8; };
template<> struct DataTypeToType<DataType::U16> { using type = U16; };
template<> struct DataTypeToType<DataType::U32> { using type = U32; };
template<> struct DataTypeToType<DataType::U64> { using type = U64; };
template<> struct DataTypeToType<DataType::CF32> { using type = CF32; };
template<> struct DataTypeToType<DataType::CF64> { using type = CF64; };
template<> struct DataTypeToType<DataType::CI8> { using type = CI8; };
template<> struct DataTypeToType<DataType::CI16> { using type = CI16; };
template<> struct DataTypeToType<DataType::CI32> { using type = CI32; };
template<> struct DataTypeToType<DataType::CI64> { using type = CI64; };
template<> struct DataTypeToType<DataType::CU8> { using type = CU8; };
template<> struct DataTypeToType<DataType::CU16> { using type = CU16; };
template<> struct DataTypeToType<DataType::CU32> { using type = CU32; };
template<> struct DataTypeToType<DataType::CU64> { using type = CU64; };

}  // namespace Jetstream::mem2

template <> struct jst::fmt::formatter<Jetstream::mem2::Location> : ostream_formatter {};
template <> struct jst::fmt::formatter<Jetstream::mem2::DataType> : ostream_formatter {};

#endif  // JETSTREAM_MEMORY2_TYPES_HH
