#ifndef JETSTREAM_MEMORY_TYPES_HH
#define JETSTREAM_MEMORY_TYPES_HH

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>
#include <ostream>
#include <cstdint>
#include <complex>
#include <unordered_map>

#include "jetstream/logger.hh"
#include "jetstream/macros.hh"

namespace Jetstream {

//
// Device
//

enum class JETSTREAM_API DeviceType : uint8_t {
    None    = 1 << 0,
    CPU     = 1 << 1,
    CUDA    = 1 << 2,
    Metal   = 1 << 3,
    Vulkan  = 1 << 4,
    WebGPU  = 1 << 5,
};

inline constexpr DeviceType operator|(DeviceType lhs, DeviceType rhs) {
    return static_cast<DeviceType>(static_cast<uint8_t>(lhs) | static_cast<uint8_t>(rhs));
}

inline constexpr DeviceType operator&(DeviceType lhs, DeviceType rhs) {
    return static_cast<DeviceType>(static_cast<uint8_t>(lhs) & static_cast<uint8_t>(rhs));
}

inline constexpr bool operator==(DeviceType lhs, DeviceType rhs) {
    return (static_cast<uint8_t>(lhs) == static_cast<uint8_t>(rhs));
}

const char* GetDeviceName(const DeviceType& device);
const char* GetDevicePrettyName(const DeviceType& device);
DeviceType StringToDevice(const std::string& device);

inline std::ostream& operator<<(std::ostream& os, const DeviceType& device) {
    return os << GetDevicePrettyName(device);
}

//
// Numeric Types
//

typedef float    F32;
typedef double   F64;
typedef int8_t   I8;
typedef int16_t  I16;
typedef int32_t  I32;
typedef int64_t  I64;
typedef uint8_t  U8;
typedef uint16_t U16;
typedef uint32_t U32;
typedef uint64_t U64;

typedef std::complex<F32> CF32;
typedef std::complex<F64> CF64;
typedef std::complex<I8>  CI8;
typedef std::complex<I16> CI16;
typedef std::complex<I32> CI32;
typedef std::complex<I64> CI64;
typedef std::complex<U8>  CU8;
typedef std::complex<U16> CU16;
typedef std::complex<U32> CU32;
typedef std::complex<U64> CU64;

template <typename T = void>
struct JETSTREAM_API NumericTypeInfo;

template<>
struct JETSTREAM_API NumericTypeInfo<F32> {
    using type = F32;
    using subtype = F32;
    using surtype = CF32;
    inline static const char* name = "F32";
};

template<>
struct JETSTREAM_API NumericTypeInfo<F64> {
    using type = F64;
    using subtype = F64;
    using surtype = CF64;
    inline static const char* name = "F64";
};

template<>
struct JETSTREAM_API NumericTypeInfo<I8> {
    using type = I8;
    using subtype = I8;
    using surtype = CI8;
    inline static const char* name = "I8";
};

template<>
struct JETSTREAM_API NumericTypeInfo<I16> {
    using type = I16;
    using subtype = I16;
    using surtype = CI16;
    inline static const char* name = "I16";
};

template<>
struct JETSTREAM_API NumericTypeInfo<I32> {
    using type = I32;
    using subtype = I32;
    using surtype = CI32;
    inline static const char* name = "I32";
};

template<>
struct JETSTREAM_API NumericTypeInfo<I64> {
    using type = I64;
    using subtype = I64;
    using surtype = CI64;
    inline static const char* name = "I64";
};

template<>
struct JETSTREAM_API NumericTypeInfo<U8> {
    using type = U8;
    using subtype = U8;
    using surtype = CU8;
    inline static const char* name = "U8";
};

template<>
struct JETSTREAM_API NumericTypeInfo<U16> {
    using type = U16;
    using subtype = U16;
    using surtype = CU16;
    inline static const char* name = "U16";
};

template<>
struct JETSTREAM_API NumericTypeInfo<U32> {
    using type = U32;
    using subtype = U32;
    using surtype = CU32;
    inline static const char* name = "U32";
};

template<>
struct JETSTREAM_API NumericTypeInfo<U64> {
    using type = U64;
    using subtype = U64;
    using surtype = CU64;
    inline static const char* name = "U64";
};

template<>
struct JETSTREAM_API NumericTypeInfo<CF32> {
    using type = CF32;
    using subtype = F32;
    using surtype = F32;
    inline static const char* name = "CF32";
};

template<>
struct JETSTREAM_API NumericTypeInfo<CF64> {
    using type = CF64;
    using subtype = F64;
    using surtype = F64;
    inline static const char* name = "CF64";
};

template<>
struct JETSTREAM_API NumericTypeInfo<CI8> {
    using type = CI8;
    using subtype = I8;
    using surtype = I8;
    inline static const char* name = "CI8";
};

template<>
struct JETSTREAM_API NumericTypeInfo<CI16> {
    using type = CI16;
    using subtype = I16;
    using surtype = I16;
    inline static const char* name = "CI16";
};

template<>
struct JETSTREAM_API NumericTypeInfo<CI32> {
    using type = CI32;
    using subtype = I32;
    using surtype = I32;
    inline static const char* name = "CI32";
};

template<>
struct JETSTREAM_API NumericTypeInfo<CI64> {
    using type = CI64;
    using subtype = I64;
    using surtype = I64;
    inline static const char* name = "CI64";
};

template<>
struct JETSTREAM_API NumericTypeInfo<CU8> {
    using type = CU8;
    using subtype = U8;
    using surtype = U8;
    inline static const char* name = "CU8";
};

template<>
struct JETSTREAM_API NumericTypeInfo<CU16> {
    using type = CU16;
    using subtype = U16;
    using surtype = U16;
    inline static const char* name = "CU16";
};

template<>
struct JETSTREAM_API NumericTypeInfo<CU32> {
    using type = CU32;
    using subtype = U32;
    using surtype = U32;
    inline static const char* name = "CU32";
};

template<>
struct JETSTREAM_API NumericTypeInfo<CU64> {
    using type = CU64;
    using subtype = U64;
    using surtype = U64;
    inline static const char* name = "CU64";
};

template<>
struct JETSTREAM_API NumericTypeInfo<void> {
    using type = void;
    using subtype = void;
    using surtype = void;
    inline static const char* name = "";
};

//
// Location
//

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

std::string_view LocationName(const Location& loc);

inline std::ostream& operator<<(std::ostream& os, const Location& location) {
    return os << LocationName(location);
}

//
// Index Shape
//

using Index = U64;
using Shape = std::vector<Index>;

//
// Data Type
//

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

std::size_t DataTypeSize(const DataType& type);
bool IsDataTypeComplex(const DataType& type);
std::string_view DataTypeToName(const DataType& type);
DataType NameToDataType(const std::string_view& name);

inline std::ostream& operator<<(std::ostream& os, const DataType& type) {
    return os << DataTypeToName(type);
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

}  // namespace Jetstream

template <> struct jst::fmt::formatter<Jetstream::DeviceType> : ostream_formatter {};
template <> struct jst::fmt::formatter<Jetstream::Location> : ostream_formatter {};
template <> struct jst::fmt::formatter<Jetstream::DataType> : ostream_formatter {};

#endif  // JETSTREAM_MEMORY_TYPES_HH
