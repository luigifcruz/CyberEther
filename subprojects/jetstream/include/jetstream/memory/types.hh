#ifndef JETSTREAM_MEMORY_TYPES_HH
#define JETSTREAM_MEMORY_TYPES_HH

#include <span>
#include <vector>
#include <complex>

#include <fmt/ranges.h>
#include <fmt/ostream.h>

#include "jetstream/logger.hh"
#include "jetstream/macros.hh"

namespace Jetstream {

inline std::string GetTypeName(const auto& type) {
    std::ostringstream oss;
    oss << type;
    return oss.str();
}

//
// Device
//

enum class JETSTREAM_API Device : uint8_t {
    None    = 1 << 0,
#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE 
    CPU     = 1 << 1,
#endif
#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE 
    CUDA    = 1 << 2,
#endif
#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE 
    Metal   = 1 << 3,
#endif
#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE 
    Vulkan  = 1 << 4,
#endif
};

inline constexpr Device operator|(Device lhs, Device rhs) {
    return static_cast<Device>(static_cast<uint8_t>(lhs) | static_cast<uint8_t>(rhs));
}

inline constexpr Device operator&(Device lhs, Device rhs) {
    return static_cast<Device>(static_cast<uint8_t>(lhs) & static_cast<uint8_t>(rhs));
}

inline std::ostream& operator<<(std::ostream& os, const Device& device) {
    switch(device) {
        case Device::None: os << "None"; break;
#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE 
        case Device::CPU: os << "CPU"; break;
#endif
#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE 
        case Device::CUDA: os << "CUDA"; break;
#endif
#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE 
        case Device::Metal: os << "Metal"; break;
#endif
#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE 
        case Device::Vulkan: os << "Vulkan"; break;
#endif
        default: os.setstate(std::ios_base::failbit);
    }
    return os;
}

//
// Range
//

template<typename T>
struct JETSTREAM_API Range {
    T min;
    T max;

    bool operator==(const Range<T>& a) const {
        return (min == a.min && max == a.max);
    }

    bool operator!=(const Range<T>& a) const {
        return (min != a.min || max != a.max);
    }

    bool operator<=(const Range<T>& a) const {
        return (min <= a.min || max <= a.max);
    }
};

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
typedef bool     BOOL;

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
struct JETSTREAM_API NumericTypeInfo<BOOL> {
    using type = BOOL;
    using subtype = BOOL;
    using surtype = BOOL;
    inline static const char* name = "BOOL";
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

}  // namespace Jetstream

template <> struct fmt::formatter<Jetstream::Device> : ostream_formatter {};

#endif
