#ifndef JETSTREAM_TYPE_HH
#define JETSTREAM_TYPE_HH

#include <map>
#include <span>
#include <vector>
#include <complex>
#include <typeindex>

namespace Jetstream {

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

template<typename T>
inline const std::string& getTypeName() {
    static std::map<std::type_index, std::string> map = {
        {typeid(CF32),  "CF32"},
        {typeid(CF64),  "CF64"},
        {typeid(CI8),   "CI8"},
        {typeid(CI16),  "CI16"},
        {typeid(CI32),  "CI32"},
        {typeid(CI64),  "CI64"},
        {typeid(CU8),   "CU8"},
        {typeid(CU16),  "CU16"},
        {typeid(CU32),  "CU32"},
        {typeid(CU64),  "CU64"},
        {typeid(F32),   "F32"},
        {typeid(F64),   "F64"},
        {typeid(I8),    "I8"},
        {typeid(I16),   "I16"},
        {typeid(I32),   "I32"},
        {typeid(I64),   "I64"},
        {typeid(U8),    "U8"},
        {typeid(U16),   "U16"},
        {typeid(U32),   "U32"},
        {typeid(U64),   "U64"},
        {typeid(void),  "N/S"}
    };

    auto& type = typeid(T);
    if (!map.contains(type)) {
        return map[typeid(void)];   
    }
    return map[type];
}

enum class Device : uint8_t {
    CPU     = 1 << 0,
    CUDA    = 1 << 1,
    Metal   = 1 << 2,
    Vulkan  = 1 << 3,
};

inline constexpr const Device operator|(Device lhs, Device rhs) {
    return static_cast<Device>(static_cast<uint8_t>(lhs) | static_cast<uint8_t>(rhs));
}

template<typename T>
struct Range {
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

template<Device T>
inline const std::string& getDeviceName() {
    static std::map<std::type_index, std::string> map = {
        {typeid(Device::CPU),    "CPU"},
        {typeid(Device::CUDA),   "CUDA"},
        {typeid(Device::Metal),  "Metal"},
        {typeid(Device::Vulkan), "Vulkan"},
        {typeid(void),           "N/S"}
    };

    auto& type = typeid(T);
    if (!map.contains(type)) {
        return map[typeid(void)];   
    }
    return map[type];
}

enum class Result : U8 {
    SUCCESS = 0,
    ERROR = 1,
    CUDA_ERROR,
    ASSERTION_ERROR,
};

enum class Direction : I64 {
    Forward = 1,
    Backward = -1,
};

}  // namespace Jetstream

#endif
