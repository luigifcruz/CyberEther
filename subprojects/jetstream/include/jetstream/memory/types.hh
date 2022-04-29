#ifndef JETSTREAM_MEMORY_TYPES_HH
#define JETSTREAM_MEMORY_TYPES_HH

#include <span>
#include <vector>
#include <complex>

namespace Jetstream {

enum class Device : uint8_t {
    CPU     = 1 << 0,
    CUDA    = 1 << 1,
    METAL   = 1 << 2,
    VULKAN  = 1 << 3,
};

inline constexpr const Device operator|(Device lhs, Device rhs) {
    return static_cast<Device>(static_cast<uint8_t>(lhs) | static_cast<uint8_t>(rhs));
}

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

}  // namespace Jetstream

#endif
