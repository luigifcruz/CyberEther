#ifndef JETSTREAM_MEMORY_TYPES_HH
#define JETSTREAM_MEMORY_TYPES_HH

#include <map>
#include <span>
#include <vector>
#include <complex>
#include <unordered_map>
#include <functional>

#include "jetstream/logger.hh"
#include "jetstream/macros.hh"

namespace Jetstream {

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

template <typename T>
struct IsComplex : std::false_type {};

template <typename T>
struct IsComplex<std::complex<T>> : std::true_type {};

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

template<>
struct JETSTREAM_API NumericTypeInfo<void> {
    using type = void;
    using subtype = void;
    using surtype = void;
    inline static const char* name = "";
};

//
// Device
//

enum class JETSTREAM_API Device : uint8_t {
    None    = 1 << 0,
    CPU     = 1 << 1,
    CUDA    = 1 << 2,
    Metal   = 1 << 3,
    Vulkan  = 1 << 4,
    WebGPU  = 1 << 5,
};

inline constexpr Device operator|(Device lhs, Device rhs) {
    return static_cast<Device>(static_cast<uint8_t>(lhs) | static_cast<uint8_t>(rhs));
}

inline constexpr Device operator&(Device lhs, Device rhs) {
    return static_cast<Device>(static_cast<uint8_t>(lhs) & static_cast<uint8_t>(rhs));
}

inline constexpr bool operator==(Device lhs, Device rhs) {
    return (static_cast<uint8_t>(lhs) == static_cast<uint8_t>(rhs));
}

inline const char* GetDeviceName(const Device& device) {
    static const std::unordered_map<Device, const char*> deviceNames = {
        {Device::None,   "none"},
        {Device::CPU,    "cpu"},
        {Device::CUDA,   "cuda"},
        {Device::Metal,  "metal"},
        {Device::Vulkan, "vulkan"},
        {Device::WebGPU, "webgpu"}
    };
    return deviceNames.at(device);
}

inline const char* GetDevicePrettyName(const Device& device) {
    static const std::unordered_map<Device, const char*> deviceNames = {
        {Device::None,   "None"},
        {Device::CPU,    "CPU"},
        {Device::CUDA,   "CUDA"},
        {Device::Metal,  "Metal"},
        {Device::Vulkan, "Vulkan"},
        {Device::WebGPU, "WebGPU"}
    };
    return deviceNames.at(device);
}

inline Device StringToDevice(const std::string& device) {
    std::string device_l = device;
    std::transform(device_l.begin(), device_l.end(), device_l.begin(), ::tolower);
    static const std::unordered_map<std::string, Device> deviceNames = {
        {"none",   Device::None},
        {"cpu",    Device::CPU},
        {"cuda",   Device::CUDA},
        {"metal",  Device::Metal},
        {"vulkan", Device::Vulkan},
        {"webgpu", Device::WebGPU}
    };
    if (deviceNames.find(device_l) == deviceNames.end()) {
        JST_ERROR("Invalid device name: {}", device);
        throw Device::None;
    }
    return deviceNames.at(device);
}

inline std::ostream& operator<<(std::ostream& os, const Device& device) {
    return os << GetDevicePrettyName(device);
}

//
// Locale
//

struct Locale {
    std::string id = "";
    std::string subId = "";
    std::string pinId = "";

    // TODO: Remove
    std::string str() const {
        return id + subId + pinId;
    }

    Locale idOnly() const {
        return Locale{id};
    }

    Locale idSub() const {
        return Locale{id, subId};
    }

    Locale idPin() const {
        return Locale{id, "", pinId};
    }

    Locale pinOnly() const {
        return Locale{id, "", pinId};
    }

    bool empty() const {
        return id.empty() && subId.empty() && pinId.empty();
    }

    bool internal() const {
        return !subId.empty();
    }

    bool operator==(const Locale& other) const {
        return id == other.id &&
               subId == other.subId &&
               pinId == other.pinId;
    }

    std::size_t hash() const {
        return Hasher()(*this);
    }

    struct Hasher {
        std::size_t operator()(const Locale& locale) const {
            std::hash<std::string> string_hasher;
            std::size_t h1 = string_hasher(locale.id);
            std::size_t h2 = string_hasher(locale.subId);
            std::size_t h3 = string_hasher(locale.pinId);
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };
};

inline std::ostream& operator<<(std::ostream& os, const Locale& locale) {
    if (!locale.id.empty() && !locale.subId.empty()) {
        os << fmt::format("{}-", locale.id);
    } else {
        os << fmt::format("{}", locale.id);
    }
    if (!locale.subId.empty()) {
        os << fmt::format("{}", locale.subId);
    }
    if (!locale.pinId.empty()) {
        os << fmt::format(".{}", locale.pinId);
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

// TODO Add print for Range.

}  // namespace Jetstream

template <> struct fmt::formatter<Jetstream::Device> : ostream_formatter {};
template <> struct fmt::formatter<Jetstream::Locale> : ostream_formatter {};

#endif
