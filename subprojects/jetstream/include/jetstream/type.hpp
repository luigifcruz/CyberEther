#ifndef JETSTREAM_TYPE_H
#define JETSTREAM_TYPE_H

#include <complex>
#include <future>
#include <iostream>
#include <vector>
#include <cstring>
#include <algorithm>
#include <map>
#include <variant>

#include "jetstream_config.hpp"
#include "tools/span.hpp"

#ifdef JETSTREAM_CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif

namespace Jetstream {

enum Result {
    SUCCESS = 0,
    ERROR = 1,
    UNKNOWN = 2,
    TIMEOUT,
    CUDA_ERROR,
    ERROR_DATA_DEPENDECY,
    ERROR_FUTURE_INVALID,
};

void print_error(Result, const char*, int, const char*);

#ifndef JETSTREAM_CHECK_THROW
#define JETSTREAM_CHECK_THROW(result) { \
    if (result != Jetstream::Result::SUCCESS) { \
        print_error(result, __PRETTY_FUNCTION__, __LINE__, __FILE__); \
        throw result; \
    } \
}
#endif

#ifndef CHECK
#define CHECK(result) { \
    if (result != Jetstream::Result::SUCCESS) { \
        print_error(result, __PRETTY_FUNCTION__, __LINE__, __FILE__); \
        return result; \
    } \
}
#endif

#ifdef JETSTREAM_CUDA_AVAILABLE
void cuda_print_error(cudaError_t, const char*, int, const char*);
#ifndef CUDA_CHECK_THROW
#define CUDA_CHECK_THROW(result) { \
    if (result != cudaSuccess) { \
        cuda_print_error(result, __PRETTY_FUNCTION__, __LINE__, __FILE__); \
        throw result; \
    } \
}
#endif

#ifndef CUDA_CHECK
#define CUDA_CHECK(result) { \
    if (result != cudaSuccess) { \
        cuda_print_error(result, __PRETTY_FUNCTION__, __LINE__, __FILE__); \
        return Jetstream::Result::CUDA_ERROR; \
    } \
}
#endif
#else
#ifndef CUDA_CHECK
#define CUDA_CHECK(result)
#endif
#ifndef CUDA_CHECK_THROW
#define CUDA_CHECK_THROW(result)
#endif
#endif

#if defined JETSTREAM_CUDA_AVAILABLE && defined JETSTREAM_DEBUG
#include <nvtx3/nvToolsExt.h>

#ifndef DEBUG_PUSH
#define DEBUG_PUSH(name) { nvtxRangePushA(name); }
#endif
#ifndef DEBUG_POP
#define DEBUG_POP() { nvtxRangePop(); }
#endif

#else

#ifndef DEBUG_PUSH
#define DEBUG_PUSH(name)
#endif
#ifndef DEBUG_POP
#define DEBUG_POP()
#endif

#endif

enum class Launch : uint8_t {
    ASYNC   = 1,
    SYNC    = 2,
};

enum class Locale : uint8_t {
    CPU     = 1 << 0,
    CUDA    = 1 << 1,
};

template<typename T>
struct Data {
    Locale location;
    T buf;
};

typedef nonstd::span<std::complex<float>> VCF32;
typedef nonstd::span<float> VF32;

typedef std::variant<Data<VCF32>, Data<VF32>> Variant;
typedef struct { std::string module; std::string port; } Tap;

typedef std::map<std::string, Variant> Manifest;
typedef std::map<std::string, std::variant<std::monostate, Variant, Tap>> VirtualManifest;

class Module;
class Scheduler;

typedef struct { Locale device; Launch mode; } Policy;
typedef std::vector<std::shared_ptr<Scheduler>> Dependencies;

} // namespace Jetstream

#ifndef GADGET_TYPE_H
#define GADGET_TYPE_H

namespace Gadget {

template<typename T>
struct Size2D {
    T width;
    T height;

    bool operator==(const Size2D<T>& a) const {
        return (width == a.width && height == a.height);
    }

    bool operator!=(const Size2D<T>& a) const {
        return (width != a.width || height != a.height);
    }

    bool operator<=(const Size2D<T>& a) const {
        return (width <= a.width || height <= a.height);
    }
};

template<typename T>
struct Range {
    T min;
    T max;

    bool operator==(const Size2D<T>& a) const {
        return (min == a.min && max == a.max);
    }

    bool operator!=(const Size2D<T>& a) const {
        return (min != a.min || max != a.max);
    }

    bool operator<=(const Size2D<T>& a) const {
        return (min <= a.min || max <= a.max);
    }
};

} // namspace Gadget

using namespace Gadget;

#endif

#endif
