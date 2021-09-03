#ifndef GADGET_H
#define GADGET_H

#include <iostream>
#include <complex>
#include "gadget/tools/span.hpp"

#if __has_include("cuda_runtime.h")
#define GADGET_HAS_CUDA
#if __has_include("nvtx3/nvToolsExt.h")
#define GADGET_HAS_NVTX
#endif
#endif

#if defined GADGET_HAS_NVTX && defined GADGET_DEBUG
#include <nvtx3/nvToolsExt.h>

#ifndef GT_DEBUG_PUSH
#define GT_DEBUG_PUSH(name) { nvtxRangePushA(name); }
#endif
#ifndef GT_DEBUG_POP
#define GT_DEBUG_POP() { nvtxRangePop(); }
#endif

#else

#ifndef GT_DEBUG_PUSH
#define GT_DEBUG_PUSH(name)
#endif
#ifndef GT_DEBUG_POP
#define GT_DEBUG_POP()
#endif

#endif

#if defined GADGET_HAS_CUDA
#include <cuda_runtime.h>

#ifndef GT_CUDA_CHECK_THROW
#define GT_CUDA_CHECK_THROW(result) { \
    if (result != cudaSuccess) { \
        cuda_print_error(result, __PRETTY_FUNCTION__, __LINE__, __FILE__); \
        throw result; \
    } \
}
#endif

#ifndef GT_CUDA_CHECK
#define GT_CUDA_CHECK(result) { \
    if (result != cudaSuccess) { \
        cuda_print_error(result, __PRETTY_FUNCTION__, __LINE__, __FILE__); \
        return Gadget::Result::CUDA_ERROR; \
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

namespace Gadget {

typedef nonstd::span<float> VF32;
typedef nonstd::span<std::complex<float>> VCF32;

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

} // namespace Gadget

#endif
