#ifndef RENDER_TYPE_H
#define RENDER_TYPE_H

#include <unistd.h>

#include <vector>
#include <memory>
#include <iostream>
#include <algorithm>
#include <cstring>

#include "render_config.hpp"

#ifdef RENDER_CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif

#include <jetstream/base.hh>

typedef unsigned int uint;

namespace Render {

enum struct Result {
    SUCCESS = 0,
    ERROR,
    CUDA_ERROR,
    RENDER_BACKEND_ERROR,
    NO_RENDER_BACKEND_FOUND,
    FAILED_TO_OPEN_SCREEN,
};

void print_error(Result, const char*, int, const char*);

#ifndef RENDER_CHECK_THROW
#define RENDER_CHECK_THROW(result) { \
    if (result != Render::Result::SUCCESS) { \
        print_error(result, __PRETTY_FUNCTION__, __LINE__, __FILE__); \
        throw result; \
    } \
}
#endif

#ifndef RENDER_ASSERT
#define RENDER_ASSERT(result) { \
    if (result == nullptr) { \
        print_error(Result::ERROR, __PRETTY_FUNCTION__, __LINE__, __FILE__); \
        return Result::ERROR; \
    } \
}
#endif

#ifndef CHECK
#define CHECK(result) { \
    if (result != Render::Result::SUCCESS) { \
        print_error(result, __PRETTY_FUNCTION__, __LINE__, __FILE__); \
        return result; \
    } \
}
#endif

#ifdef RENDER_CUDA_AVAILABLE
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
        return Render::Result::CUDA_ERROR; \
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

enum struct Backend {
    GLES,
    Vulkan,
    DX12,
    Metal,
    WebGPU,
};

enum class PixelFormat : uint {
    RGBA,
    RED,
};

enum class DataFormat : uint {
    RGBA,
    UI8,
    F32,
};

enum class PixelType : uint {
    UI8,
    F32,
};

template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

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

}  // namespace Render

#endif
