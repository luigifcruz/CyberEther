#ifndef RENDER_TYPE_H
#define RENDER_TYPE_H

#include <vector>
#include <memory>
#include <iostream>
#include <algorithm>
#include <unistd.h>
#include <cstring>

#include "render_config.hpp"

#ifdef RENDER_CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif

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

#ifndef CHECK
#define CHECK(result) { \
    if (result != Render::Result::SUCCESS) { \
        print_error(result, __PRETTY_FUNCTION__, __LINE__, __FILE__); \
        return result; \
    } \
}
#endif

#ifdef RENDER_CUDA_AVAILABLE
#ifndef CUDA_CHECK
#define CUDA_CHECK(result) { \
    if (result != cudaSuccess) { \
        std::cout << "CUDA error thrown: " << result << std::endl; \
        return Render::Result::CUDA_ERROR; \
    } \
}
#endif
#ifndef CUDA_CHECK_THROW
#define CUDA_CHECK_THROW(result) { \
    if (result != cudaSuccess) { \
        std::cout << "CUDA error thrown: " << result << std::endl; \
        throw result; \
    } \
}
#endif
#endif

enum struct API {
	GLES,
	VULKAN,
	D3D12,
	METAL,
	WEBGPU,
};

enum class PixelFormat : uint {
    RGB,
    RED,
};

enum class DataFormat : uint {
    RGB,
    UI8,
    F32,
};

enum class PixelType : uint {
    UI8,
    F32,
};

} // namespace Render

#endif
