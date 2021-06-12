#ifndef RENDER_TYPES_H
#define RENDER_TYPES_H

#include <vector>
#include <memory>
#include <iostream>
#include <algorithm>
#include <unistd.h>

#include "magic_enum.hpp"
#include "render_config.hpp"

#ifdef RENDER_CUDA_INTEROP_AVAILABLE
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#ifndef CUDA_ASSERT_SUCCESS
#define CUDA_ASSERT_SUCCESS(result) { \
    if (result != cudaSuccess) { \
        std::cerr << "CUDA encountered an exception (" <<  magic_enum::enum_name(result) << ") in " \
            << __PRETTY_FUNCTION__ << " in line " << __LINE__ << " of file " << __FILE__ << "." << std::endl; \
        throw result; \
    } \
}
#endif
#endif

typedef unsigned int uint;

#ifndef RENDER_ASSERT_SUCCESS
#define RENDER_ASSERT_SUCCESS(result) { \
    if (result != Render::Result::SUCCESS) { \
        std::cerr << "Render encountered an exception (" <<  magic_enum::enum_name(result) << ") in " \
            << __PRETTY_FUNCTION__ << " in line " << __LINE__ << " of file " << __FILE__ << "." << std::endl; \
        throw result; \
    } \
}
#endif

namespace Render {

enum struct Result {
    SUCCESS = 0,
    ERROR,
    CUDA_ERROR,
    RENDER_BACKEND_ERROR,
    NO_RENDER_BACKEND_FOUND,
    FAILED_TO_OPEN_SCREEN,
};

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
