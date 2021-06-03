#ifndef RENDER_TYPES_H
#define RENDER_TYPES_H

#include <vector>
#include <memory>
#include <iostream>
#include <algorithm>
#include <unistd.h>

#include "magic_enum.hpp"

typedef unsigned int uint;

#ifndef RENDER_ASSERT_SUCCESS
#define RENDER_ASSERT_SUCCESS(result) { \
    if (result != Result::SUCCESS) { \
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
    UINT8,
    F32,
};

} // namespace Render

#endif
