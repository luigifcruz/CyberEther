#ifndef RENDER_TYPES_H
#define RENDER_TYPES_H

#include <vector>
#include <memory>
#include <iostream>
#include <unistd.h>

#include "magic_enum.hpp"

namespace Render {

#ifndef ASSERT_SUCCESS
#define ASSERT_SUCCESS(result) { \
    if (result != Result::SUCCESS) { \
        std::cerr << "Render encountered an exception (" <<  magic_enum::enum_name(result) << ") in line " \
            << __LINE__ << " of file " << __FILE__ << "." << std::endl; \
        return result; \
    } \
}
#endif

enum struct Result {
    SUCCESS = 0,
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

} // namespace Render

#endif
