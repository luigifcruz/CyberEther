#ifndef RENDER_TYPES_H
#define RENDER_TYPES_H

#include <iostream>

#include "magic_enum.hpp"

namespace Render {

#ifndef ASSERT_SUCCESS
#define ASSERT_SUCCESS(result) { \
    if (result != Result::SUCCESS) { \
        std::cerr << "Render encountered an exception (" <<  magic_enum::enum_name(result) << ") in line " \
            << __LINE__ << " of file " << __FILE__ << "." << std::endl; \
        throw result; \
    } \
}
#endif

enum struct Result {
    SUCCESS = 0,
    RENDER_BACKEND_ERROR,
    NO_RENDER_BACKEND_FOUND,
    FAILED_TO_OPEN_SCREEN,
};

enum struct BackendId {
    OPENGL,
    VULKAN,
    D3D12,
    METAL,
    WEBGPU,
};

} // namespace Render

#endif
