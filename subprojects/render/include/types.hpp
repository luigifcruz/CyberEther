#ifndef RENDER_TYPES_H
#define RENDER_TYPES_H

namespace Render {

#define ASSERT_SUCCESS(result) { \
    if (result != Result::SUCCESS) { \
        std::cerr << "Render encountered an exception (" <<  magic_enum::enum_name(result) << ") in line " \
            << __LINE__ << " of file " << __FILE__ << "." << std::endl; \
        throw result; \
    } \
}

enum struct Result {
    SUCCESS = 0,
    NO_RENDER_BACKEND_FOUND,
    FAILED_TO_OPEN_SCREEN,
};

enum struct Backend {
    OPENGL,
    VULKAN,
    D3D12,
    METAL,
    WEBGPU,
};

} // namespace Render

#endif