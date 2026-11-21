#ifndef JETSTREAM_RENDER_VULKAN_CREATION_HH
#define JETSTREAM_RENDER_VULKAN_CREATION_HH

#include <vulkan/vulkan.h>

namespace Jetstream::Render::Detail {

template<typename Handle, typename Create>
VkResult CreateHandle(Handle& owned, Create&& create) {
    Handle candidate = VK_NULL_HANDLE;
    const VkResult result = create(&candidate);
    if (result == VK_SUCCESS) {
        owned = candidate;
    }
    return result;
}

}  // namespace Jetstream::Render::Detail

#endif
