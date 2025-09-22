#ifndef JETSTREAM_MEMORY_DEVICES_VULKAN_BUFFER_HH
#define JETSTREAM_MEMORY_DEVICES_VULKAN_BUFFER_HH

#include <vulkan/vulkan.h>

namespace Jetstream {

class VulkanBufferBackend {
 public:
    virtual ~VulkanBufferBackend() = default;
    virtual VkBuffer buffer() const = 0;
    virtual VkDeviceMemory memory() const = 0;
};

}  // namespace Jetstream

#endif  // JETSTREAM_MEMORY_DEVICES_VULKAN_BUFFER_HH
