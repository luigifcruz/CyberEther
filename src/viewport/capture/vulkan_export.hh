#pragma once

#include "jetstream/viewport/capture.hh"
#include "jetstream/backend/base.hh"
#include "jetstream/backend/devices/vulkan/base.hh"

#include <memory>

namespace Jetstream::Viewport {

inline Result FindExportMemoryType(VkPhysicalDevice physicalDevice, U32 typeBits, U32& index) {
    VkPhysicalDeviceMemoryProperties properties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &properties);

    for (const bool deviceLocal : {true, false}) {
        for (U32 i = 0; i < properties.memoryTypeCount; ++i) {
            const bool local = (properties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
            if ((typeBits & (1u << i)) && (!deviceLocal || local)) {
                index = i;
                return Result::SUCCESS;
            }
        }
    }

    return Result::ERROR;
}

class FrameExport {
 public:
    virtual ~FrameExport() = default;

    virtual Result create(const VkExtent2D& extent, U32 family, U32 count) = 0;
    virtual void destroy() = 0;
    virtual void record(VkCommandBuffer commandBuffer, VkImage source, U32 index) = 0;
    virtual const FrameCapture::SharedFrame& frame(U32 index) const = 0;
};

#if defined(JST_OS_LINUX)
std::unique_ptr<FrameExport> CreateBufferExport(FrameCapture::Export exporting);
#endif

#if defined(JST_OS_WINDOWS)
std::unique_ptr<FrameExport> CreateD3D11Export();
#endif

}  // namespace Jetstream::Viewport
