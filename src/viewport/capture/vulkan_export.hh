#pragma once

#include "jetstream/viewport/capture.hh"
#include "jetstream/backend/base.hh"
#include "jetstream/backend/devices/vulkan/base.hh"
#include "jetstream/backend/devices/vulkan/helpers.hh"

#include <memory>

namespace Jetstream::Viewport {

inline Result FindExportMemoryType(VkPhysicalDevice physicalDevice, U32 typeBits, U32& index) {
    for (const VkMemoryPropertyFlags flags : {VkMemoryPropertyFlags{VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT},
                                              VkMemoryPropertyFlags{0}}) {
        if (Backend::HasMemoryType(physicalDevice, typeBits, flags)) {
            index = Backend::FindMemoryType(physicalDevice, typeBits, flags);
            return Result::SUCCESS;
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
