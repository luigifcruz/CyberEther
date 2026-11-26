#include "vulkan_export.hh"

#include "jetstream/backend/devices/vulkan/helpers.hh"
#include "jetstream/logger.hh"

#include <unistd.h>

#include <vector>

namespace Jetstream::Viewport {

namespace {

class BufferExport : public FrameExport {
 public:
    explicit BufferExport(FrameCapture::Export exporting)
        : handleType(exporting == FrameCapture::Export::DmaBuf ? VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT
                                                               : VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT),
          alignment(exporting == FrameCapture::Export::DmaBuf ? 256 : 4),
          owner(exporting == FrameCapture::Export::DmaBuf ? VK_QUEUE_FAMILY_FOREIGN_EXT : VK_QUEUE_FAMILY_EXTERNAL) {}

    ~BufferExport() override {
        destroy();
    }

    Result create(const VkExtent2D& extent, U32 family, U32 count) override {
        this->extent = extent;
        this->family = family;
        targets.resize(count);

        for (auto& target : targets) {
            JST_CHECK(createTarget(target));
        }

        return Result::SUCCESS;
    }

    void destroy() override {
        auto& device = Backend::State<DeviceType::Vulkan>()->getDevice();

        for (auto& target : targets) {
            if (target.buffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(device, target.buffer, nullptr);
            }
            if (target.memory != VK_NULL_HANDLE) {
                vkFreeMemory(device, target.memory, nullptr);
            }
            if (target.exported) {
                close(static_cast<int>(target.frame.handle));
            }
        }
        targets.clear();
    }

    void record(VkCommandBuffer commandBuffer, VkImage source, U32 index) override {
        const Target& target = targets[index];

        VkBufferMemoryBarrier acquire = {};
        acquire.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        acquire.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        acquire.srcQueueFamilyIndex = owner;
        acquire.dstQueueFamilyIndex = family;
        acquire.buffer = target.buffer;
        acquire.size = VK_WHOLE_SIZE;

        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 1, &acquire, 0, nullptr);

        VkBufferImageCopy region = {};
        region.bufferRowLength = static_cast<U32>(target.frame.pitch / 4);
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = {extent.width, extent.height, 1};

        vkCmdCopyImageToBuffer(commandBuffer, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, target.buffer, 1, &region);

        VkBufferMemoryBarrier release = acquire;
        release.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        release.dstAccessMask = 0;
        release.srcQueueFamilyIndex = family;
        release.dstQueueFamilyIndex = owner;

        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                             0, 0, nullptr, 1, &release, 0, nullptr);
    }

    const FrameCapture::SharedFrame& frame(U32 index) const override {
        return targets[index].frame;
    }

 private:
    struct Target {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        FrameCapture::SharedFrame frame;
        bool exported = false;
    };

    Result createTarget(Target& target) {
        auto& device = Backend::State<DeviceType::Vulkan>()->getDevice();
        auto& physicalDevice = Backend::State<DeviceType::Vulkan>()->getPhysicalDevice();

        const U64 pitch = ((static_cast<U64>(extent.width) * 4 + alignment - 1) / alignment) * alignment;

        VkExternalMemoryBufferCreateInfo externalInfo = {};
        externalInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
        externalInfo.handleTypes = handleType;

        VkBufferCreateInfo bufferInfo = {};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.pNext = &externalInfo;
        bufferInfo.size = pitch * extent.height;
        bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        JST_VK_CHECK(vkCreateBuffer(device, &bufferInfo, nullptr, &target.buffer), [&]{
            JST_ERROR("[CAPTURE] Failed to create the shared frame buffer.");
        });

        VkMemoryRequirements requirements;
        vkGetBufferMemoryRequirements(device, target.buffer, &requirements);

        VkMemoryDedicatedAllocateInfo dedicatedInfo = {};
        dedicatedInfo.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
        dedicatedInfo.buffer = target.buffer;

        VkExportMemoryAllocateInfo exportInfo = {};
        exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
        exportInfo.pNext = &dedicatedInfo;
        exportInfo.handleTypes = handleType;

        VkMemoryAllocateInfo allocateInfo = {};
        allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.pNext = &exportInfo;
        allocateInfo.allocationSize = requirements.size;
        if (FindExportMemoryType(physicalDevice, requirements.memoryTypeBits, allocateInfo.memoryTypeIndex) != Result::SUCCESS) {
            JST_ERROR("[CAPTURE] No memory type can hold the shared frame.");
            return Result::ERROR;
        }

        JST_VK_CHECK(vkAllocateMemory(device, &allocateInfo, nullptr, &target.memory), [&]{
            JST_ERROR("[CAPTURE] Failed to allocate the shared frame memory.");
        });

        JST_VK_CHECK(vkBindBufferMemory(device, target.buffer, target.memory, 0), [&]{
            JST_ERROR("[CAPTURE] Failed to bind the shared frame memory.");
        });

        auto getMemoryFd = reinterpret_cast<PFN_vkGetMemoryFdKHR>(vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR"));
        if (!getMemoryFd) {
            JST_ERROR("[CAPTURE] Failed to load vkGetMemoryFdKHR.");
            return Result::ERROR;
        }

        VkMemoryGetFdInfoKHR fdInfo = {};
        fdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
        fdInfo.memory = target.memory;
        fdInfo.handleType = handleType;

        int fd = -1;
        JST_VK_CHECK(getMemoryFd(device, &fdInfo, &fd), [&]{
            JST_ERROR("[CAPTURE] Failed to export the shared frame memory.");
        });

        target.exported = true;
        target.frame = {
            .handle = static_cast<U64>(fd),
            .pitch = pitch,
            .size = requirements.size,
        };

        return Result::SUCCESS;
    }

    const VkExternalMemoryHandleTypeFlagBits handleType;
    const U64 alignment;
    const U32 owner;
    VkExtent2D extent = {};
    U32 family = 0;
    std::vector<Target> targets;
};

}  // namespace

std::unique_ptr<FrameExport> CreateBufferExport(FrameCapture::Export exporting) {
    return std::make_unique<BufferExport>(exporting);
}

}  // namespace Jetstream::Viewport
