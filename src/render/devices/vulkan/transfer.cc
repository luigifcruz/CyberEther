#include "jetstream/render/devices/vulkan/transfer.hh"

#include <algorithm>
#include <limits>

#include "jetstream/backend/devices/vulkan/helpers.hh"
#include "jetstream/render/devices/vulkan/buffer.hh"
#include "jetstream/render/devices/vulkan/texture.hh"

namespace Jetstream::Render {

using Implementation = TransferImp<DeviceType::Vulkan>;

void Implementation::create(const size_t framesInFlight) {
    arenas.resize(framesInFlight);
}

Result Implementation::ensureCapacity(Arena& arena, const U64& required) {
    if (required <= arena.capacity) {
        return Result::SUCCESS;
    }

    auto& backend = Backend::State<DeviceType::Vulkan>();
    auto& device = backend->getDevice();

    U64 capacity = 0;
    if (!calculateCapacity(required, 1, capacity)) {
        return Result::ERROR;
    }

    Arena replacement;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = capacity;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &bufferInfo, nullptr, &replacement.buffer) != VK_SUCCESS) {
        JST_ERROR("[VULKAN] Failed to create a {} byte transfer arena.", capacity);
        return Result::ERROR;
    }

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(device, replacement.buffer, &memoryRequirements);

    VkMemoryAllocateInfo allocationInfo{};
    allocationInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocationInfo.allocationSize = memoryRequirements.size;
    allocationInfo.memoryTypeIndex = Backend::FindMemoryType(
        backend->getPhysicalDevice(),
        memoryRequirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (vkAllocateMemory(device, &allocationInfo, nullptr, &replacement.memory) != VK_SUCCESS) {
        vkDestroyBuffer(device, replacement.buffer, nullptr);
        JST_ERROR("[VULKAN] Failed to allocate transfer arena memory.");
        return Result::ERROR;
    }

    if (vkBindBufferMemory(device, replacement.buffer, replacement.memory, 0) != VK_SUCCESS) {
        vkDestroyBuffer(device, replacement.buffer, nullptr);
        vkFreeMemory(device, replacement.memory, nullptr);
        JST_ERROR("[VULKAN] Failed to bind transfer arena memory.");
        return Result::ERROR;
    }

    void* mapped = nullptr;
    if (vkMapMemory(device, replacement.memory, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS) {
        vkDestroyBuffer(device, replacement.buffer, nullptr);
        vkFreeMemory(device, replacement.memory, nullptr);
        JST_ERROR("[VULKAN] Failed to map transfer arena memory.");
        return Result::ERROR;
    }

    replacement.mapped = static_cast<U8*>(mapped);
    replacement.capacity = capacity;

    destroyArena(arena);
    arena = replacement;

    JST_DEBUG("[VULKAN] Grew frame transfer arena to {:.2f} MB.",
              static_cast<F32>(capacity) / JST_MB);
    return Result::SUCCESS;
}

Result Implementation::encode(Transfer::Batch& batch,
                              VkCommandBuffer commandBuffer,
                              const size_t frameIndex) {
    if (commandBuffer == VK_NULL_HANDLE || frameIndex >= arenas.size()) {
        return Result::ERROR;
    }

    struct BufferCopy {
        std::shared_ptr<BufferImp<DeviceType::Vulkan>> destination;
        U64 sourceOffset;
        U64 destinationOffset;
        U64 size;
        const U8* source;
    };

    struct TextureCopy {
        std::shared_ptr<TextureImp<DeviceType::Vulkan>> destination;
        U64 sourceOffset;
        U64 destinationRow;
        U64 rowCount;
        U64 width;
        U64 rowByteSize;
        const U8* source;
    };

    std::vector<BufferCopy> bufferCopies;
    std::vector<TextureCopy> textureCopies;
    U64 required = 0;

    for (const auto& transfer : batch.buffers()) {
        const U64 size = transfer.upload.data.size();
        if ((transfer.destinationOffset % 4) != 0 || (size % 4) != 0) {
            JST_ERROR("[VULKAN] Buffer transfer offsets and sizes must be four-byte aligned.");
            return Result::ERROR;
        }

        auto destination = std::dynamic_pointer_cast<BufferImp<DeviceType::Vulkan>>(
            transfer.destination);
        if (!destination) {
            JST_ERROR("[VULKAN] Cannot encode a buffer from another render device.");
            return Result::ERROR;
        }

        U64 sourceOffset = 0;
        if (!reserveRange(required, size, 4, sourceOffset)) {
            return Result::ERROR;
        }

        bufferCopies.push_back({destination,
                                sourceOffset,
                                transfer.destinationOffset,
                                size,
                                transfer.upload.data.data()});
    }

    for (const auto& transfer : batch.textures()) {
        auto destination = std::dynamic_pointer_cast<TextureImp<DeviceType::Vulkan>>(
            transfer.destination);
        if (!destination) {
            JST_ERROR("[VULKAN] Cannot encode a texture from another render device.");
            return Result::ERROR;
        }

        U64 rowCount = 0;
        if (!validateTexture(transfer, rowCount)) {
            JST_ERROR("[VULKAN] Invalid texture transfer.");
            return Result::ERROR;
        }

        if (transfer.destinationSize.x > std::numeric_limits<U32>::max() ||
            transfer.destinationRow > std::numeric_limits<I32>::max() ||
            rowCount > std::numeric_limits<U32>::max()) {
            return Result::ERROR;
        }

        U64 sourceOffset = 0;
        const U64 pixelByteSize = transfer.rowByteSize / transfer.destinationSize.x;
        if (!reserveRange(required,
                          transfer.upload.data.size(),
                          std::max<U64>(4, pixelByteSize),
                          sourceOffset)) {
            return Result::ERROR;
        }

        textureCopies.push_back({destination,
                                 sourceOffset,
                                 transfer.destinationRow,
                                 rowCount,
                                 transfer.destinationSize.x,
                                 transfer.rowByteSize,
                                 transfer.upload.data.data()});
    }

    auto& arena = arenas[frameIndex];
    JST_CHECK(ensureCapacity(arena, required));

    for (const auto& copy : bufferCopies) {
        if (!copyBuffer(arena.mapped + copy.sourceOffset,
                        copy.source,
                        copy.size)) {
            return Result::ERROR;
        }
    }
    for (const auto& copy : textureCopies) {
        if (!copyTextureRows(arena.mapped + copy.sourceOffset,
                             copy.source,
                             copy.rowCount,
                             copy.rowByteSize,
                             copy.rowByteSize)) {
            return Result::ERROR;
        }
    }

    VkMemoryBarrier hostBarrier{};
    hostBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    hostBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    hostBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(commandBuffer,
                         VK_PIPELINE_STAGE_HOST_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &hostBarrier, 0, nullptr, 0, nullptr);

    if (!bufferCopies.empty()) {
        VkMemoryBarrier producerBarrier{};
        producerBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        producerBarrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT |
                                        VK_ACCESS_MEMORY_WRITE_BIT;
        producerBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(commandBuffer,
                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 1, &producerBarrier, 0, nullptr, 0, nullptr);
    }

    for (const auto& copy : bufferCopies) {
        VkBufferCopy region{};
        region.srcOffset = copy.sourceOffset;
        region.dstOffset = copy.destinationOffset;
        region.size = copy.size;
        vkCmdCopyBuffer(commandBuffer, arena.buffer, copy.destination->buffer, 1, &region);
    }

    size_t textureIndex = 0;
    while (textureIndex < textureCopies.size()) {
        auto& destination = textureCopies[textureIndex].destination;

        VkPipelineStageFlags sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        VkAccessFlags sourceAccess = 0;
        if (destination->layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            sourceStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            sourceAccess = VK_ACCESS_SHADER_READ_BIT;
        } else if (destination->layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
            sourceStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            sourceAccess = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        }

        VkImageMemoryBarrier before{};
        before.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        before.srcAccessMask = sourceAccess;
        before.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        before.oldLayout = destination->layout;
        before.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        before.image = destination->texture;
        before.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        before.subresourceRange.baseMipLevel = 0;
        before.subresourceRange.levelCount = 1;
        before.subresourceRange.baseArrayLayer = 0;
        before.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(commandBuffer,
                             sourceStage,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &before);

        size_t nextTexture = textureIndex;
        while (nextTexture < textureCopies.size() &&
               textureCopies[nextTexture].destination == destination) {
            const auto& copy = textureCopies[nextTexture];
            VkBufferImageCopy region{};
            region.bufferOffset = copy.sourceOffset;
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.mipLevel = 0;
            region.imageSubresource.baseArrayLayer = 0;
            region.imageSubresource.layerCount = 1;
            region.imageOffset = {0, static_cast<I32>(copy.destinationRow), 0};
            region.imageExtent = {
                static_cast<U32>(copy.width),
                static_cast<U32>(copy.rowCount),
                1,
            };
            vkCmdCopyBufferToImage(commandBuffer,
                                   arena.buffer,
                                   destination->texture,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   1,
                                   &region);
            ++nextTexture;
        }

        VkImageMemoryBarrier after = before;
        after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        after.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        after.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        after.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vkCmdPipelineBarrier(commandBuffer,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &after);

        textureIndex = nextTexture;
    }

    if (!bufferCopies.empty()) {
        VkMemoryBarrier consumerBarrier{};
        consumerBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        consumerBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        consumerBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                        VK_ACCESS_SHADER_WRITE_BIT |
                                        VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT |
                                        VK_ACCESS_INDEX_READ_BIT |
                                        VK_ACCESS_INDIRECT_COMMAND_READ_BIT |
                                        VK_ACCESS_UNIFORM_READ_BIT;
        vkCmdPipelineBarrier(commandBuffer,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                             VK_PIPELINE_STAGE_VERTEX_INPUT_BIT |
                             VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                             VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                             0, 1, &consumerBarrier, 0, nullptr, 0, nullptr);
    }

    return Result::SUCCESS;
}

void Implementation::commit(const Transfer::Batch& batch) {
    for (const auto& transfer : batch.textures()) {
        auto texture = std::dynamic_pointer_cast<TextureImp<DeviceType::Vulkan>>(
            transfer.destination);
        if (texture) {
            texture->layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
    }
}

void Implementation::destroyArena(Arena& arena) {
    if (arena.buffer == VK_NULL_HANDLE) {
        return;
    }

    auto& device = Backend::State<DeviceType::Vulkan>()->getDevice();
    vkUnmapMemory(device, arena.memory);
    vkDestroyBuffer(device, arena.buffer, nullptr);
    vkFreeMemory(device, arena.memory, nullptr);
    arena = {};
}

void Implementation::destroy() {
    for (auto& arena : arenas) {
        destroyArena(arena);
    }
    arenas.clear();
}

}  // namespace Jetstream::Render
