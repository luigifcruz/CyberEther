#include "jetstream/viewport/platforms/headless/vulkan.hh"
#include "jetstream/backend/devices/vulkan/helpers.hh"

namespace Jetstream::Viewport {

using Implementation = Headless<Device::Vulkan>;

Implementation::Headless(const Config& config) : Adapter(config) {
    JST_DEBUG("[VULKAN] Creating Headless viewport.");
}

Implementation::~Headless() {
    JST_DEBUG("[VULKAN] Destroying Headless viewport.");
}

Result Implementation::create() {
    swapchainImageFormat = VK_FORMAT_B8G8R8A8_UNORM;
    _currentDrawableIndex = 0;

    JST_CHECK(createSwapchain());

    framebufferPipe = std::ofstream("/tmp/cefb", std::ios::binary | std::ios::out);

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_CHECK(destroySwapchain());

    return Result::SUCCESS;
}

Result Implementation::createSwapchain() {
    auto& physicalDevice = Backend::State<Device::Vulkan>()->getPhysicalDevice();
    auto& device = Backend::State<Device::Vulkan>()->getDevice();

    // Create extent.

    swapchainExtent = {
        static_cast<U32>(config.size.width),
        static_cast<U32>(config.size.height)
    };

    // Create image.

    VkImageCreateInfo imageCreateInfo = {};
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.extent.width = config.size.width;
    imageCreateInfo.extent.height = config.size.height;
    imageCreateInfo.extent.depth = 1;
    imageCreateInfo.mipLevels = 1;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.format = swapchainImageFormat;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageCreateInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                            VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                            VK_IMAGE_USAGE_SAMPLED_BIT;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    JST_VK_CHECK(vkCreateImage(device, &imageCreateInfo, nullptr, &swapchainImage), [&]{
        JST_ERROR("[VULKAN] Failed to create swapchain image.");   
    });

    // Allocate backing memory.

    VkMemoryRequirements memoryRequirements;
    vkGetImageMemoryRequirements(device, swapchainImage, &memoryRequirements);

    VkMemoryAllocateInfo memoryAllocateInfo = {};
    memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryAllocateInfo.allocationSize = memoryRequirements.size;
    memoryAllocateInfo.memoryTypeIndex = Backend::FindMemoryType(physicalDevice,
                                                                 memoryRequirements.memoryTypeBits,
                                                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    JST_VK_CHECK(vkAllocateMemory(device, &memoryAllocateInfo, nullptr, &swapchainMemory), [&]{
        JST_ERROR("[VULKAN] Failed to allocate swapchain image memory.");
    });

    JST_VK_CHECK(vkBindImageMemory(device, swapchainImage, swapchainMemory, 0), [&]{
        JST_ERROR("[VULKAN] Failed to bind memory to the swapchain image.");
    });

    // Create image view.

    swapchainImageViews.resize(1);

    VkImageViewCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    createInfo.image = swapchainImage;
    createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    createInfo.format = swapchainImageFormat;

    createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

    createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    createInfo.subresourceRange.baseMipLevel = 0;
    createInfo.subresourceRange.levelCount = 1;
    createInfo.subresourceRange.baseArrayLayer = 0;
    createInfo.subresourceRange.layerCount = 1;

    JST_VK_CHECK(vkCreateImageView(device, &createInfo, NULL, &swapchainImageViews[0]), [&]{
        JST_ERROR("[VULKAN] Failed to create swapchain image view."); 
    });

    return Result::SUCCESS;
}

Result Implementation::destroySwapchain() {
    auto& device = Backend::State<Device::Vulkan>()->getDevice();

    vkDestroyImageView(device, swapchainImageViews[0], nullptr);
    vkDestroyImage(device, swapchainImage, nullptr);
    vkFreeMemory(device, swapchainMemory, nullptr);

    return Result::SUCCESS;
}

Result Implementation::createImgui() {
    // TODO: Implement interaction.

    return Result::SUCCESS;
}

F32 Implementation::scale(const F32& scale) const {
    // TODO: Implement HiDPI support.

    return scale;
}

Result Implementation::destroyImgui() {
    return Result::SUCCESS;
}

Result Implementation::nextDrawable(VkSemaphore&) {
    // Dump framebuffer image to disk.

    auto& backend = Backend::State<Device::Vulkan>();

    JST_CHECK(Backend::ExecuteOnce(backend->getDevice(),
                                   backend->getComputeQueue(),
                                   backend->getDefaultFence(),
                                   backend->getDefaultCommandBuffer(),
        [&](VkCommandBuffer& commandBuffer){
            JST_CHECK(Backend::TransitionImageLayout(commandBuffer, 
                                                     swapchainImage, 
                                                     VK_IMAGE_LAYOUT_UNDEFINED,
                                                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL));

            // Copy image to staging buffer.

            VkBufferImageCopy region{};
            region.bufferOffset = 0;
            region.bufferRowLength = 0;
            region.bufferImageHeight = 0;
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.mipLevel = 0;
            region.imageSubresource.baseArrayLayer = 0;
            region.imageSubresource.layerCount = 1;
            region.imageOffset = {
                    0,
                    0,
                    0
                };
            region.imageExtent = {
                    static_cast<U32>(config.size.width),
                    static_cast<U32>(config.size.height),
                    1
                };

            vkCmdCopyImageToBuffer(
                commandBuffer,
                swapchainImage,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                backend->getStagingBuffer(),
                1,
                &region
            );

            JST_CHECK(Backend::TransitionImageLayout(commandBuffer, 
                                                     swapchainImage, 
                                                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));

            return Result::SUCCESS;
        }
    ));

    // Save staging buffer memory to disk.

    void* data = backend->getStagingBufferMappedMemory();
    framebufferPipe.write(static_cast<char*>(data), config.size.width * config.size.height * 4);
    
    // Block execution for 16ms.
    std::this_thread::sleep_for(std::chrono::milliseconds(14));

    // Update ImGui state.
    
    ImGui::GetIO().DisplaySize = ImVec2(config.size.width, config.size.height);

    JST_INFO("FRAME!");

    return Result::SUCCESS;
}

Result Implementation::commitDrawable(std::vector<VkSemaphore>&) {
    // TODO: Add synchronization.

    return Result::SUCCESS;
}

const VkFormat& Implementation::getSwapchainImageFormat() const {
    return swapchainImageFormat;
}

std::vector<VkImageView>& Implementation::getSwapchainImageViews() {
    return swapchainImageViews;
}

const VkExtent2D& Implementation::getSwapchainExtent() const {
    return swapchainExtent;       
}

Result Implementation::pollEvents() {
    // TODO: Implement input handling.
    return Result::SUCCESS;
}

bool Implementation::keepRunning() {
    return true;
}

}  // namespace Jetstream::Viewport 