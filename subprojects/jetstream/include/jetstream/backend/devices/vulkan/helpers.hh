#ifndef JETSTREAM_BACKEND_DEVICE_VULKAN_HELPERS_HH
#define JETSTREAM_BACKEND_DEVICE_VULKAN_HELPERS_HH

#include <set>
#include <optional>
#include <vector>
#include <thread>
#include <functional>

#include "jetstream/types.hh"
#include "jetstream/macros.hh"

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

namespace Jetstream::Backend {

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicFamily;
    std::optional<uint32_t> computeFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() {
        return graphicFamily.has_value() &&
               computeFamily.has_value() &&
               presentFamily.has_value();
    }
};

inline QueueFamilyIndices FindQueueFamilies(const VkPhysicalDevice& device) {
    QueueFamilyIndices indices;

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    int i = 0;
    for (const auto& queueFamily : queueFamilies) {
        if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            indices.graphicFamily = i;
        }

        if (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT) {
            indices.computeFamily = i;
        }

        indices.presentFamily = i;

        if (indices.isComplete()) {
            break;
        }

        i++;
    }

    return indices;
}

inline U32 FindMemoryType(const VkPhysicalDevice& device, const U32& typeFilter, const VkMemoryPropertyFlags& properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(device, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    JST_FATAL("[VULKAN] Failed to find suitable memory type!");
    JST_CHECK_THROW(Result::ERROR);

    return 0;
}

inline VkShaderModule LoadShader(const U8& code, const U64& size, VkDevice device) {
    VkShaderModule shaderModule;
    VkShaderModuleCreateInfo moduleCreateInfo{};
    moduleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    moduleCreateInfo.codeSize = size;
    moduleCreateInfo.pCode = reinterpret_cast<const U32*>(&code);

    JST_VK_CHECK_THROW(vkCreateShaderModule(device, &moduleCreateInfo, nullptr, &shaderModule), [&]{
        JST_FATAL("[VULKAN] Can't create shader module.");  
    });

    return shaderModule;
}

inline Result ExecuteOnce(VkDevice& device,
                          VkQueue& queue,
                          VkCommandBuffer& commandBuffer,
                          VkCommandPool& commandPool,
                          std::function<void(VkCommandBuffer&)> func) {
    VkCommandBufferBeginInfo cmdBeginInfo = {};
    cmdBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cmdBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    JST_VK_CHECK(vkBeginCommandBuffer(commandBuffer, &cmdBeginInfo), [&]{
        JST_FATAL("[VULKAN] Failed to begin one time command buffer.");    
    });

    func(commandBuffer);

    JST_VK_CHECK(vkEndCommandBuffer(commandBuffer), [&]{
        JST_FATAL("[VULKAN] Failed to end one time command buffer.");   
    });

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext = nullptr;
    submitInfo.waitSemaphoreCount = 0;
    submitInfo.pWaitSemaphores = nullptr;
    submitInfo.pWaitDstStageMask = nullptr;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    submitInfo.signalSemaphoreCount = 0;
    submitInfo.pSignalSemaphores = nullptr;

    VkFence fence;
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    JST_VK_CHECK(vkCreateFence(device, &fenceInfo, nullptr, &fence), [&]{
        JST_FATAL("[VULKAN] Can't create one time fence.");            
    });

    JST_VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, fence), [&]{
        JST_FATAL("[VULKAN] Can't submit one time queue.");            
    });

    vkWaitForFences(device, 1, &fence, true, UINT64_MAX);
    vkResetFences(device, 1, &fence);
    vkDestroyFence(device, fence, nullptr);

    vkResetCommandPool(device, commandPool, 0);

    return Result::SUCCESS;
}

}  // namespace Jetstream::Backend

#endif