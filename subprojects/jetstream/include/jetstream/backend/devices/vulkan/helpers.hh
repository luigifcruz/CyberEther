#ifndef JETSTREAM_BACKEND_DEVICE_VULKAN_HELPERS_HH
#define JETSTREAM_BACKEND_DEVICE_VULKAN_HELPERS_HH

#include <set>
#include <optional>
#include <vector>

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicFamily;
    std::optional<uint32_t> computeFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete(bool headlessEnabled = false) {
        if (headlessEnabled) {
            return computeFamily.has_value() &&
                   graphicFamily.has_value();
        } else {
            return graphicFamily.has_value() &&
                   computeFamily.has_value() &&
                   presentFamily.has_value();
        }
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

        if (indices.isComplete()) {
            break;
        }

        i++;
    }

    return indices;
}

#endif