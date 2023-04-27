#include "jetstream/logger.hh"

#include "jetstream/backend/devices/vulkan/base.hh"
#include "jetstream/backend/devices/vulkan/helpers.hh"
#include <vulkan/vulkan_core.h>

namespace Jetstream::Backend {

std::vector<const char*> Vulkan::getRequiredInstanceExtensions() {
    std::vector<const char*> extensions;

    extensions.push_back("VK_KHR_get_physical_device_properties2");

    if (!config.headless) {
        extensions.push_back("VK_KHR_surface");

#ifdef __linux__
        extensions.push_back("VK_KHR_xcb_surface");
#endif
        // TODO: Add Windows and Android support.
    }

    if (config.validationEnabled) {
        extensions.push_back("VK_EXT_debug_report");
    }

    return extensions;
}

std::vector<const char*> Vulkan::getRequiredValidationLayers() {
    std::vector<const char*> layers;

    layers.push_back("VK_LAYER_KHRONOS_validation");

    return layers;
}

std::vector<const char*> Vulkan::getRequiredDeviceExtensions() {
    std::vector<const char*> extensions;

    extensions.push_back("VK_EXT_memory_budget");

    return extensions;
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debugMessageCallback(VkDebugReportFlagsEXT flags,
                                                           VkDebugReportObjectTypeEXT objectType,
                                                           uint64_t object,
                                                           size_t location,
                                                           int32_t messageCode,
                                                           const char *pLayerPrefix,
                                                           const char *pMessage,
                                                           void *pUserData) {
    JST_DEBUG("[VULKAN] {}", pMessage);
    return VK_FALSE;
}

bool Vulkan::checkValidationLayerSupport() {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
    const auto validationLayers = getRequiredValidationLayers();
    std::set<std::string> requiredLayers(validationLayers.begin(), validationLayers.end());

    for (const auto& layer : availableLayers) {
        if (requiredLayers.count(layer.layerName) && config.validationEnabled) {
            JST_DEBUG("[VULKAN] Required layer found: {}", layer.layerName);
        }
        requiredLayers.erase(layer.layerName);
    }

    return requiredLayers.empty();
}

bool Vulkan::checkDeviceExtensionSupport(const VkPhysicalDevice& device) {
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());
    const auto deviceExtensions = getRequiredDeviceExtensions();
    std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

    for (const auto& extension : availableExtensions) {
        if (requiredExtensions.count(extension.extensionName) && config.validationEnabled) {
            JST_DEBUG("[VULKAN] Device supports required extension: {}", extension.extensionName);
        }
        requiredExtensions.erase(extension.extensionName);
    }

    auto indices = FindQueueFamilies(device, config.headless);

    return indices.isComplete(config.headless) && requiredExtensions.empty();
}

Vulkan::Vulkan(const Config& config) : config(config) {
    // Create application.

    {
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Jetstream";
        appInfo.applicationVersion = VK_MAKE_VERSION(0, 0, 1);
        appInfo.pEngineName = "Jetstream Vulkan Backend";
        appInfo.engineVersion = VK_MAKE_VERSION(0, 0, 1);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        // Create instance.

        VkInstanceCreateInfo instanceCreateInfo{};
        instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instanceCreateInfo.pApplicationInfo = &appInfo;

        if (config.validationEnabled && !checkValidationLayerSupport()) {
            JST_FATAL("[VULKAN] Couldn't find validation layers.");
            JST_CHECK_THROW(Result::VULKAN_ERROR);
        }

        const auto extensions = getRequiredInstanceExtensions();
        instanceCreateInfo.enabledExtensionCount = extensions.size();
        instanceCreateInfo.ppEnabledExtensionNames = extensions.data();
        instanceCreateInfo.enabledLayerCount = 0;

        const auto validationLayers = getRequiredValidationLayers();
        if (config.validationEnabled) {
            instanceCreateInfo.enabledLayerCount = validationLayers.size();
            instanceCreateInfo.ppEnabledLayerNames = validationLayers.data();
        }

        JST_VK_CHECK_THROW(vkCreateInstance(&instanceCreateInfo, nullptr, &instance), [&]{
            JST_FATAL("[VULKAN] Couldn't create instance.");        
        });
    }

    // Setup validation layers.

    if (config.validationEnabled) {
        VkDebugReportCallbackCreateInfoEXT debugReportCreateInfo{};
        debugReportCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
        debugReportCreateInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT;
        debugReportCreateInfo.pfnCallback = (PFN_vkDebugReportCallbackEXT)debugMessageCallback;

        PFN_vkCreateDebugReportCallbackEXT vkCreateDebugReportCallbackEXT =
            reinterpret_cast<PFN_vkCreateDebugReportCallbackEXT>(vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT"));
        if (!vkCreateDebugReportCallbackEXT) {
            JST_FATAL("[VULKAN] Failed to create validation.");
            JST_CHECK_THROW(Result::VULKAN_ERROR);       
        }
        JST_VK_CHECK_THROW(vkCreateDebugReportCallbackEXT(instance, &debugReportCreateInfo, nullptr, &debugReportCallback), [&]{
            JST_FATAL("[VULKAN] Failed to create validation.");
        });
    }

    // Get physical device.

    {
        U32 physicalDeviceCount = 0;
        JST_VK_CHECK_THROW(vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, nullptr), [&]{
            JST_FATAL("[VULKAN] Can't enumerate physical devices.");     
        });
        if (physicalDeviceCount == 0) {
            JST_FATAL("[VULKAN] No physical devices found.");
            JST_CHECK_THROW(Result::VULKAN_ERROR);
        }
        if (physicalDeviceCount <= config.deviceId) {
            JST_FATAL("[VULKAN] Can't find desired device ID.");
            JST_CHECK_THROW(Result::VULKAN_ERROR);
        }

        std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
        vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, physicalDevices.data());

        std::vector<VkPhysicalDevice> validPhysicalDevices;
        for (const auto& candidatePhysicalDevice : physicalDevices) {
            if (checkDeviceExtensionSupport(candidatePhysicalDevice)) {
                validPhysicalDevices.push_back(candidatePhysicalDevice);
            }
        }
        if (validPhysicalDevices.size() == 0) {
            JST_FATAL("[VULKAN] No valid physical devices found.");
            JST_CHECK_THROW(Result::VULKAN_ERROR);
        }
        if (validPhysicalDevices.size() <= config.deviceId) {
            JST_FATAL("[VULKAN] Can't find desired device ID.");
            JST_CHECK_THROW(Result::VULKAN_ERROR);
        }
        physicalDevice = validPhysicalDevices[config.deviceId];

        vkGetPhysicalDeviceProperties(physicalDevice, &properties);
    }

    // Populate information cache.

    cache.deviceName = properties.deviceName;
    cache.totalProcessorCount = std::thread::hardware_concurrency();
    cache.getThermalState = 0;  // TODO: Wire implementation.
    cache.lowPowerStatus = false;  // TODO: Wire implementation.

    {
        uint32_t major = VK_VERSION_MAJOR(properties.apiVersion);
        uint32_t minor = VK_VERSION_MINOR(properties.apiVersion);
        uint32_t patch = VK_VERSION_PATCH(properties.apiVersion);
        cache.apiVersion = fmt::format("{}.{}.{}", major, minor, patch);
    }

    {
        switch (properties.deviceType) {
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
                cache.physicalDeviceType = PhysicalDeviceType::INTEGRATED;
                break;
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                cache.physicalDeviceType = PhysicalDeviceType::DISCRETE;
                break;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
            case VK_PHYSICAL_DEVICE_TYPE_CPU:
            case VK_PHYSICAL_DEVICE_TYPE_OTHER:
                cache.physicalDeviceType = PhysicalDeviceType::OTHER;
                break;
            case VK_PHYSICAL_DEVICE_TYPE_MAX_ENUM:
                cache.physicalDeviceType = PhysicalDeviceType::UNKNOWN;
                break;
        }
    }

    {
        VkPhysicalDeviceMemoryProperties memoryProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
        for (uint32_t i = 0; i < memoryProperties.memoryHeapCount; i++) {
            VkMemoryHeap memoryHeap = memoryProperties.memoryHeaps[i];

            if (memoryHeap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                cache.physicalMemory += memoryHeap.size;
            }
        }
    }

    {
        VkPhysicalDeviceMemoryProperties2 memProperties = {};
        memProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
        memProperties.pNext = nullptr;

        vkGetPhysicalDeviceMemoryProperties2(physicalDevice, &memProperties);

        bool hasDeviceLocalMemory = false;
        bool hasHostVisibleMemory = false;

        for (uint32_t i = 0; i < memProperties.memoryProperties.memoryTypeCount; i++) {
            VkMemoryType memoryType = memProperties.memoryProperties.memoryTypes[i];

            if ((memoryType.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0) {
                hasDeviceLocalMemory = true;
            }

            if ((memoryType.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
                hasHostVisibleMemory = true;
            }
        }
        cache.hasUnifiedMemory = hasDeviceLocalMemory && hasHostVisibleMemory;
    }

    // Create logical device.

    {
        QueueFamilyIndices indices = FindQueueFamilies(physicalDevice, config.headless);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = {
            indices.graphicFamily.value(),
            indices.computeFamily.value(),
        };

        if (!config.headless) {
            uniqueQueueFamilies.insert(indices.presentFamily.value());
        }

        float queuePriority = 0.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        VkPhysicalDeviceFeatures deviceFeatures{};

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();

        createInfo.pEnabledFeatures = &deviceFeatures;

        const auto deviceExtensions = getRequiredDeviceExtensions();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        const auto validationLayers = getRequiredValidationLayers();
        if (config.validationEnabled) {
            createInfo.enabledLayerCount = validationLayers.size();
            createInfo.ppEnabledLayerNames = validationLayers.data();
        } else {
            createInfo.enabledLayerCount = 0;
        }

        JST_VK_CHECK_THROW(vkCreateDevice(physicalDevice, &createInfo, nullptr, &device), [&]{
            JST_FATAL("[VULKAN] Can't create logical device.");     
        });

        vkGetDeviceQueue(device, indices.graphicFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.computeFamily.value(), 0, &computeQueue);
        if (!config.headless) {
            vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
        }
    }

    // Print device information.

    JST_INFO("—————————————————————————————————————————————————————");
    JST_INFO("Jetstream Heterogeneous Backend [VULKAN]")
    JST_INFO("—————————————————————————————————————————————————————");
    JST_INFO("Device Name:     {}", getDeviceName());
    JST_INFO("Device Type:     {}", getPhysicalDeviceType());
    JST_INFO("API Version:     {}", getApiVersion())
    JST_INFO("Unified Memory:  {}", hasUnifiedMemory() ? "YES" : "NO");
    JST_INFO("Processor Count: {}", getTotalProcessorCount());
    JST_INFO("Physical Memory: {:.2f} GB", static_cast<F32>(getPhysicalMemory()) / (1024*1024*1024));
    JST_INFO("—————————————————————————————————————————————————————");
}

Vulkan::~Vulkan() {
    if (debugReportCallback != VK_NULL_HANDLE) {
        PFN_vkDestroyDebugReportCallbackEXT vkDestroyDebugReportCallbackEXT =
                reinterpret_cast<PFN_vkDestroyDebugReportCallbackEXT>(
                    vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT"));
        vkDestroyDebugReportCallbackEXT(instance, debugReportCallback, nullptr);
    }

    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
}

const std::string Vulkan::getDeviceName() const {
    return cache.deviceName;
}

const std::string Vulkan::getApiVersion() const {
    return cache.apiVersion;       
}

const PhysicalDeviceType Vulkan::getPhysicalDeviceType() const {
    return cache.physicalDeviceType; 
}

const bool Vulkan::hasUnifiedMemory() const {
    return cache.hasUnifiedMemory;
}

const U64 Vulkan::getPhysicalMemory() const {
    return cache.physicalMemory;
}

const U64 Vulkan::getTotalProcessorCount() const {
    return cache.totalProcessorCount;
}

const bool Vulkan::getLowPowerStatus() const {
    // TODO: Pool power status periodically.
    return cache.lowPowerStatus;
}

const U64 Vulkan::getThermalState() const {
    // TODO: Pool thermal state periodically.
    return cache.getThermalState;
}

}  // namespace Jetstream::Backend
