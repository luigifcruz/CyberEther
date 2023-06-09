#include "jetstream/logger.hh"

#include "jetstream/backend/devices/vulkan/base.hh"
#include "jetstream/backend/devices/vulkan/helpers.hh"

namespace Jetstream::Backend {

std::vector<const char*> Vulkan::getRequiredInstanceExtensions() {
    std::vector<const char*> extensions;

    extensions.push_back("VK_KHR_get_physical_device_properties2");
    extensions.push_back("VK_KHR_surface");

#if defined(__linux__)
    extensions.push_back("VK_KHR_xcb_surface");
    extensions.push_back("VK_KHR_wayland_surface");
#endif
#if defined(__APPLE__)
    extensions.push_back("VK_MVK_macos_surface");
    extensions.push_back("VK_EXT_metal_surface");
#endif
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
    extensions.push_back("VK_KHR_win32_surface");
#endif
#if defined(__ANDROID__)
    extensions.push_back("VK_KHR_android_surface");
#endif

    if (config.validationEnabled) {
        extensions.push_back("VK_EXT_debug_report");
    }

    JST_DEBUG("[VULKAN] Requested Extensions: {}", extensions);

    return extensions;
}

std::vector<const char*> Vulkan::getRequiredValidationLayers() {
    std::vector<const char*> layers;

    layers.push_back("VK_LAYER_KHRONOS_validation");

    JST_DEBUG("[VULKAN] Requested Validation Layers: {}", layers);

    return layers;
}

std::vector<const char*> Vulkan::getRequiredDeviceExtensions() {
    std::vector<const char*> extensions;

    extensions.push_back("VK_EXT_memory_budget");
    extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

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

    auto indices = FindQueueFamilies(device);

    return indices.isComplete() && requiredExtensions.empty();
}

Vulkan::Vulkan(const Config& _config) : config(_config) {
    // Create application.

    {
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Jetstream";
        appInfo.applicationVersion = VK_MAKE_VERSION(0, 0, 1);
        appInfo.pEngineName = "Jetstream Vulkan Backend";
        appInfo.engineVersion = VK_MAKE_VERSION(0, 0, 1);

        // Reasons why this is Vulkan 1.1:
        // 1. Negative viewport support for compatibility with Metal.
        appInfo.apiVersion = VK_API_VERSION_1_1;

        // Create instance.

        VkInstanceCreateInfo instanceCreateInfo{};
        instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instanceCreateInfo.pApplicationInfo = &appInfo;

        if (config.validationEnabled && !checkValidationLayerSupport()) {
            JST_WARN("[VULKAN] Couldn't find validation layers. Disabling Vulkan debug.");
            config.validationEnabled = false;
        }

        const auto extensions = getRequiredInstanceExtensions();
        instanceCreateInfo.enabledExtensionCount = extensions.size();
        instanceCreateInfo.ppEnabledExtensionNames = extensions.data();
        instanceCreateInfo.enabledLayerCount = 0;

        if (config.validationEnabled) {
            const auto validationLayers = getRequiredValidationLayers();
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
        QueueFamilyIndices indices = FindQueueFamilies(physicalDevice);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = {
            indices.graphicFamily.value(),
            indices.computeFamily.value(),
            indices.presentFamily.value(),
        };

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

        createInfo.queueCreateInfoCount = static_cast<U32>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();

        createInfo.pEnabledFeatures = &deviceFeatures;

        const auto deviceExtensions = getRequiredDeviceExtensions();
        createInfo.enabledExtensionCount = static_cast<U32>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        if (config.validationEnabled) {
            const auto validationLayers = getRequiredValidationLayers();
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
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
    }

    // Create descriptor pool.

    {
        VkDescriptorPoolSize poolSizes[] = {
            { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
            { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
            { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
        };

        VkDescriptorPoolCreateInfo descriptorPoolInfo{};
        descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptorPoolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        descriptorPoolInfo.maxSets = 1000;
        descriptorPoolInfo.poolSizeCount = std::size(poolSizes);
        descriptorPoolInfo.pPoolSizes = poolSizes;

        JST_VK_CHECK_THROW(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool), [&]{
            JST_FATAL("[VULKAN] Can't create descriptor pool.")
        });
    }

    // Create staging buffer.

    {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = config.stagingBufferSize;
        bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        JST_VK_CHECK_THROW(vkCreateBuffer(device, &bufferInfo, nullptr, &stagingBuffer), [&]{
            JST_FATAL("[VULKAN] Failed to create staging buffer.");     
        });

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, stagingBuffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = Backend::FindMemoryType(physicalDevice,
                                                            memRequirements.memoryTypeBits,
                                                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        JST_VK_CHECK_THROW(vkAllocateMemory(device, &allocInfo, nullptr, &stagingBufferMemory), [&]{
            JST_FATAL("[VULKAN] Failed to allocate staging buffer memory.");
        });

        JST_VK_CHECK_THROW(vkBindBufferMemory(device, stagingBuffer, stagingBufferMemory, 0), [&]{
            JST_FATAL("[VULKAN] Failed to bind memory to staging buffer.");
        });
    }

    // Create transfer pool.

    {
        Backend::QueueFamilyIndices indices = Backend::FindQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = indices.computeFamily.value();
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT |
                         VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;

        JST_VK_CHECK_THROW(vkCreateCommandPool(device, &poolInfo, nullptr, &transferCommandPool), [&]{
            JST_FATAL("[VULKAN] Failed to create transfer command pool.");
        });
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
    JST_INFO("Staging Buffer:  {:.2f} MB", static_cast<F32>(config.stagingBufferSize) / (1024*1024));
    JST_INFO("—————————————————————————————————————————————————————");
}

Vulkan::~Vulkan() {
    vkDestroyCommandPool(device, transferCommandPool, nullptr);
    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);

    if (debugReportCallback != VK_NULL_HANDLE) {
        PFN_vkDestroyDebugReportCallbackEXT vkDestroyDebugReportCallbackEXT =
                reinterpret_cast<PFN_vkDestroyDebugReportCallbackEXT>(
                    vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT"));
        vkDestroyDebugReportCallbackEXT(instance, debugReportCallback, nullptr);
    }

    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
}

std::string Vulkan::getDeviceName() const {
    return cache.deviceName;
}

std::string Vulkan::getApiVersion() const {
    return cache.apiVersion;       
}

PhysicalDeviceType Vulkan::getPhysicalDeviceType() const {
    return cache.physicalDeviceType; 
}

bool Vulkan::hasUnifiedMemory() const {
    return cache.hasUnifiedMemory;
}

U64 Vulkan::getPhysicalMemory() const {
    return cache.physicalMemory;
}

U64 Vulkan::getTotalProcessorCount() const {
    return cache.totalProcessorCount;
}

bool Vulkan::getLowPowerStatus() const {
    // TODO: Pool power status periodically.
    return cache.lowPowerStatus;
}

U64 Vulkan::getThermalState() const {
    // TODO: Pool thermal state periodically.
    return cache.getThermalState;
}

}  // namespace Jetstream::Backend
