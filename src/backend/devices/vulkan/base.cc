#include <fstream>

#include "jetstream/logger.hh"

#include "jetstream/backend/devices/vulkan/base.hh"
#include "jetstream/backend/devices/vulkan/helpers.hh"

#include "../selection.hh"

namespace Jetstream::Backend {

static VKAPI_ATTR VkBool32 VKAPI_CALL debugMessageCallback(VkDebugReportFlagsEXT,
                                                           VkDebugReportObjectTypeEXT,
                                                           uint64_t,
                                                           size_t,
                                                           int32_t,
                                                           const char*,
                                                           const char *pMessage,
                                                           void*) {
    JST_DEBUG("[VULKAN] {}", pMessage);
    return VK_FALSE;
}

std::set<std::string> Vulkan::getRequiredInstanceExtensions() {
    std::set<std::string> extensions;

    // Presentation extensions.

    if (!config.headless) {
        extensions.insert(VK_KHR_SURFACE_EXTENSION_NAME);

#if defined(JST_OS_LINUX)
        extensions.insert(VK_KHR_XCB_SURFACE_EXTENSION_NAME);
        if (Backend::WindowMightBeWayland()) {
            extensions.insert(VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME);
        }
#endif
#if defined(JST_OS_MAC) || defined(JST_OS_IOS)
        extensions.insert(VK_EXT_METAL_SURFACE_EXTENSION_NAME);
#endif
#if defined(JST_OS_WINDOWS)
        extensions.insert(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#endif
#if defined(JST_OS_ANDROID)
        extensions.insert(VK_KHR_ANDROID_SURFACE_EXTENSION_NAME);
#endif
    }

    // Validation extensions.

    if (config.validationEnabled) {
        extensions.insert(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
    }

    return extensions;
}

std::set<std::string> Vulkan::getOptionalInstanceExtensions() {
    std::set<std::string> extensions;

#if defined(VK_KHR_portability_enumeration)
    extensions.insert(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#endif

    return extensions;
}

std::set<std::string> Vulkan::getRequiredValidationLayers() {
    std::set<std::string> layers;

    layers.insert("VK_LAYER_KHRONOS_validation");

    return layers;
}

std::set<std::string> Vulkan::getRequiredDeviceExtensions() {
    std::set<std::string> extensions;

    if (!config.headless) {
        extensions.insert(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }

    return extensions;
}

std::set<std::string> Vulkan::getOptionalDeviceExtensions() {
    std::set<std::string> extensions;

    if (!config.headless) {
        extensions.insert(VK_KHR_PRESENT_ID_EXTENSION_NAME);
        extensions.insert(VK_KHR_PRESENT_WAIT_EXTENSION_NAME);
    }

#if defined(VK_KHR_external_memory_fd)
    extensions.insert(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
    extensions.insert(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
#endif

#if defined(JST_OS_LINUX)
    extensions.insert(VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME);
    extensions.insert(VK_EXT_PHYSICAL_DEVICE_DRM_EXTENSION_NAME);
    extensions.insert(VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME);
#endif

#if defined(JST_OS_WINDOWS)
    extensions.insert(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
#endif

#if defined(VK_KHR_portability_subset)
    extensions.insert(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
#endif

    return extensions;
}

std::set<std::string> Vulkan::checkInstanceExtensionSupport(const std::set<std::string>& extensions) {
    uint32_t extensionCount;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, availableExtensions.data());

    std::set<std::string> supportedExtensions;

    for (const auto& extension : availableExtensions) {
        if (extensions.contains(extension.extensionName)) {
            supportedExtensions.insert(extension.extensionName);
        }
    }

    return supportedExtensions;
}

std::set<std::string> Vulkan::checkValidationLayerSupport(const std::set<std::string>& layers) {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    std::set<std::string> supportedLayers;

    for (const auto& layer : availableLayers) {
        if (layers.contains(layer.layerName)) {
            supportedLayers.insert(layer.layerName);
        }
    }

    return supportedLayers;
}

std::set<std::string> Vulkan::checkDeviceExtensionSupport(const VkPhysicalDevice& device, const std::set<std::string>& extensions) {
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

    std::set<std::string> supportedExtensions;

    for (const auto& extension : availableExtensions) {
        if (extensions.contains(extension.extensionName)) {
            supportedExtensions.insert(extension.extensionName);
        }
    }

    return supportedExtensions;
}

PhysicalDevice Vulkan::describePhysicalDevice(const VkPhysicalDevice& device, const VkPhysicalDeviceProperties& properties) {
    PhysicalDevice description;
    description.name = properties.deviceName;

    switch (properties.deviceType) {
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
            description.type = PhysicalDeviceType::INTEGRATED;
            break;
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
            description.type = PhysicalDeviceType::DISCRETE;
            break;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
            description.type = PhysicalDeviceType::VIRTUAL;
            break;
        case VK_PHYSICAL_DEVICE_TYPE_CPU:
            description.type = PhysicalDeviceType::CPU;
            break;
        case VK_PHYSICAL_DEVICE_TYPE_OTHER:
            description.type = PhysicalDeviceType::OTHER;
            break;
        default:
            description.type = PhysicalDeviceType::UNKNOWN;
            break;
    }

    if (properties.apiVersion < VK_API_VERSION_1_1) {
        return description;
    }

    const bool hasPciBusInfo = !checkDeviceExtensionSupport(device, {VK_EXT_PCI_BUS_INFO_EXTENSION_NAME}).empty();

    VkPhysicalDevicePCIBusInfoPropertiesEXT pciBusInfo{};
    pciBusInfo.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT;

    VkPhysicalDeviceIDProperties idProperties{};
    idProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
    idProperties.pNext = hasPciBusInfo ? &pciBusInfo : nullptr;

    VkPhysicalDeviceProperties2 properties2{};
    properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties2.pNext = &idProperties;

    vkGetPhysicalDeviceProperties2(device, &properties2);

    description.uuid = FormatDeviceUuid(idProperties.deviceUUID);

    if (hasPciBusInfo) {
        description.pciAddress = jst::fmt::format("{:04x}:{:02x}:{:02x}.{:x}",
                                                  pciBusInfo.pciDomain,
                                                  pciBusInfo.pciBus,
                                                  pciBusInfo.pciDevice,
                                                  pciBusInfo.pciFunction);
#if defined(JST_OS_LINUX)
        if (description.type == PhysicalDeviceType::DISCRETE) {
            std::ifstream removable(jst::fmt::format("/sys/bus/pci/devices/{}/removable", description.pciAddress));
            std::string value;
            if (removable && std::getline(removable, value) && value == "removable") {
                description.type = PhysicalDeviceType::EXTERNAL;
            }
        }
#endif
    }

    return description;
}

Vulkan::Vulkan(const Config& _config) : config(_config), cache({}) {
    // Gather instance extensions.

    std::vector<std::string> instanceExtensions;

    {
        const auto& requiredExtensions = getRequiredInstanceExtensions();
        const auto& optionalExtensions = getOptionalInstanceExtensions();

        JST_DEBUG("[VULKAN] Required instance extensions: {}", requiredExtensions);
        JST_DEBUG("[VULKAN] Optional instance extensions: {}", optionalExtensions);

        supportedInstanceExtensions.merge(checkInstanceExtensionSupport(requiredExtensions));

        std::set<std::string> unsupportedInstanceExtensions;

        for (const auto& extension : requiredExtensions) {
            if (!supportedInstanceExtensions.contains(extension)) {
                unsupportedInstanceExtensions.insert(extension);
            }
        }

        if (!unsupportedInstanceExtensions.empty()) {
            JST_FATAL("[VULKAN] Required instance extensions are not supported: {}.", unsupportedInstanceExtensions);
            JST_CHECK_THROW(Result::FATAL);
        }

        supportedInstanceExtensions.merge(checkInstanceExtensionSupport(optionalExtensions));

        for (const auto& extension : optionalExtensions) {
            if (!supportedInstanceExtensions.contains(extension)) {
                JST_WARN("[VULKAN] Optional instance extension '{}' is not supported.", extension);
            }
        }

        instanceExtensions.insert(instanceExtensions.end(), supportedInstanceExtensions.begin(), supportedInstanceExtensions.end());
    }

    // Gather validation layers.

    std::vector<std::string> validationLayers;

    {
        const auto& requiredValidationLayers = getRequiredValidationLayers();
        const auto& supportedValidationLayers = checkValidationLayerSupport(requiredValidationLayers);
        const auto& validationLayerCheck = requiredValidationLayers.size() == supportedValidationLayers.size();

        if (config.validationEnabled && !validationLayerCheck) {
            JST_WARN("[VULKAN] Couldn't find validation layers. Disabling Vulkan debug.");
            config.validationEnabled = false;
        }

        validationLayers.insert(validationLayers.end(), supportedValidationLayers.begin(), supportedValidationLayers.end());
    }

    // Create application.

    {
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Jetstream";
        appInfo.applicationVersion = VK_MAKE_VERSION(0, 0, 1);
        appInfo.pEngineName = "Jetstream Vulkan Backend";
        appInfo.engineVersion = VK_MAKE_VERSION(0, 0, 1);
        appInfo.apiVersion = VK_API_VERSION_1_1;

        VkInstanceCreateInfo instanceCreateInfo{};
        instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instanceCreateInfo.pApplicationInfo = &appInfo;
#if defined(VK_KHR_portability_enumeration)
        if (supportedInstanceExtensions.contains(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
            JST_DEBUG("[VULKAN] Enabling portability enumeration.");
            instanceCreateInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        }
#endif

        std::vector<const char*> vulkanInstanceExtensions(instanceExtensions.size());
        std::vector<const char*> vulkanValidationLayers(validationLayers.size());

        // Enable instance extensions.

        std::transform(instanceExtensions.begin(), instanceExtensions.end(), vulkanInstanceExtensions.begin(),
                       [](const std::string& str) { return str.c_str(); });

        instanceCreateInfo.enabledExtensionCount = static_cast<U32>(vulkanInstanceExtensions.size());
        instanceCreateInfo.ppEnabledExtensionNames = vulkanInstanceExtensions.data();
        instanceCreateInfo.enabledLayerCount = 0;

        // Enable validation layers.

        if (config.validationEnabled) {
            std::transform(validationLayers.begin(), validationLayers.end(), vulkanValidationLayers.begin(),
                           [](const std::string& str) { return str.c_str(); });

            instanceCreateInfo.enabledLayerCount = static_cast<U32>(vulkanValidationLayers.size());
            instanceCreateInfo.ppEnabledLayerNames = vulkanValidationLayers.data();
        }

        // Create instance.

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
            JST_CHECK_THROW(Result::FATAL);
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
            JST_CHECK_THROW(Result::FATAL);
        }

        std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
        JST_VK_CHECK_THROW(vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, physicalDevices.data()), [&]{
            JST_FATAL("[VULKAN] Can't enumerate physical devices.");
        });

        std::vector<VkPhysicalDevice> validPhysicalDevices;
        for (const auto& candidatePhysicalDevice : physicalDevices) {
            VkPhysicalDeviceProperties candidateProperties;
            vkGetPhysicalDeviceProperties(candidatePhysicalDevice, &candidateProperties);
            const auto& requiredExtensions = getRequiredDeviceExtensions();
            const auto& supportedExtensions = checkDeviceExtensionSupport(candidatePhysicalDevice, requiredExtensions);
            const auto& extensionCheck = requiredExtensions.size() == supportedExtensions.size();

            const auto& queueFamilyIndices = FindQueueFamilies(candidatePhysicalDevice);
            const auto& queueFamilyCheck = queueFamilyIndices.isComplete();

            auto candidate = describePhysicalDevice(candidatePhysicalDevice, candidateProperties);

            JST_INFO("[VULKAN] Candidate device '{}' ({}) - Extension check: {}, Queue family check: {}",
                     candidate.name, candidate.type,
                     extensionCheck ? "OK" : "FAIL", queueFamilyCheck ? "OK" : "FAIL");

            if (extensionCheck && queueFamilyCheck) {
                candidate.index = validPhysicalDevices.size();
                JST_INFO("[VULKAN] Device ID {}: {}", candidate.index, candidate.name);
                validPhysicalDevices.push_back(candidatePhysicalDevice);
                devices.push_back(std::move(candidate));
            }
        }
        if (validPhysicalDevices.size() == 0) {
            JST_FATAL("[VULKAN] No valid physical devices found.");
            JST_CHECK_THROW(Result::FATAL);
        }

        const auto chosen = ChooseDevice(devices, config.deviceId, config.deviceUuid);
        if (!chosen.has_value()) {
            JST_FATAL("[VULKAN] Can't find desired device ID ({}).", config.deviceId.value_or(0));
            JST_CHECK_THROW(Result::FATAL);
        }
        deviceIndex = chosen->index;
        selection = chosen->selection;
        physicalDevice = validPhysicalDevices[deviceIndex];

        const auto& selected = devices[deviceIndex];
        switch (selection) {
            case DeviceSelection::Override:
                JST_INFO("[VULKAN] Using '{}' from the requested device index.", selected.name);
                break;
            case DeviceSelection::Saved:
                JST_INFO("[VULKAN] Using the saved GPU '{}'.", selected.name);
                break;
            case DeviceSelection::Missing:
                JST_WARN("[VULKAN] The saved GPU ({}) is not connected. Using '{}' for this run.",
                         config.deviceUuid, selected.name);
                break;
            case DeviceSelection::Automatic:
                JST_INFO("[VULKAN] Using '{}' (automatic).", selected.name);
                break;
        }

        if (const auto discrete = UnusedDiscreteDevice(devices, chosen.value()); discrete.has_value()) {
            JST_INFO("[VULKAN] The discrete GPU '{}' is available but not in use. "
                     "Select it in Settings > General > Graphics Device and restart CyberEther.",
                     devices[discrete.value()].name);
        }

        vkGetPhysicalDeviceProperties(physicalDevice, &properties);
    }

    // Gather device extensions.

    std::vector<std::string> deviceExtensions;

    {
        const auto& requiredExtensions = getRequiredDeviceExtensions();
        const auto& optionalExtensions = getOptionalDeviceExtensions();

        JST_DEBUG("[VULKAN] Required device extensions: {}", requiredExtensions);
        JST_DEBUG("[VULKAN] Optional device extensions: {}", optionalExtensions);

        supportedDeviceExtensions.merge(checkDeviceExtensionSupport(physicalDevice, requiredExtensions));
        supportedDeviceExtensions.merge(checkDeviceExtensionSupport(physicalDevice, optionalExtensions));

        for (const auto& extension : optionalExtensions) {
            if (!supportedDeviceExtensions.contains(extension)) {
                JST_WARN("[VULKAN] Optional device extension '{}' is not supported.", extension);
            }
        }

        if (!supportedDeviceExtensions.contains(VK_KHR_PRESENT_ID_EXTENSION_NAME) ||
            !supportedDeviceExtensions.contains(VK_KHR_PRESENT_WAIT_EXTENSION_NAME)) {
            supportedDeviceExtensions.erase(VK_KHR_PRESENT_ID_EXTENSION_NAME);
            supportedDeviceExtensions.erase(VK_KHR_PRESENT_WAIT_EXTENSION_NAME);
        }

        deviceExtensions.insert(deviceExtensions.end(), supportedDeviceExtensions.begin(), supportedDeviceExtensions.end());
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
        cache.apiVersion = jst::fmt::format("{}.{}.{}", major, minor, patch);
    }

    cache.deviceUuid = devices[deviceIndex].uuid;
    cache.physicalDeviceType = devices[deviceIndex].type;

    {
        cache.hasUnifiedMemory = true;
        VkPhysicalDeviceMemoryProperties memoryProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
        for (uint32_t i = 0; i < memoryProperties.memoryHeapCount; i++) {
            VkMemoryHeap memoryHeap = memoryProperties.memoryHeaps[i];

            if (memoryHeap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                cache.physicalMemory += memoryHeap.size;
            }

            cache.hasUnifiedMemory &= (memoryHeap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT);
        }
    }

#if defined(JST_OS_WINDOWS)
    cache.canImportDeviceMemory = false;
    cache.canExportDeviceMemory = false;
#else
    cache.canImportDeviceMemory = supportedDeviceExtensions.contains(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
    cache.canExportDeviceMemory = supportedDeviceExtensions.contains(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
#endif
    cache.canImportHostMemory = supportedDeviceExtensions.contains(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
#if defined(JST_OS_LINUX)
    cache.canShareFrames = supportedDeviceExtensions.contains(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME) &&
                           supportedDeviceExtensions.contains(VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME) &&
                           supportedDeviceExtensions.contains(VK_EXT_PHYSICAL_DEVICE_DRM_EXTENSION_NAME) &&
                           supportedDeviceExtensions.contains(VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME);
#elif defined(JST_OS_WINDOWS)
    cache.canShareFrames = supportedDeviceExtensions.contains(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
#else
    cache.canShareFrames = false;
#endif

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

        // TODO: Add validation for these features.
        VkPhysicalDeviceFeatures supportedDeviceFeatures = {};
        vkGetPhysicalDeviceFeatures(physicalDevice, &supportedDeviceFeatures);

        cache.supportsMultiDrawIndirect = supportedDeviceFeatures.multiDrawIndirect;
        if (!cache.supportsMultiDrawIndirect) {
            JST_WARN("[VULKAN] VkPhysicalDeviceFeatures.multiDrawIndirect not supported. Falling back to single indirect draws.");
        }

        VkPhysicalDeviceFeatures deviceFeatures = {};
        deviceFeatures.multiDrawIndirect = cache.supportsMultiDrawIndirect;

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

        createInfo.queueCreateInfoCount = static_cast<U32>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.pEnabledFeatures = &deviceFeatures;

        VkPhysicalDevicePresentWaitFeaturesKHR presentWaitFeatures{};
        presentWaitFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES_KHR;
        VkPhysicalDevicePresentIdFeaturesKHR presentIdFeatures{};
        presentIdFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR;
        presentIdFeatures.pNext = &presentWaitFeatures;

        if (supportedDeviceExtensions.contains(VK_KHR_PRESENT_WAIT_EXTENSION_NAME)) {
            VkPhysicalDeviceFeatures2 features{};
            features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            features.pNext = &presentIdFeatures;
            vkGetPhysicalDeviceFeatures2(physicalDevice, &features);
            cache.supportsPresentWait = presentIdFeatures.presentId && presentWaitFeatures.presentWait;
            if (cache.supportsPresentWait) {
                createInfo.pNext = &presentIdFeatures;
            }
        }

        std::vector<const char*> vulkanDeviceExtensions(deviceExtensions.size());

        std::transform(deviceExtensions.begin(), deviceExtensions.end(), vulkanDeviceExtensions.begin(),
                       [](const std::string& str) { return str.c_str(); });

        createInfo.enabledExtensionCount = static_cast<U32>(vulkanDeviceExtensions.size());
        createInfo.ppEnabledExtensionNames = vulkanDeviceExtensions.data();
        createInfo.enabledLayerCount = 0;

        JST_VK_CHECK_THROW(vkCreateDevice(physicalDevice, &createInfo, nullptr, &device), [&]{
            JST_FATAL("[VULKAN] Can't create logical device.");
        });

        vkGetDeviceQueue(device, indices.graphicFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.computeFamily.value(), 0, &computeQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
    }

    // Validate multisampling level from configuration.

    {
        VkSampleCountFlags counts = properties.limits.framebufferColorSampleCounts &
                                    properties.limits.framebufferDepthSampleCounts;

        const U64 maxSamples = counts & VK_SAMPLE_COUNT_64_BIT ? 64 :
                               counts & VK_SAMPLE_COUNT_32_BIT ? 32 :
                               counts & VK_SAMPLE_COUNT_16_BIT ? 16 :
                               counts & VK_SAMPLE_COUNT_8_BIT  ?  8 :
                               counts & VK_SAMPLE_COUNT_4_BIT  ?  4 :
                               counts & VK_SAMPLE_COUNT_2_BIT  ?  2 : 1;

        if (config.multisampling > maxSamples) {
            JST_WARN("[VULKAN] Requested multisampling level ({}) is not supported. Using {} instead.", config.multisampling, maxSamples);
            config.multisampling = maxSamples;
        }
    }

    // Create descriptor pool.

    {
        VkDescriptorPoolSize poolSizes[] = {
            { VK_DESCRIPTOR_TYPE_SAMPLER, 8192 },
            { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 8192 },
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 8192 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 8192 },
            { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 8192 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 8192 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8192 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 8192 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 8192 },
            { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 8192 }
        };

        VkDescriptorPoolCreateInfo descriptorPoolInfo{};
        descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptorPoolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        descriptorPoolInfo.maxSets = 8192;
        descriptorPoolInfo.poolSizeCount = std::size(poolSizes);
        descriptorPoolInfo.pPoolSizes = poolSizes;

        JST_VK_CHECK_THROW(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool), [&]{
            JST_FATAL("[VULKAN] Can't create descriptor pool.")
        });
    }

    // Create pipeline cache.

    {
        VkPipelineCacheCreateInfo pipelineCacheInfo{};
        pipelineCacheInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;

        JST_VK_CHECK_THROW(vkCreatePipelineCache(device, &pipelineCacheInfo, nullptr, &pipelineCache), [&]{
            JST_FATAL("[VULKAN] Can't create pipeline cache.")
        });
    }

    // Create default command pool.

    {
        Backend::QueueFamilyIndices indices = Backend::FindQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = indices.computeFamily.value();
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT |
                         VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;

        JST_VK_CHECK_THROW(vkCreateCommandPool(device, &poolInfo, nullptr, &defaultCommandPool), [&]{
            JST_FATAL("[VULKAN] Failed to create default command pool.");
        });
    }

    // Create default command buffer.

    {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = defaultCommandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        JST_VK_CHECK_THROW(vkAllocateCommandBuffers(device, &allocInfo, &defaultCommandBuffer), [&]{
            JST_ERROR("[VULKAN] Failed to create default command buffer.");
        });
    }

    // Create default fence.

    {
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

        JST_VK_CHECK_THROW(vkCreateFence(device, &fenceInfo, nullptr, &defaultFence), [&]{
            JST_ERROR("[VULKAN] Failed to create default fence.");
        });
    }

    // Signal device is available.

    _isAvailable = true;

    // Print device information.

    JST_INFO("-----------------------------------------------------");
    JST_INFO("Jetstream Heterogeneous Backend [VULKAN]")
    JST_INFO("-----------------------------------------------------");
    JST_INFO("Device ID:        {}", getDeviceId());
    JST_INFO("Device Name:      {}", getDeviceName());
    JST_INFO("Device Type:      {}", getPhysicalDeviceType());
    JST_INFO("API Version:      {}", getApiVersion());
    JST_INFO("Unified Memory:   {}", hasUnifiedMemory() ? "YES" : "NO");
    JST_INFO("Processor Count:  {}", getTotalProcessorCount());
    JST_INFO("Device Memory:    {:.2f} GB", static_cast<F32>(getPhysicalMemory()) / (1024*1024*1024));
    JST_INFO("Interoperability:");
    JST_INFO("  - Can Import Device Memory: {}", canImportDeviceMemory() ? "YES" : "NO");
    JST_INFO("  - Can Export Device Memory: {}", canExportDeviceMemory() ? "YES" : "NO");
    JST_INFO("  - Can Import Host Memory:   {}", canImportHostMemory() ? "YES" : "NO");
    JST_INFO("  - Can Share Frames:         {}", canShareFrames() ? "YES" : "NO");
    JST_INFO("-----------------------------------------------------");
}

Vulkan::~Vulkan() {
    vkDestroyFence(device, defaultFence, nullptr);
    vkFreeCommandBuffers(device, defaultCommandPool, 1, &defaultCommandBuffer);
    vkDestroyCommandPool(device, defaultCommandPool, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    if (pipelineCache != VK_NULL_HANDLE) {
        vkDestroyPipelineCache(device, pipelineCache, nullptr);
    }

    if (debugReportCallback != VK_NULL_HANDLE) {
        PFN_vkDestroyDebugReportCallbackEXT vkDestroyDebugReportCallbackEXT =
                reinterpret_cast<PFN_vkDestroyDebugReportCallbackEXT>(
                    vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT"));
        vkDestroyDebugReportCallbackEXT(instance, debugReportCallback, nullptr);
    }

    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
}

VkSampleCountFlagBits Vulkan::getMultisampling() const {
    switch (config.multisampling) {
        case  1: return VK_SAMPLE_COUNT_1_BIT;
        case  2: return VK_SAMPLE_COUNT_2_BIT;
        case  4: return VK_SAMPLE_COUNT_4_BIT;
        case  8: return VK_SAMPLE_COUNT_8_BIT;
        case 16: return VK_SAMPLE_COUNT_16_BIT;
        case 32: return VK_SAMPLE_COUNT_32_BIT;
        case 64: return VK_SAMPLE_COUNT_64_BIT;
    }
    return VK_SAMPLE_COUNT_1_BIT;
}

bool Vulkan::isAvailable() const {
    return _isAvailable;
}

U64 Vulkan::getDeviceId() const {
    return deviceIndex;
}

std::string Vulkan::getDeviceUuid() const {
    return cache.deviceUuid;
}

DeviceSelection Vulkan::getDeviceSelection() const {
    return selection;
}

const std::vector<PhysicalDevice>& Vulkan::getDevices() const {
    return devices;
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

bool Vulkan::supportsMultiDrawIndirect() const {
    return cache.supportsMultiDrawIndirect;
}

bool Vulkan::supportsPresentWait() const {
    return cache.supportsPresentWait;
}

bool Vulkan::canExportDeviceMemory() const {
    return cache.canExportDeviceMemory;
}

bool Vulkan::canImportDeviceMemory() const {
    return cache.canImportDeviceMemory;
}

bool Vulkan::canImportHostMemory() const {
    return cache.canImportHostMemory;
}

bool Vulkan::canShareFrames() const {
    return cache.canShareFrames;
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
