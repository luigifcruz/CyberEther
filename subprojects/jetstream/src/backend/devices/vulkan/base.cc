#include "jetstream/logger.hh"

#include "jetstream/backend/devices/vulkan/base.hh"
#include "jetstream/backend/devices/vulkan/helpers.hh"

namespace Jetstream::Backend {

std::vector<const char*> Vulkan::getRequiredExtensions() {
    std::vector<const char*> extensions;

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
    JST_DEBUG("[VULKAN] {} - {}", pLayerPrefix, pMessage);
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
            JST_INFO("[VULKAN] Required layer found: {}", layer.layerName);
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

    return requiredExtensions.empty();
}

Vulkan::Vulkan(const Config& config) : config(config) {
    JST_DEBUG("Initializing Vulkan backend.");

    // Create application.

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

    if (config.validationEnabled && checkValidationLayerSupport()) {
        JST_FATAL("[VULKAN] Couldn't find validation layers.");
        JST_CHECK_THROW(Result::VULKAN_ERROR);
    }

    const auto extensions = getRequiredExtensions();
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

    // Populate information cache.
    cache.deviceName = properties.deviceName;

    uint32_t major = VK_VERSION_MAJOR(properties.apiVersion);
    uint32_t minor = VK_VERSION_MINOR(properties.apiVersion);
    uint32_t patch = VK_VERSION_PATCH(properties.apiVersion);
    cache.apiVersion = fmt::format("{}.{}.{}", major, minor, patch);

    // PhysicalDeviceType physicalDeviceType;
    // bool hasUnifiedMemory;
    // U64 physicalMemory;
    // U64 totalProcessorCount;

    cache.getThermalState = 0;  // TODO: Wire implementation.
    cache.lowPowerStatus = false;  // TODO: Wire implementation.

    // Print device information.

    JST_INFO("—————————————————————————————————————————————————————");
    JST_INFO("Jetstream Heterogeneous Backend [VULKAN]")
    JST_INFO("—————————————————————————————————————————————————————");
    JST_INFO("Device Name: {}", getDeviceName());
    JST_INFO("API Version: {}", getApiVersion())
    JST_INFO("—————————————————————————————————————————————————————");
}

Vulkan::~Vulkan() {
    JST_DEBUG("Destroying Vulkan backend.");

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

const U64 Vulkan::physicalMemory() const {
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
