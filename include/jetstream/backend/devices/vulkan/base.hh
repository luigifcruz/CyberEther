#ifndef JETSTREAM_BACKEND_DEVICE_VULKAN_HH
#define JETSTREAM_BACKEND_DEVICE_VULKAN_HH

#include <set>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_beta.h>
#if defined(JST_OS_LINUX)
#include <xcb/xcb.h>
#include <vulkan/vulkan_xcb.h>
#include <vulkan/vulkan_wayland.h>
#endif
#if defined(JST_OS_WINDOWS)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef ERROR
#undef FATAL
#include <vulkan/vulkan_win32.h>
#endif
#if defined(JST_OS_MAC) || defined(JST_OS_IOS)
#include <vulkan/vulkan_metal.h>
#endif
#if defined(JST_OS_ANDROID)
#include <vulkan/vulkan_android.h>
#endif

#include "jetstream/backend/config.hh"

namespace Jetstream::Backend {

class Vulkan {
 public:
    explicit Vulkan(const Config& config);
    ~Vulkan();

    bool isAvailable() const;
    std::string getDeviceName() const;
    std::string getApiVersion() const;
    PhysicalDeviceType getPhysicalDeviceType() const;
    VkSampleCountFlagBits getMultisampling() const;
    U64 getPhysicalMemory() const;
    U64 getTotalProcessorCount() const;
    bool getLowPowerStatus() const;
    U64 getThermalState() const;

    bool hasUnifiedMemory() const;
    bool canExportDeviceMemory() const;
    bool canImportDeviceMemory() const;
    bool canImportHostMemory() const;

    constexpr const U64& getDeviceId() const {
        return config.deviceId;
    }

    constexpr const bool& headless() const {
        return config.headless;
    }

    constexpr VkDevice& getDevice() {
        return device;
    }

    constexpr VkPhysicalDevice& getPhysicalDevice() {
        return physicalDevice;
    }

    constexpr VkInstance& getInstance() {
        return instance;
    }

    constexpr VkQueue& getGraphicsQueue() {
        return graphicsQueue;
    }

    constexpr VkQueue& getPresentQueue() {
        return presentQueue;
    }

    constexpr VkQueue& getComputeQueue() {
        return computeQueue;
    }

    constexpr VkDescriptorPool& getDescriptorPool() {
        return descriptorPool;
    }

    constexpr void* getStagingBufferMappedMemory() {
        return stagingBufferMappedMemory;       
    }

    constexpr VkDeviceMemory& getStagingBufferMemory() {
        return stagingBufferMemory;       
    }

    constexpr VkBuffer& getStagingBuffer() {
        return stagingBuffer;       
    }

    constexpr const U64& getStagingBufferSize() {
        return config.stagingBufferSize;       
    }

    constexpr VkFence& getDefaultFence() {
        return defaultFence;
    }

    constexpr VkCommandPool& getDefaultCommandPool() {
        return defaultCommandPool;
    }

    constexpr VkCommandBuffer& getDefaultCommandBuffer() {
        return defaultCommandBuffer;
    }

 private:
    Config config;
    VkDevice device;
    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkPhysicalDeviceProperties properties;
    VkDescriptorPool descriptorPool;
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    void* stagingBufferMappedMemory;
    VkFence defaultFence;
    VkCommandBuffer defaultCommandBuffer;
    VkCommandPool defaultCommandPool;
    VkQueue graphicsQueue;
    VkQueue computeQueue;
    VkQueue presentQueue;
    std::set<std::string> availableOptionalDeviceCapabilities;
    std::set<std::string> availableOptionalInstanceCapabilities;
    bool _isAvailable = false;

    struct {
        std::string deviceName;
        std::string apiVersion;
        PhysicalDeviceType physicalDeviceType;
        bool hasUnifiedMemory;
        U64 physicalMemory;
        U64 totalProcessorCount;
        bool lowPowerStatus;
        U64 getThermalState;
        bool canImportDeviceMemory;
        bool canImportHostMemory;
        bool canExportDeviceMemory;
    } cache;

    VkDebugReportCallbackEXT debugReportCallback{};
    
    std::vector<const char*> getRequiredInstanceExtensions();
    bool checkRequiredInstanceExtensionSupport();

    std::vector<const char*> getOptionalInstanceExtensions();
    std::set<std::string> checkOptionalInstanceExtensionSupport();

    std::vector<const char*> getRequiredValidationLayers();
    bool checkRequiredValidationLayerSupport();

    std::vector<std::string> getRequiredDeviceExtensions();
    bool checkRequiredDeviceExtensionSupport(const VkPhysicalDevice& device);

    std::vector<std::string> getOptionalDeviceExtensions();
    std::set<std::string> checkOptionalDeviceExtensionSupport();

    bool isDeviceSuitable(const VkPhysicalDevice& device);
};

}  // namespace Jetstream::Backend

#endif
