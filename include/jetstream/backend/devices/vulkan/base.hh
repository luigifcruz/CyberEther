#ifndef JETSTREAM_BACKEND_DEVICE_VULKAN_HH
#define JETSTREAM_BACKEND_DEVICE_VULKAN_HH

#include <set>

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#include "jetstream/backend/config.hh"

namespace Jetstream::Backend {

class Vulkan {
 public:
    explicit Vulkan(const Config& config);
    ~Vulkan();

    std::string getDeviceName() const;
    std::string getApiVersion() const;
    PhysicalDeviceType getPhysicalDeviceType() const;
    bool hasUnifiedMemory() const;
    U64 getPhysicalMemory() const;
    U64 getTotalProcessorCount() const;
    bool getLowPowerStatus() const;
    U64 getThermalState() const;

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

    constexpr VkDeviceMemory& getStagingBufferMemory() {
        return stagingBufferMemory;       
    }

    constexpr VkBuffer& getStagingBuffer() {
        return stagingBuffer;       
    }

    constexpr const U64& getStagingBufferSize() {
         return config.stagingBufferSize;       
    }

    constexpr VkCommandPool& getTransferCommandPool() {
         return transferCommandPool;
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
    VkCommandPool transferCommandPool;
    VkQueue graphicsQueue;
    VkQueue computeQueue;
    VkQueue presentQueue;

    struct {
        std::string deviceName;
        std::string apiVersion;
        PhysicalDeviceType physicalDeviceType;
        bool hasUnifiedMemory;
        U64 physicalMemory;
        U64 totalProcessorCount;
        bool lowPowerStatus;
        U64 getThermalState;
    } cache;

    VkDebugReportCallbackEXT debugReportCallback{};
        
    bool checkValidationLayerSupport();
    std::vector<const char*> getRequiredInstanceExtensions();
    std::vector<const char*> getRequiredDeviceExtensions();
    std::vector<const char*> getRequiredValidationLayers();
    bool checkDeviceExtensionSupport(const VkPhysicalDevice& device);
    bool isDeviceSuitable(const VkPhysicalDevice& device);
};

}  // namespace Jetstream::Backend

#endif
