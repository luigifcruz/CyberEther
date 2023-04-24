#ifndef JETSTREAM_BACKEND_DEVICE_VULKAN_HH
#define JETSTREAM_BACKEND_DEVICE_VULKAN_HH

#include <set>

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#include "jetstream/backend/config.hh"

#ifndef JST_VK_CHECK
#define JST_VK_CHECK(f, callback) { \
    VkResult err = (f); \
    if (err != VK_SUCCESS) { \
    callback(); \
    return Result::ERROR; \
    } \
}
#endif  // JST_VK_CHECK

#ifndef JST_VK_CHECK_THROW
#define JST_VK_CHECK_THROW(f, callback) { \
    VkResult err = (f); \
    if (err != VK_SUCCESS) { \
    callback(); \
    JST_CHECK_THROW(Result::ERROR); \
    } \
}
#endif  // JST_VK_CHECK_THROW

namespace Jetstream::Backend {

class Vulkan {
 public:
    explicit Vulkan(const Config& config);
    ~Vulkan();

    const std::string getDeviceName() const;
    const std::string getApiVersion() const;
    const PhysicalDeviceType getPhysicalDeviceType() const;
    const bool hasUnifiedMemory() const;
    const U64 physicalMemory() const;
    const U64 getTotalProcessorCount() const;
    const bool getLowPowerStatus() const;
    const U64 getThermalState() const;

    constexpr VkDevice& getDevice() {
        return device;
    }

 private:
    const Config& config;

    VkDevice device;
    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkPhysicalDeviceProperties properties;

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
    std::vector<const char*> getRequiredExtensions();
    std::vector<const char*> getRequiredDeviceExtensions();
    std::vector<const char*> getRequiredValidationLayers();
    bool checkDeviceExtensionSupport(const VkPhysicalDevice& device);
    bool isDeviceSuitable(const VkPhysicalDevice& device);
};

}  // namespace Jetstream::Backend

#endif
