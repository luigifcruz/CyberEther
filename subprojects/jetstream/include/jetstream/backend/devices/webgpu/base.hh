#ifndef JETSTREAM_BACKEND_DEVICE_WEBGPU_HH
#define JETSTREAM_BACKEND_DEVICE_WEBGPU_HH

#include <set>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <memory>

#include <emscripten.h>
#include <emscripten/html5.h>
#include <webgpu/webgpu_cpp.h>
#include <emscripten/html5_webgpu.h>

#include "jetstream/backend/config.hh"

// #ifndef JST_WG_CHECK
// #define JST_WG_CHECK(f, callback) { \
//     VkResult err = (f); \
//     if (err != VK_SUCCESS) { \
//     callback(); \
//     JST_FATAL("[VULKAN] Error code: {}", string_VkResult(err)); \
//     return Result::ERROR; \
//     } \
// }
// #endif  // JST_WG_CHECK

// #ifndef JST_WG_CHECK_THROW
// #define JST_WG_CHECK_THROW(f, callback) { \
//     VkResult err = (f); \
//     if (err != VK_SUCCESS) { \
//     callback(); \
//     JST_FATAL("[VULKAN] Error code: {}", string_VkResult(err)); \
//     JST_CHECK_THROW(Result::ERROR); \
//     } \
// }
// #endif  // JST_WG_CHECK_THROW

namespace Jetstream::Backend {

class WebGPU {
 public:
    explicit WebGPU(const Config& config);
    ~WebGPU();

    std::string getDeviceName() const;
    std::string getApiVersion() const;
    PhysicalDeviceType getPhysicalDeviceType() const;
    bool hasUnifiedMemory() const;
    U64 getPhysicalMemory() const;
    U64 getTotalProcessorCount() const;
    bool getLowPowerStatus() const;
    U64 getThermalState() const;

    constexpr wgpu::Device& getDevice() {
        return device;
    }

    // constexpr VkPhysicalDevice& getPhysicalDevice() {
    //     return physicalDevice;
    // }

    // constexpr VkInstance& getInstance() {
    //     return instance;
    // }

    // constexpr VkQueue& getGraphicsQueue() {
    //     return graphicsQueue;
    // }

    // constexpr VkQueue& getPresentQueue() {
    //     return presentQueue;
    // }

    // constexpr VkQueue& getComputeQueue() {
    //     return computeQueue;
    // }

    // constexpr VkDescriptorPool& getDescriptorPool() {
    //     return descriptorPool;
    // }

    // constexpr VkDeviceMemory& getStagingBufferMemory() {
    //     return stagingBufferMemory;       
    // }

    // constexpr VkBuffer& getStagingBuffer() {
    //     return stagingBuffer;       
    // }

    // constexpr const U64& getStagingBufferSize() {
    //      return config.stagingBufferSize;       
    // }

    // constexpr VkCommandPool& getTransferCommandPool() {
    //      return transferCommandPool;
    // }

 private:
    Config config;

    wgpu::Adapter adapter;
    wgpu::Device device;
    WGPUSurface surface;
    // VkDevice device;
    // VkInstance instance;
    // VkPhysicalDevice physicalDevice;
    // VkPhysicalDeviceProperties properties;
    // VkDescriptorPool descriptorPool;
    // VkBuffer stagingBuffer;
    // VkDeviceMemory stagingBufferMemory;
    // VkCommandPool transferCommandPool;
    // VkQueue graphicsQueue;
    // VkQueue computeQueue;
    // VkQueue presentQueue;

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

    // VkDebugReportCallbackEXT debugReportCallback{};
        
    bool checkValidationLayerSupport();
    std::vector<const char*> getRequiredInstanceExtensions();
    std::vector<const char*> getRequiredDeviceExtensions();
    std::vector<const char*> getRequiredValidationLayers();
    // bool checkDeviceExtensionSupport(const VkPhysicalDevice& device);
    // bool isDeviceSuitable(const VkPhysicalDevice& device);
};

}  // namespace Jetstream::Backend

#endif
