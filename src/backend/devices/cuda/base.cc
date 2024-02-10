#include "nvml.h"

#include "jetstream/backend/devices/cuda/base.hh"
#include "jetstream/backend/devices/cuda/helpers.hh"
#include "jetstream/logger.hh"

namespace Jetstream::Backend {

CUDA::CUDA(const Config& config) : config(config), cache({}) {
    // Initialize CUDA.

    JST_CUDA_CHECK_THROW(cuInit(0), [&]{
        JST_FATAL("[CUDA] Cannot initialize CUDA: {}", err);
    });

    // Check if device ID is valid.

    I32 deviceCount;
    JST_CUDA_CHECK_THROW(cuDeviceGetCount(&deviceCount), [&]{
        JST_FATAL("[CUDA] Cannot get device count: {}", err);
    });
    if (config.deviceId >= static_cast<U64>(deviceCount)) {
       JST_FATAL("[CUDA] Cannot get desired device ID ({}).", config.deviceId);
    }

    // Setup device.

    JST_CUDA_CHECK_THROW(cuDeviceGet(&device, config.deviceId), [&]{
        JST_FATAL("[CUDA] Cannot get desired device ID ({}): {}", config.deviceId, err);
    });
    JST_CUDA_CHECK_THROW(cudaSetDevice(config.deviceId), [&]{
        JST_FATAL("[CUDA] Cannot get desired device ID ({}): {}", config.deviceId, err);
    });
    JST_CUDA_CHECK_THROW(cuCtxCreate(&context, 0, device), [&]{
        JST_FATAL("[CUDA] Cannot create context for device ID ({}): {}", config.deviceId, err);
    });
    _isAvailable = true;

    // Parse device information.

    {
        char deviceName[256];
        cuDeviceGetName(deviceName, 256, device);
        cache.deviceName = deviceName;
    }

    {
        int ccVersionMajor;
        cuDeviceGetAttribute(&ccVersionMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
        int ccVersionMinor;
        cuDeviceGetAttribute(&ccVersionMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
        cache.computeCapability = jst::fmt::format("{}.{}", ccVersionMajor, ccVersionMinor);
    }

    {
        int runtimeVersion;
        cudaRuntimeGetVersion(&runtimeVersion);

        int major = runtimeVersion / 1000;
        int minor = runtimeVersion % 1000 / 10;
        int patch = runtimeVersion % 10;

        cache.apiVersion = jst::fmt::format("{}.{}.{}", major, minor, patch);
    }

    {
        int isDeviceIntegrated;
        cuDeviceGetAttribute(&isDeviceIntegrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, device);

        if (isDeviceIntegrated) {
            cache.physicalDeviceType = PhysicalDeviceType::INTEGRATED;
            cache.hasUnifiedMemory = true;
        } else {
            cache.physicalDeviceType = PhysicalDeviceType::DISCRETE;
            cache.hasUnifiedMemory = false;
        }
    }

    {
        size_t physicalMemory;
        cuDeviceTotalMem(&physicalMemory, device);
        cache.physicalMemory = physicalMemory;
    }

    {
        nvmlInit();
        char driverVersion[256];
        nvmlSystemGetDriverVersion(driverVersion, 256);
        cache.driverVersion = driverVersion;
        nvmlShutdown();
    }

    {
        int query = 0;

        // TODO: Find a valid attribute for this.
        cache.canImportDeviceMemory = true;

        JST_CUDA_CHECK_THROW(cuDeviceGetAttribute(&query, 
                                                  CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, 
                                                  device), [&]{
            JST_FATAL("[CUDA] Cannot get device attribute: {}", err);
        });
        cache.canExportDeviceMemory = query;

        JST_CUDA_CHECK_THROW(cuDeviceGetAttribute(&query, 
                                                  CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM, 
                                                  device), [&]{
            JST_FATAL("[CUDA] Cannot get device attribute: {}", err);
        });
        cache.canImportHostMemory = query;
    }

    // Print device information.

    JST_INFO("-----------------------------------------------------");
    JST_INFO("Jetstream Heterogeneous Backend [CUDA]")
    JST_INFO("-----------------------------------------------------");
    JST_INFO("Device Name:        {}", getDeviceName());
    JST_INFO("Device Type:        {}", getPhysicalDeviceType());
    JST_INFO("API Version:        {}", getApiVersion());
    JST_INFO("Compute Capability: {}", getComputeCapability());
    JST_INFO("Driver Version:     {}", getDriverVersion());
    JST_INFO("Unified Memory:     {}", hasUnifiedMemory() ? "YES" : "NO");
    JST_INFO("Device Memory:      {:.2f} GB", static_cast<F32>(getPhysicalMemory()) / (1024*1024*1024));
    JST_INFO("Interoperability:");
    JST_INFO("  - Can Import Device Memory: {}", canImportDeviceMemory() ? "YES" : "NO");
    JST_INFO("  - Can Export Device Memory: {}", canExportDeviceMemory() ? "YES" : "NO");
    JST_INFO("  - Can Export Host Memory:   {}", canImportHostMemory() ? "YES" : "NO");
    JST_INFO("-----------------------------------------------------");
}

CUDA::~CUDA() {
    if (_isAvailable) {
        cuCtxDestroy(context);
    }
}

bool CUDA::isAvailable() const {
    return _isAvailable;
}

std::string CUDA::getDeviceName() const {
    return cache.deviceName;
}

std::string CUDA::getApiVersion() const {
    return cache.apiVersion;
}

std::string CUDA::getDriverVersion() const {
    return cache.driverVersion;
}

std::string CUDA::getComputeCapability() const {
    return cache.computeCapability;
}

PhysicalDeviceType CUDA::getPhysicalDeviceType() const {
    return cache.physicalDeviceType;
}

bool CUDA::hasUnifiedMemory() const {
    return cache.hasUnifiedMemory;
}

bool CUDA::canExportDeviceMemory() const {
    return cache.canExportDeviceMemory;
}

bool CUDA::canImportDeviceMemory() const {
    return cache.canImportDeviceMemory;
}

bool CUDA::canImportHostMemory() const {
    return cache.canImportHostMemory;
}

U64 CUDA::getPhysicalMemory() const {
    return cache.physicalMemory;
}

}  // namespace Jetstream::Backend
