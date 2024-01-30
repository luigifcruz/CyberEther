#ifndef JETSTREAM_BACKEND_DEVICE_CUDA_HH
#define JETSTREAM_BACKEND_DEVICE_CUDA_HH

#include <cuda.h>
#include <cuda_runtime.h>

#include "jetstream/backend/config.hh"

namespace Jetstream::Backend {

class CUDA {
 public:
    explicit CUDA(const Config& config);
    ~CUDA();

    bool isAvailable() const;
    std::string getDeviceName() const;
    std::string getApiVersion() const;
    std::string getDriverVersion() const;
    std::string getComputeCapability() const;
    PhysicalDeviceType getPhysicalDeviceType() const;
    U64 getPhysicalMemory() const;

    bool hasUnifiedMemory() const;
    bool canExportDeviceMemory() const;
    bool canImportDeviceMemory() const;
    bool canImportHostMemory() const;

    constexpr const U64& getDeviceId() const {
        return config.deviceId;
    }
    
 private:
    Config config;
    CUdevice device;
    CUcontext context;
    bool _isAvailable = false;

    struct {
        std::string deviceName;
        std::string apiVersion;
        std::string driverVersion;
        std::string computeCapability;
        PhysicalDeviceType physicalDeviceType;
        bool hasUnifiedMemory;
        U64 physicalMemory;
        bool canImportDeviceMemory;
        bool canImportHostMemory;
        bool canExportDeviceMemory;
    } cache;
};

}  // namespace Jetstream::Backend

#endif
