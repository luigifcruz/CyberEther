#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "jetstream/backend/devices/metal/base.hh"

#include "jetstream/logger.hh"
#include "jetstream/macros.hh"

namespace Jetstream::Backend {

Metal::Metal(const Config& config) : config(config) {
    // Get default Metal device.
    // TODO: Respect config.deviceId.
    if (!(device = MTL::CreateSystemDefaultDevice())) {
        JST_FATAL("Cannot create Metal device.");
        JST_CHECK_THROW(Result::FATAL);
    }

    // Import generic information.
    info = NS::ProcessInfo::processInfo();

    // Signal device is available.

    _isAvailable = true;

    // Print device information.
    JST_INFO("-----------------------------------------------------");
    JST_INFO("Jetstream Heterogeneous Backend [METAL]")
    JST_INFO("-----------------------------------------------------");
    JST_INFO("Device Name:     {}", getDeviceName());
    JST_INFO("Device Type:     {}", getPhysicalDeviceType());
    JST_INFO("API Version:     {}", getApiVersion());
    JST_INFO("Unified Memory:  {}", hasUnifiedMemory() ? "YES" : "NO");
    JST_INFO("Processor Count: {}/{}", getActiveProcessorCount(), getTotalProcessorCount());
    JST_INFO("Device Memory:   {:.2f} GB", static_cast<F32>(getPhysicalMemory()) / (1024*1024*1024));
    JST_INFO("-----------------------------------------------------");
}

Metal::~Metal() {
    info->release();
    device->release();
}

bool Metal::isAvailable() const {
    return _isAvailable;
}

std::string Metal::getDeviceName() const {
    return device->name()->utf8String();
}

std::string Metal::getApiVersion() const {
    return device->supportsFamily(MTL::GPUFamilyMetal3) ? "Metal 3" : "Metal <2";       
}

bool Metal::getLowPowerStatus() const {
    return info->isLowPowerModeEnabled();
}

bool Metal::hasUnifiedMemory() const {
    return device->hasUnifiedMemory();
}

U64 Metal::getMultisampling() const {
    return 4;
}

U64 Metal::getPhysicalMemory() const {
    return info->physicalMemory();
}

U64 Metal::getActiveProcessorCount() const {
    return info->activeProcessorCount();
}

U64 Metal::getTotalProcessorCount() const {
    return info->processorCount();
}

U64 Metal::getThermalState() const {
    return info->thermalState();
}

PhysicalDeviceType Metal::getPhysicalDeviceType() const {
    PhysicalDeviceType deviceType = PhysicalDeviceType::INTEGRATED;

#ifndef JST_OS_IOS
    if (device->removable()) {
        deviceType = PhysicalDeviceType::DISCRETE;
    }
#endif

    return deviceType;
}

}  // namespace Jetstream::Backend
