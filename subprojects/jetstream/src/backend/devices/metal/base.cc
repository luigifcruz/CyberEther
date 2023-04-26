#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "jetstream/backend/devices/metal/base.hh"

#include "jetstream/logger.hh"
#include "jetstream/macros.hh"

namespace Jetstream::Backend {

Metal::Metal(const Config& config) {
    // Get default Metal device.
    if (!(device = MTL::CreateSystemDefaultDevice())) {
        JST_FATAL("Cannot create Metal device.");
        JST_CHECK_THROW(Result::ERROR);
    }

    // Import generic information.
    info = NS::ProcessInfo::processInfo();

    // Print device information.
    JST_INFO("—————————————————————————————————————————————————————");
    JST_INFO("Jetstream Heterogeneous Backend [METAL]")
    JST_INFO("—————————————————————————————————————————————————————");
    JST_INFO("Device Name:     {}", getDeviceName());
    JST_INFO("Device Type:     {}", getPhysicalDeviceType());
    JST_INFO("API Version:     {}", getApiVersion())
    JST_INFO("Unified Memory:  {}", hasUnifiedMemory() ? "YES" : "NO");
    JST_INFO("Processor Count: {}/{}", getActiveProcessorCount(), getTotalProcessorCount());
    JST_INFO("Physical Memory: {:.2f} GB", static_cast<F32>(getPhysicalMemory()) / (1024*1024*1024));
    JST_INFO("—————————————————————————————————————————————————————");
}

Metal::~Metal() {
    info->release();
    device->release();
}

const std::string Metal::getDeviceName() const {
    return device->name()->utf8String();
}

const std::string Metal::getApiVersion() const {
    return device->supportsFamily(MTL::GPUFamilyMetal3) ? "Metal 3" : "Metal <2";       
}

const bool Metal::getLowPowerStatus() const {
    return info->isLowPowerModeEnabled();
}

const bool Metal::hasUnifiedMemory() const {
    return device->hasUnifiedMemory();
}

const U64 Metal::getPhysicalMemory() const {
    return info->physicalMemory();
}

const U64 Metal::getActiveProcessorCount() const {
    return info->activeProcessorCount();
}

const U64 Metal::getTotalProcessorCount() const {
    return info->processorCount();
}

const U64 Metal::getThermalState() const {
    return info->thermalState();
}

const PhysicalDeviceType Metal::getPhysicalDeviceType() const {
    PhysicalDeviceType deviceType = PhysicalDeviceType::INTEGRATED;

    if (device->removable()) {
        deviceType = PhysicalDeviceType::DISCRETE;
    }

    return deviceType;
}

}  // namespace Jetstream::Backend
