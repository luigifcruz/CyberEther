#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "jetstream/backend/devices/metal/base.hh"

#include "jetstream/logger.hh"
#include "jetstream/macros.hh"

namespace Jetstream::Backend {

Metal::Metal(const Config& config) {
    JST_DEBUG("Initializing Metal backend.");

    // TODO: Add validation layer.

    if (!(device = MTL::CreateSystemDefaultDevice())) {
        JST_FATAL("Cannot create Metal device.");
        JST_CHECK_THROW(Result::ERROR);
    }

    info = NS::ProcessInfo::processInfo();

    JST_INFO("===== Metal Backend Configuration");
    JST_INFO("GPU Name: {}", this->device->name()->utf8String());
    JST_INFO("Has Unified Memory: {}", 
             this->device->hasUnifiedMemory() ? "YES" : "NO");
}

Metal::~Metal() {
    JST_DEBUG("Destroying Metal backend.");

    info->release();
    device->release();
}

const std::string Metal::getDeviceName() const {
    return device->name()->utf8String();
}

const bool Metal::getLowPowerStatus() const {
    return info->isLowPowerModeEnabled();
}

const bool Metal::hasUnifiedMemory() const {
    return device->hasUnifiedMemory();
}

const U64 Metal::physicalMemory() const {
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

}  // namespace Jetstream::Backend
