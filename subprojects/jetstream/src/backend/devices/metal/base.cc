#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "jetstream/backend/devices/metal/base.hh"

#include "jetstream/logger.hh"
#include "jetstream/macros.hh"

namespace Jetstream::Backend {

Metal::Metal(const Config& config) {
    JST_DEBUG("Initializing Metal backend.");

    if (!(device = MTL::CreateSystemDefaultDevice())) {
        JST_FATAL("Cannot create Metal device.");
        throw Result::ERROR;
    }

    JST_INFO("===== Metal Backend Configuration");
    JST_INFO("GPU Name: {}", this->device->name()->utf8String());
    JST_INFO("Has Unified Memory: {}", 
             this->device->hasUnifiedMemory() ? "YES" : "NO");
}

Metal::~Metal() {
    JST_DEBUG("Destroying Metal backend.");

    device->release();
}

}  // namespace Jetstream::Backend
