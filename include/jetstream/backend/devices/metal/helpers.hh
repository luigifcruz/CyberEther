#ifndef JETSTREAM_BACKEND_DEVICE_METAL_HELPERS_HH
#define JETSTREAM_BACKEND_DEVICE_METAL_HELPERS_HH

#include <vector>

#include "jetstream/backend/devices/metal/base.hh"
#include "jetstream/logger.hh"
#include "jetstream/types.hh"

namespace Jetstream::Backend {

inline MTL::Function* LoadShader(const std::vector<U8>& data, MTL::Device* device) {
    MTL::CompileOptions* options = MTL::CompileOptions::alloc()->init();
    options->setFastMathEnabled(true);
    options->setLanguageVersion(MTL::LanguageVersion3_0);
    options->setLibraryType(MTL::LibraryTypeExecutable);

    NS::Error* error = nullptr;
    auto* text = NS::String::alloc()->init(const_cast<U8*>(data.data()), data.size(), NS::UTF8StringEncoding, false);
    MTL::Library* library = device->newLibrary(text, options, &error);
    text->release();
    options->release();
    if (!library) {
        JST_ERROR("[METAL] Library error:\n{}", error ? error->description()->utf8String() : "unknown");
        return nullptr;
    }

    MTL::Function* function = library->newFunction(NS::String::string("main0", NS::UTF8StringEncoding));
    library->release();
    if (!function) {
        JST_ERROR("[METAL] Failed to create shader function.");
    }
    return function;
}

}  // namespace Jetstream::Backend

#endif  // JETSTREAM_BACKEND_DEVICE_METAL_HELPERS_HH
