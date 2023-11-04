#include "jetstream/memory/devices/metal/buffer.hh"

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
#include "jetstream/memory/devices/cpu/buffer.hh"
#endif

namespace Jetstream {

using Implementation = TensorBuffer<Device::Metal>;

Implementation::TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                             const std::shared_ptr<TensorPrototypeMetadata>& prototype) {
    JST_TRACE("[METAL:BUFFER] Allocating new buffer.");

    // Check platform.

    if (!Backend::State<Device::Metal>()->hasUnifiedMemory()) {
        JST_ERROR("[METAL:BUFFER] Platform is not unified. Cannot allocate Metal memory.");
        JST_CHECK_THROW(Result::ERROR);
    }

    // Initialize storage.

    storage->root_device = Device::Metal;
    storage->compatible_devices = {
        Device::Metal,
        Device::CPU
    };

    // Check size.

    if (prototype->size_bytes == 0) {
        return;
    }

    // Allocate memory buffer.

    auto device = Backend::State<Device::Metal>()->getDevice();
    const auto alignedSizeBytes = JST_PAGE_ALIGNED_SIZE(prototype->size_bytes);
    buffer = device->newBuffer(alignedSizeBytes, MTL::ResourceStorageModeShared);
    if (!buffer) {
        JST_ERROR("[METAL:BUFFER] Failed to allocate memory.");
        JST_CHECK_THROW(Result::ERROR);
    }
    owns_data = true;

    // Null out array.
    memset(buffer->contents(), 0, prototype->size_bytes);
}

Implementation::TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                             const std::shared_ptr<TensorPrototypeMetadata>& prototype,
                             const std::shared_ptr<TensorBuffer<Device::CPU>>& root_buffer) {
    JST_TRACE("[METAL:BUFFER] Cloning from CPU buffer.");

    // Check platform.

    if (!Backend::State<Device::Metal>()->hasUnifiedMemory()) {
        JST_ERROR("[METAL:BUFFER] Platform is not unified. Cannot clone CPU memory.");
        JST_CHECK_THROW(Result::ERROR);
    }

    // Check size.

    if (prototype->size_bytes == 0) {
        return;
    }

    // Check alignment.

    if (!JST_IS_ALIGNED(root_buffer->data()) && root_buffer->data() != nullptr) {
        JST_TRACE("[METAL:BUFFER] Buffer is not aligned. Cannot clone CPU memory.");
        JST_CHECK_THROW(Result::ERROR);
    }

    // Allocate memory buffer.

    auto device = Backend::State<Device::Metal>()->getDevice();
    auto* ptr = root_buffer->data(); 
    const auto alignedSizeBytes = JST_PAGE_ALIGNED_SIZE(prototype->size_bytes);
    buffer = device->newBuffer(ptr, alignedSizeBytes, MTL::ResourceStorageModeShared, nullptr);
    if (!buffer) {
        JST_ERROR("[METAL:BUFFER] Failed to allocate memory.");
        JST_CHECK_THROW(Result::ERROR);
    }
    owns_data = false;
}

Implementation::~TensorBuffer() {
    JST_TRACE("[METAL:BUFFER] Releasing buffer {}.", fmt::ptr(buffer));

    if (buffer) {
        buffer->release();
    }
}

}  // namespace Jetstream