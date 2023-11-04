#ifndef JETSTREAM_MEMORY_STORAGE_HH
#define JETSTREAM_MEMORY_STORAGE_HH

#include <any>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <cassert>
#include <cstdlib>

#include "jetstream/types.hh"
#include "jetstream/memory/prototype.hh"
#include "jetstream/memory/devices/base/buffer.hh"

namespace Jetstream {

template<typename T>
class TensorStorage : public TensorPrototype {
 public:
    const Device& root_device() const noexcept {
        return storage->root_device;
    }

    const std::unordered_set<Device>& compatible_devices() const noexcept {
        return storage->compatible_devices;
    }

    U64 references() const {
        assert(storage.use_count() == prototype.use_count());
        return storage.use_count();
    }

 protected:
    std::shared_ptr<TensorStorageMetadata> storage;

    TensorStorage() {
        JST_TRACE("[STORAGE] Creating new empty storage.");

        // Initialize storage.
        storage = std::make_shared<TensorStorageMetadata>();
    }

    explicit TensorStorage(const std::vector<U64>& shape) {
        JST_TRACE("[STORAGE] Creating new storage with shape ({}).", shape);

        // Initialize storage.
        storage = std::make_shared<TensorStorageMetadata>();

        // Calculate prototype metadata.

        prototype->shape = shape;

        prototype->size = 1;
        for (const auto& dim : prototype->shape) {
            prototype->size *= dim;
        }

        // TODO: Update for loop to use U64.
        prototype->strides.resize(prototype->shape.size());
        prototype->strides.back() = 1;
        for (int i = prototype->shape.size() - 2; i >= 0; i--) {
            prototype->strides[i] = prototype->strides[i + 1] * prototype->shape[i + 1];
        }

        assert(prototype->strides.size() == prototype->shape.size());

        prototype->type_size = sizeof(T);
        prototype->hash = rand();
        prototype->size_bytes = prototype->size *
                                prototype->type_size;
    }

    template<Device RootDevice, typename... Args>
    std::shared_ptr<TensorBuffer<RootDevice>> create_buffer(Args... args) {
        JST_TRACE("[STORAGE] Creating new buffer on {}.", GetDevicePrettyName(RootDevice));

        // Create a new buffer.
        return make_buffer<RootDevice>(args...);
    }

    template<Device TargetDevice, Device RootDevice>
    std::shared_ptr<TensorBuffer<TargetDevice>> clone_buffer() {
        JST_TRACE("[STORAGE] Cloning buffer from {} to {}.", GetDevicePrettyName(RootDevice), 
                                                             GetDevicePrettyName(TargetDevice));

        // Check if root buffer is created.

        if (root_device() == Device::None) {
            JST_ERROR("[STORAGE] Root device not created.");
            JST_CHECK_THROW(Result::ERROR);
        }

        // Check if the root buffer can be cloned to the target device.

        if (!compatible_devices().contains(TargetDevice)) {
            JST_ERROR("[STORAGE] Device not compatible.");
            JST_CHECK_THROW(Result::ERROR);
        }

        // Check if the buffer has already been cloned.

        if (storage->clones.contains(TargetDevice)) {
            return get_buffer<TargetDevice>();
        }

        // Create a new buffer.
        return make_buffer<TargetDevice>(get_buffer<RootDevice>());
    }

    template<Device TargetDevice, Device RootDevice>
    std::shared_ptr<TensorBuffer<TargetDevice>> clone_buffer(const TensorStorage& other) {
        // Copy metadata.

        storage = other.storage;
        prototype = other.prototype;

        // Clone buffer.
        return clone_buffer<TargetDevice, RootDevice>();
    }

 private:
    template<Device D>
    std::shared_ptr<TensorBuffer<D>> get_buffer() {
        return std::any_cast<std::shared_ptr<TensorBuffer<D>>>(storage->clones.at(D));
    }

    template<Device D, typename... Args>
    std::shared_ptr<TensorBuffer<D>> make_buffer(Args... args) {
        auto buffer = std::make_shared<TensorBuffer<D>>(storage, this->prototype, args...);
        storage->clones[D] = buffer;
        return buffer;
    }
};

}  // namespace Jetstream

#endif
