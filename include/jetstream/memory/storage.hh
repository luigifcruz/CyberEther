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
    TensorStorage(const TensorStorage& other) {
        storage = other.storage;
        prototype = other.prototype;
    }

    TensorStorage(TensorStorage&& other) noexcept {
        storage = std::move(other.storage);
        prototype = std::move(other.prototype);
    }

    TensorStorage& operator=(const TensorStorage& other) {
        storage = other.storage;
        prototype = other.prototype;
        return *this;
    }

    TensorStorage& operator=(TensorStorage&& other) noexcept {
        storage = std::move(other.storage);
        prototype = std::move(other.prototype);
        return *this;
    }

    const Device& root_device() const noexcept {
        return storage->root_device;
    }

    const std::unordered_set<Device>& compatible_devices() const noexcept {
        return storage->compatible_devices;
    }

    U64 references() const {
        return storage.use_count();
    }

    const TensorStorageMetadata::AttributeMap& attributes() const noexcept {
        return storage->attributes;
    }

    const TensorStorageMetadata::Attribute& attribute(const std::string& key) const {
        return storage->attributes.at(key);
    }

    TensorStorageMetadata::Attribute& attribute(const std::string& key) {
        return storage->attributes[key];
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

        prototype.shape = shape;

        prototype.size = 1;
        for (const auto& dim : prototype.shape) {
            prototype.size *= dim;
        }

        prototype.strides.resize(prototype.shape.size());
        for (U64 i = 0; i < prototype.shape.size(); i++) {
            prototype.strides[i] = 1;
            for (U64 j = i + 1; j < prototype.shape.size(); j++) {
                prototype.strides[i] *= prototype.shape[j];
            }
        }

        assert(prototype.strides.size() == prototype.shape.size());

        prototype.type_size = sizeof(T);
        prototype.hash = std::rand() + 1;
        prototype.size_bytes = prototype.size *
                                prototype.type_size;
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
