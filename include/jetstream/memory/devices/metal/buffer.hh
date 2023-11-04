#ifndef JETSTREAM_MEMORY_METAL_BUFFER_HH
#define JETSTREAM_MEMORY_METAL_BUFFER_HH

#include <memory>

#include "jetstream/memory/devices/base/buffer.hh"

namespace Jetstream {

template<>
class TensorBuffer<Device::Metal> {
 public:
    explicit TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                          const std::shared_ptr<TensorPrototypeMetadata>& prototype);

    explicit TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                          const std::shared_ptr<TensorPrototypeMetadata>& prototype,
                          const std::shared_ptr<TensorBuffer<Device::CPU>>& root_buffer);

    ~TensorBuffer();

    TensorBuffer(const TensorBuffer&) = delete;
    TensorBuffer& operator=(const TensorBuffer&) = delete;

    const MTL::Buffer* data() const noexcept {
        return buffer;
    }

    MTL::Buffer* data() noexcept {
        return buffer;
    }

 private:
    MTL::Buffer* buffer;
    bool owns_data = false;
};

}  // namespace Jetstream

#endif
