#ifndef JETSTREAM_MEMORY_BUFFER_HH
#define JETSTREAM_MEMORY_BUFFER_HH

#include <memory>

#include "jetstream/types.hh"
#include "jetstream/memory/types.hh"

namespace Jetstream {

class Buffer {
 public:
    struct Config {
        bool hostAccessible;
    };

    Buffer();
    Buffer(const Buffer& other) = default;
    Buffer(Buffer&& other) noexcept = default;
    Buffer& operator=(const Buffer& other) = default;
    Buffer& operator=(Buffer&& other) noexcept = default;
    ~Buffer();

    Result create(const DeviceType& device, const U64& sizeBytes, const Config& config = {});
    Result create(const DeviceType& device, void* pointer, const U64& sizeBytes);
    Result create(const DeviceType& device, const Buffer& source);

    Result copyFrom(const Buffer& source, void* context = nullptr);

    Result destroy();

    bool valid() const;
    bool isBorrowed() const;

    U64 sizeBytes() const;
    DeviceType device() const;
    DeviceType nativeDevice() const;
    Location location() const;

    void* data();
    const void* data() const;

    void* backend() const;

 private:
    struct Impl;
    std::shared_ptr<Impl> impl;

    void ensureImpl();
};

}  // namespace Jetstream

#endif  // JETSTREAM_MEMORY_BUFFER_HH
