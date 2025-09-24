#ifndef JETSTREAM_MEMORY2_BUFFER_HH
#define JETSTREAM_MEMORY2_BUFFER_HH

#include <memory>

#include "jetstream/types.hh"
#include "jetstream/memory2/types.hh"

namespace Jetstream::mem2 {

class Buffer {
 public:
    Buffer();
    Buffer(const Buffer& other) = default;
    Buffer(Buffer&& other) noexcept = default;
    Buffer& operator=(const Buffer& other) = default;
    Buffer& operator=(Buffer&& other) noexcept = default;
    ~Buffer();

    Result create(const Device& device, const U64& size_bytes);
    Result create(const Device& device, const Buffer& source);

    Result copy_from(const Buffer& source);

    Result destroy();

    bool valid() const;
    bool is_borrowed() const;

    U64 size_bytes() const;
    Device device() const;
    Device native_device() const;
    Location location() const;

    void* data();
    const void* data() const;

 private:
    struct Impl;
    std::shared_ptr<Impl> impl;

    void ensure_impl();
};

}  // namespace Jetstream::mem2

#endif  // JETSTREAM_MEMORY2_BUFFER_HH
