#ifndef JETSTREAM_MEMORY_BASE_BUFFER_HH
#define JETSTREAM_MEMORY_BASE_BUFFER_HH

#include "jetstream/memory/metadata.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream {

class TensorBufferBase {
 public:
    constexpr bool allocated() const noexcept {
        return _allocated;
    }

    constexpr bool host_accessible() const noexcept {
        return _host_accessible;
    }

    constexpr bool device_native() const noexcept {
        return _device_native;
    }

    constexpr bool host_native() const noexcept {
        return _host_native;
    }

    constexpr Device external_memory_device() const noexcept {
        return _external_memory_device;
    }

 protected:
    void set_allocated() noexcept {
        _allocated = true;
    }

    void set_host_accessible() noexcept {
        _host_accessible = true;
    }

    void set_device_native() noexcept {
        _device_native = true;
    }

    void set_host_native() noexcept {
        _host_native = true;
    }

    void set_external_memory_device(const Device& device) noexcept {
        _external_memory_device = device;
    }

 private:
    bool _allocated = false;

    bool _host_accessible = false;
    bool _device_native = false;
    bool _host_native = false;

    Device _external_memory_device = Device::None;
};

template<Device D>
class TensorBuffer : public TensorBufferBase {
 public:
    virtual ~TensorBuffer() = default;

    virtual bool host_accessible() const noexcept = 0;
    virtual bool device_native() const noexcept = 0;
    virtual bool host_native() const noexcept = 0;
};

}  // namespace Jetstream

#endif
