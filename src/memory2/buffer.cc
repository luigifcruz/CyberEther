#include "jetstream/memory2/buffer.hh"

#include "jetstream/logger.hh"
#include "buffer_backend.hh"
#include "jetstream/types.hh"

namespace Jetstream::mem2 {

namespace {

std::unique_ptr<detail::Backend> MakeBackend(Device device) {
    switch (device) {
        case Device::CPU:
            return detail::CreateCpuBackend();
#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
        case Device::Metal:
            return detail::CreateMetalBackend();
#endif
        default:
            return nullptr;
    }
}

}  // namespace

struct Buffer::Impl {
    Device native_device = Device::None;
    std::unique_ptr<detail::Backend> backend;
};

Buffer::Buffer() {
    ensure_impl();
}

Buffer::~Buffer() = default;

void Buffer::ensure_impl() {
    if (!impl) {
        impl = std::make_shared<Impl>();
    }
}

Result Buffer::create(const Device& device, const U64& size_bytes) {
    ensure_impl();

    if (impl->backend) {
        JST_ERROR("[MEM2:BUFFER] Buffer already initialized on {}.", impl->backend->device());
        return Result::ERROR;
    }

    if (!(impl->backend = MakeBackend(device))) {
        JST_ERROR("[MEM2:BUFFER] Unsupported device {}.", device);
        return Result::ERROR;
    }

    impl->native_device = device;
    JST_CHECK(impl->backend->create(size_bytes));

    return Result::SUCCESS;
}

Result Buffer::create(const Device& device, const Buffer& source) {
    ensure_impl();

    if (!source.valid()) {
        JST_ERROR("[MEM2:BUFFER] Source buffer not initialized.");
        return Result::ERROR;
    }

    if (impl->backend) {
        JST_ERROR("[MEM2:BUFFER] Buffer already initialized ({}).", impl->backend->device());
        return Result::ERROR;
    }

    if (source.device() == device) {
        JST_ERROR("[MEM2:BUFFER] Source and destination buffer devices are the same.");
        return Result::ERROR;
    }

    if (!(impl->backend = MakeBackend(device))) {
        JST_ERROR("[MEM2:BUFFER] Unsupported device {}.", device);
        return Result::ERROR;
    }

    impl->native_device = source.native_device();
    
    if (source.size_bytes() == 0) {
        return Result::SUCCESS;
    }
    JST_CHECK(impl->backend->create(*source.impl->backend));

    return Result::SUCCESS;
}

Result Buffer::copy_from(const Buffer& source) {
    ensure_impl();

    // Check if source buffer is valid.

    if (!source.valid()) {
        JST_ERROR("[MEM2:BUFFER] Source buffer not initialized.");
        return Result::ERROR;
    }

    // Check if destination buffer is valid.

    if (impl->backend) {
        // Backend already allocated, reuse it.

        if (source.size_bytes() != size_bytes()) {
            JST_ERROR("[MEM2:BUFFER] Source buffer size does not match destination buffer size.");
            return Result::ERROR;
        }

        if (source.device() != device()) {
            JST_ERROR("[MEM2:BUFFER] Source and destination buffer devices are not the same.");
            return Result::ERROR;
        }
    } else {
        // Backend not allocated, create a new one and allocate memory for it.
        JST_CHECK(create(source.device(), source.size_bytes()));
    }

    // Return success if source buffer is empty.

    if (source.size_bytes() == 0) {
        return Result::SUCCESS;
    }

    // Copy data from source buffer to destination buffer.
    JST_CHECK(impl->backend->copy_from(*source.impl->backend));

    return Result::SUCCESS;
}

Result Buffer::destroy() {
    if (!impl) {
        return Result::SUCCESS;
    }
    impl->backend.reset();
    impl->native_device = Device::None;
    return Result::SUCCESS;
}

bool Buffer::valid() const {
    return impl && impl->backend != nullptr;
}

bool Buffer::is_borrowed() const {
    return impl && impl->backend ? impl->backend->is_borrowed() : false;
}

U64 Buffer::size_bytes() const {
    return impl && impl->backend ? impl->backend->size() : 0;
}

Device Buffer::device() const {
    return impl && impl->backend ? impl->backend->device() : Device::None;
}

Device Buffer::native_device() const {
    return impl ? impl->native_device : Device::None;
}

Location Buffer::location() const {
    return impl && impl->backend ? impl->backend->location() : Location::None;
}

void* Buffer::data() {
    return impl && impl->backend ? impl->backend->raw_handle() : nullptr;
}

const void* Buffer::data() const {
    return impl && impl->backend ? impl->backend->raw_handle() : nullptr;
}

}  // namespace Jetstream::mem2
