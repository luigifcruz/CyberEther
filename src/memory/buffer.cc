#include "jetstream/memory/buffer.hh"

#include "jetstream/logger.hh"
#include "buffer_backend.hh"
#include "jetstream/types.hh"

namespace Jetstream {

namespace {

std::unique_ptr<detail::Backend> MakeBackend(DeviceType device) {
    switch (device) {
        case DeviceType::CPU:
            return detail::CreateCpuBackend();
#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
        case DeviceType::CUDA:
            return detail::CreateCudaBackend();
#endif
#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
        case DeviceType::Metal:
            return detail::CreateMetalBackend();
#endif
#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
        case DeviceType::Vulkan:
            return detail::CreateVulkanBackend();
#endif
        default:
            return nullptr;
    }
}

}  // namespace

struct Buffer::Impl {
    DeviceType nativeDevice = DeviceType::None;
    std::unique_ptr<detail::Backend> backend;
};

Buffer::Buffer() {
    ensureImpl();
}

Buffer::~Buffer() = default;

void Buffer::ensureImpl() {
    if (!impl) {
        impl = std::make_shared<Impl>();
    }
}

Result Buffer::create(const DeviceType& device, const U64& sizeBytes, const Config& config) {
    ensureImpl();

    if (impl->backend) {
        JST_ERROR("[MEMORY:BUFFER] Buffer already initialized on {}.", impl->backend->device());
        return Result::ERROR;
    }

    if (!(impl->backend = MakeBackend(device))) {
        JST_ERROR("[MEMORY:BUFFER] Unsupported device {}.", device);
        return Result::ERROR;
    }

    impl->nativeDevice = device;
    JST_CHECK(impl->backend->create(sizeBytes, config));

    return Result::SUCCESS;
}

Result Buffer::create(const DeviceType& device, void* pointer, const U64& sizeBytes) {
    ensureImpl();

    if (impl->backend) {
        JST_ERROR("[MEMORY:BUFFER] Buffer already initialized on {}.", impl->backend->device());
        return Result::ERROR;
    }

    if (!(impl->backend = MakeBackend(device))) {
        JST_ERROR("[MEMORY:BUFFER] Unsupported device {}.", device);
        return Result::ERROR;
    }

    impl->nativeDevice = device;
    JST_CHECK(impl->backend->create(pointer, sizeBytes));

    return Result::SUCCESS;
}

Result Buffer::create(const DeviceType& device, const Buffer& source) {
    ensureImpl();

    if (!source.valid()) {
        JST_ERROR("[MEMORY:BUFFER] Source buffer not initialized.");
        return Result::ERROR;
    }

    if (impl->backend) {
        JST_ERROR("[MEMORY:BUFFER] Buffer already initialized ({}).", impl->backend->device());
        return Result::ERROR;
    }

    if (source.device() == device) {
        JST_ERROR("[MEMORY:BUFFER] Source and destination buffer devices are the same.");
        return Result::ERROR;
    }

    if (!(impl->backend = MakeBackend(device))) {
        JST_ERROR("[MEMORY:BUFFER] Unsupported device {}.", device);
        return Result::ERROR;
    }

    impl->nativeDevice = source.nativeDevice();

    if (source.sizeBytes() == 0) {
        return Result::SUCCESS;
    }
    JST_CHECK(impl->backend->create(*source.impl->backend));

    return Result::SUCCESS;
}

Result Buffer::copyFrom(const Buffer& source, void* context) {
    ensureImpl();

    // Check if source buffer is valid.

    if (!source.valid()) {
        JST_ERROR("[MEMORY:BUFFER] Source buffer not initialized.");
        return Result::ERROR;
    }

    // Check if destination buffer is valid.

    if (impl->backend) {
        // Backend already allocated, reuse it.

        if (source.sizeBytes() != sizeBytes()) {
            JST_ERROR("[MEMORY:BUFFER] Source buffer size does not match destination buffer size.");
            return Result::ERROR;
        }

        if (source.device() != device()) {
            JST_ERROR("[MEMORY:BUFFER] Source and destination buffer devices are not the same.");
            return Result::ERROR;
        }
    } else {
        // Backend not allocated, create a new one and allocate memory for it.
        JST_CHECK(create(source.device(), source.sizeBytes()));
    }

    // Return success if source buffer is empty.

    if (source.sizeBytes() == 0) {
        return Result::SUCCESS;
    }

    // Copy data from source buffer to destination buffer.
    JST_CHECK(impl->backend->copyFrom(*source.impl->backend, context));

    return Result::SUCCESS;
}

Result Buffer::destroy() {
    if (!impl) {
        return Result::SUCCESS;
    }
    impl->backend.reset();
    impl->nativeDevice = DeviceType::None;
    return Result::SUCCESS;
}

bool Buffer::valid() const {
    return impl && impl->backend != nullptr;
}

bool Buffer::isBorrowed() const {
    return impl && impl->backend ? impl->backend->isBorrowed() : false;
}

U64 Buffer::sizeBytes() const {
    return impl && impl->backend ? impl->backend->size() : 0;
}

DeviceType Buffer::device() const {
    return impl && impl->backend ? impl->backend->device() : DeviceType::None;
}

DeviceType Buffer::nativeDevice() const {
    return impl ? impl->nativeDevice : DeviceType::None;
}

Location Buffer::location() const {
    return impl && impl->backend ? impl->backend->location() : Location::None;
}

void* Buffer::data() {
    return impl && impl->backend ? impl->backend->rawHandle() : nullptr;
}

const void* Buffer::data() const {
    return impl && impl->backend ? impl->backend->rawHandle() : nullptr;
}

void* Buffer::backend() const {
    return impl && impl->backend ? dynamic_cast<void*>(impl->backend.get()) : nullptr;
}

}  // namespace Jetstream
