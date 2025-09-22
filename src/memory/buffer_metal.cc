#include "buffer_backend.hh"

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE

#include <cstring>

#include "jetstream/backend/base.hh"
#include "jetstream/backend/devices/metal/base.hh"
#include "jetstream/logger.hh"
#include "jetstream/memory/macros.hh"

namespace Jetstream::detail {

namespace {

class MetalBackend final : public Backend {
 public:
    MetalBackend() = default;
    ~MetalBackend() override {
        destroy();
    }

    DeviceType device() const override {
        return DeviceType::Metal;
    }

    Result create(const U64& bytes, const Buffer::Config&) override {
        destroy();

        sizeBytes = bytes;
        borrowed = false;

        if (bytes == 0) {
            return Result::SUCCESS;
        }

        auto* metalDevice = FetchDevice();
        if (!metalDevice) {
            return Result::ERROR;
        }

        const auto aligned = JST_PAGE_ALIGNED_SIZE(bytes);
        buffer = metalDevice->newBuffer(aligned, MTL::ResourceStorageModeShared);
        if (!buffer) {
            JST_ERROR("[MEMORY:BUFFER:METAL] Failed to allocate {} bytes.", bytes);
            return Result::ERROR;
        }

        if (auto* contents = buffer->contents()) {
            std::memset(contents, 0, bytes);
        }

        return Result::SUCCESS;
    }

    Result create(void* pointer, const U64& bytes) override {
        (void)pointer;
        (void)bytes;
        JST_ERROR("[MEMORY:BUFFER:METAL] Borrowed host-pointer create is only supported on CPU buffers.");
        return Result::ERROR;
    }

    Result create(const Backend& source) override {
        // TODO: Implement Vulkan -> Metal.

        if (source.device() == DeviceType::CPU) {
            JST_TRACE("[MEMORY:BUFFER:METAL] Mirroring CPU buffer.");

            if (!JST_IS_ALIGNED(source.rawHandle())) {
                JST_ERROR("[MEMORY:BUFFER:METAL] Mirroring requires page-aligned host memory.");
                return Result::ERROR;
            }

            auto* metalDevice = FetchDevice();
            if (!metalDevice) {
                return Result::ERROR;
            }
            const auto aligned = JST_PAGE_ALIGNED_SIZE(source.size());
            buffer = metalDevice->newBuffer(const_cast<void*>(source.rawHandle()), aligned, MTL::ResourceStorageModeShared, nullptr);
            if (!buffer) {
                JST_ERROR("[MEMORY:BUFFER:METAL] Failed to wrap shared host memory.");
                return Result::ERROR;
            }

            sizeBytes = source.size();
            borrowed = true;

            return Result::SUCCESS;
        }

        JST_ERROR("[MEMORY:BUFFER:METAL] Cannot mirror from device {}.", source.device());
        return Result::ERROR;
    }

    Result copyFrom(const Backend& source) override {
        JST_TRACE("[MEMORY:BUFFER:METAL] Copying buffer.");

        auto* metalDevice = FetchDevice();
        if (!metalDevice) {
            return Result::ERROR;
        }

        // TODO: Cache the command queue for reuse.

        auto* commandQueue = metalDevice->newCommandQueue();
        if (!commandQueue) {
            JST_ERROR("[MEMORY:BUFFER:METAL] Failed to create command queue for copy.");
            return Result::ERROR;
        }

        auto* commandBuffer = commandQueue->commandBuffer();
        if (!commandBuffer) {
            JST_ERROR("[MEMORY:BUFFER:METAL] Failed to create command buffer for copy.");
            commandQueue->release();
            return Result::ERROR;
        }

        auto* blitEncoder = commandBuffer->blitCommandEncoder();
        if (!blitEncoder) {
            JST_ERROR("[MEMORY:BUFFER:METAL] Failed to create blit encoder for copy.");
            commandBuffer->release();
            commandQueue->release();
            return Result::ERROR;
        }

        const auto* sourceBuffer = static_cast<const MTL::Buffer*>(source.rawHandle());
        blitEncoder->copyFromBuffer(sourceBuffer, 0, buffer, 0, source.size());
        blitEncoder->endEncoding();

        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();

        blitEncoder->release();
        commandBuffer->release();
        commandQueue->release();

        return Result::SUCCESS;
    }

    void* rawHandle() override {
        return buffer;
    }

    const void* rawHandle() const override {
        return buffer;
    }

    bool isBorrowed() const override {
        return borrowed;
    }

    Location location() const override {
        return Location::Unified;
    }

    U64 size() const override {
        return sizeBytes;
    }

    void destroy() override {
        if (buffer) {
            buffer->release();
            buffer = nullptr;
        }
        borrowed = false;
        sizeBytes = 0;
    }

 private:
    static MTL::Device* FetchDevice() {
        const auto& state = Jetstream::Backend::State<DeviceType::Metal>();
        if (!state || !state->isAvailable()) {
            JST_ERROR("[MEMORY:BUFFER:METAL] Metal backend is not available.");
            return nullptr;
        }

        auto* metalDevice = state->getDevice();
        if (!metalDevice) {
            JST_ERROR("[MEMORY:BUFFER:METAL] Failed to retrieve Metal device.");
            return nullptr;
        }

        return metalDevice;
    }

    MTL::Buffer* buffer = nullptr;
    U64 sizeBytes = 0;
    bool borrowed = false;
};

}  // namespace

std::unique_ptr<Backend> CreateMetalBackend() {
    return std::make_unique<MetalBackend>();
}

}  // namespace Jetstream::detail

#endif  // JETSTREAM_BACKEND_METAL_AVAILABLE
