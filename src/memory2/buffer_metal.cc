#include "buffer_backend.hh"

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE

#include <cstring>

#include "jetstream/backend/base.hh"
#include "jetstream/backend/devices/metal/base.hh"
#include "jetstream/logger.hh"
#include "jetstream/memory2/macros.hh"

namespace Jetstream::mem2::detail {

namespace {

class MetalBackend final : public Backend {
 public:
    MetalBackend() = default;
    ~MetalBackend() override {
        destroy();
    }

    Device device() const override {
        return Device::Metal;
    }

    Result create(const U64& bytes) override {
        destroy();

        size_bytes = bytes;
        borrowed = false;

        if (bytes == 0) {
            return Result::SUCCESS;
        }

        auto* metal_device = FetchDevice();
        if (!metal_device) {
            return Result::ERROR;
        }

        const auto aligned = JST_PAGE_ALIGNED_SIZE(bytes);
        buffer = metal_device->newBuffer(aligned, MTL::ResourceStorageModeShared);
        if (!buffer) {
            JST_ERROR("[MEM2:BUFFER:METAL] Failed to allocate {} bytes.", bytes);
            return Result::ERROR;
        }

        if (auto* contents = buffer->contents()) {
            std::memset(contents, 0, bytes);
        }

        return Result::SUCCESS;
    }

    Result create(const Backend& source) override {
        // TODO: Implement Vulkan -> Metal.

        if (source.device() == Device::CPU) {
            JST_TRACE("[MEM2:BUFFER:METAL] Mirroring CPU buffer.");

            if (!JST_IS_ALIGNED(source.raw_handle())) {
                JST_ERROR("[MEM2:BUFFER:METAL] Mirroring requires page-aligned host memory.");
                return Result::ERROR;
            }

            auto* metal_device = FetchDevice();
            if (!metal_device) {
                return Result::ERROR;
            }
            const auto aligned = JST_PAGE_ALIGNED_SIZE(source.size());
            buffer = metal_device->newBuffer(const_cast<void*>(source.raw_handle()), aligned, MTL::ResourceStorageModeShared, nullptr);
            if (!buffer) {
                JST_ERROR("[MEM2:BUFFER:METAL] Failed to wrap shared host memory.");
                return Result::ERROR;
            }

            size_bytes = source.size();
            borrowed = true;

            return Result::SUCCESS;
        }

        JST_ERROR("[MEM2:BUFFER:METAL] Cannot mirror from device {}.", source.device());
        return Result::ERROR;
    }

    Result copy_from(const Backend& source) override {
        JST_TRACE("[MEM2:BUFFER:METAL] Copying buffer.");

        auto* metal_device = FetchDevice();
        if (!metal_device) {
            return Result::ERROR;
        }

        // TODO: Cache the command queue for reuse.

        auto* command_queue = metal_device->newCommandQueue();
        if (!command_queue) {
            JST_ERROR("[MEM2:BUFFER:METAL] Failed to create command queue for copy.");
            return Result::ERROR;
        }

        auto* command_buffer = command_queue->commandBuffer();
        if (!command_buffer) {
            JST_ERROR("[MEM2:BUFFER:METAL] Failed to create command buffer for copy.");
            command_queue->release();
            return Result::ERROR;
        }

        auto* blit_encoder = command_buffer->blitCommandEncoder();
        if (!blit_encoder) {
            JST_ERROR("[MEM2:BUFFER:METAL] Failed to create blit encoder for copy.");
            command_buffer->release();
            command_queue->release();
            return Result::ERROR;
        }

        const auto* source_buffer = static_cast<const MTL::Buffer*>(source.raw_handle());
        blit_encoder->copyFromBuffer(source_buffer, 0, buffer, 0, source.size());
        blit_encoder->endEncoding();

        command_buffer->commit();
        command_buffer->waitUntilCompleted();

        blit_encoder->release();
        command_buffer->release();
        command_queue->release();

        return Result::SUCCESS;
    }

    void* raw_handle() override {
        return buffer;
    }

    const void* raw_handle() const override {
        return buffer;
    }

    bool is_borrowed() const override {
        return borrowed;
    }

    Location location() const override {
        return Location::Unified;
    }

    U64 size() const override {
        return size_bytes;
    }

    void destroy() override {
        if (buffer) {
            buffer->release();
            buffer = nullptr;
        }
        borrowed = false;
        size_bytes = 0;
    }

 private:
    static MTL::Device* FetchDevice() {
        const auto& state = Jetstream::Backend::State<Device::Metal>();
        if (!state || !state->isAvailable()) {
            JST_ERROR("[MEM2:BUFFER:METAL] Metal backend is not available.");
            return nullptr;
        }

        auto* metal_device = state->getDevice();
        if (!metal_device) {
            JST_ERROR("[MEM2:BUFFER:METAL] Failed to retrieve Metal device.");
            return nullptr;
        }

        return metal_device;
    }

    MTL::Buffer* buffer = nullptr;
    U64 size_bytes = 0;
    bool borrowed = false;
};

}  // namespace

std::unique_ptr<Backend> CreateMetalBackend() {
    return std::make_unique<MetalBackend>();
}

}  // namespace Jetstream::mem2::detail

#endif  // JETSTREAM_BACKEND_METAL_AVAILABLE
