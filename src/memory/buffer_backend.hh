#ifndef JETSTREAM_MEMORY_BUFFER_BACKEND_HH
#define JETSTREAM_MEMORY_BUFFER_BACKEND_HH

#include <memory>

#include "jetstream/types.hh"
#include "jetstream/memory/types.hh"
#include "jetstream/memory/buffer.hh"

namespace Jetstream::detail {

struct Backend {
    virtual ~Backend() = default;

    virtual DeviceType device() const = 0;

    virtual Result create(const U64& bytes, const Buffer::Config& config) = 0;
    virtual Result create(void* pointer, const U64& bytes) = 0;
    virtual Result create(const Backend& source) = 0;
    virtual void destroy() = 0;

    virtual Result copyFrom(const Backend& source, void* context) = 0;

    virtual void* rawHandle() = 0;
    virtual const void* rawHandle() const = 0;

    virtual bool isBorrowed() const = 0;
    virtual Location location() const = 0;
    virtual U64 size() const = 0;
};

std::unique_ptr<Backend> CreateCpuBackend();
#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
std::unique_ptr<Backend> CreateCudaBackend();
#endif
#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
std::unique_ptr<Backend> CreateMetalBackend();
#endif
#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
std::unique_ptr<Backend> CreateVulkanBackend();
#endif

}  // namespace Jetstream::detail

#endif  // JETSTREAM_MEMORY_BUFFER_BACKEND_HH
