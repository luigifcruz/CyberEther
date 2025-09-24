#ifndef JETSTREAM_MEMORY2_BUFFER_BACKEND_HH
#define JETSTREAM_MEMORY2_BUFFER_BACKEND_HH

#include <memory>

#include "jetstream/types.hh"
#include "jetstream/memory2/types.hh"

namespace Jetstream::mem2::detail {

struct Backend {
    virtual ~Backend() = default;

    virtual Device device() const = 0;

    virtual Result create(const U64& bytes) = 0;
    virtual Result create(const Backend& source) = 0;
    virtual void destroy() = 0;

    virtual Result copy_from(const Backend& source) = 0;

    virtual void* raw_handle() = 0;
    virtual const void* raw_handle() const = 0;

    virtual bool is_borrowed() const = 0;
    virtual Location location() const = 0;
    virtual U64 size() const = 0;
};

std::unique_ptr<Backend> CreateBackend(Device device);
std::unique_ptr<Backend> CreateCpuBackend();
#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
std::unique_ptr<Backend> CreateMetalBackend();
#endif

}  // namespace Jetstream::mem2::detail

#endif  // JETSTREAM_MEMORY2_BUFFER_BACKEND_HH
