#ifndef JETSTREAM_MEMORY_BASE_BUFFER_HH
#define JETSTREAM_MEMORY_BASE_BUFFER_HH

#include "jetstream/memory/metadata.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream {

template<Device D>
class TensorBuffer {
 public:
    virtual bool host_accessible() const noexcept = 0;
    virtual bool device_native() const noexcept = 0;
    virtual bool host_native() const noexcept = 0;
};

}  // namespace Jetstream

#endif
