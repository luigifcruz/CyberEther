#ifndef JETSTREAM_MEMORY_DEVICES_CUDA_BUFFER_HH
#define JETSTREAM_MEMORY_DEVICES_CUDA_BUFFER_HH

#include <cuda.h>

namespace Jetstream {

class CudaBufferBackend {
 public:
    virtual ~CudaBufferBackend() = default;

    virtual bool hostAccessible() const = 0;
    virtual bool deviceNative() const = 0;
    virtual bool exportableDeviceMemory() const = 0;
    virtual CUmemGenericAllocationHandle allocationHandle() const = 0;
};

}  // namespace Jetstream

#endif  // JETSTREAM_MEMORY_DEVICES_CUDA_BUFFER_HH
