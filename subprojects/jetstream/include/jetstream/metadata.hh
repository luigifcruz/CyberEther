#ifndef JETSTREAM_METADATA_HH
#define JETSTREAM_METADATA_HH

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
#include "jetstream/backend/base.hh"
#endif

#include "jetstream/types.hh"

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif

namespace Jetstream {

struct RuntimeMetadata {
#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
    struct {
    } cpu;
#endif

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
    struct {
       const cudaStream_t stream = 0; 
    } cuda;
#endif

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
    struct {
        MTL::CommandBuffer* commandBuffer;
    } metal;
#endif
};

}  // namespace Jetstream

#endif
