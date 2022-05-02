#ifndef JETSTREAM_METADATA_HH
#define JETSTREAM_METADATA_HH

#include "jetstream/types.hh"

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif

namespace Jetstream {

struct RuntimeMetadata {
#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
    struct {
    } CPU;
#endif

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
    struct {
       const cudaStream_t stream = 0; 
    } CUDA;
#endif
};

}  // namespace Jetstream

#endif
