#ifndef JETSTREAM_METADATA_HH
#define JETSTREAM_METADATA_HH

#include "jetstream/types.hh"

#ifdef JETSTREAM_CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif

namespace Jetstream {

struct RuntimeMetadata {
    struct {
        const I64 magicNumber = 42;
    } CPU;

#ifdef JETSTREAM_CUDA_AVAILABLE
    struct {
       const cudaStream_t stream = 0; 
    } CUDA;
#endif
};

}  // namespace Jetstream

#endif
