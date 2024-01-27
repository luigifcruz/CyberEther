#ifndef JETSTREAM_METADATA_HH
#define JETSTREAM_METADATA_HH

// TODO: Rename to RuntimeContext.

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
#include "jetstream/backend/base.hh"
#endif

#include "jetstream/types.hh"

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif

namespace Jetstream {

class CPU;
class CUDA;

struct RuntimeMetadata {
#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
    struct {
        CPU* graph;
    } cpu;
#endif

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
    struct {
        CUDA* graph;
    } cuda;
#endif

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
    // TODO: Replace with Metal class pointer.
    struct {
        MTL::CommandQueue* commandQueue;
        MTL::CommandBuffer* commandBuffer;
    } metal;
#endif
};

}  // namespace Jetstream

#endif
