#ifndef JETSTREAM_MEMORY_CUDA_HELPER_HH
#define JETSTREAM_MEMORY_CUDA_HELPER_HH

#include <cuda_runtime.h>

#include "jetstream/types.hh"
#include "jetstream/memory/vector.hh"

namespace Jetstream::Memory {

template<typename T>
static Result PageLock(const Vector<Device::CPU, T>& vec,
                        const bool& readOnly = false) {
    cudaPointerAttributes attr;
    JST_CUDA_CHECK(cudaPointerGetAttributes(&attr, vec.data()), [&]{
        BL_FATAL("Failed to get pointer attributes: {}", err);
    });

    if (attr.type != cudaMemoryTypeUnregistered) {
        JST_WARN("Memory already registered.");
        return Result::SUCCESS;
    }

    unsigned int kind = cudaHostRegisterDefault;
    if (readOnly) {
        kind = cudaHostRegisterReadOnly;
    }

    JST_CUDA_CHECK(cudaHostRegister(vec.data(), vec.size_bytes(), kind), [&]{
        JST_FATAL("Failed to register CPU memory: {}", err);
    });

    return Result::SUCCESS;
}

}  // namespace Jetstream::Memory

#endif
