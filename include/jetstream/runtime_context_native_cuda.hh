#ifndef JETSTREAM_RUNTIME_CONTEXT_NATIVE_CUDA_HH
#define JETSTREAM_RUNTIME_CONTEXT_NATIVE_CUDA_HH

#include "jetstream/runtime.hh"
#include "jetstream/runtime_context.hh"

#include <cuda_runtime.h>

namespace Jetstream {

struct NativeCudaRuntimeContext : Runtime::Context {
 public:
    NativeCudaRuntimeContext();
    ~NativeCudaRuntimeContext();

    Result createKernel(const std::string& name,
                        const std::string& source,
                        const std::vector<std::string>& headers = {});
    Result scheduleKernel(const std::string& name,
                          const cudaStream_t& stream,
                          const Extent3D<U64>& grid,
                          const Extent3D<U64>& block,
                          void** arguments);
    Result destroyKernel(const std::string& name);

    virtual Result computeInitialize();
    virtual Result computeSubmit(const cudaStream_t& stream);
    virtual Result computeDeinitialize();

 private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

}  // namespace Jetstream

#endif  // JETSTREAM_RUNTIME_CONTEXT_NATIVE_CUDA_HH
