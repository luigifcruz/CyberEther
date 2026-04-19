#ifndef JETSTREAM_RUNTIME_CONTEXT_NATIVE_CPU_HH
#define JETSTREAM_RUNTIME_CONTEXT_NATIVE_CPU_HH

#include "jetstream/runtime.hh"
#include "jetstream/runtime_context.hh"

namespace Jetstream {

struct NativeCpuRuntimeContext : Runtime::Context {
 public:
    virtual Result computeInitialize();
    virtual Result computeSubmit();
    virtual Result computeDeinitialize();
};

}  // namespace Jetstream

#endif  // JETSTREAM_RUNTIME_CONTEXT_NATIVE_CPU_HH
