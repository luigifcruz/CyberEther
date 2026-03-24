#ifndef JETSTREAM_RUNTIME_CONTEXT_NATIVE_CPU_HH
#define JETSTREAM_RUNTIME_CONTEXT_NATIVE_CPU_HH

#include "jetstream/runtime.hh"

namespace Jetstream {

struct Runtime::Context {
 public:
    virtual Result computeInitialize();
    virtual Result computeSubmit();
    virtual Result computeDeinitialize();
};

}  // namespace Jetstream

#endif  // JETSTREAM_RUNTIME_CONTEXT_NATIVE_CPU_HH
