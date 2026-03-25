#ifndef JETSTREAM_RUNTIME_CONTEXT_HH
#define JETSTREAM_RUNTIME_CONTEXT_HH

#include "jetstream/runtime.hh"

namespace Jetstream {

struct Runtime::Context {
 public:
    virtual ~Context() = default;
};

}  // namespace Jetstream

#endif  // JETSTREAM_RUNTIME_CONTEXT_HH
