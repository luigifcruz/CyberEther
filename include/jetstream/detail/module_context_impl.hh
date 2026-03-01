#ifndef JETSTREAM_DETAIL_MODULE_CONTEXT_IMPL_HH
#define JETSTREAM_DETAIL_MODULE_CONTEXT_IMPL_HH

#include "../module_context.hh"

namespace Jetstream {

struct Module::Context::Impl {
    std::shared_ptr<Runtime::Context> runtime;
    std::shared_ptr<Scheduler::Context> scheduler;
};

}  // namespace Jetstream

#endif  // JETSTREAM_DETAIL_MODULE_CONTEXT_IMPL_HH
