#ifndef JETSTREAM_MODULE_CONTEXT_HH
#define JETSTREAM_MODULE_CONTEXT_HH

#include "jetstream/module.hh"
#include "jetstream/scheduler.hh"

namespace Jetstream {

struct Module::Context {
 public:
    Context(const std::shared_ptr<Runtime::Context>& runtime,
            const std::shared_ptr<Scheduler::Context>& scheduler);
    ~Context();

    const std::shared_ptr<Runtime::Context>& runtime();
    const std::shared_ptr<Scheduler::Context>& scheduler();

 private:
    struct Impl;
    std::shared_ptr<Impl> impl;

    friend struct Module::Impl;
};

}  // namespace Jetstream

#endif  // JETSTREAM_MODULE_CONTEXT_HH
