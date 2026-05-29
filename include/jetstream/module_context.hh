#ifndef JETSTREAM_MODULE_CONTEXT_HH
#define JETSTREAM_MODULE_CONTEXT_HH

#include <memory>

#include "jetstream/module.hh"
#include "jetstream/flowgraph_environment.hh"
#include "jetstream/scheduler.hh"

namespace Jetstream {

struct JETSTREAM_API Module::Context {
 public:
    Context(const std::shared_ptr<Runtime::Context>& runtime,
            const std::shared_ptr<Scheduler::Context>& scheduler,
            const std::shared_ptr<Flowgraph::Environment>& environment = nullptr);
    ~Context();

    const std::shared_ptr<Runtime::Context>& runtime() const;
    const std::shared_ptr<Scheduler::Context>& scheduler() const;
    const std::shared_ptr<Flowgraph::Environment>& environment() const;

 private:
    struct Impl;
    std::shared_ptr<Impl> impl;

    friend struct Module::Impl;
};

}  // namespace Jetstream

#endif  // JETSTREAM_MODULE_CONTEXT_HH
