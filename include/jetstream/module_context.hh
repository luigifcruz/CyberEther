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
            const std::shared_ptr<Flowgraph::Environment>& environment,
            const std::shared_ptr<Flowgraph::View>& view);
    ~Context();

    const std::shared_ptr<Runtime::Context>& runtime() const;
    const std::shared_ptr<Scheduler::Context>& scheduler() const;
    const std::shared_ptr<Flowgraph::Environment>& environment() const;
    const std::shared_ptr<Flowgraph::View>& view() const;

 private:
    struct Impl;
    std::shared_ptr<Impl> impl;

    friend struct Module::Impl;
};

}  // namespace Jetstream

#endif  // JETSTREAM_MODULE_CONTEXT_HH
