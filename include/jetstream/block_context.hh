#ifndef JETSTREAM_BLOCK_CONTEXT_HH
#define JETSTREAM_BLOCK_CONTEXT_HH

#include <memory>

#include "jetstream/block.hh"
#include "jetstream/flowgraph_environment.hh"

namespace Jetstream {

struct JETSTREAM_API Block::Context {
 public:
    Context(const std::shared_ptr<Instance>& instance,
            const std::shared_ptr<Render::Window>& render,
            const std::shared_ptr<Scheduler>& scheduler,
            const std::shared_ptr<Flowgraph::Environment>& environment);
    ~Context();

    const std::shared_ptr<Instance>& instance() const;
    const std::shared_ptr<Render::Window>& render() const;
    const std::shared_ptr<Scheduler>& scheduler() const;
    const std::shared_ptr<Flowgraph::Environment>& environment() const;

 private:
    struct Impl;
    std::shared_ptr<Impl> impl;

    friend struct Block::Impl;
};

}  // namespace Jetstream

#endif  // JETSTREAM_BLOCK_CONTEXT_HH
