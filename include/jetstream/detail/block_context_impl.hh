#ifndef JETSTREAM_DETAIL_BLOCK_CONTEXT_IMPL_HH
#define JETSTREAM_DETAIL_BLOCK_CONTEXT_IMPL_HH

#include "../block_context.hh"

namespace Jetstream {

struct Block::Context::Impl {
    std::shared_ptr<Instance> instance;
    std::shared_ptr<Render::Window> render;
    std::shared_ptr<Scheduler> scheduler;
    std::shared_ptr<Flowgraph::Environment> environment;
};

}  // namespace Jetstream

#endif  // JETSTREAM_DETAIL_BLOCK_CONTEXT_IMPL_HH
