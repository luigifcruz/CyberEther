#include <jetstream/detail/block_context_impl.hh>

namespace Jetstream {

Block::Context::Context(const std::shared_ptr<Instance>& instance,
                        const std::shared_ptr<Render::Window>& render,
                        const std::shared_ptr<Scheduler>& scheduler,
                        const std::shared_ptr<Flowgraph::Environment>& environment) {
    impl = std::make_shared<Impl>();
    impl->instance = instance;
    impl->render = render;
    impl->scheduler = scheduler;
    impl->environment = environment;
}

Block::Context::~Context() {
    impl.reset();
}

const std::shared_ptr<Instance>& Block::Context::instance() const {
    return impl->instance;
}

const std::shared_ptr<Render::Window>& Block::Context::render() const {
    return impl->render;
}

const std::shared_ptr<Scheduler>& Block::Context::scheduler() const {
    return impl->scheduler;
}

const std::shared_ptr<Flowgraph::Environment>& Block::Context::environment() const {
    return impl->environment;
}

}  // namespace Jetstream
