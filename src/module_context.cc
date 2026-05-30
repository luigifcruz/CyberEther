#include <jetstream/detail/module_context_impl.hh>

namespace Jetstream {

Module::Context::Context(const std::shared_ptr<Runtime::Context>& runtime,
                         const std::shared_ptr<Scheduler::Context>& scheduler,
                         const std::shared_ptr<Flowgraph::Environment>& environment,
                         const std::shared_ptr<Flowgraph::View>& view) {
    impl = std::make_shared<Impl>();
    impl->runtime = runtime;
    impl->scheduler = scheduler;
    impl->environment = environment;
    impl->view = view;
}

Module::Context::~Context() {
    impl.reset();
}

const std::shared_ptr<Runtime::Context>& Module::Context::runtime() const {
    return impl->runtime;
}

const std::shared_ptr<Scheduler::Context>& Module::Context::scheduler() const {
    return impl->scheduler;
}

const std::shared_ptr<Flowgraph::Environment>& Module::Context::environment() const {
    return impl->environment;
}

const std::shared_ptr<Flowgraph::View>& Module::Context::view() const {
    return impl->view;
}

}  // namespace Jetstream
