#include <jetstream/detail/module_context_impl.hh>

namespace Jetstream {

Module::Context::Context(const std::shared_ptr<Runtime::Context>& runtime,
                         const std::shared_ptr<Scheduler::Context>& scheduler) {
    impl = std::make_shared<Impl>();
    impl->runtime = runtime;
    impl->scheduler = scheduler;
}

Module::Context::~Context() {
    impl.reset();
}

const std::shared_ptr<Runtime::Context>& Module::Context::runtime() {
    return impl->runtime;
}

const std::shared_ptr<Scheduler::Context>& Module::Context::scheduler() {
    return impl->scheduler;
}

}  // namespace Jetstream
