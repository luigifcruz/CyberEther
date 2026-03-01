#include "jetstream/scheduler.hh"
#include "jetstream/detail/scheduler_impl.hh"
#include "jetstream/module.hh"

namespace Jetstream {

std::shared_ptr<Scheduler::Impl> SynchronousSchedulerFactory();

Scheduler::Scheduler(const SchedulerType& type) {
    switch (type) {
        case SchedulerType::SYNCHRONOUS:
            impl = SynchronousSchedulerFactory();
            break;
        default:
            JST_FATAL("[SHEDULER] Unknown scheduler type.");
            throw Result::FATAL;
    }
}

Result Scheduler::create(const std::shared_ptr<Instance>& instance) {
    impl->instance = instance;

    return impl->create();
}

Result Scheduler::destroy() {
    return impl->destroy();
}

Result Scheduler::add(const std::shared_ptr<Module>& module) {
    JST_DEBUG("[SCHEDULER] Adding module '{}'.", module->name());
    return impl->add(module);
}

Result Scheduler::remove(const std::shared_ptr<Module>& module) {
    JST_DEBUG("[SCHEDULER] Removing module '{}'.", module->name());
    return impl->remove(module);
}

Result Scheduler::reload(const std::shared_ptr<Module>& module) {
    JST_DEBUG("[SCHEDULER] Reloading module '{}'.", module->name());
    return impl->reload(module);
}

Result Scheduler::present() {
    return impl->present();
}

Result Scheduler::compute() {
    return impl->compute();
}

const std::unordered_map<std::string, std::shared_ptr<Runtime::Metrics>>& Scheduler::metrics() const {
    return impl->metrics();
}

const char* GetSchedulerName(const SchedulerType& scheduler) {
    switch (scheduler) {
        case SchedulerType::SYNCHRONOUS:
            return "synchronous";
        default:
            return "none";
    }
}

const char* GetSchedulerPrettyName(const SchedulerType& scheduler) {
    switch (scheduler) {
        case SchedulerType::SYNCHRONOUS:
            return "Synchronous";
        default:
            return "None";
    }
}

SchedulerType StringToScheduler(const std::string& scheduler) {
    if (scheduler == "synchronous") {
        return SchedulerType::SYNCHRONOUS;
    } else {
        return SchedulerType::NONE;
    }
}

}  // namespace Jetstream
