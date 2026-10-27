#ifndef JETSTREAM_DOMAINS_CORE_THROTTLE_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_CORE_THROTTLE_MODULE_IMPL_HH

#include <chrono>

#include <jetstream/domains/core/throttle/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct ThrottleImpl : public Module::Impl, public DynamicConfig<Throttle> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result reconfigure() override;

 protected:
    Tensor input;
    std::chrono::steady_clock::duration interval{};
    std::chrono::steady_clock::time_point deadline;

    void resetDeadline();
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_THROTTLE_MODULE_IMPL_HH
