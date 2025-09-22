#ifndef JETSTREAM_DOMAINS_CORE_ARITHMETIC_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_CORE_ARITHMETIC_MODULE_IMPL_HH

#include <jetstream/domains/core/arithmetic/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct ArithmeticImpl : public Module::Impl,
                        public DynamicConfig<Arithmetic> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;

 protected:
    Tensor input;
    Tensor output;
    Tensor broadcastedOutput;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_ARITHMETIC_MODULE_IMPL_HH
