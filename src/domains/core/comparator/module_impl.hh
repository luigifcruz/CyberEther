#ifndef JETSTREAM_DOMAINS_CORE_COMPARATOR_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_CORE_COMPARATOR_MODULE_IMPL_HH

#include <string>
#include <vector>

#include <jetstream/domains/core/comparator/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/tools/snapshot.hh>

namespace Jetstream::Modules {

struct ComparatorImpl : public Module::Impl, public DynamicConfig<Comparator> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

    F64 getMaxDiff() const;
    F64 getMeanDiff() const;
    F64 getMse() const;
    bool getMatch() const;

 protected:
    static std::string inputPortName(U64 index);

    std::vector<Tensor> inputTensors;
    Tensor error;

    Tools::Snapshot<F64> maxDiffState{0.0};
    Tools::Snapshot<F64> meanDiffState{0.0};
    Tools::Snapshot<F64> mseState{0.0};
    Tools::Snapshot<bool> matchState{true};
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_COMPARATOR_MODULE_IMPL_HH
