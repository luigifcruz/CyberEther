#ifndef JETSTREAM_DOMAINS_ML_FRBNN_DETECT_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_ML_FRBNN_DETECT_MODULE_IMPL_HH

#include <jetstream/domains/ml/frbnn_detect/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/tools/snapshot.hh>

namespace Jetstream::Modules {

struct FrbnnDetectImpl : public Module::Impl, public DynamicConfig<FrbnnDetect> {
 public:
    Result define() override;
    Result create() override;
    Result reconfigure() override;

    U64 getTotalCandidates() const;
    F32 getLatestProbability() const;

 protected:
    Tensor input;
    Tensor output;

    bool inputIs2D = false;

    Tools::Snapshot<U64> snapshotTotalCandidates{0};
    Tools::Snapshot<F32> snapshotLatestProbability{0.0f};
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_ML_FRBNN_DETECT_MODULE_IMPL_HH
