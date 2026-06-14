#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct FrbnnDetectImplNativeCpu : public FrbnnDetectImpl,
                                  public NativeCpuRuntimeContext,
                                  public Scheduler::Context {
 public:
    Result create() final;
    Result computeSubmit() override;
};

Result FrbnnDetectImplNativeCpu::create() {
    JST_CHECK(FrbnnDetectImpl::create());
    return Result::SUCCESS;
}

Result FrbnnDetectImplNativeCpu::computeSubmit() {
    const U64 batchDim  = input.shape()[0];
    const U64 nclasses  = inputIs2D ? input.shape()[1] : 1;
    const F32* inPtr    = input.data<F32>();
    F32*       outPtr   = output.data<F32>();

    U64 newCandidates = 0;
    F32 maxProb       = 0.0f;

    for (U64 b = 0; b < batchDim; ++b) {
        const F32 prob = inputIs2D ? inPtr[b * nclasses + classIndex] : inPtr[b];
        outPtr[b] = prob;

        if (prob > maxProb) {
            maxProb = prob;
        }

        if (prob >= threshold) {
            newCandidates++;
            JST_WARN("[FRBNN_DETECT] FRB candidate! batch_idx={} p={:.4f}", b, prob);
        }
    }

    if (newCandidates > 0) {
        snapshotTotalCandidates.publish(snapshotTotalCandidates.get() + newCandidates);
    }
    snapshotLatestProbability.publish(maxProb);

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(FrbnnDetectImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
