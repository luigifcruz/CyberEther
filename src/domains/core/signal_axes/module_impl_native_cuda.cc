#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cuda.hh>
#include <jetstream/scheduler_context.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct SignalAxesImplNativeCuda : public SignalAxesImpl,
                                  public NativeCudaRuntimeContext,
                                  public Scheduler::Context {
    Result computeSubmit(const cudaStream_t&) override;
};

Result SignalAxesImplNativeCuda::computeSubmit(const cudaStream_t&) {
    return Result::SUCCESS;
}

JST_REGISTER_MODULE(SignalAxesImplNativeCuda, DeviceType::CUDA, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
