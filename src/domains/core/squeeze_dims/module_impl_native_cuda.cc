#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cuda.hh>
#include <jetstream/scheduler_context.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct SqueezeDimsImplNativeCuda : public SqueezeDimsImpl,
                                   public NativeCudaRuntimeContext,
                                   public Scheduler::Context {
 public:
    Result computeSubmit(const cudaStream_t&) override;
};

Result SqueezeDimsImplNativeCuda::computeSubmit(const cudaStream_t&) {
    return Result::SUCCESS;
}

JST_REGISTER_MODULE(SqueezeDimsImplNativeCuda, DeviceType::CUDA, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
