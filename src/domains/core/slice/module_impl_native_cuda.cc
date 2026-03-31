#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cuda.hh>
#include <jetstream/scheduler_context.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct SliceImplNativeCuda : public SliceImpl,
                             public NativeCudaRuntimeContext,
                             public Scheduler::Context {
 public:
    Result computeSubmit(const cudaStream_t&) override;
};

Result SliceImplNativeCuda::computeSubmit(const cudaStream_t&) {
    return Result::SUCCESS;
}

JST_REGISTER_MODULE(SliceImplNativeCuda, DeviceType::CUDA, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
