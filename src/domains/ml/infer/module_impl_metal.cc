#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

// CoreML EP routes ops to the Apple Neural Engine / GPU.
// It accepts CPU-side tensors and handles data transfer internally,
// so this file uses NativeCpuRuntimeContext (not Metal).

namespace Jetstream::Modules {

struct InferImplCoreML : public InferImpl,
                         public NativeCpuRuntimeContext,
                         public Scheduler::Context {
 public:
    Result create() final { return InferImpl::create(); }

    Result computeSubmit() override {
        session->Run(Ort::RunOptions{nullptr},
                     inputNames.data(),  inputValues.data(),  inputValues.size(),
                     outputNames.data(), outputValues.data(), outputValues.size());
        return Result::SUCCESS;
    }
};

JST_REGISTER_MODULE(InferImplCoreML, DeviceType::CPU, RuntimeType::NATIVE, "coreml");

}  // namespace Jetstream::Modules
