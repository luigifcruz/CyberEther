#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct InferImplNativeCpu : public InferImpl,
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

JST_REGISTER_MODULE(InferImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
