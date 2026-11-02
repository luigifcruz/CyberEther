#include <cmath>

#include <jetstream/tools/automatic_iterator.hh>
#include <jetstream/memory/macros.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct ComplexMergeImplNativeCpu : public ComplexMergeImpl,
                                   public NativeCpuRuntimeContext,
                                   public Scheduler::Context {
 public:
    Result computeSubmit() override;

 private:
    Result kernelRectangular();
    Result kernelPolar();
};

Result ComplexMergeImplNativeCpu::computeSubmit() {
    return polar ? kernelPolar() : kernelRectangular();
}

Result ComplexMergeImplNativeCpu::kernelRectangular() {
    return AutomaticIterator<const F32, const F32, CF32>(
        [](const auto& a, const auto& b, auto& out) {
            out = CF32(a, b);
        },
    inputA, inputB, output);
}

Result ComplexMergeImplNativeCpu::kernelPolar() {
    return AutomaticIterator<const F32, const F32, CF32>(
        [](const auto& a, const auto& b, auto& out) {
            out = CF32(a * std::cos(b), a * std::sin(b));
        },
    inputA, inputB, output);
}

JST_REGISTER_MODULE(ComplexMergeImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
