#include <cmath>

#include <jetstream/tools/automatic_iterator.hh>
#include <jetstream/memory/macros.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct ComplexSplitImplNativeCpu : public ComplexSplitImpl,
                                   public NativeCpuRuntimeContext,
                                   public Scheduler::Context {
 public:
    Result computeSubmit() override;

 private:
    Result kernelRectangular();
    Result kernelPolar();
};

Result ComplexSplitImplNativeCpu::computeSubmit() {
    return polar ? kernelPolar() : kernelRectangular();
}

Result ComplexSplitImplNativeCpu::kernelRectangular() {
    return AutomaticIterator<const CF32, F32, F32>(
        [](const auto& in, auto& a, auto& b) {
            a = in.real();
            b = in.imag();
        },
    input, outputA, outputB);
}

Result ComplexSplitImplNativeCpu::kernelPolar() {
    return AutomaticIterator<const CF32, F32, F32>(
        [](const auto& in, auto& a, auto& b) {
            a = std::hypot(in.real(), in.imag());
            b = std::atan2(in.imag(), in.real());
        },
    input, outputA, outputB);
}

JST_REGISTER_MODULE(ComplexSplitImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
