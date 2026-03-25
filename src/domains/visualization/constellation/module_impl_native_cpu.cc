#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct ConstellationImplNativeCpu : public ConstellationImpl,
                                    public NativeCpuRuntimeContext,
                                    public Scheduler::Context {
 public:
    Result create() final;

    Result presentInitialize() override;
    Result presentSubmit() override;
    Result computeSubmit() override;
};

Result ConstellationImplNativeCpu::create() {
    // Create parent.

    JST_CHECK(ConstellationImpl::create());

    // Validate input dtype.

    if (input.dtype() != DataType::CF32) {
        JST_ERROR("[MODULE_CONSTELLATION_NATIVE_CPU] "
                  "Unsupported input data type: {}.",
                  input.dtype());
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result ConstellationImplNativeCpu::presentInitialize() {
    return createPresent();
}

Result ConstellationImplNativeCpu::presentSubmit() {
    return present();
}

Result ConstellationImplNativeCpu::computeSubmit() {
    if (!shapes) {
        return Result::SUCCESS;
    }

    // Get position buffer from shapes component.

    std::span<Extent2D<F32>> positions;
    JST_CHECK(shapes->getPositions("constellation_points", positions));

    // Map complex values to 2D positions.

    const CF32* inputData = input.data<CF32>();

    for (U64 i = 0; i < numberOfPoints; i++) {
        positions[i] = {inputData[i].real(), inputData[i].imag()};
    }

    // Schedule positions for update.

    updatePositionsFlag = true;

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(ConstellationImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
