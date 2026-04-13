#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include <algorithm>

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

    return updatePointPositions();
}

Result ConstellationImpl::updatePointPositions() {
    if (!shapes) {
        return Result::SUCCESS;
    }

    std::span<Extent2D<F32>> positions;
    JST_CHECK(shapes->getPositions("constellation_points", positions));

    const auto paddingScale = axis ? axis->paddingScale() : Extent2D<F32>{1.0f, 1.0f};
    const Extent2D<F32> pointMargin = {
        kConstellationPointSize * ((2.0f * interaction.scale) / interaction.viewSize.x),
        kConstellationPointSize * ((2.0f * interaction.scale) / interaction.viewSize.y),
    };
    const Extent2D<F32> safeScale = {
        std::max(0.0f, paddingScale.x - pointMargin.x),
        std::max(0.0f, paddingScale.y - pointMargin.y),
    };
    const CF32* inputData = input.data<CF32>();

    for (U64 i = 0; i < numberOfPoints; i++) {
        positions[i] = {
            inputData[i].real() * safeScale.x * interaction.zoom,
            inputData[i].imag() * safeScale.y * interaction.zoom,
        };
    }

    updatePositionsFlag = true;

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(ConstellationImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
