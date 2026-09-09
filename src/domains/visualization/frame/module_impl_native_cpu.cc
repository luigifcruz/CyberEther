#include <algorithm>
#include <cmath>
#include <limits>

#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct FrameImplNativeCpu : public FrameImpl,
                            public NativeCpuRuntimeContext,
                            public Scheduler::Context {
 public:
    Result validate() final;
    Result create() final;

    Result presentInitialize() override;
    Result presentSubmit() override;
    Result computeSubmit() override;
};

Result FrameImplNativeCpu::validate() {
    JST_CHECK(FrameImpl::validate());

    if (!inputs().contains("frame")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("frame").tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    if (inputTensor.dtype() != DataType::F32) {
        JST_ERROR("[MODULE_FRAME_NATIVE_CPU] Unsupported input data type: {}.",
                  inputTensor.dtype());
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result FrameImplNativeCpu::create() {
    JST_CHECK(FrameImpl::create());
    return Result::SUCCESS;
}

Result FrameImplNativeCpu::presentInitialize() {
    return createPresent();
}

Result FrameImplNativeCpu::presentSubmit() {
    return present();
}

Result FrameImplNativeCpu::computeSubmit() {
    if (!autoRange || input.size() == 0 || channels == 0) {
        return Result::SUCCESS;
    }

    const F32* data = input.data<F32>();
    const U64 count = input.size();
    const U64 colorChannels = (channels == 4) ? 3 : channels;

    F32 lower = std::numeric_limits<F32>::infinity();
    F32 upper = -std::numeric_limits<F32>::infinity();

    for (U64 i = 0; i + channels <= count; i += channels) {
        for (U64 c = 0; c < colorChannels; ++c) {
            const F32 value = data[i + c];
            if (!std::isfinite(value)) {
                continue;
            }
            lower = std::min(lower, value);
            upper = std::max(upper, value);
        }
    }

    if (lower <= upper) {
        autoRangeMin.store(lower);
        autoRangeMax.store(upper);
    }

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(FrameImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
