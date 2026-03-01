#include <cmath>
#include <complex>

#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct FilterTapsImplNativeCpu : public FilterTapsImpl,
                                 public Runtime::Context,
                                 public Scheduler::Context {
 public:
    Result create() final;

    Result computeSubmit() override;

 private:
    bool baked = false;

    Result generateCoeffs();
};

Result FilterTapsImplNativeCpu::create() {
    JST_CHECK(FilterTapsImpl::create());

    if (coeffs.dtype() != DataType::CF32) {
        JST_ERROR("[MODULE_FILTER_TAPS_NATIVE_CPU] Only CF32 output is supported.");
        return Result::ERROR;
    }

    baked = false;

    return Result::SUCCESS;
}

Result FilterTapsImplNativeCpu::computeSubmit() {
    if (baked) {
        return Result::SUCCESS;
    }

    JST_CHECK(generateCoeffs());

    baked = true;

    return Result::SUCCESS;
}

Result FilterTapsImplNativeCpu::generateCoeffs() {
    const F64 filterWidth = (bandwidth / sampleRate) / 2.0;
    const std::complex<F64> j(0.0, 1.0);
    const U64 heads = center.size();

    for (U64 c = 0; c < heads; ++c) {
        const F64 filterOffset = (center[c] / (sampleRate / 2.0)) / 2.0;

        for (U64 i = 0; i < taps; ++i) {
            const F64 fi = static_cast<F64>(i);
            const F64 halfLen = static_cast<F64>(taps - 1) / 2.0;
            const F64 n = fi - halfLen;

            // Sinc function.
            const F64 sincVal = (n == 0.0) ? (2.0 * filterWidth) :
                std::sin(2.0 * JST_PI * filterWidth * n) / (JST_PI * n);

            // Blackman window.
            const F64 windowVal = 0.42 -
                                  0.50 * std::cos(2.0 * JST_PI * fi / (taps - 1)) +
                                  0.08 * std::cos(4.0 * JST_PI * fi / (taps - 1));

            // Upconversion to center frequency.
            const auto upconvert = std::exp(j * 2.0 * JST_PI * n * filterOffset);

            const auto result = sincVal * windowVal * upconvert;
            const CF32 val(static_cast<F32>(result.real()),
                           static_cast<F32>(result.imag()));

            coeffs.at<CF32>(c, i) = val;
        }
    }

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(FilterTapsImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
