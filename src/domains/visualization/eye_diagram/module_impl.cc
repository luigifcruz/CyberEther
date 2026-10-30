#include "module_impl.hh"

#include <algorithm>
#include <cmath>

namespace Jetstream::Modules {

Result EyeDiagramImpl::validate() {
    const auto& config = *candidate();

    if (config.samplesPerSymbol < 1 || config.samplesPerSymbol > 256) {
        JST_ERROR("[MODULE_EYE_DIAGRAM] Samples per symbol ({}) must be between 1 and 256.",
                  config.samplesPerSymbol);
        return Result::ERROR;
    }

    if (config.interpolation != 1 && config.interpolation != 2 &&
        config.interpolation != 4 && config.interpolation != 8 &&
        config.interpolation != 16) {
        JST_ERROR("[MODULE_EYE_DIAGRAM] Interpolation ({}) must be 1, 2, 4, 8, or 16.",
                  config.interpolation);
        return Result::ERROR;
    }

    if (config.component != "real" && config.component != "imag") {
        JST_ERROR("[MODULE_EYE_DIAGRAM] Invalid component '{}'.", config.component);
        return Result::ERROR;
    }

    if (!std::isfinite(config.rangeMin) || !std::isfinite(config.rangeMax) ||
        !std::isfinite(config.rangeMax - config.rangeMin) ||
        config.rangeMax <= config.rangeMin) {
        JST_ERROR("[MODULE_EYE_DIAGRAM] Invalid amplitude range ({} to {}).",
                  config.rangeMin, config.rangeMax);
        return Result::ERROR;
    }

    if (!inputs().contains("signal")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("signal").tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    if (inputTensor.rank() != 1) {
        JST_ERROR("[MODULE_EYE_DIAGRAM] Invalid input rank ({}), expected 1.",
                  inputTensor.rank());
        return Result::ERROR;
    }

    if (inputTensor.dtype() == DataType::F32 && config.component != "real") {
        JST_ERROR("[MODULE_EYE_DIAGRAM] Real input supports only the real component.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result EyeDiagramImpl::define() {
    JST_CHECK(defineInterfaceInput("signal"));
    JST_CHECK(defineInterfaceOutput("signal"));

    return Result::SUCCESS;
}

Result EyeDiagramImpl::create() {
    input = inputs().at("signal").tensor;

    const U64 points = 2 * samplesPerSymbol * interpolation + 1;
    JST_CHECK(output.create(DeviceType::CPU, DataType::F32, {points}));
    std::fill_n(output.data<F32>(), points, 0.5f);

    pending.assign(points, 0.0f);
    interval = 0;
    initialized = false;

    outputs()["signal"].produced(name(), "signal", output);

    return Result::SUCCESS;
}

Result EyeDiagramImpl::reconfigure() {
    return Result::RECREATE;
}

}  // namespace Jetstream::Modules
