#include "module_impl.hh"

#include <algorithm>
#include <cmath>

namespace Jetstream::Modules {

Result EyeDiagramImpl::validate() {
    const auto& config = *candidate();
    if (config.samplesPerSymbol < 1 || config.samplesPerSymbol > 256 ||
        (config.interpolation != 1 && config.interpolation != 2 &&
         config.interpolation != 4 && config.interpolation != 8 &&
         config.interpolation != 16)) {
        JST_ERROR("[MODULE_EYE_DIAGRAM] Invalid samples per symbol or interpolation.");
        return Result::ERROR;
    }
    if ((config.component != "real" && config.component != "imag") ||
        !std::isfinite(config.rangeMin) || !std::isfinite(config.rangeMax) ||
        !std::isfinite(config.rangeMax - config.rangeMin) ||
        config.rangeMax <= config.rangeMin) {
        JST_ERROR("[MODULE_EYE_DIAGRAM] Invalid component or amplitude range.");
        return Result::ERROR;
    }
    if (!inputs().contains("signal")) {
        return Result::SUCCESS;
    }
    const auto& tensor = inputs().at("signal").tensor;
    if (!tensor.validShape() || tensor.size() == 0) {
        return Result::SUCCESS;
    }
    if (tensor.rank() != 1 ||
        (tensor.dtype() != DataType::F32 && tensor.dtype() != DataType::CF32) ||
        (tensor.dtype() == DataType::F32 && config.component != "real")) {
        JST_ERROR("[MODULE_EYE_DIAGRAM] Expected rank-1 F32 (real) or CF32 input.");
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
    // Deliberately omit RF attributes: this is normalized symbol time, not frequency.
    outputs()["signal"].produced(name(), "signal", output);
    return Result::SUCCESS;
}

Result EyeDiagramImpl::reconfigure() {
    return Result::RECREATE;
}

}  // namespace Jetstream::Modules
