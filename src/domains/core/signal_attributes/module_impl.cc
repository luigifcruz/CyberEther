#include "module_impl.hh"

#include <cmath>
#include <string>

#include <jetstream/memory/attributes.hh>

namespace Jetstream::Modules {

Result SignalAttributesImpl::validate() {
    const auto& config = *candidate();

    if (!std::isfinite(config.sampleRate) || config.sampleRate < 0.0f) {
        JST_ERROR("[MODULE_SIGNAL_ATTRIBUTES] Sample rate must be zero or positive.");
        return Result::ERROR;
    }

    if (!std::isfinite(config.frequency)) {
        JST_ERROR("[MODULE_SIGNAL_ATTRIBUTES] Frequency must be finite.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result SignalAttributesImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::DISCONTIGUOUS |
                          Module::Taint::STATELESS));

    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result SignalAttributesImpl::create() {
    input = inputs().at("buffer").tensor;
    output = input.clone();
    JST_CHECK(output.propagateAttributes(input));

    if (sampleRate > 0.0f) {
        JST_CHECK(output.setAttribute(std::string(SampleRateAttribute), sampleRate));
    }
    if (frequency != 0.0f) {
        JST_CHECK(output.setAttribute(std::string(FrequencyAttribute), frequency));
    }

    outputs()["buffer"].produced(name(), "buffer", output);
    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
