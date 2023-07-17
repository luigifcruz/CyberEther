#include "jetstream/modules/scale.hh"

namespace Jetstream {

template<Device D, typename T>
Scale<D, T>::Scale(const Config& config,
                   const Input& input) 
         : config(config), input(input) {
    JST_DEBUG("Initializing Scale module.");
    
    // Initialize output.
    JST_CHECK_THROW(Module::initInput(this->input.buffer));
    JST_CHECK_THROW(Module::initOutput(this->output.buffer, this->input.buffer.shape()));
}

template<Device D, typename T>
void Scale<D, T>::summary() const {
    JST_INFO("    Amplitude (min, max): ({}, {})", config.range.min, config.range.max);
}

template<Device D, typename T>
Result Scale<D, T>::Factory(std::unordered_map<std::string, std::any>& configMap,
                            std::unordered_map<std::string, std::any>& inputMap,
                            std::unordered_map<std::string, std::any>& outputMap,
                            std::shared_ptr<Scale<D, T>>& module, 
                            const bool& castFromString) {
    using Module = Scale<D, T>;

    Module::Config config{};

    JST_CHECK(Module::BindVariable(configMap, "range", config.range, castFromString));

    Module::Input input{};

    JST_CHECK(Module::BindVariable(inputMap, "buffer", input.buffer));

    module = std::make_shared<Module>(config, input);

    JST_CHECK(Module::RegisterVariable(outputMap, "buffer", module->getOutputBuffer()));

    return Result::SUCCESS;
}
    
}  // namespace Jetstream
