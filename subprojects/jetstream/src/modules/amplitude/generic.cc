#include "jetstream/modules/amplitude.hh"

namespace Jetstream { 

template<Device D, typename IT, typename OT>
Amplitude<D, IT, OT>::Amplitude(const Config& config,
                                const Input& input)
         : config(config), input(input) {
    JST_DEBUG("Initializing Amplitude module.");
    
    // Initialize output.
    JST_CHECK_THROW(Module::initInput(this->input.buffer));
    JST_CHECK_THROW(Module::initOutput(this->output.buffer, this->input.buffer.shape()));
}

template<Device D, typename IT, typename OT>
void Amplitude<D, IT, OT>::summary() const {
    JST_INFO("    None");
}

template<Device D, typename IT, typename OT>
Result Amplitude<D, IT, OT>::Factory(std::unordered_map<std::string, std::any>& configMap,
                                     std::unordered_map<std::string, std::any>& inputMap,
                                     std::unordered_map<std::string, std::any>& outputMap,
                                     std::shared_ptr<Amplitude<D, IT, OT>>& module) {
    using Module = Amplitude<D, IT, OT>;

    Module::Config config{};
    Module::Input input{};

    JST_CHECK(Module::BindVariable(inputMap, "buffer", input.buffer));

    module = std::make_shared<Module>(config, input);

    JST_CHECK(Module::RegisterVariable(outputMap, "buffer", module->getOutputBuffer()));

    return Result::SUCCESS;
}

}  // namespace Jetstream
