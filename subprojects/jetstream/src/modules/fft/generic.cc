#include "jetstream/modules/fft.hh"

namespace Jetstream { 

template<Device D, typename T>
FFT<D, T>::FFT(const Config& config,
               const Input& input) 
         : config(config), input(input) {
    JST_DEBUG("Initializing FFT module.");

    JST_CHECK_THROW(Module::initInput(this->input.buffer));
    JST_CHECK_THROW(Module::initOutput(this->output.buffer, this->input.buffer.shape()));
}

template<Device D, typename T>
void FFT<D, T>::summary() const {
    JST_INFO("    Direction: {}", static_cast<I64>(config.direction));
}

template<Device D, typename T>
Result FFT<D, T>::Factory(std::unordered_map<std::string, std::any>& configMap,
                          std::unordered_map<std::string, std::any>& inputMap,
                          std::unordered_map<std::string, std::any>& outputMap,
                          std::shared_ptr<FFT<D, T>>& module) {
    using Module = FFT<D, T>;

    Module::Config config{};

    JST_CHECK(Module::BindVariable(configMap, "direction", config.direction));

    Module::Input input{};

    JST_CHECK(Module::BindVariable(inputMap, "buffer", input.buffer));

    module = std::make_shared<Module>(config, input);

    JST_CHECK(Module::RegisterVariable(outputMap, "buffer", module->getOutputBuffer()));

    return Result::SUCCESS;
}

}  // namespace Jetstream
