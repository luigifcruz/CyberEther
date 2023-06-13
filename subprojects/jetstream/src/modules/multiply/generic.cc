#include "jetstream/modules/multiply.hh"

namespace Jetstream { 

template<Device D, typename T>
Multiply<D, T>::Multiply(const Config& config, 
                         const Input& input) 
         : config(config), input(input) {
    JST_DEBUG("Initializing Multiply module.");
    
    // Initialize output.
    JST_CHECK_THROW(Module::initInput(this->input.factorA));
    JST_CHECK_THROW(Module::initInput(this->input.factorB));
    JST_CHECK_THROW(Module::initOutput(this->output.product, this->input.factorA.shape()));

    // Check parameters.
    if (this->input.factorA.shape(1) != this->input.factorB.shape(1)) {
        JST_FATAL("Input A shape ({}) is different than the" \
            "Input B shape ({}).",
            this->input.factorA.shape(),
            this->input.factorB.shape());
        JST_CHECK_THROW(Result::ERROR);
    }
}

template<Device D, typename T>
void Multiply<D, T>::summary() const {
    JST_INFO("     None");
}

template<Device D, typename T>
Result Multiply<D, T>::Factory(std::unordered_map<std::string, std::any>& configMap,
                               std::unordered_map<std::string, std::any>& inputMap,
                               std::unordered_map<std::string, std::any>& outputMap,
                               std::shared_ptr<Multiply<D, T>>& module) {
    using Module = Multiply<D, T>;

    Module::Config config{};
    Module::Input input{};

    JST_CHECK(Module::BindVariable(inputMap, "factorA", input.factorA));
    JST_CHECK(Module::BindVariable(inputMap, "factorB", input.factorB));

    module = std::make_shared<Module>(config, input);

    JST_CHECK(Module::RegisterVariable(outputMap, "product", module->getProductBuffer()));

    return Result::SUCCESS;
}

}  // namespace Jetstream
