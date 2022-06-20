#include "jetstream/modules/multiply.hh"

namespace Jetstream {

template<Device D, typename T>
Multiply<D, T>::Multiply(const Config& config, const Input& input) 
    : config(config), input(input) {
    JST_DEBUG("Initializing Multiply module with CPU backend.");

    // Intialize output.
    this->InitInput(this->input.factorA, getBufferSize());
    this->InitInput(this->input.factorB, getBufferSize());
    this->InitOutput(this->output.product, getBufferSize());

    // Check parameters. 
    if (this->input.factorA.size() != this->config.size) {
        JST_FATAL("Input A size ({}) is different than the" \
            "configuration size ({}).", this->input.factorA.size(),
            this->config.size);
        throw Result::ERROR;
    }

    if (this->input.factorB.size() != this->config.size) {
        JST_FATAL("Input B size ({}) is different than the" \
            "configuration size ({}).", this->input.factorB.size(), 
            this->config.size);
        throw Result::ERROR;
    }

    JST_INFO("===== Multiply Module Configuration");
    JST_INFO("Size: {}", this->config.size);
    JST_INFO("Input Type: {}", getTypeName<T>());
}

template<Device D, typename T>
const Result Multiply<D, T>::compute(const RuntimeMetadata& meta) {
    for (U64 i = 0; i < this->config.size; i++) {
        this->output.product[i] = 
            this->input.factorA[i] * this->input.factorB[i];
    }

    return Result::SUCCESS;
}

template class Multiply<Device::CPU, CF64>;
template class Multiply<Device::CPU, CF32>;
template class Multiply<Device::CPU, F64>;
template class Multiply<Device::CPU, F32>;
    
}  // namespace Jetstream
