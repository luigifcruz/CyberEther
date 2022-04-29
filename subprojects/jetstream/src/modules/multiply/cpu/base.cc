#include "jetstream/modules/multiply.hh"

namespace Jetstream {

template<>
Multiply<Device::CPU>::Multiply(const Config& config, const Input& input) 
    : config(config), input(input) {
    JST_DEBUG("Initializing Multiply module with CPU backend.");

    // Intialize output.
    this->InitInput(this->input.A, getBufferSize());
    this->InitInput(this->input.B, getBufferSize());
    this->InitOutput(this->output.product, getBufferSize());

    // Check parameters. 
    if (this->input.A.size() != this->config.size) {
        JST_FATAL("Input A size ({}) is different than the" \
            "configuration size ({}).", this->input.A.size(), this->config.size);
        throw Result::ERROR;
    }

    if (this->input.B.size() != this->config.size) {
        JST_FATAL("Input B size ({}) is different than the" \
            "configuration size ({}).", this->input.B.size(), this->config.size);
        throw Result::ERROR;
    }

    JST_INFO("===== Multiply Module Configuration");
    JST_INFO("Buffer Size: {}", this->config.size);
}

template<>
const Result Multiply<Device::CPU>::compute() {
    for (U64 i = 0; i < this->config.size; i++) {
        this->output.product[i] = this->input.A[i] * this->input.B[i];
    }

    return Result::SUCCESS;
}
    
}  // namespace Jetstream
