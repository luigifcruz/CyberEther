#include "jetstream/modules/scale.hh"

namespace Jetstream {

template<>
Scale<Device::CPU>::Scale(const Config& config, const Input& input) 
    : config(config), input(input) {
    JST_DEBUG("Initializing Scale module with CPU backend.");

    // Intialize output.
    this->InitInput(this->input.buffer, getBufferSize());
    this->InitOutput(this->output.buffer, getBufferSize());

    // Check parameters. 
    if (this->input.buffer.size() != this->config.size) {
        JST_FATAL("Input buffer size ({}) is different than the" \
            "configuration size ({}).", this->input.buffer.size(), 
            this->config.size);
        throw Result::ERROR;
    }

    JST_INFO("===== Scale Module Configuration");
    JST_INFO("Buffer Size: {}", this->config.size);
    JST_INFO("FFT Amplitude (min, max): ({}, {})", config.range.min, config.range.max);
}

static inline F32 scale(const F32 x, const F32 min, const F32 max) {
    return (x - min) / (max - min);
}

template<>
const Result Scale<Device::CPU>::compute() {
    auto [min, max] = this->config.range;

    for (U64 i = 0; i < this->config.size; i++) {
        this->output.buffer[i] = scale(this->input.buffer[i], min, max);
    }
    return Result::SUCCESS;
}
    
}  // namespace Jetstream
