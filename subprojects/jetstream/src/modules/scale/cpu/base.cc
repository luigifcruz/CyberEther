#include "jetstream/modules/scale.hh"

namespace Jetstream {

template<Device D, typename T>
Scale<D, T>::Scale(const Config& config, const Input& input) 
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
    JST_INFO("Size: {}", this->config.size);
    JST_INFO("Amplitude (min, max): ({}, {})", config.range.min, config.range.max);
    JST_INFO("Input Type: {}", getTypeName<T>());
}

template<typename T>
static inline T scale(const T x, const T min, const T max) {
    return (x - min) / (max - min);
}

template<Device D, typename T>
const Result Scale<D, T>::compute(const RuntimeMetadata& meta) {
    auto [min, max] = this->config.range;

    for (U64 i = 0; i < this->config.size; i++) {
        this->output.buffer[i] = scale<T>(this->input.buffer[i], min, max);
    }
    return Result::SUCCESS;
}

template class Scale<Device::CPU, F64>;
template class Scale<Device::CPU, F32>;
    
}  // namespace Jetstream
