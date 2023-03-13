#include "jetstream/modules/amplitude.hh"

namespace Jetstream {

template<Device D, typename IT, typename OT>
Amplitude<D, IT, OT>::Amplitude(const Config& config,
                                const Input& input)
         : config(config), input(input) {
    JST_DEBUG("Initializing Amplitude module.");
    
    // Intialize output.
    JST_CHECK_THROW(this->initInput(this->input.buffer, getBufferSize()));
    JST_CHECK_THROW(this->initOutput(this->output.buffer, getBufferSize()));

    // Check parameters. 
    if (this->input.buffer.size() != this->config.size) {
        JST_FATAL("Input buffer size ({}) is different than the" \
            "configuration size ({}).", this->input.buffer.size(), 
            this->config.size);
        JST_CHECK_THROW(Result::ERROR);
    }
}

template<Device D, typename IT, typename OT>
void Amplitude<D, IT, OT>::summary() const {
    JST_INFO("===== Amplitude Module Configuration");
    JST_INFO("Size: {}", this->config.size);
    JST_INFO("Device: {}", DeviceTypeInfo<D>().name);
    JST_INFO("Input Type: {}", NumericTypeInfo<IT>().name);
    JST_INFO("Output Type: {}", NumericTypeInfo<OT>().name);
}

// Faster Log10 by http://openaudio.blogspot.com/2017/02/faster-log10-and-pow.html
template<typename T>
static inline T log10(T X) {
    T Y, F;
    int E;
    F = frexpf(fabs(X), &E);
    Y = 1.23149591368684f;
    Y *= F;
    Y += -4.11852516267426f;
    Y *= F;
    Y += 6.02197014179219f;
    Y *= F;
    Y += -3.13396450166353f;
    Y += E;
    return Y * 0.3010299956639812f;
}

template<Device D, typename IT, typename OT>
const Result Amplitude<D, IT, OT>::createCompute(const RuntimeMetadata& meta) {
    JST_TRACE("Create Amplitude compute core using CPU backend.");
    return Result::SUCCESS;
}

template<Device D, typename IT, typename OT>
const Result Amplitude<D, IT, OT>::compute(const RuntimeMetadata& meta) {
    for (U64 i = 0; i < this->config.size; i++) {
        this->output.buffer[i] = 20.0 * log10(abs(this->input.buffer[i]) / this->config.size);
    }
    return Result::SUCCESS;
}

template class Amplitude<Device::CPU, CF32>;
template class Amplitude<Device::CPU, CF64>;
    
}  // namespace Jetstream
