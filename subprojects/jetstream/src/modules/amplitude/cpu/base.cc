#include "../generic.cc"

namespace Jetstream {

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
Result Amplitude<D, IT, OT>::createCompute(const RuntimeMetadata& meta) {
    JST_TRACE("Create Amplitude compute core using CPU backend.");
    return Result::SUCCESS;
}

template<Device D, typename IT, typename OT>
Result Amplitude<D, IT, OT>::compute(const RuntimeMetadata& meta) {
    const auto& fftSize = input.buffer.shape(1);

    for (U64 i = 0; i < input.buffer.size(); i++) {
        output.buffer[i] = 20.0 * log10(abs(input.buffer[i]) / fftSize);
    }

    return Result::SUCCESS;
}

template class Amplitude<Device::CPU, CF32>;
template class Amplitude<Device::CPU, CF64>;
    
}  // namespace Jetstream
