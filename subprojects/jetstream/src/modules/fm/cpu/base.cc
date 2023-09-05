#include "../generic.cc"

#include "jetstream/backend/devices/cpu/helpers.hh"

namespace Jetstream {

template<Device D, typename T>
Result FM<D, T>::createCompute(const RuntimeMetadata&) {
    JST_TRACE("Create FM compute core.");
    return Result::SUCCESS;
}

template<Device D, typename T>
Result FM<D, T>::compute(const RuntimeMetadata&) {
    std::vector<CF32> tmp(input.buffer.size());

    for (U64 i = 0; i < input.buffer.size(); i++) {
        tmp[i] = {input.buffer[i].real(), input.buffer[i].imag()};
        tmp[i] *= (i % 2) == 0 ? CF32{1.0, 0.0} : CF32{-1.0, 0.0};
    }

    // Direct implementation of https://www.embedded.com/dsp-tricks-frequency-demodulation-algorithms/
    for (size_t n = 1; n < tmp.size() - 1; ++n) {
        F32 i_prime = tmp[n + 1].real() - tmp[n - 1].real();
        F32 q_prime = tmp[n + 1].imag() - tmp[n - 1].imag();

        F32 i = tmp[n].real();
        F32 q = tmp[n].imag();

        output.buffer[n - 1] = (i * q_prime - q * i_prime) / (i * i + q * q);
    }

    return Result::SUCCESS;
}

template class FM<Device::CPU, CF32>;
    
}  // namespace Jetstream
