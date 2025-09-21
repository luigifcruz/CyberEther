#include "../generic.cc"

#include "jetstream/backend/devices/cpu/helpers.hh"

namespace Jetstream {

template<Device D, typename IT, typename OT>
struct Cast<D, IT, OT>::Impl {};

template<Device D, typename IT, typename OT>
Cast<D, IT, OT>::Cast() {
    impl = std::make_unique<Impl>();
}

template<Device D, typename IT, typename OT>
Cast<D, IT, OT>::~Cast() {
    impl.reset();
}

template<Device D, typename IT, typename OT>
Result Cast<D, IT, OT>::createCompute(const Context&) {
    JST_TRACE("Create Cast compute core using CPU backend.");
    return Result::SUCCESS;
}

template<Device D, typename IT, typename OT>
Result Cast<D, IT, OT>::compute(const Context&) {
    for (U64 i = 0; i < input.buffer.size(); i++) {
        // CF32 to F32: Take real part and discard imaginary.

        if constexpr (std::is_same<IT, CF32>::value && std::is_same<OT, F32>::value) {
            output.buffer[i] = input.buffer[i].real();
        }

        // CI8 to CF32: Convert integer complex to float complex.

        if constexpr (std::is_same<IT, CI8>::value && std::is_same<OT, CF32>::value) {

            F32 realPart = static_cast<F32>(input.buffer[i].real());
            F32 imagPart = static_cast<F32>(input.buffer[i].imag());
            if (config.scaler != 0.0f) {
                realPart /= config.scaler;
                imagPart /= config.scaler;
            }
            output.buffer[i] = CF32(realPart, imagPart);
        }
    }

    return Result::SUCCESS;
}

JST_CAST_CPU(JST_INSTANTIATION)
JST_CAST_CPU(JST_BENCHMARK)

}  // namespace Jetstream
