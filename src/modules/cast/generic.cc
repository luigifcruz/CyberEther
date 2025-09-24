#include "jetstream/modules/cast.hh"

#include "benchmark.cc"

namespace Jetstream {

template<Device D, typename IT, typename OT>
Result Cast<D, IT, OT>::create() {
    JST_DEBUG("Initializing Cast module.");
    JST_INIT_IO();

    // Configure scaler.

    if (config.scaler == 0.0f) {
        if constexpr (std::is_same<IT, CI8>::value && std::is_same<OT, CF32>::value) {
            config.scaler = 128.0f;
        } else if constexpr (std::is_same<IT, CF32>::value && std::is_same<OT, F32>::value) {
            config.scaler = 1.0f;
        } else {
            JST_ERROR("[CAST] No default scaler for the cast operation.");
            return Result::ERROR;
        }
    }

    // Allocate output.

    JST_CHECK(output.buffer.create(D, mem2::TypeToDataType<OT>(), input.buffer.shape()));

    return Result::SUCCESS;
}

template<Device D, typename IT, typename OT>
void Cast<D, IT, OT>::info() const {
    JST_DEBUG("  Scaler:         {}", config.scaler);
    JST_DEBUG("  Cast Operation: {} -> {}", NumericTypeInfo<IT>::name, NumericTypeInfo<OT>::name);
}

}  // namespace Jetstream
