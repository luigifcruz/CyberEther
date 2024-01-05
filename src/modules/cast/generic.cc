#include "jetstream/modules/cast.hh"

namespace Jetstream {

template<Device D, typename IT, typename OT>
Result Cast<D, IT, OT>::create() {
    JST_DEBUG("Initializing Cast module.");
    JST_INIT_IO();

    // Configure scaler.

    if (config.scaler == 0.0f) {
        if constexpr (std::is_same<IT, F32>::value && std::is_same<IT, I32>::value) {
            config.scaler = 32768.0f;
        } else {
            JST_ERROR("[CAST] No default scaler for the cast operation.");
            return Result::ERROR;
        }
    }

    // Allocate output.

    output.buffer = Tensor<D, OT>(input.buffer.shape());

    return Result::SUCCESS;
}

template<Device D, typename IT, typename OT>
void Cast<D, IT, OT>::info() const {
    JST_INFO("  Scaler:         {}", config.scaler);
    JST_INFO("  Cast Operation: {} -> {}", NumericTypeInfo<IT>::name, NumericTypeInfo<OT>::name);
}

}  // namespace Jetstream
