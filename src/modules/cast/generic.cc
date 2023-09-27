#include "jetstream/modules/cast.hh"

namespace Jetstream {

template<Device D, typename IT, typename OT>
Result Cast<D, IT, OT>::create() {
    JST_DEBUG("Initializing Cast module.");

    // Initialize output.
    JST_INIT(
        JST_INIT_INPUT("buffer", input.buffer);
        JST_INIT_OUTPUT("buffer", output.buffer, input.buffer.shape());
    );

    // Configure scaler.
    if (config.scaler == 0.0f) {
        if constexpr (std::is_same<IT, F32>::value && std::is_same<IT, I32>::value) {
            config.scaler = 32768.0f;
        } else {
            JST_ERROR("[CAST] No default scaler for the cast operation.");
            return Result::ERROR;
        }
    }

    return Result::SUCCESS;
}

template<Device D, typename IT, typename OT>
void Cast<D, IT, OT>::summary() const {
    JST_INFO("  Scaler:         {}", config.scaler);
    JST_INFO("  Cast Operation: {} -> {}", NumericTypeInfo<IT>::name, NumericTypeInfo<OT>::name);
}

}  // namespace Jetstream
