#include "jetstream/modules/fm.hh"

namespace Jetstream {

template<Device D, typename IT, typename OT>
Result FM<D, IT, OT>::create() {
    JST_DEBUG("Initializing FM module.");
    JST_INIT_IO();

    // Initialize constant coefficients.

    impl->kf = 100e3f / config.sampleRate;
    impl->ref = 1.0f / (2.0f * JST_PI * impl->kf);

    // Allocate output.

    JST_CHECK(output.buffer.create(D, mem2::TypeToDataType<OT>(), input.buffer.shape()));

    return Result::SUCCESS;
}

template<Device D, typename IT, typename OT>
void FM<D, IT, OT>::info() const {
    JST_DEBUG("  Sample Rate: {:.2f} MHz", config.sampleRate / JST_MHZ);
}

}  // namespace Jetstream
