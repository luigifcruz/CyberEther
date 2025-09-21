#include "../generic.cc"

#include "jetstream/backend/devices/cpu/helpers.hh"

namespace Jetstream {

template<Device D, typename IT, typename OT>
struct FM<D, IT, OT>::Impl {
    F32 kf;
    F32 ref;
};

template<Device D, typename IT, typename OT>
FM<D, IT, OT>::FM() {
    impl = std::make_unique<Impl>();
}

template<Device D, typename IT, typename OT>
FM<D, IT, OT>::~FM() {
    impl.reset();
}

template<Device D, typename IT, typename OT>
Result FM<D, IT, OT>::createCompute(const Context&) {
    JST_TRACE("Create FM compute core.");
    return Result::SUCCESS;
}

template<Device D, typename IT, typename OT>
Result FM<D, IT, OT>::compute(const Context&) {
    for (size_t n = 1; n < input.buffer.size(); n++) {
        output.buffer[n] = std::arg(std::conj(input.buffer[n - 1]) * input.buffer[n]) * impl->ref;
    }

    return Result::SUCCESS;
}

JST_FM_CPU(JST_INSTANTIATION)

}  // namespace Jetstream
