#include "../generic.cc"

#include "jetstream/memory/devices/cpu/helpers.hh"
#include "jetstream/memory2/helpers.hh"

namespace Jetstream {

template<Device D, typename T>
struct Duplicate<D, T>::Impl {
};

template<Device D, typename T>
Duplicate<D, T>::Duplicate() {
    pimpl = std::make_unique<Impl>();
}

template<Device D, typename T>
Duplicate<D, T>::~Duplicate() {
    pimpl.reset();
}

template<Device D, typename T>
Result Duplicate<D, T>::createCompute(const Context&) {
    JST_TRACE("Create Duplicate compute core using CPU backend.");

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Duplicate<D, T>::compute(const Context&) {
    if (input.buffer.contiguous()) {
        Memory::Copy(output.buffer, input.buffer);
    } else {
        mem2::AutomaticIterator([](const auto& in, auto& out) {
            out = in;
        }, input.buffer, output.buffer);
    }

    return Result::SUCCESS;
}

JST_DUPLICATE_CPU(JST_INSTANTIATION)
JST_DUPLICATE_CPU(JST_BENCHMARK)

}  // namespace Jetstream
