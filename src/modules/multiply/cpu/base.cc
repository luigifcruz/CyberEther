#include "../generic.cc"

#pragma GCC optimize("unroll-loops")
#include "jetstream/memory/devices/cpu/helpers.hh"

namespace Jetstream {

template<Device D, typename T>
Result Multiply<D, T>::createCompute(const Context&) {
    JST_TRACE("Create Multiply compute core using CPU backend.");
    return Result::SUCCESS;
}

template<Device D, typename T>
Result Multiply<D, T>::compute(const Context&) {
    Memory::CPU::AutomaticIterator([](const auto& a, const auto& b, auto& c) {
        if constexpr (std::is_same_v<T, CF32>) {
            c = std::complex<F32>(a.real() * b.real() - a.imag() * b.imag(),
                                  a.real() * b.imag() + a.imag() * b.real());
        } else {
            c = a * b;
        }
    }, a, b, c);

    return Result::SUCCESS;
}

JST_MULTIPLY_CPU(JST_INSTANTIATION)
JST_MULTIPLY_CPU(JST_BENCHMARK)

}  // namespace Jetstream
