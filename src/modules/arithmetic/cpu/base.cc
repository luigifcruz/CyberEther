#include "../generic.cc"

#pragma GCC optimize("unroll-loops")
#include "jetstream/memory/devices/cpu/helpers.hh"

namespace Jetstream {

template<Device D, typename T>
struct Arithmetic<D, T>::Impl {
};

template<Device D, typename T>
Arithmetic<D, T>::Arithmetic() {
    pimpl = std::make_unique<Impl>();
}

template<Device D, typename T>
Arithmetic<D, T>::~Arithmetic() {
    pimpl.reset();
}

template<Device D, typename T>
Result Arithmetic<D, T>::createCompute(const Context&) {
    JST_TRACE("Create Arithmetic compute core using CPU backend.");

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Arithmetic<D, T>::compute(const Context&) {
    for (U64 i = 0; i < output.buffer.size(); i++) {
        output.buffer[i] = 0;
    }

    switch (config.operation) {
        case ArithmeticOp::Add:
            Memory::CPU::AutomaticIterator([](auto& a, const auto& b) {
                a += b;
            }, broadcasted_output, input.buffer);
            break;
        case ArithmeticOp::Sub:
            Memory::CPU::AutomaticIterator([](auto& a, const auto& b) {
                a -= b;
            }, broadcasted_output, input.buffer);
            break;
        case ArithmeticOp::Mul:
            Memory::CPU::AutomaticIterator([](auto& a, const auto& b) {
                a *= b;
            }, broadcasted_output, input.buffer);
            break;
        case ArithmeticOp::Div:
            Memory::CPU::AutomaticIterator([](auto& a, const auto& b) {
                a /= b;
            }, broadcasted_output, input.buffer);
            break;
    }

    return Result::SUCCESS;
}

JST_ARITHMETIC_CPU(JST_INSTANTIATION)
JST_ARITHMETIC_CPU(JST_BENCHMARK)

}  // namespace Jetstream
