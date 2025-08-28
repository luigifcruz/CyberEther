#include "../generic.cc"

namespace Jetstream {

template<Device D, typename T>
struct MultiplyConstant<D, T>::Impl {};

template<Device D, typename T>
MultiplyConstant<D, T>::MultiplyConstant() {
    impl = std::make_unique<Impl>();
}

template<Device D, typename T>
MultiplyConstant<D, T>::~MultiplyConstant() {
    impl.reset();
}

template<Device D, typename T>
Result MultiplyConstant<D, T>::createCompute(const Context&) {
    JST_TRACE("Create Multiply Constant compute core using CPU backend.");
    return Result::SUCCESS;
}

template<Device D, typename T>
Result MultiplyConstant<D, T>::compute(const Context&) {
    for (U64 i = 0; i < input.factor.size(); i++) {
        output.product[i] = input.factor[i] * config.constant;
    }

    return Result::SUCCESS;
}

JST_MULTIPLY_CONSTANT_CPU(JST_INSTANTIATION)
JST_MULTIPLY_CONSTANT_CPU(JST_BENCHMARK)

}  // namespace Jetstream
