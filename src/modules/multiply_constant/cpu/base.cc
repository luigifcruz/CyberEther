#include "../generic.cc"

namespace Jetstream {

template<Device D, typename T>
Result MultiplyConstant<D, T>::createCompute(const RuntimeMetadata&) {
    JST_TRACE("Create Multiply Constant compute core using CPU backend.");
    return Result::SUCCESS;
}

template<Device D, typename T>
Result MultiplyConstant<D, T>::compute(const RuntimeMetadata&) {
    for (U64 i = 0; i < input.factor.size(); i++) {
        output.product[i] = input.factor[i] * config.constant;
    }

    return Result::SUCCESS;
}

template class MultiplyConstant<Device::CPU, CF32>;
template class MultiplyConstant<Device::CPU, F32>;

}  // namespace Jetstream
