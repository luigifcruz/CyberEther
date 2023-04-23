#include "../generic.cc"

namespace Jetstream {

template<Device D, typename T>
const Result Multiply<D, T>::createCompute(const RuntimeMetadata& meta) {
    JST_TRACE("Create Multiply compute core using CPU backend.");
    return Result::SUCCESS;
}

template<Device D, typename T>
const Result Multiply<D, T>::compute(const RuntimeMetadata& meta) {
    for (U64 i = 0; i < input.factorA.size(); i++) {
        output.product[i] = input.factorA[i] * input.factorB[i];
    }

    return Result::SUCCESS;
}

template class Multiply<Device::CPU, CF32>;
template class Multiply<Device::CPU, F32>;
    
}  // namespace Jetstream
