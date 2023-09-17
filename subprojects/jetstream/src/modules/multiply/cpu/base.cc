#include "../generic.cc"

namespace Jetstream {

template<Device D, typename T>
Result Multiply<D, T>::createCompute(const RuntimeMetadata&) {
    JST_TRACE("Create Multiply compute core using CPU backend.");
    return Result::SUCCESS;
}

template<Device D, typename T>
Result Multiply<D, T>::compute(const RuntimeMetadata&) {
    for (U64 i = 0; i < input.factorA.size(); i++) {
       output.product.at(i) = input.factorA.at(i) * input.factorB.at(i);
    }

    return Result::SUCCESS;
}

// TODO: Remove in favor of module manifest.
template class Multiply<Device::CPU, CF32>;
template class Multiply<Device::CPU, F32>;
    
}  // namespace Jetstream
