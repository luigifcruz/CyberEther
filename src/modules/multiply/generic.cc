#include "jetstream/modules/multiply.hh"

namespace Jetstream {

template<Device D, typename T>
Result Multiply<D, T>::create() {
    JST_DEBUG("Initializing Multiply module.");

    // Initialize output.
    JST_INIT(
        JST_INIT_INPUT("factorA", input.factorA);
        JST_INIT_INPUT("factorB", input.factorB);
        JST_INIT_OUTPUT("product", output.product, input.factorA.shape());
    );

    // Check parameters.
    if (input.factorA.shape()[1] != input.factorB.shape()[1]) {
        JST_ERROR("Input A shape ({}) is different than the " \
            "Input B shape ({}).",
            input.factorA.shape(),
            input.factorB.shape());
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
void Multiply<D, T>::summary() const {
    JST_INFO("  None");
}

}  // namespace Jetstream
