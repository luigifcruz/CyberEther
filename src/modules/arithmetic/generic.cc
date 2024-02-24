#include "jetstream/modules/arithmetic.hh"

#include "benchmark.cc"

namespace Jetstream {

template<Device D, typename T>
Result Arithmetic<D, T>::create() {
    JST_DEBUG("Initializing Arithmetic module.");
    JST_INIT_IO();

    // Check parameters.

    if (input.buffer.rank() == 0) {
        JST_ERROR("Input buffer rank is 0.");
        return Result::ERROR;
    }

    if (input.buffer.rank() < config.axis) {
        JST_ERROR("Input buffer rank {} is less than axis {}.", input.buffer.rank(), config.axis);
        return Result::ERROR;
    }

    if (input.buffer.shape()[config.axis] == 0) {
        JST_ERROR("Input buffer axis {} is 0.", config.axis);
        return Result::ERROR;
    }

    if (config.operation != ArithmeticOp::Add && D == Device::CUDA) {
        JST_ERROR("Only addition is supported for CUDA arithmetic.");
        return Result::ERROR;
    }

    // Calculate output shape.

    std::vector<U64> output_shape(input.buffer.shape());
    output_shape[config.axis] = 1;

    // Allocate output.

    output.buffer = Tensor<D, T>(output_shape);

    broadcasted_output = output.buffer;
    JST_CHECK(broadcasted_output.broadcast_to(input.buffer.shape()));

    return Result::SUCCESS;
}

template<Device D, typename T>
void Arithmetic<D, T>::info() const {
    JST_DEBUG("  Operation: {}", config.operation);
    JST_DEBUG("  Axis: {}", config.axis);
}

}  // namespace Jetstream
