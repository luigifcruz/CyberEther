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

    mem2::Shape output_shape(input.buffer.shape());
    output_shape[config.axis] = 1;

    // Allocate output.

    JST_CHECK(output.buffer.create(D, mem2::TypeToDataType<T>(), output_shape));

    JST_CHECK(pimpl->broadcasted_output.create(D, output.buffer));
    JST_CHECK(pimpl->broadcasted_output.broadcast_to(input.buffer.shape()));

    // Apply squeeze if requested.
    if (config.squeeze) {
        output.buffer.squeeze_dims(config.axis);
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
void Arithmetic<D, T>::info() const {
    JST_DEBUG("  Operation: {}", config.operation);
    JST_DEBUG("  Axis: {}", config.axis);
    JST_DEBUG("  Squeeze: {}", config.squeeze);
}

}  // namespace Jetstream
