#include "../generic.cc"

#include <cufft.h>

namespace Jetstream {

template<Device D, typename IT, typename OT>
struct FFT<D, IT, OT>::Impl {
    cufftHandle plan;

    Tensor<Device::CUDA, IT> input;
};

template<Device D, typename IT, typename OT>
FFT<D, IT, OT>::FFT() {
    pimpl = std::make_unique<Impl>();
}

template<Device D, typename IT, typename OT>
FFT<D, IT, OT>::~FFT() {
    pimpl.reset();
}

template<Device D, typename IT, typename OT>
Result FFT<D, IT, OT>::createCompute(const Context& ctx) {
    JST_TRACE("Create FFT compute core using CUDA backend.");

    // Initialize kernel input.

    if (!input.buffer.device_native()) {
        pimpl->input = Tensor<Device::CUDA, IT>(input.buffer.shape());
    } else {
        pimpl->input = input.buffer;
    }

    // Initialize cuFFT.

    JST_CUFFT_CHECK(cufftCreate(&pimpl->plan), [&](){
        JST_FATAL("Failed to create cuFFT instance: {}", err);
    });

    // Create FFT plan.
    // This should get the shape() and stride() of the input buffer and create a plan for the FFT.

    const U64 last_axis = input.buffer.rank() - 1;

    int rank = 1;

    // FFT size for each dimension.
    int n[] = { static_cast<int>(input.buffer.shape()[last_axis]) }; 

    // Distance between successive input element and output element.
    int istride = input.buffer.stride()[last_axis];
    int ostride = 1;

    // Number of batched FFTs. 
    U64 numberOfOperations = 1;
    for (U64 i = 0; i < last_axis; i++) {
        numberOfOperations *= input.buffer.shape()[i];
    }

    // Distance between input batches and output batches.
    int idist = input.buffer.shape()[last_axis];
    int odist = input.buffer.shape()[last_axis];

    // Input size with pitch, this is ignored for 1D tansformations.
    int inembed[] = { 0 }; 
    int onembed[] = { 0 };

    JST_CUFFT_CHECK(cufftPlanMany(&pimpl->plan,
                                  rank,
                                  n,
                                  inembed,
                                  istride,
                                  idist,
                                  onembed,
                                  ostride,
                                  odist,
                                  CUFFT_C2C,
                                  numberOfOperations), [&]{
        JST_ERROR("Failed to create FFT plan: {}.", err);
    });

    // Set cuFFT stream.

    JST_CUFFT_CHECK(cufftSetStream(pimpl->plan, ctx.cuda->stream()), [&](){
        JST_FATAL("Failed to set cuFFT stream: {}", err);
    });

    return Result::SUCCESS;
}

template<Device D, typename IT, typename OT>
Result FFT<D, IT, OT>::destroyCompute(const Context&) {
    JST_TRACE("Destroy FFT compute core using CUDA backend.");

    JST_CUFFT_CHECK(cufftDestroy(pimpl->plan), [&](){
        JST_ERROR("Failed to destroy FFT plan: {}.", err);
    });

    return Result::SUCCESS;
}

template<>
Result FFT<Device::CUDA, CF32, CF32>::compute(const Context& ctx) {
    if (!input.buffer.device_native()) {
        JST_CHECK(Memory::Copy(pimpl->input, input.buffer, ctx.cuda->stream()));
    }

    const auto input = reinterpret_cast<cufftComplex*>(pimpl->input.data());
    const auto output = reinterpret_cast<cufftComplex*>(this->output.buffer.data());
    const auto direction = (config.forward) ? CUFFT_FORWARD : CUFFT_INVERSE;

    JST_CUFFT_CHECK(cufftExecC2C(pimpl->plan, input, output, direction), [&](){
        JST_ERROR("Failed to execute FFT: {}.", err);
    });

    return Result::SUCCESS;
}

JST_FFT_CUDA(JST_INSTANTIATION)
JST_FFT_CUDA(JST_BENCHMARK)

}  // namespace Jetstream
