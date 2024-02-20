#include <regex>

#include "../generic.cc"

#include "jetstream/memory/devices/cuda/copy.hh"

namespace Jetstream {

template<Device D, typename T>
struct Arithmetic<D, T>::Impl {
    std::vector<U64> grid;
    std::vector<U64> block;

    struct Meta {
        void* ptr;
        size_t rank;
        size_t shape[8];
        size_t strides[8];
    };

    Meta inputMeta;
    Meta outputMeta;
    U64 size;

    std::vector<void*> arguments;

    Tensor<Device::CUDA, T> input;
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
Result Arithmetic<D, T>::createCompute(const Context& ctx) {
    JST_TRACE("Create Arithmetic compute core using CUDA backend.");

    // Create CUDA kernel.

    std::string kernel = R"""(
        struct Meta {
            void* ptr;
            size_t rank;
            size_t shape[8];
            size_t strides[8];
        };

        // TODO: Improve this naive implementation.
        // TODO: Implement global stride handler.
        // TODO: Implement all arithmetic operations.

        __global__ void arithmetic(Meta input, Meta output, size_t size) {
            size_t id = blockIdx.x * blockDim.x + threadIdx.x;

            // Return if ID is out of bounds.

            if (id > size) {
                return;
            }
            
            // Calculate shape from ID.

            size_t shape[8];
            for (size_t i = 0; i < input.rank; i++) {
                shape[i] = id % input.shape[i];
                id /= input.shape[i];
            }

            // Calculate input and output offset from shape.

            size_t input_offset = 0;
            size_t output_offset = 0;

            for (size_t i = 0; i < input.rank; i++) {
                input_offset += shape[i] * input.strides[i];
                output_offset += shape[i] * output.strides[i];
            }

            // Reinterpret input and output pointers.

            [//CAST//]

            // Perform arithmetic operation.

            [//OP//]
        }
    )""";

    if constexpr (std::is_same_v<T, F32>) {
        const std::string cast = R"""(
            const auto* input_ptr = reinterpret_cast<float*>(input.ptr);
            auto* output_ptr = reinterpret_cast<float*>(output.ptr);
        )""";
        kernel = std::regex_replace(kernel, std::regex(R"(\[\/\/CAST\/\/\])"), cast);

        const std::string operation = R"""(
            atomicAdd(&output_ptr[output_offset], input_ptr[input_offset]);
        )""";
        kernel = std::regex_replace(kernel, std::regex(R"(\[\/\/OP\/\/\])"), operation);
    } else if constexpr (std::is_same_v<T, CF32>) {
        const std::string cast = R"""(
            const auto* input_ptr = reinterpret_cast<float2*>(input.ptr);
            auto* output_ptr = reinterpret_cast<float2*>(output.ptr);
        )""";
        kernel = std::regex_replace(kernel, std::regex(R"(\[\/\/CAST\/\/\])"), cast);

        const std::string operation = R"""(
            atomicAdd(&output_ptr[output_offset].x, input_ptr[input_offset].x);
            atomicAdd(&output_ptr[output_offset].y, input_ptr[input_offset].y);
        )""";
        kernel = std::regex_replace(kernel, std::regex(R"(\[\/\/OP\/\/\])"), operation);
    }

    ctx.cuda->createKernel("arithmetic", kernel);

    // Initialize kernel size.

    U64 threadsPerBlock = 512;
    U64 blocksPerGrid = (input.buffer.size() + threadsPerBlock - 1) / threadsPerBlock;

    pimpl->grid = { blocksPerGrid, 1, 1 };
    pimpl->block = { threadsPerBlock, 1, 1 };

    // Initialize kernel input.

    if (!input.buffer.device_native() && input.buffer.contiguous()) {
        pimpl->input = Tensor<Device::CUDA, T>(input.buffer.shape());
    } else {
        pimpl->input = input.buffer;
    }

    // Initialize kernel arguments.

    pimpl->inputMeta = {
        pimpl->input.data(),
        pimpl->input.rank(),
    };

    for (U64 i = 0; i < pimpl->input.rank(); i++) {
        pimpl->inputMeta.shape[i] = pimpl->input.shape()[i];
        pimpl->inputMeta.strides[i] = pimpl->input.stride()[i];
    }

    pimpl->outputMeta = {
        broadcasted_output.data(),
        broadcasted_output.rank(),
    };

    for (U64 i = 0; i < broadcasted_output.rank(); i++) {
        pimpl->outputMeta.shape[i] = broadcasted_output.shape()[i];
        pimpl->outputMeta.strides[i] = broadcasted_output.stride()[i];
    }

    pimpl->size = input.buffer.size();

    pimpl->arguments = {
        &pimpl->inputMeta,
        &pimpl->outputMeta,
        &pimpl->size,
    };

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Arithmetic<D, T>::compute(const Context& ctx) {
    if (!input.buffer.device_native() && input.buffer.contiguous()) {
        JST_CHECK(Memory::Copy(pimpl->input, input.buffer, ctx.cuda->stream()));
    }

    JST_CUDA_CHECK(cudaMemsetAsync(output.buffer.data(), 0, output.buffer.size_bytes(), ctx.cuda->stream()), [&]{
        JST_ERROR("Failed to clear output buffer: {}", err);
    });

    JST_CHECK(ctx.cuda->launchKernel("arithmetic", 
                                     pimpl->grid, 
                                     pimpl->block, 
                                     pimpl->arguments.data()));

    return Result::SUCCESS;
}

JST_ARITHMETIC_CUDA(JST_INSTANTIATION)
JST_ARITHMETIC_CUDA(JST_BENCHMARK)

}  // namespace Jetstream