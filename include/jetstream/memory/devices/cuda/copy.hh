#ifndef JETSTREAM_MEMORY_CUDA_COPY_HH
#define JETSTREAM_MEMORY_CUDA_COPY_HH

#include "jetstream/memory/devices/cuda/buffer.hh"
#include "jetstream/memory/devices/cuda/copy.hh"
#include "jetstream/memory/devices/cuda/tensor.hh"

#include "jetstream/backend/devices/cuda/helpers.hh"

namespace Jetstream::Memory {

template<typename T>
inline Result Copy(Tensor<Device::CUDA, T>& dst, const Tensor<Device::CUDA, T>& src, const cudaStream_t& stream = 0) {
    if ((dst.size() == src.size()) &&
        (dst.shape() == src.shape()) &&
        (dst.contiguous() && src.contiguous())) {

        if (dst.host_native() && src.host_native()) {
            if (stream != 0) {
                JST_CUDA_CHECK(cudaMemcpyAsync(dst.data() + dst.offset_bytes(), src.data() + src.offset_bytes(), dst.size_bytes(), cudaMemcpyHostToHost, stream), [&]{
                    JST_ERROR("[CUDA:COPY] Failed to copy host to host: {}", err);
                });
            } else {
                JST_CUDA_CHECK(cudaMemcpy(dst.data() + dst.offset_bytes(), src.data() + src.offset_bytes(), dst.size_bytes(), cudaMemcpyHostToHost), [&]{
                    JST_ERROR("[CUDA:COPY] Failed to copy host to host: {}", err);
                });
            }
            return Result::SUCCESS;
        }

        if (dst.device_native() && src.device_native()) {
            if (stream != 0) {
                JST_CUDA_CHECK(cudaMemcpyAsync(dst.data() + dst.offset_bytes(), src.data() + src.offset_bytes(), dst.size_bytes(), cudaMemcpyDeviceToDevice, stream), [&]{
                    JST_ERROR("[CUDA:COPY] Failed to copy device to device: {}", err);
                });
            } else {
                JST_CUDA_CHECK(cudaMemcpy(dst.data() + dst.offset_bytes(), src.data() + src.offset_bytes(), dst.size_bytes(), cudaMemcpyDeviceToDevice), [&]{
                    JST_ERROR("[CUDA:COPY] Failed to copy device to device: {}", err);
                });
            }
            return Result::SUCCESS;
        }

        if (dst.host_native() && src.device_native()) {
            if (stream != 0) {
                JST_CUDA_CHECK(cudaMemcpyAsync(dst.data() + dst.offset_bytes(), src.data() + src.offset_bytes(), dst.size_bytes(), cudaMemcpyDeviceToHost, stream), [&]{
                    JST_ERROR("[CUDA:COPY] Failed to copy device to host: {}", err);
                });
            } else {
                JST_CUDA_CHECK(cudaMemcpy(dst.data() + dst.offset_bytes(), src.data() + src.offset_bytes(), dst.size_bytes(), cudaMemcpyDeviceToHost), [&]{
                    JST_ERROR("[CUDA:COPY] Failed to copy device to host: {}", err);
                });
            }
            return Result::SUCCESS;
        }

        if (dst.device_native() && src.host_native()) {
            if (stream != 0) {
                JST_CUDA_CHECK(cudaMemcpyAsync(dst.data() + dst.offset_bytes(), src.data() + src.offset_bytes(), dst.size_bytes(), cudaMemcpyHostToDevice, stream), [&]{
                    JST_ERROR("[CUDA:COPY] Failed to copy host to device: {}", err);
                });
            } else {
                JST_CUDA_CHECK(cudaMemcpy(dst.data() + dst.offset_bytes(), src.data() + src.offset_bytes(), dst.size_bytes(), cudaMemcpyHostToDevice), [&]{
                    JST_ERROR("[CUDA:COPY] Failed to copy host to device: {}", err);
                });
            }
            return Result::SUCCESS;
        }
    }

    JST_ERROR("[CUDA:COPY] Copy not implemented for non-contiguous tensors.");
    return Result::ERROR;
}

template<typename T>
inline Result Copy(Tensor<Device::CPU, T>& dst, const Tensor<Device::CUDA, T>& src, const cudaStream_t& stream = 0) {
    if ((dst.size() == src.size()) &&
        (dst.shape() == src.shape()) &&
        (dst.contiguous() && src.contiguous())) {
        
        if (src.host_native()) {
            if (stream != 0) {
                JST_CUDA_CHECK(cudaMemcpyAsync(dst.data() + dst.offset_bytes(), src.data() + src.offset_bytes(), dst.size_bytes(), cudaMemcpyHostToHost, stream), [&]{
                    JST_ERROR("[CUDA:COPY] Failed to copy host to host: {}", err);
                });
            } else {
                JST_CUDA_CHECK(cudaMemcpy(dst.data() + dst.offset_bytes(), src.data() + src.offset_bytes(), dst.size_bytes(), cudaMemcpyHostToHost), [&]{
                    JST_ERROR("[CUDA:COPY] Failed to copy host to host: {}", err);
                });
            }
            return Result::SUCCESS;
        }

        if (src.device_native()) {
            if (stream != 0) {
                JST_CUDA_CHECK(cudaMemcpyAsync(dst.data() + dst.offset_bytes(), src.data() + src.offset_bytes(), dst.size_bytes(), cudaMemcpyDeviceToHost, stream), [&]{
                    JST_ERROR("[CUDA:COPY] Failed to copy device to host: {}", err);
                });
            } else {
                JST_CUDA_CHECK(cudaMemcpy(dst.data() + dst.offset_bytes(), src.data() + src.offset_bytes(), dst.size_bytes(), cudaMemcpyDeviceToHost), [&]{
                    JST_ERROR("[CUDA:COPY] Failed to copy device to host: {}", err);
                });
            }
            return Result::SUCCESS;
        }
    }

    JST_ERROR("[CUDA:COPY] Copy not implemented for non-contiguous tensors.");
    return Result::ERROR;
}

}  // namespace Jetstream::Memory

#endif
