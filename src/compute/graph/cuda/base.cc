#include "jetstream/compute/graph/cuda.hh"
#include "jetstream/backend/devices/cuda/helpers.hh"

namespace Jetstream {

CUDA::CUDA() {
    JST_DEBUG("Creating new CUDA compute graph.");
    metadata = std::make_shared<RuntimeMetadata>();
}

Result CUDA::create() {
    // Create CUDA stream.

    JST_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), [&]{
        JST_ERROR("[CUDA] Can't create stream.");
    });

    // Copy params to metadata.

    metadata->cuda.stream = stream;

    // Create blocks.

    for (const auto& block : blocks) {
        JST_CHECK(block->createCompute(*metadata));
    }

    return Result::SUCCESS;
}

Result CUDA::computeReady() {
    for (const auto& block : blocks) {
        JST_CHECK(block->computeReady());
    }
    return Result::SUCCESS;
}

Result CUDA::compute() {
    // Execute blocks.

    for (const auto& block : blocks) {
        JST_CHECK(block->compute(*metadata));

        // Check for CUDA errors.

        JST_CUDA_CHECK(cudaGetLastError(), [&]{
            JST_ERROR("[CUDA] Module kernel execution failed.");
        });
    }

    // Wait for all blocks to finish.

    JST_CUDA_CHECK(cudaStreamSynchronize(stream), [&]{
        JST_ERROR("[CUDA] Can't synchronize stream.");
    });

    return Result::SUCCESS;
}

Result CUDA::destroy() {
    // Destroy blocks.

    for (const auto& block : blocks) {
        JST_CHECK(block->destroyCompute(*metadata));
    }
    blocks.clear();

    // Destroy CUDA stream.

    JST_CUDA_CHECK(cudaStreamDestroy(stream), [&]{
        JST_ERROR("[CUDA] Can't destroy stream.");
    });

    return Result::SUCCESS;
}

}  // namespace Jetstream
