#include "jetstream/render/devices/metal/buffer.hh"
#include "jetstream/render/devices/metal/kernel.hh"
#include "jetstream/backend/devices/metal/helpers.hh"

namespace Jetstream::Render {

using Implementation = KernelImp<DeviceType::Metal>;

Implementation::KernelImp(const Config& config) : Kernel(config) {
    for (auto& [buffer, _] : config.buffers) {
        buffers.push_back(
            std::dynamic_pointer_cast<BufferImp<DeviceType::Metal>>(buffer)
        );
    }
}

Result Implementation::create() {
    JST_DEBUG("[METAL] Creating kernel.");

    if (config.kernels.contains(DeviceType::Metal) == 0) {
        JST_ERROR("[METAL] Module doesn't have necessary kernel.");
        return Result::ERROR;
    }

    const auto& kernels = config.kernels[DeviceType::Metal];
    auto device = Backend::State<DeviceType::Metal>()->getDevice();

    auto func = NS::TransferPtr(Backend::LoadShader(kernels[0], device));
    if (!func) {
        return Result::ERROR;
    }

    NS::Error* err = nullptr;
    pipelineState = device->newComputePipelineState(func.get(), &err);
    if (!pipelineState) {
        JST_ERROR("[METAL] Failed to create pipeline state:\n{}", err ? err->description()->utf8String() : "unknown");
        return Result::ERROR;
    }

    update();

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    if (pipelineState) {
        pipelineState->release();
        pipelineState = nullptr;
    }

    return Result::SUCCESS;
}

Result Implementation::encode(MTL::ComputeCommandEncoder* encoder) {
    // Check if data needs to be updated.

    if (!shouldEncode()) {
        return Result::SUCCESS;
    }

    // Set pipeline state.

    encoder->setComputePipelineState(pipelineState);

    // Attach compute buffers.

    for (U64 i = 0; i < buffers.size(); i++) {
        encoder->setBuffer(buffers[i]->getHandle(), 0, i);
    }

    // Dispatch threads.

    const auto& [x, y, z] = config.gridSize;

    // TODO: Implement 2D and 3D grid sizes.
    if (y != 1 || z != 1 || config.workgroupSize == 0 ||
        config.workgroupSize > pipelineState->maxTotalThreadsPerThreadgroup()) {
        JST_ERROR("[METAL] Invalid render kernel grid or workgroup size.");
        return Result::ERROR;
    }

    const U64 groupSize = config.workgroupSize > 1 ? config.workgroupSize :
                         pipelineState->maxTotalThreadsPerThreadgroup();
    encoder->dispatchThreads(MTL::Size(x, y, z),
                             MTL::Size(groupSize, 1, 1));

    markScheduled();

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render
