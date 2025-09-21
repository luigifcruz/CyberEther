#include "jetstream/render/devices/webgpu/buffer.hh"
#include "jetstream/render/devices/webgpu/kernel.hh"
#include "jetstream/backend/devices/webgpu/helpers.hh"

namespace Jetstream::Render {

using Implementation = KernelImp<Device::WebGPU>;

Implementation::KernelImp(const Config& config) : Kernel(config) {
    for (auto& [buffer, mode] : config.buffers) {
        buffers.push_back(
            {std::dynamic_pointer_cast<BufferImp<Device::WebGPU>>(buffer), mode}
        );
    }
}

Result Implementation::create() {
    JST_DEBUG("[WebGPU] Creating kernel.");

    // Load kernel from memory.

    if (config.kernels.contains(Device::WebGPU) == 0) {
        JST_ERROR("[WebGPU] Module doesn't have necessary kernel.");
        return Result::ERROR;
    }

    auto device = Backend::State<Device::WebGPU>()->getDevice();

    const auto& kernels = config.kernels[Device::WebGPU];
    WGPUShaderModule kernelModule = Backend::LoadShader(kernels[0], device);

    // Create bind group layout.

    for (U64 i = 0; i < buffers.size(); i++) {
        auto& [buffer, mode] = buffers[i];

        WGPUBindGroupLayoutEntry entry = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
        entry.binding = i;
        entry.visibility = WGPUShaderStage_Compute;
        entry.buffer.type = BufferDescriptorType(buffer, mode);

        bindings.push_back(entry);
    }

    WGPUBindGroupLayoutDescriptor layoutDesc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    layoutDesc.entryCount = static_cast<U32>(bindings.size());
    layoutDesc.entries = bindings.data();

    bindGroupLayout = wgpuDeviceCreateBindGroupLayout(device, &layoutDesc);

    // Create bind group.

    for (U64 i = 0; i < buffers.size(); i++) {
        auto& [buffer, _] = buffers[i];

        WGPUBindGroupEntry entry = WGPU_BIND_GROUP_ENTRY_INIT;
        entry.binding = i;
        entry.buffer = buffer->getHandle();
        entry.offset = 0;
        entry.size = buffer->byteSize();

        bindGroupEntries.push_back(entry);
    }

    WGPUBindGroupDescriptor bindGroupDesc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bindGroupDesc.layout = bindGroupLayout;
    bindGroupDesc.entryCount = static_cast<U32>(bindGroupEntries.size());
    bindGroupDesc.entries = bindGroupEntries.data();

    bindGroup = wgpuDeviceCreateBindGroup(device, &bindGroupDesc);

    // Create pipeline layout.

    WGPUPipelineLayoutDescriptor pipelineLayoutDesc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pipelineLayoutDesc.bindGroupLayoutCount = 1;
    pipelineLayoutDesc.bindGroupLayouts = &bindGroupLayout;

    pipelineLayout = wgpuDeviceCreatePipelineLayout(device, &pipelineLayoutDesc);

    // Create compute pipeline.

    WGPUComputePipelineDescriptor computePipelineDescriptor = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    computePipelineDescriptor.compute.module = kernelModule;
    computePipelineDescriptor.compute.entryPoint = {"main", WGPU_STRLEN};
    computePipelineDescriptor.layout = pipelineLayout;

    pipeline = wgpuDeviceCreateComputePipeline(device, &computePipelineDescriptor);

    // Clean up.

    this->updated = true;

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    bindings.clear();
    bindGroupEntries.clear();

    return Result::SUCCESS;
}

Result Implementation::encode(WGPUComputePassEncoder& computePassEncoder) {
    // Check if data needs to be updated.

    if (!this->updated) {
        return Result::SUCCESS;
    }
    this->updated = false;

    // Create compute pass.

    wgpuComputePassEncoderSetPipeline(computePassEncoder, pipeline);
    wgpuComputePassEncoderSetBindGroup(computePassEncoder, 0, bindGroup, 0, nullptr);

    // Dispatch compute work.

    const auto& [x, y, z] = config.gridSize;

    if (y != 1 || z != 1) {
        JST_ERROR("[WebGPU] 2D and 3D grid sizes are not implemented.");
        return Result::ERROR;
    }

    wgpuComputePassEncoderDispatchWorkgroups(computePassEncoder, x, y, z);

    return Result::SUCCESS;
}

WGPUBufferBindingType Implementation::BufferDescriptorType(const std::shared_ptr<Buffer>& buffer,
                                                           const Kernel::AccessMode& mode) {
    const auto& bufferType = buffer->getConfig().target;

    if ((bufferType & Buffer::Target::UNIFORM) == Buffer::Target::UNIFORM) {
        return WGPUBufferBindingType_Uniform;
    }

    if ((bufferType & Buffer::Target::STORAGE) == Buffer::Target::STORAGE) {
        if ((mode & Kernel::AccessMode::READ) == Kernel::AccessMode::READ &&
            (mode & Kernel::AccessMode::WRITE) != Kernel::AccessMode::WRITE) {
            return WGPUBufferBindingType_ReadOnlyStorage;
        }
        return WGPUBufferBindingType_Storage;
    }

    JST_ERROR("[WebGPU] Invalid buffer usage.");
    throw Result::ERROR;
}

}  // namespace Jetstream::Render
