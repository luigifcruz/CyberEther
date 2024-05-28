#include "jetstream/render/webgpu/buffer.hh"
#include "jetstream/render/webgpu/kernel.hh"
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
    wgpu::ShaderModule kernelModule = Backend::LoadShader(kernels[0], device);

    // Create bind group layout.

    for (U64 i = 0; i < buffers.size(); i++) {
        auto& [buffer, mode] = buffers[i];

        wgpu::BindGroupLayoutEntry entry{};
        entry.binding = i;
        entry.visibility = wgpu::ShaderStage::Compute;
        entry.buffer.type = BufferDescriptorType(buffer, mode);

        bindings.push_back(entry);
    }

    wgpu::BindGroupLayoutDescriptor layoutDesc{};
    layoutDesc.entryCount = static_cast<U32>(bindings.size());
    layoutDesc.entries = bindings.data();

    bindGroupLayout = device.CreateBindGroupLayout(&layoutDesc);

    // Create bind group.

    for (U64 i = 0; i < buffers.size(); i++) {
        auto& [buffer, _] = buffers[i];

        wgpu::BindGroupEntry entry{};
        entry.binding = i;
        entry.buffer = buffer->getHandle();
        entry.offset = 0;
        entry.size = buffer->byteSize();

        bindGroupEntries.push_back(entry);
    }

    wgpu::BindGroupDescriptor bindGroupDesc{};
    bindGroupDesc.layout = bindGroupLayout;
    bindGroupDesc.entryCount = static_cast<U32>(bindGroupEntries.size());
    bindGroupDesc.entries = bindGroupEntries.data();

    bindGroup = device.CreateBindGroup(&bindGroupDesc);

    // Create pipeline layout.

    wgpu::PipelineLayoutDescriptor pipelineLayoutDesc{};
    pipelineLayoutDesc.bindGroupLayoutCount = 1;
    pipelineLayoutDesc.bindGroupLayouts = &bindGroupLayout;

    pipelineLayout = device.CreatePipelineLayout(&pipelineLayoutDesc);

    // Create compute pipeline.

    wgpu::ComputePipelineDescriptor computePipelineDescriptor{};
    computePipelineDescriptor.compute.module = kernelModule;
    computePipelineDescriptor.compute.entryPoint = "main";
    computePipelineDescriptor.layout = pipelineLayout;

    pipeline = device.CreateComputePipeline(&computePipelineDescriptor);

    // Clean up.

    this->updated = true;

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    bindings.clear();
    bindGroupEntries.clear();

    return Result::SUCCESS;
}

Result Implementation::encode(wgpu::ComputePassEncoder& computePassEncoder) {
    // Check if data needs to be updated.

    if (!this->updated) {
        return Result::SUCCESS;
    }
    this->updated = false;

    // Create compute pass.

    computePassEncoder.SetPipeline(pipeline);
    computePassEncoder.SetBindGroup(0, bindGroup);

    // Dispatch compute work.

    const auto& [x, y, z] = config.gridSize;

    if (y != 1 || z != 1) {
        JST_ERROR("[WebGPU] 2D and 3D grid sizes are not implemented.");
        return Result::ERROR;
    }

    computePassEncoder.DispatchWorkgroups(x, y, z);

    return Result::SUCCESS;
}

wgpu::BufferBindingType Implementation::BufferDescriptorType(const std::shared_ptr<Buffer>& buffer, 
                                                             const Kernel::AccessMode& mode) {
    const auto& bufferType = buffer->getConfig().target;

    if ((bufferType & Buffer::Target::UNIFORM) == Buffer::Target::UNIFORM) {
        return wgpu::BufferBindingType::Uniform;
    }

    if ((bufferType & Buffer::Target::STORAGE) == Buffer::Target::STORAGE) {
        if ((mode & Kernel::AccessMode::READ) == Kernel::AccessMode::READ &&
            (mode & Kernel::AccessMode::WRITE) != Kernel::AccessMode::WRITE) {
            return wgpu::BufferBindingType::ReadOnlyStorage;
        }
        return wgpu::BufferBindingType::Storage;
    }

    JST_ERROR("[WebGPU] Invalid buffer usage.");
    throw Result::ERROR;
}

}  // namespace Jetstream::Render
