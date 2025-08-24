#include "jetstream/render/devices/webgpu/buffer.hh"
#include "jetstream/render/devices/webgpu/draw.hh"
#include "jetstream/render/devices/webgpu/texture.hh"
#include "jetstream/render/devices/webgpu/program.hh"
#include "jetstream/backend/devices/webgpu/helpers.hh"

namespace Jetstream::Render {

using Implementation = ProgramImp<Device::WebGPU>;

Implementation::ProgramImp(const Config& config) : Program(config) {
    for (auto& draw : config.draws) {
        draws.push_back(
            std::dynamic_pointer_cast<DrawImp<Device::WebGPU>>(draw)
        );
    }

    for (auto& texture : config.textures) {
        textures.push_back(
            std::dynamic_pointer_cast<TextureImp<Device::WebGPU>>(texture)
        );
    }

    for (auto& [buffer, target] : config.buffers) {
        buffers.push_back(
            {std::dynamic_pointer_cast<BufferImp<Device::WebGPU>>(buffer), target}
        );
    }
}

Result Implementation::create(const WGPUTextureFormat& pixelFormat) {
    JST_DEBUG("[WebGPU] Creating program.");

    // Load shaders from memory.

    if (config.shaders.contains(Device::WebGPU) == 0) {
        JST_ERROR("[WebGPU] Module doesn't have necessary shader.");
        return Result::ERROR;
    }

    auto device = Backend::State<Device::WebGPU>()->getDevice();

    const auto& shader = config.shaders[Device::WebGPU];
    WGPUShaderModule vertShaderModule = Backend::LoadShader(shader[0], device);
    WGPUShaderModule fragShaderModule = Backend::LoadShader(shader[1], device);

    // Enumerate the bindings of program targers.

    U32 bindingOffset = 0;
    for (U64 i = 0; i < buffers.size(); i++) {
        auto& [buffer, target] = buffers[i];

        WGPUBufferBindingLayout bindingLayout = WGPU_BUFFER_BINDING_LAYOUT_INIT;
        bindingLayout.type = BufferDescriptorType(buffer);

        WGPUBindGroupLayoutEntry binding = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
        binding.binding = bindingOffset++;
        binding.visibility = TargetToShaderStage(target);
        binding.buffer = bindingLayout;
        bindings.push_back(binding);
    }

    for (U64 i = 0; i < textures.size(); i++) {
        auto& texture = textures[i];

        {
            WGPUBindGroupLayoutEntry binding = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
            binding.binding = bindingOffset++;
            binding.visibility = WGPUShaderStage_Fragment;
            binding.texture = texture->getTextureBindingLayout();
            bindings.push_back(binding);
        }

        {
            WGPUBindGroupLayoutEntry binding = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
            binding.binding = bindingOffset++;
            binding.visibility = WGPUShaderStage_Fragment;
            binding.sampler = texture->getSamplerBindingLayout();
            bindings.push_back(binding);
        }
    }

    if (!bindings.empty()) {
        WGPUBindGroupLayoutDescriptor bglDesc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
        bglDesc.entryCount = static_cast<uint32_t>(bindings.size());
        bglDesc.entries = bindings.data();
        bindGroupLayout = wgpuDeviceCreateBindGroupLayout(device, &bglDesc);

        WGPUPipelineLayoutDescriptor layoutDesc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
        layoutDesc.bindGroupLayoutCount = 1;
        layoutDesc.bindGroupLayouts = &bindGroupLayout;
        pipelineLayout = wgpuDeviceCreatePipelineLayout(device, &layoutDesc);
    }

    bindingOffset = 0;
    for (U64 i = 0; i < buffers.size(); i++) {
        auto& [buffer, target] = buffers[i];

        WGPUBindGroupEntry bindGroupEntry = WGPU_BIND_GROUP_ENTRY_INIT;
        bindGroupEntry.binding = bindingOffset++;
        bindGroupEntry.buffer = buffer->getHandle();
        bindGroupEntry.size = buffer->byteSize();
        bindGroupEntry.offset = 0;

        bindGroupEntries.push_back(bindGroupEntry);
    }

    for (U64 i = 0; i < textures.size(); i++) {
        auto& texture = textures[i];

        {
            WGPUBindGroupEntry bindGroupEntry = WGPU_BIND_GROUP_ENTRY_INIT;
            bindGroupEntry.binding = bindingOffset++;
            bindGroupEntry.textureView = texture->getViewHandle();
            bindGroupEntry.offset = 0;

            bindGroupEntries.push_back(bindGroupEntry);
        }

        {
            WGPUBindGroupEntry bindGroupEntry = WGPU_BIND_GROUP_ENTRY_INIT;
            bindGroupEntry.binding = bindingOffset++;
            bindGroupEntry.sampler = texture->getSamplerHandle();
            bindGroupEntry.offset = 0;

            bindGroupEntries.push_back(bindGroupEntry);
        }
    }

    if (!bindings.empty()) {
        WGPUBindGroupDescriptor bindGroupDescriptor = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        bindGroupDescriptor.layout = bindGroupLayout;
        bindGroupDescriptor.entryCount = static_cast<uint32_t>(bindGroupEntries.size());
        bindGroupDescriptor.entries = bindGroupEntries.data();

        bindGroup = wgpuDeviceCreateBindGroup(device, &bindGroupDescriptor);
    }

    WGPUBlendState blendState = WGPU_BLEND_STATE_INIT;
    blendState.color.srcFactor = WGPUBlendFactor_SrcAlpha;
    blendState.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
    blendState.color.operation = WGPUBlendOperation_Add;
    blendState.alpha.srcFactor = WGPUBlendFactor_One;
    blendState.alpha.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
    blendState.alpha.operation = WGPUBlendOperation_Add;

    WGPUColorTargetState colorTarget = WGPU_COLOR_TARGET_STATE_INIT;
    colorTarget.format = pixelFormat;
    colorTarget.writeMask = WGPUColorWriteMask_All;

    if (config.enableAlphaBlending) {
        colorTarget.blend = &blendState;
    }

    WGPUFragmentState fragment = WGPU_FRAGMENT_STATE_INIT;
    fragment.module = fragShaderModule;
    fragment.entryPoint = {"main", WGPU_STRLEN};
    fragment.targetCount = 1;
    fragment.targets = &colorTarget;

    WGPURenderPipelineDescriptor renderPipelineDescriptor = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
    renderPipelineDescriptor.fragment = &fragment;
    renderPipelineDescriptor.layout = pipelineLayout;
    renderPipelineDescriptor.depthStencil = nullptr;

    renderPipelineDescriptor.multisample.count = 1;
    renderPipelineDescriptor.multisample.mask = 0xFFFFFFFF;
    renderPipelineDescriptor.multisample.alphaToCoverageEnabled = false;

    for (auto& draw : draws) {
        JST_CHECK(draw->create(renderPipelineDescriptor));
    }

    renderPipelineDescriptor.vertex.module = vertShaderModule;
    renderPipelineDescriptor.vertex.entryPoint = {"main", WGPU_STRLEN};

    pipeline = wgpuDeviceCreateRenderPipeline(device, &renderPipelineDescriptor);

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    for (auto& draw : draws) {
        JST_CHECK(draw->destroy());
    }

    bindings.clear();
    bindGroupEntries.clear();

    return Result::SUCCESS;
}

Result Implementation::draw(WGPURenderPassEncoder& renderPassEncoder) {
    wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipeline);

    if (!bindings.empty()) {
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 0, bindGroup, 0, nullptr);
    }

    for (auto& draw : draws) {
        JST_CHECK(draw->encode(renderPassEncoder));
    }

    return Result::SUCCESS;
}

WGPUShaderStage Implementation::TargetToShaderStage(const Program::Target& target) {
    WGPUShaderStage flags = WGPUShaderStage_None;

    if ((target & Program::Target::VERTEX) == Program::Target::VERTEX) {
        flags |= WGPUShaderStage_Vertex;
    }

    if ((target & Program::Target::FRAGMENT) == Program::Target::FRAGMENT) {
        flags |= WGPUShaderStage_Fragment;
    }

    return flags;
}

WGPUBufferBindingType Implementation::BufferDescriptorType(const std::shared_ptr<Buffer>& buffer) {
    const auto& bufferType = buffer->getConfig().target;

    if ((bufferType & Buffer::Target::UNIFORM) == Buffer::Target::UNIFORM) {
        return WGPUBufferBindingType_Uniform;
    }

    if ((bufferType & Buffer::Target::STORAGE) == Buffer::Target::STORAGE) {
        return WGPUBufferBindingType_ReadOnlyStorage;
    }

    JST_ERROR("[WebGPU] Invalid buffer usage.");
    throw Result::ERROR;
}

}  // namespace Jetstream::Render
