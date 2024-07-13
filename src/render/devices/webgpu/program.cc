#include "jetstream/render/devices/webgpu/buffer.hh"
#include "jetstream/render/devices/webgpu/draw.hh"
#include "jetstream/render/devices/webgpu/texture.hh"
#include "jetstream/render/devices/webgpu/program.hh"
#include "jetstream/backend/devices/webgpu/helpers.hh"

namespace Jetstream::Render {

using Implementation = ProgramImp<Device::WebGPU>;

Implementation::ProgramImp(const Config& config) : Program(config) {
    _draw = std::dynamic_pointer_cast<DrawImp<Device::WebGPU>>(config.draw);

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

Result Implementation::create(const wgpu::TextureFormat& pixelFormat) {
    JST_DEBUG("[WebGPU] Creating program.");

    // Load shaders from memory.

    if (config.shaders.contains(Device::WebGPU) == 0) {
        JST_ERROR("[WebGPU] Module doesn't have necessary shader.");       
        return Result::ERROR;
    }

    auto device = Backend::State<Device::WebGPU>()->getDevice();

    const auto& shader = config.shaders[Device::WebGPU];
    wgpu::ShaderModule vertShaderModule = Backend::LoadShader(shader[0], device);
    wgpu::ShaderModule fragShaderModule = Backend::LoadShader(shader[1], device);

    // Enumerate the bindings of program targers.

    U32 bindingOffset = 0;
    for (U64 i = 0; i < buffers.size(); i++) {
        auto& [buffer, target] = buffers[i];

        wgpu::BufferBindingLayout bindingLayout{};
        bindingLayout.type = BufferDescriptorType(buffer);

        wgpu::BindGroupLayoutEntry binding{};
        binding.binding = bindingOffset++;
        binding.visibility = TargetToShaderStage(target);
        binding.buffer = bindingLayout;
        bindings.push_back(binding);
    }

    for (U64 i = 0; i < textures.size(); i++) {
        auto& texture = textures[i];

        {
            wgpu::BindGroupLayoutEntry binding{};
            binding.binding = bindingOffset++;
            binding.visibility = wgpu::ShaderStage::Fragment;
            binding.texture = texture->getTextureBindingLayout();
            bindings.push_back(binding);
        }

        {
            wgpu::BindGroupLayoutEntry binding{};
            binding.binding = bindingOffset++;
            binding.visibility = wgpu::ShaderStage::Fragment;
            binding.sampler = texture->getSamplerBindingLayout();
            bindings.push_back(binding);
        }
    }

    if (!bindings.empty()) {
        wgpu::BindGroupLayoutDescriptor bglDesc{};
        bglDesc.entryCount = bindings.size();
        bglDesc.entries = bindings.data();
        bindGroupLayout = device.CreateBindGroupLayout(&bglDesc);

        wgpu::PipelineLayoutDescriptor layoutDesc{};
        layoutDesc.bindGroupLayoutCount = 1;
        layoutDesc.bindGroupLayouts = &bindGroupLayout;
        pipelineLayout = device.CreatePipelineLayout(&layoutDesc);
    }

    bindingOffset = 0;
    for (U64 i = 0; i < buffers.size(); i++) {
        auto& [buffer, target] = buffers[i];

        wgpu::BindGroupEntry bindGroupEntry{};
        bindGroupEntry.binding = bindingOffset++;
        bindGroupEntry.buffer = buffer->getHandle();
        bindGroupEntry.size = buffer->byteSize();
        bindGroupEntry.offset = 0;

        bindGroupEntries.push_back(bindGroupEntry);
    }

    for (U64 i = 0; i < textures.size(); i++) {
        auto& texture = textures[i];

        {
            wgpu::BindGroupEntry bindGroupEntry;
            bindGroupEntry.binding = bindingOffset++;
            bindGroupEntry.textureView = texture->getViewHandle();
            bindGroupEntry.offset = 0;

            bindGroupEntries.push_back(bindGroupEntry);
        }

        {
            wgpu::BindGroupEntry bindGroupEntry;
            bindGroupEntry.binding = bindingOffset++;
            bindGroupEntry.sampler = texture->getSamplerHandle();
            bindGroupEntry.offset = 0;

            bindGroupEntries.push_back(bindGroupEntry);
        }
    }

    if (!bindings.empty()) {
        wgpu::BindGroupDescriptor bindGroupDescriptor{};
        bindGroupDescriptor.layout = bindGroupLayout;
        bindGroupDescriptor.entryCount = bindGroupEntries.size();
        bindGroupDescriptor.entries = bindGroupEntries.data();

        bindGroup = device.CreateBindGroup(&bindGroupDescriptor);
    }

    wgpu::BlendState blendState;
    blendState.color.srcFactor = wgpu::BlendFactor::SrcAlpha;
    blendState.color.dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha;
    blendState.color.operation = wgpu::BlendOperation::Add;
    blendState.alpha.srcFactor = wgpu::BlendFactor::One;
    blendState.alpha.dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha;
    blendState.alpha.operation = wgpu::BlendOperation::Add;

    wgpu::ColorTargetState colorTarget{};
    colorTarget.format = pixelFormat;
    colorTarget.writeMask = wgpu::ColorWriteMask::All;

    if (config.enableAlphaBlending) {
        colorTarget.blend = &blendState;
    }

    wgpu::FragmentState fragment{};
    fragment.module = fragShaderModule;
    fragment.entryPoint = "main";
    fragment.targetCount = 1;
    fragment.targets = &colorTarget;

    wgpu::RenderPipelineDescriptor renderPipelineDescriptor;
    renderPipelineDescriptor.fragment = &fragment;
    renderPipelineDescriptor.layout = pipelineLayout;
    renderPipelineDescriptor.depthStencil = nullptr;

    renderPipelineDescriptor.multisample.count = 1;
    renderPipelineDescriptor.multisample.mask = 0xFFFFFFFF;
    renderPipelineDescriptor.multisample.alphaToCoverageEnabled = false;

    JST_CHECK(_draw->create(renderPipelineDescriptor));

    renderPipelineDescriptor.vertex.module = vertShaderModule;
    renderPipelineDescriptor.vertex.entryPoint = "main";

    pipeline = device.CreateRenderPipeline(&renderPipelineDescriptor);

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_CHECK(_draw->destroy());

    bindings.clear();
    bindGroupEntries.clear();

    return Result::SUCCESS;
}

Result Implementation::draw(wgpu::RenderPassEncoder& renderPassEncoder) {
    renderPassEncoder.SetPipeline(pipeline);

    if (!bindings.empty()) {
        renderPassEncoder.SetBindGroup(0, bindGroup, 0, 0);
    }

    JST_CHECK(_draw->encode(renderPassEncoder));

    return Result::SUCCESS;
}

wgpu::ShaderStage Implementation::TargetToShaderStage(const Program::Target& target) {
    auto flags = wgpu::ShaderStage::None;

    if ((target & Program::Target::VERTEX) == Program::Target::VERTEX) {
        flags |= wgpu::ShaderStage::Vertex;
    }

    if ((target & Program::Target::FRAGMENT) == Program::Target::FRAGMENT) {
        flags |= wgpu::ShaderStage::Fragment;
    }
        
    return flags;
}

wgpu::BufferBindingType Implementation::BufferDescriptorType(const std::shared_ptr<Buffer>& buffer) {
    const auto& bufferType = buffer->getConfig().target;

    if ((bufferType & Buffer::Target::UNIFORM) == Buffer::Target::UNIFORM) {
        return wgpu::BufferBindingType::Uniform;
    }

    if ((bufferType & Buffer::Target::STORAGE) == Buffer::Target::STORAGE) {
        return wgpu::BufferBindingType::ReadOnlyStorage;
    }

    JST_ERROR("[WebGPU] Invalid buffer usage.");
    throw Result::ERROR;
}

}  // namespace Jetstream::Render
