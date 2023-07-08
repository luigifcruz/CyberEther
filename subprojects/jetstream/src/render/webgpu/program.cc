#include "jetstream/render/webgpu/buffer.hh"
#include "jetstream/render/webgpu/draw.hh"
#include "jetstream/render/webgpu/texture.hh"
#include "jetstream/render/webgpu/program.hh"
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

    // Create program targets.

    for (const auto& texture : textures) {
        JST_CHECK(texture->create());
    }

    for (const auto& [buffer, _] : buffers) {
        JST_CHECK(buffer->create());
    }

    // Load shaders from memory.

    if (config.shaders.count(Device::WebGPU) == 0) {
        JST_FATAL("[WebGPU] Module doesn't have necessary shader.");       
        JST_CHECK(Result::ERROR);
    }

    auto device = Backend::State<Device::WebGPU>()->getDevice();

    const auto& shader = config.shaders[Device::WebGPU];
    wgpu::ShaderModule vertShaderModule = Backend::LoadShaderWebGPU(shader[0], device);
    wgpu::ShaderModule fragShaderModule = Backend::LoadShaderWebGPU(shader[1], device);


    // Enumerate the bindings of program targers.

    U32 bindingOffset = 0;
    for (U64 i = 0; i < buffers.size(); i++) {
        auto& [buffer, target] = buffers[i];

        wgpu::BindGroupLayoutEntry binding{};
        binding.binding = bindingOffset++;
        binding.visibility = TargetToWebGPU(target);
        binding.buffer = buffer->getBufferBindingLayout();
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

    // Configure render pipeline.

    renderPipelineDescriptor = {};

    wgpu::BlendState blend{};
    blend.color.operation = wgpu::BlendOperation::Add;
    blend.color.srcFactor = wgpu::BlendFactor::One;
    blend.color.dstFactor = wgpu::BlendFactor::One;
    blend.alpha.operation = wgpu::BlendOperation::Add;
    blend.alpha.srcFactor = wgpu::BlendFactor::One;
    blend.alpha.dstFactor = wgpu::BlendFactor::One;

    wgpu::ColorTargetState colorTarget{};
    colorTarget.format = pixelFormat;
    colorTarget.blend = &blend;
    colorTarget.writeMask = wgpu::ColorWriteMask::All;

    wgpu::FragmentState fragment{};
    fragment.module = fragShaderModule;
    fragment.entryPoint = "main";
    fragment.targetCount = 1;
    fragment.targets = &colorTarget;

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

    for (const auto& texture : textures) {
        JST_CHECK(texture->destroy());
    }

    for (const auto& [buffer, _] : buffers) {
        JST_CHECK(buffer->destroy());
    }

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

wgpu::ShaderStage Implementation::TargetToWebGPU(const Program::Target& target) {
    auto flags = wgpu::ShaderStage::None;

    if (static_cast<U8>(target & Program::Target::VERTEX) > 0) {
        flags |= wgpu::ShaderStage::Vertex;
    }

    if (static_cast<U8>(target & Program::Target::FRAGMENT) > 0) {
        flags |= wgpu::ShaderStage::Fragment;
    }
        
    return flags;
}

}  // namespace Jetstream::Render
