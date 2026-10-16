#include <algorithm>
#include <utility>

#include "jetstream/render/devices/webgpu/buffer.hh"
#include "jetstream/render/devices/webgpu/draw.hh"
#include "jetstream/render/devices/webgpu/texture.hh"
#include "jetstream/render/devices/webgpu/program.hh"
#include "jetstream/backend/devices/webgpu/helpers.hh"

namespace Jetstream::Render {

using Implementation = ProgramImp<DeviceType::WebGPU>;

Implementation::ProgramImp(const Config& config) : Program(config) {
    for (auto& draw : config.draws) {
        draws.push_back(
            std::dynamic_pointer_cast<DrawImp<DeviceType::WebGPU>>(draw)
        );
    }

    for (auto& texture : config.textures) {
        textures.push_back(
            std::dynamic_pointer_cast<TextureImp<DeviceType::WebGPU>>(texture)
        );
    }

    for (auto& [buffer, target] : config.buffers) {
        buffers.push_back(
            {std::dynamic_pointer_cast<BufferImp<DeviceType::WebGPU>>(buffer), target}
        );
    }
}

Result Implementation::create(const WGPUTextureFormat& pixelFormat) {
    JST_DEBUG("[WebGPU] Creating program.");

    // Load shaders from memory.

    if (config.shaders.contains(DeviceType::WebGPU) == 0) {
        JST_ERROR("[WebGPU] Module doesn't have necessary shader.");
        return Result::ERROR;
    }

    auto device = Backend::State<DeviceType::WebGPU>()->getDevice();

    const auto& shader = config.shaders[DeviceType::WebGPU];
    vertShaderModule = Backend::LoadShader(shader[0], device);
    fragShaderModule = Backend::LoadShader(shader[1], device);

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

    bindGroup = createBindGroup();
    JST_ASSERT(bindings.empty() || bindGroup, "Failed to create bind group.");

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

WGPUBindGroup Implementation::createBindGroup() {
    bindGroupEntries.clear();

    U32 bindingOffset = 0;
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
        auto device = Backend::State<DeviceType::WebGPU>()->getDevice();

        WGPUBindGroupDescriptor bindGroupDescriptor = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        bindGroupDescriptor.layout = bindGroupLayout;
        bindGroupDescriptor.entryCount = static_cast<uint32_t>(bindGroupEntries.size());
        bindGroupDescriptor.entries = bindGroupEntries.data();

        return wgpuDeviceCreateBindGroup(device, &bindGroupDescriptor);
    }
    return nullptr;
}

Implementation::BindGroupCandidate::~BindGroupCandidate() {
    if (handle) {
        wgpuBindGroupRelease(handle);
    }
}

Result Implementation::rebindTextures() {
    if (bindings.empty() || textures.empty()) {
        return Result::SUCCESS;
    }

    const auto allocations = textureAllocations();
    if (pendingBindGroup && pendingBindGroup->allocations != allocations) {
        pendingBindGroup.reset();
    }
    if (!pendingBindGroup) {
        for (const auto& texture : textures) {
            if (!texture || !texture->allocated() || !texture->getViewHandle() ||
                !texture->getSamplerHandle()) {
                return Result::ERROR;
            }
        }

        auto candidate = std::make_shared<BindGroupCandidate>();
        candidate->allocations = allocations;
        auto device = Backend::State<DeviceType::WebGPU>()->getDevice();
        wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation);
        candidate->handle = createBindGroup();

        WGPUPopErrorScopeCallbackInfo callback = WGPU_POP_ERROR_SCOPE_CALLBACK_INFO_INIT;
        callback.mode = WGPUCallbackMode_AllowSpontaneous;
        callback.userdata1 = new std::shared_ptr<BindGroupCandidate>(candidate);
        callback.callback = [](WGPUPopErrorScopeStatus status, WGPUErrorType type,
                               WGPUStringView, void* userdata1, void*) {
            std::unique_ptr<std::shared_ptr<BindGroupCandidate>> state(
                static_cast<std::shared_ptr<BindGroupCandidate>*>(userdata1));
            const bool valid = status == WGPUPopErrorScopeStatus_Success &&
                               type == WGPUErrorType_NoError && (*state)->handle;
            (*state)->result.store(valid ? Result::SUCCESS : Result::ERROR,
                                   std::memory_order_release);
        };
        (void)wgpuDevicePopErrorScope(device, callback);
        pendingBindGroup = std::move(candidate);
    }

    const Result result = pendingBindGroup->result.load(std::memory_order_acquire);
    if (result == Result::SKIP) {
        return result;
    }
    if (result != Result::SUCCESS) {
        pendingBindGroup.reset();
        return result;
    }

    const auto previous = bindGroup;
    bindGroup = std::exchange(pendingBindGroup->handle, nullptr);
    pendingBindGroup.reset();
    if (previous) {
        wgpuBindGroupRelease(previous);
    }

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    pendingBindGroup.reset();
    for (auto& draw : draws) {
        JST_CHECK(draw->destroy());
    }

    if (!bindings.empty()) {
        if (bindGroup) {
            wgpuBindGroupRelease(bindGroup);
            bindGroup = nullptr;
        }
        wgpuBindGroupLayoutRelease(bindGroupLayout);
        wgpuPipelineLayoutRelease(pipelineLayout);
    }
    wgpuRenderPipelineRelease(pipeline);
    wgpuShaderModuleRelease(fragShaderModule);
    wgpuShaderModuleRelease(vertShaderModule);

    bindings.clear();
    bindGroupEntries.clear();

    return Result::SUCCESS;
}

Result Implementation::draw(WGPURenderPassEncoder& renderPassEncoder,
                            const Extent2D<U64>& framebufferSize) {
    wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipeline);

    // Set scissor rect.
    if (config.scissorRect) {
        const auto& sr = *config.scissorRect;
        if (sr.x >= framebufferSize.x || sr.y >= framebufferSize.y) {
            return Result::SUCCESS;
        }

        const U32 width = static_cast<U32>(std::min<U64>(
            sr.width, framebufferSize.x - sr.x));
        const U32 height = static_cast<U32>(std::min<U64>(
            sr.height, framebufferSize.y - sr.y));
        if (width == 0 || height == 0) {
            return Result::SUCCESS;
        }

        wgpuRenderPassEncoderSetScissorRect(renderPassEncoder,
                                            sr.x,
                                            sr.y,
                                            width,
                                            height);
    }

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
