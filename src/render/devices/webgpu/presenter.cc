#include "presenter.hh"

#include "jetstream/backend/devices/webgpu/helpers.hh"
#include "resources/shaders/global_shaders.hh"

#include "../../present_impl.hh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace Jetstream::Render {

namespace {

constexpr U64 kInitialVertices = 6 * 16;

WGPURenderPipeline CreatePipeline(WGPUDevice device,
                                  WGPUPipelineLayout layout,
                                  WGPUShaderModule vertex,
                                  WGPUShaderModule fragment,
                                  WGPUTextureFormat format,
                                  PresentList::Blend blend) {
    std::array<WGPUVertexAttribute, 2> attributes = {WGPU_VERTEX_ATTRIBUTE_INIT, WGPU_VERTEX_ATTRIBUTE_INIT};
    attributes[0].format = WGPUVertexFormat_Float32x2;
    attributes[0].offset = offsetof(PresentGeometry::Vertex, x);
    attributes[0].shaderLocation = 0;
    attributes[1].format = WGPUVertexFormat_Float32x2;
    attributes[1].offset = offsetof(PresentGeometry::Vertex, u);
    attributes[1].shaderLocation = 1;

    WGPUVertexBufferLayout vertexLayout = WGPU_VERTEX_BUFFER_LAYOUT_INIT;
    vertexLayout.arrayStride = sizeof(PresentGeometry::Vertex);
    vertexLayout.stepMode = WGPUVertexStepMode_Vertex;
    vertexLayout.attributeCount = static_cast<uint32_t>(attributes.size());
    vertexLayout.attributes = attributes.data();

    WGPUBlendState blendState = WGPU_BLEND_STATE_INIT;
    blendState.color.srcFactor = blend == PresentList::Blend::Premultiplied ? WGPUBlendFactor_One
                                                                            : WGPUBlendFactor_SrcAlpha;
    blendState.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
    blendState.color.operation = WGPUBlendOperation_Add;
    blendState.alpha.srcFactor = WGPUBlendFactor_One;
    blendState.alpha.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
    blendState.alpha.operation = WGPUBlendOperation_Add;

    WGPUColorTargetState target = WGPU_COLOR_TARGET_STATE_INIT;
    target.format = format;
    target.blend = &blendState;
    target.writeMask = WGPUColorWriteMask_All;

    WGPUFragmentState fragmentState = WGPU_FRAGMENT_STATE_INIT;
    fragmentState.module = fragment;
    fragmentState.entryPoint = {"main", WGPU_STRLEN};
    fragmentState.targetCount = 1;
    fragmentState.targets = &target;

    WGPURenderPipelineDescriptor descriptor = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
    descriptor.layout = layout;
    descriptor.vertex.module = vertex;
    descriptor.vertex.entryPoint = {"main", WGPU_STRLEN};
    descriptor.vertex.bufferCount = 1;
    descriptor.vertex.buffers = &vertexLayout;
    descriptor.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    descriptor.primitive.cullMode = WGPUCullMode_None;
    descriptor.multisample.count = 1;
    descriptor.multisample.mask = 0xFFFFFFFF;
    descriptor.fragment = &fragmentState;

    return wgpuDeviceCreateRenderPipeline(device, &descriptor);
}

}  // namespace

Result WindowImp<DeviceType::WebGPU>::Presenter::create(WGPUTextureFormat format) {
    device = Backend::State<DeviceType::WebGPU>()->getDevice();

    std::array<WGPUBindGroupLayoutEntry, 2> entries = {WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT,
                                                       WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT};
    entries[0].binding = 0;
    entries[0].visibility = WGPUShaderStage_Fragment;
    entries[0].texture.sampleType = WGPUTextureSampleType_Float;
    entries[0].texture.viewDimension = WGPUTextureViewDimension_2D;
    entries[1].binding = 1;
    entries[1].visibility = WGPUShaderStage_Fragment;
    entries[1].sampler.type = WGPUSamplerBindingType_Filtering;

    WGPUBindGroupLayoutDescriptor layoutDescriptor = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    layoutDescriptor.entryCount = static_cast<uint32_t>(entries.size());
    layoutDescriptor.entries = entries.data();
    bindGroupLayout = wgpuDeviceCreateBindGroupLayout(device, &layoutDescriptor);

    WGPUPipelineLayoutDescriptor pipelineLayoutDescriptor = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pipelineLayoutDescriptor.bindGroupLayoutCount = 1;
    pipelineLayoutDescriptor.bindGroupLayouts = &bindGroupLayout;
    pipelineLayout = wgpuDeviceCreatePipelineLayout(device, &pipelineLayoutDescriptor);

    const auto& shaders = GlobalShadersPackage["present"][DeviceType::WebGPU];
    WGPUShaderModule vertex = Backend::LoadShader(shaders[0], device);
    WGPUShaderModule fragment = Backend::LoadShader(shaders[1], device);
    if (vertex && fragment && pipelineLayout) {
        straight = CreatePipeline(device, pipelineLayout, vertex, fragment, format, PresentList::Blend::Straight);
        premultiplied = CreatePipeline(device, pipelineLayout, vertex, fragment, format,
                                       PresentList::Blend::Premultiplied);
    }
    if (vertex) {
        wgpuShaderModuleRelease(vertex);
    }
    if (fragment) {
        wgpuShaderModuleRelease(fragment);
    }

    WGPUSamplerDescriptor samplerDescriptor = WGPU_SAMPLER_DESCRIPTOR_INIT;
    samplerDescriptor.minFilter = WGPUFilterMode_Linear;
    samplerDescriptor.magFilter = WGPUFilterMode_Linear;
    samplerDescriptor.mipmapFilter = WGPUMipmapFilterMode_Linear;
    samplerDescriptor.addressModeU = WGPUAddressMode_ClampToEdge;
    samplerDescriptor.addressModeV = WGPUAddressMode_ClampToEdge;
    samplerDescriptor.addressModeW = WGPUAddressMode_ClampToEdge;
    samplerDescriptor.maxAnisotropy = 1;
    sampler = wgpuDeviceCreateSampler(device, &samplerDescriptor);

    if (!bindGroupLayout || !pipelineLayout || !straight || !premultiplied || !sampler) {
        JST_ERROR("[WebGPU] Failed to create presentation pipeline.");
        destroy();
        return Result::ERROR;
    }

    bindings.emplace([this](const Texture& texture) {
        std::array<WGPUBindGroupEntry, 2> entries = {WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT};
        entries[0].binding = 0;
        entries[0].textureView = reinterpret_cast<WGPUTextureView>(static_cast<uintptr_t>(texture.raw()));
        entries[1].binding = 1;
        entries[1].sampler = sampler;

        WGPUBindGroupDescriptor descriptor = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        descriptor.layout = bindGroupLayout;
        descriptor.entryCount = static_cast<uint32_t>(entries.size());
        descriptor.entries = entries.data();
        return wgpuDeviceCreateBindGroup(device, &descriptor);
    }, [](WGPUBindGroup group) { wgpuBindGroupRelease(group); });

    return Result::SUCCESS;
}

void WindowImp<DeviceType::WebGPU>::Presenter::destroy() {
    bindings.reset();

    if (vertexBuffer) {
        wgpuBufferRelease(vertexBuffer);
        vertexBuffer = nullptr;
        vertexCapacity = 0;
    }
    if (sampler) {
        wgpuSamplerRelease(sampler);
        sampler = nullptr;
    }
    if (straight) {
        wgpuRenderPipelineRelease(straight);
        straight = nullptr;
    }
    if (premultiplied) {
        wgpuRenderPipelineRelease(premultiplied);
        premultiplied = nullptr;
    }
    if (pipelineLayout) {
        wgpuPipelineLayoutRelease(pipelineLayout);
        pipelineLayout = nullptr;
    }
    if (bindGroupLayout) {
        wgpuBindGroupLayoutRelease(bindGroupLayout);
        bindGroupLayout = nullptr;
    }
}

Result WindowImp<DeviceType::WebGPU>::Presenter::reserve(U64 vertices) {
    if (vertexCapacity >= vertices) {
        return Result::SUCCESS;
    }
    if (vertexBuffer) {
        wgpuBufferRelease(vertexBuffer);
        vertexBuffer = nullptr;
    }

    U64 capacity = kInitialVertices;
    while (capacity < vertices) {
        capacity *= 2;
    }

    WGPUBufferDescriptor descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
    descriptor.size = capacity * sizeof(PresentGeometry::Vertex);
    descriptor.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst;
    vertexBuffer = wgpuDeviceCreateBuffer(device, &descriptor);
    if (!vertexBuffer) {
        vertexCapacity = 0;
        return Result::ERROR;
    }

    vertexCapacity = capacity;
    return Result::SUCCESS;
}

Result WindowImp<DeviceType::WebGPU>::Presenter::encode(WGPURenderPassEncoder pass,
                                                        WGPUQueue queue,
                                                        const PresentList& list,
                                                        const WindowMetrics& metrics,
                                                        const Extent2D<U64>& target) {
    const auto size = metrics.presentationSize();
    if (!list.quads.empty() && size.x > 0 && size.y > 0 && target.x > 0 && target.y > 0) {
        const U64 vertexCount = list.quads.size() * 6;
        JST_CHECK(reserve(vertexCount));

        std::vector<PresentGeometry::Vertex> vertices;
        vertices.reserve(vertexCount);
        for (const auto& quad : list.quads) {
            const auto triangles = PresentGeometry::Triangles(quad, metrics.displaySize, false);
            vertices.insert(vertices.end(), triangles.begin(), triangles.end());
        }
        wgpuQueueWriteBuffer(queue, vertexBuffer, 0, vertices.data(), vertices.size() * sizeof(PresentGeometry::Vertex));

        wgpuRenderPassEncoderSetViewport(pass, 0.0f, 0.0f, static_cast<F32>(size.x), static_cast<F32>(size.y),
                                         0.0f, 1.0f);
        wgpuRenderPassEncoderSetVertexBuffer(pass, 0, vertexBuffer, 0, vertices.size() * sizeof(PresentGeometry::Vertex));

        std::optional<PresentList::Blend> bound;
        for (U64 index = 0; index < list.quads.size(); ++index) {
            const auto& quad = list.quads[index];
            const ScissorRect scissor = PresentGeometry::Clamp(quad.scissor, target);
            if (scissor.width == 0 || scissor.height == 0) {
                continue;
            }
            if (bound != quad.blend) {
                wgpuRenderPassEncoderSetPipeline(pass, quad.blend == PresentList::Blend::Premultiplied ? premultiplied
                                                                                                       : straight);
                bound = quad.blend;
            }
            wgpuRenderPassEncoderSetBindGroup(pass, 0, bindings->acquire(quad.texture), 0, nullptr);
            wgpuRenderPassEncoderSetScissorRect(pass, scissor.x, scissor.y, scissor.width, scissor.height);
            wgpuRenderPassEncoderDraw(pass, 6, 1, static_cast<uint32_t>(index * 6), 0);
        }
    }

    bindings->advance();

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render
