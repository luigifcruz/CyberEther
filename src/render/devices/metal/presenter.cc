#include "presenter.hh"

#include "jetstream/backend/devices/metal/helpers.hh"
#include "resources/shaders/global_shaders.hh"

#include "../../present_impl.hh"

#include <cstddef>
#include <optional>

namespace Jetstream::Render {

namespace {

MTL::RenderPipelineState* CreatePipeline(MTL::Device* device,
                                         MTL::Function* vertex,
                                         MTL::Function* fragment,
                                         MTL::PixelFormat pixelFormat,
                                         PresentList::Blend blend) {
    auto* vertexDescriptor = MTL::VertexDescriptor::alloc()->init();
    auto* position = vertexDescriptor->attributes()->object(0);
    position->setFormat(MTL::VertexFormatFloat2);
    position->setOffset(offsetof(PresentGeometry::Vertex, x));
    position->setBufferIndex(0);
    auto* texcoord = vertexDescriptor->attributes()->object(1);
    texcoord->setFormat(MTL::VertexFormatFloat2);
    texcoord->setOffset(offsetof(PresentGeometry::Vertex, u));
    texcoord->setBufferIndex(0);
    vertexDescriptor->layouts()->object(0)->setStride(sizeof(PresentGeometry::Vertex));

    auto* descriptor = MTL::RenderPipelineDescriptor::alloc()->init();
    descriptor->setVertexFunction(vertex);
    descriptor->setFragmentFunction(fragment);
    descriptor->setVertexDescriptor(vertexDescriptor);

    auto* attachment = descriptor->colorAttachments()->object(0);
    attachment->setPixelFormat(pixelFormat);
    attachment->setBlendingEnabled(true);
    attachment->setRgbBlendOperation(MTL::BlendOperationAdd);
    attachment->setSourceRGBBlendFactor(blend == PresentList::Blend::Premultiplied ? MTL::BlendFactorOne
                                                                                   : MTL::BlendFactorSourceAlpha);
    attachment->setDestinationRGBBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
    attachment->setAlphaBlendOperation(MTL::BlendOperationAdd);
    attachment->setSourceAlphaBlendFactor(MTL::BlendFactorOne);
    attachment->setDestinationAlphaBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);

    NS::Error* error = nullptr;
    MTL::RenderPipelineState* pipeline = device->newRenderPipelineState(descriptor, &error);
    descriptor->release();
    vertexDescriptor->release();
    if (!pipeline) {
        JST_ERROR("[METAL] Failed to create presentation pipeline: {}",
                  error ? error->description()->utf8String() : "unknown");
    }
    return pipeline;
}

}  // namespace

Result WindowImp<DeviceType::Metal>::Presenter::create(MTL::PixelFormat pixelFormat) {
    auto* device = Backend::State<DeviceType::Metal>()->getDevice();
    const auto& shaders = GlobalShadersPackage["present"][DeviceType::Metal];

    MTL::Function* vertex = Backend::LoadShader(shaders[0], device);
    MTL::Function* fragment = Backend::LoadShader(shaders[1], device);
    if (vertex && fragment) {
        straight = CreatePipeline(device, vertex, fragment, pixelFormat, PresentList::Blend::Straight);
        premultiplied = CreatePipeline(device, vertex, fragment, pixelFormat, PresentList::Blend::Premultiplied);
    }
    if (vertex) {
        vertex->release();
    }
    if (fragment) {
        fragment->release();
    }

    auto* samplerDescriptor = MTL::SamplerDescriptor::alloc()->init();
    samplerDescriptor->setMinFilter(MTL::SamplerMinMagFilterLinear);
    samplerDescriptor->setMagFilter(MTL::SamplerMinMagFilterLinear);
    samplerDescriptor->setMipFilter(MTL::SamplerMipFilterLinear);
    samplerDescriptor->setSAddressMode(MTL::SamplerAddressModeClampToEdge);
    samplerDescriptor->setTAddressMode(MTL::SamplerAddressModeClampToEdge);
    samplerDescriptor->setRAddressMode(MTL::SamplerAddressModeClampToEdge);
    sampler = device->newSamplerState(samplerDescriptor);
    samplerDescriptor->release();

    if (!straight || !premultiplied || !sampler) {
        JST_ERROR("[METAL] Failed to create presenter.");
        destroy();
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

void WindowImp<DeviceType::Metal>::Presenter::destroy() {
    if (straight) {
        straight->release();
        straight = nullptr;
    }
    if (premultiplied) {
        premultiplied->release();
        premultiplied = nullptr;
    }
    if (sampler) {
        sampler->release();
        sampler = nullptr;
    }
}

void WindowImp<DeviceType::Metal>::Presenter::encode(MTL::RenderCommandEncoder* encoder,
                                                     const PresentList& list,
                                                     const WindowMetrics& metrics,
                                                     const Extent2D<U64>& target) {
    const auto viewport = metrics.presentationSize();
    if (list.quads.empty() || viewport.x == 0 || viewport.y == 0 || target.x == 0 || target.y == 0) {
        return;
    }

    encoder->setCullMode(MTL::CullModeNone);
    encoder->setViewport(MTL::Viewport{0.0, 0.0, static_cast<double>(viewport.x), static_cast<double>(viewport.y),
                                       0.0, 1.0});
    encoder->setFragmentSamplerState(sampler, 0);

    std::optional<PresentList::Blend> bound;
    for (const auto& quad : list.quads) {
        const ScissorRect scissor = PresentGeometry::Clamp(quad.scissor, target);
        if (scissor.width == 0 || scissor.height == 0) {
            continue;
        }
        if (bound != quad.blend) {
            encoder->setRenderPipelineState(quad.blend == PresentList::Blend::Premultiplied ? premultiplied : straight);
            bound = quad.blend;
        }
        encoder->setScissorRect(MTL::ScissorRect{scissor.x, scissor.y, scissor.width, scissor.height});
        encoder->setFragmentTexture(reinterpret_cast<MTL::Texture*>(quad.handle), 0);

        const auto vertices = PresentGeometry::Triangles(quad, metrics.displaySize, false);
        encoder->setVertexBytes(vertices.data(), sizeof(vertices), 0);
        encoder->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(vertices.size()));
    }
}

}  // namespace Jetstream::Render
