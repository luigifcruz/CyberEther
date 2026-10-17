#include "jetstream/render/devices/metal/buffer.hh"
#include "jetstream/render/devices/metal/draw.hh"
#include "jetstream/render/devices/metal/texture.hh"
#include "jetstream/render/devices/metal/program.hh"

#include "program_cache.hh"

namespace Jetstream::Render {

namespace {

using Detail::MetalProgramCache;

MetalProgramCache& ProgramCache() {
    static MetalProgramCache cache;
    return cache;
}

NS::SharedPtr<MTL::Function> CachedFunction(MetalProgramCache::Device& cache, const std::vector<U8>& source) {
    std::string key(reinterpret_cast<const char*>(source.data()), source.size());
    if (auto function = cache.findFunction(key)) {
        return function;
    }

    auto opts = NS::TransferPtr(MTL::CompileOptions::alloc()->init());
    opts->setFastMathEnabled(true);
    opts->setLanguageVersion(MTL::LanguageVersion3_0);
    opts->setLibraryType(MTL::LibraryTypeExecutable);

    NS::Error* err = nullptr;
    auto text = NS::TransferPtr(NS::String::alloc()->init(const_cast<U8*>(source.data()), source.size(), NS::UTF8StringEncoding, false));
    auto library = NS::TransferPtr(cache.handle->newLibrary(text.get(), opts.get(), &err));
    if (!library) {
        JST_ERROR("Library error:\n{}", err ? err->description()->utf8String() : "unknown");
        return {};
    }

    auto function = NS::TransferPtr(library->newFunction(NS::String::string("main0", NS::UTF8StringEncoding)));
    if (!function) {
        JST_ERROR("[METAL] Failed to create shader function.");
        return {};
    }

    cache.insertFunction(std::move(key), function);
    return function;
}

}  // namespace

using Implementation = ProgramImp<DeviceType::Metal>;

Implementation::ProgramImp(const Config& config) : Program(config) {
    for (auto& draw : config.draws) {
        draws.push_back(
            std::dynamic_pointer_cast<DrawImp<DeviceType::Metal>>(draw)
        );
    }

    for (auto& texture : config.textures) {
        textures.push_back(
            std::dynamic_pointer_cast<TextureImp<DeviceType::Metal>>(texture)
        );
    }

    for (auto& [buffer, target] : config.buffers) {
        buffers.push_back(
            {std::dynamic_pointer_cast<BufferImp<DeviceType::Metal>>(buffer), target}
        );
    }
}

Result Implementation::create(const MTL::PixelFormat& pixelFormat, bool multisampled) {
    JST_DEBUG("[METAL] Creating program.");

    if (config.shaders.contains(DeviceType::Metal) == 0) {
        JST_ERROR("[METAL] Module doesn't have necessary shader.");
        return Result::ERROR;
    }

    const auto& shaders = config.shaders[DeviceType::Metal];
    auto device = Backend::State<DeviceType::Metal>()->getDevice();

    auto& programCache = ProgramCache();
    std::lock_guard<std::mutex> lock(programCache.mutex);
    auto& cache = programCache.forDevice(device);

    auto vertFunc = CachedFunction(cache, shaders[0]);
    auto fragFunc = CachedFunction(cache, shaders[1]);
    if (!vertFunc || !fragFunc) {
        return Result::ERROR;
    }

    U64 indexOffset = 0;
    for (U64 i = 0; i < buffers.size(); i++) {
        auto& [_, target] = buffers[i];
        if (static_cast<U8>(target & Program::Target::VERTEX) > 0) {
            indexOffset++;
        }
    }
    auto vertDesc = NS::TransferPtr(MTL::VertexDescriptor::alloc()->init());

    for (auto& draw : draws) {
        JST_CHECK(draw->create(vertDesc.get(), indexOffset));
    }

    const U64 sampleCount = multisampled ? Backend::State<DeviceType::Metal>()->getMultisampling() : 1;
    std::string key = jst::fmt::format("{}:{}:{}:{}:{}:{}",
                                       static_cast<const void*>(vertFunc.get()),
                                       static_cast<const void*>(fragFunc.get()),
                                       static_cast<U64>(pixelFormat),
                                       sampleCount,
                                       config.enableAlphaBlending,
                                       indexOffset);
    key += config.eraseBlending ? "|erase" : "";
    for (const auto& draw : config.draws) {
        const auto& vertex = draw->getConfig().buffer->getConfig();
        key += "|v";
        for (const auto& [_, stride] : vertex.vertices) {
            key += jst::fmt::format(":{}", stride);
        }
        key += "|i";
        for (const auto& [_, stride] : vertex.instances) {
            key += jst::fmt::format(":{}", stride);
        }
    }

    if (auto pipeline = cache.findPipeline(key)) {
        renderPipelineState = pipeline.get();
        renderPipelineState->retain();
        return Result::SUCCESS;
    }

    auto renderPipelineDescriptor = NS::TransferPtr(MTL::RenderPipelineDescriptor::alloc()->init());
    JST_ASSERT(renderPipelineDescriptor, "Failed to create render pipeline descriptor.");
    renderPipelineDescriptor->setVertexDescriptor(vertDesc.get());
    renderPipelineDescriptor->setVertexFunction(vertFunc.get());
    renderPipelineDescriptor->setFragmentFunction(fragFunc.get());
    if (multisampled) {
        renderPipelineDescriptor->setSampleCount(sampleCount);
    }

    const auto& colorAttachment = renderPipelineDescriptor->colorAttachments()->object(0)->init();
    colorAttachment->setPixelFormat(pixelFormat);

    if (config.eraseBlending) {
        colorAttachment->setBlendingEnabled(true);
        colorAttachment->setSourceRGBBlendFactor(MTL::BlendFactorZero);
        colorAttachment->setDestinationRGBBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
        colorAttachment->setRgbBlendOperation(MTL::BlendOperationAdd);
        colorAttachment->setSourceAlphaBlendFactor(MTL::BlendFactorZero);
        colorAttachment->setDestinationAlphaBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
        colorAttachment->setAlphaBlendOperation(MTL::BlendOperationAdd);
    } else if (config.enableAlphaBlending) {
        colorAttachment->setBlendingEnabled(true);
        colorAttachment->setSourceRGBBlendFactor(MTL::BlendFactorSourceAlpha);
        colorAttachment->setDestinationRGBBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
        colorAttachment->setRgbBlendOperation(MTL::BlendOperationAdd);
        colorAttachment->setSourceAlphaBlendFactor(MTL::BlendFactorOne);
        colorAttachment->setDestinationAlphaBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
        colorAttachment->setAlphaBlendOperation(MTL::BlendOperationAdd);
    }

    NS::Error* err = nullptr;
    auto pipeline = NS::TransferPtr(device->newRenderPipelineState(renderPipelineDescriptor.get(), &err));
    if (!pipeline) {
        JST_ERROR("Failed to create pipeline state:\n{}", err ? err->description()->utf8String() : "unknown");
        return Result::ERROR;
    }

    cache.insertPipeline(std::move(key), vertFunc, fragFunc, pipeline);
    renderPipelineState = pipeline.get();
    renderPipelineState->retain();

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    for (auto& draw : draws) {
        JST_CHECK(draw->destroy());
    }

    if (renderPipelineState) {
        renderPipelineState->release();
        renderPipelineState = nullptr;
    }

    return Result::SUCCESS;
}

Result Implementation::draw(MTL::RenderCommandEncoder* renderCmdEncoder) {
    renderCmdEncoder->setRenderPipelineState(renderPipelineState);

    // Set scissor rect.
    if (config.scissorRect) {
        const auto& sr = *config.scissorRect;
        MTL::ScissorRect rect = {sr.x, sr.y, sr.width, sr.height};
        renderCmdEncoder->setScissorRect(rect);
    }

    // Attach frame fragment-shader buffers.
    for (U64 i = 0; i < buffers.size(); i++) {
        auto& [buffer, target] = buffers[i];
        if (static_cast<U8>(target & Program::Target::VERTEX) > 0) {
            renderCmdEncoder->setVertexBuffer(buffer->getHandle(), 0, i);
        }
        if (static_cast<U8>(target & Program::Target::FRAGMENT) > 0) {
            renderCmdEncoder->setFragmentBuffer(buffer->getHandle(), 0, i);
        }
    }

    // Attach frame textures.
    for (U64 i = 0; i < textures.size(); i++) {
        renderCmdEncoder->setFragmentTexture(textures[i]->getHandle(), i);
        renderCmdEncoder->setFragmentSamplerState(textures[i]->getSamplerStateHandle(), i);
    }

    // Attach frame encoder.
    for (auto& draw : draws) {
        JST_CHECK(draw->encode(renderCmdEncoder));
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render
