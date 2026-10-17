#include "jetstream/render/devices/metal/buffer.hh"
#include "jetstream/render/devices/metal/draw.hh"
#include "jetstream/render/devices/metal/texture.hh"
#include "jetstream/render/devices/metal/program.hh"

namespace Jetstream::Render {

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

    NS::Error* err = nullptr;
    const auto& shaders = config.shaders[DeviceType::Metal];
    auto device = Backend::State<DeviceType::Metal>()->getDevice();

    auto opts = NS::TransferPtr(MTL::CompileOptions::alloc()->init());
    opts->setFastMathEnabled(true);
    opts->setLanguageVersion(MTL::LanguageVersion3_0);
    opts->setLibraryType(MTL::LibraryTypeExecutable);

    auto vertSource = NS::TransferPtr(NS::String::alloc()->init((char*)shaders[0].data(), shaders[0].size(), NS::UTF8StringEncoding, false));
    auto vertLibrary = NS::TransferPtr(device->newLibrary(vertSource.get(), opts.get(), &err));

    auto fragSource = NS::TransferPtr(NS::String::alloc()->init((char*)shaders[1].data(), shaders[1].size(), NS::UTF8StringEncoding, false));
    auto fragLibrary = NS::TransferPtr(device->newLibrary(fragSource.get(), opts.get(), &err));

    if (!vertLibrary || !fragLibrary) {
        JST_ERROR("Library error:\n{}", err->description()->utf8String());
        return Result::ERROR;
    }

    auto vertFunc = NS::TransferPtr(vertLibrary->newFunction(
        NS::String::string("main0", NS::UTF8StringEncoding)
    ));
    JST_ASSERT(vertFunc, "Failed to create vertex function.");

    auto fragFunc = NS::TransferPtr(fragLibrary->newFunction(
        NS::String::string("main0", NS::UTF8StringEncoding)
    ));
    JST_ASSERT(fragFunc, "Failed to create fragment function.");

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

    auto renderPipelineDescriptor = NS::TransferPtr(MTL::RenderPipelineDescriptor::alloc()->init());
    JST_ASSERT(renderPipelineDescriptor, "Failed to create render pipeline descriptor.");
    renderPipelineDescriptor->setVertexDescriptor(vertDesc.get());
    renderPipelineDescriptor->setVertexFunction(vertFunc.get());
    renderPipelineDescriptor->setFragmentFunction(fragFunc.get());
    if (multisampled) {
        renderPipelineDescriptor->setSampleCount(Backend::State<DeviceType::Metal>()->getMultisampling());
    }

    const auto& colorAttachment = renderPipelineDescriptor->colorAttachments()->object(0)->init();
    colorAttachment->setPixelFormat(pixelFormat);

    if (config.enableAlphaBlending) {
        colorAttachment->setBlendingEnabled(true);
        colorAttachment->setSourceRGBBlendFactor(MTL::BlendFactorSourceAlpha);
        colorAttachment->setDestinationRGBBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
        colorAttachment->setRgbBlendOperation(MTL::BlendOperationAdd);
        colorAttachment->setSourceAlphaBlendFactor(MTL::BlendFactorOne);
        colorAttachment->setDestinationAlphaBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
        colorAttachment->setAlphaBlendOperation(MTL::BlendOperationAdd);
    }

    renderPipelineState = device->newRenderPipelineState(renderPipelineDescriptor.get(), &err);
    if (!renderPipelineState) {
        JST_ERROR("Failed to create pipeline state:\n{}", err->description()->utf8String());
        return Result::ERROR;
    }

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
