#include "jetstream/render/devices/metal/buffer.hh"
#include "jetstream/render/devices/metal/draw.hh"
#include "jetstream/render/devices/metal/texture.hh"
#include "jetstream/render/devices/metal/program.hh"

namespace Jetstream::Render {

using Implementation = ProgramImp<Device::Metal>;

Implementation::ProgramImp(const Config& config) : Program(config) {
    _draw = std::dynamic_pointer_cast<DrawImp<Device::Metal>>(config.draw);

    for (auto& texture : config.textures) {
        textures.push_back(
            std::dynamic_pointer_cast<TextureImp<Device::Metal>>(texture)
        );
    }

    for (auto& [buffer, target] : config.buffers) {
        buffers.push_back(
            {std::dynamic_pointer_cast<BufferImp<Device::Metal>>(buffer), target}
        );
    }
}

Result Implementation::create(const std::shared_ptr<TextureImp<Device::Metal>>& framebuffer) {
    JST_DEBUG("Creating Metal program.");

    if (config.shaders.contains(Device::Metal) == 0) {
        JST_ERROR("[Metal] Module doesn't have necessary shader.");       
        return Result::ERROR;
    }

    NS::Error* err = nullptr;
    const auto& shaders = config.shaders[Device::Metal];
    auto device = Backend::State<Device::Metal>()->getDevice();

    MTL::CompileOptions* opts = MTL::CompileOptions::alloc()->init();
    opts->setFastMathEnabled(true);
    opts->setLanguageVersion(MTL::LanguageVersion3_0);
    opts->setLibraryType(MTL::LibraryTypeExecutable);

    auto vertSource = NS::String::alloc()->init((char*)shaders[0].data(), shaders[0].size(), NS::UTF8StringEncoding, false);
    auto vertLibrary = device->newLibrary(vertSource, opts, &err);

    auto fragSource = NS::String::alloc()->init((char*)shaders[1].data(), shaders[1].size(), NS::UTF8StringEncoding, false);
    auto fragLibrary = device->newLibrary(fragSource, opts, &err);

    if (!vertLibrary || !fragLibrary) {
        JST_ERROR("Library error:\n{}", err->description()->utf8String());
        return Result::ERROR;
    }

    MTL::Function* vertFunc = vertLibrary->newFunction(
        NS::String::string("main0", NS::UTF8StringEncoding)
    );
    JST_ASSERT(vertFunc);

    MTL::Function* fragFunc = fragLibrary->newFunction(
        NS::String::string("main0", NS::UTF8StringEncoding)
    );
    JST_ASSERT(fragFunc);

    U64 indexOffset = 0;
    for (U64 i = 0; i < buffers.size(); i++) {
        auto& [_, target] = buffers[i];
        if (static_cast<U8>(target & Program::Target::VERTEX) > 0) {
            indexOffset++;
        }
    }
    auto vertDesc = MTL::VertexDescriptor::alloc()->init();
    JST_CHECK(_draw->create(vertDesc, indexOffset));

    auto renderPipelineDescriptor = MTL::RenderPipelineDescriptor::alloc()->init();
    JST_ASSERT(renderPipelineDescriptor);
    renderPipelineDescriptor->setVertexDescriptor(vertDesc);
    renderPipelineDescriptor->setVertexFunction(vertFunc);
    renderPipelineDescriptor->setFragmentFunction(fragFunc);
    if (framebuffer->multisampled()) {
        renderPipelineDescriptor->setSampleCount(Backend::State<Device::Metal>()->getMultisampling());
    }

    const auto& colorAttachment = renderPipelineDescriptor->colorAttachments()->object(0)->init();
    colorAttachment->setPixelFormat(framebuffer->getPixelFormat());

    if (config.enableAlphaBlending) {
        colorAttachment->setBlendingEnabled(true);
        colorAttachment->setSourceRGBBlendFactor(MTL::BlendFactorSourceAlpha);
        colorAttachment->setDestinationRGBBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
        colorAttachment->setRgbBlendOperation(MTL::BlendOperationAdd);
        colorAttachment->setSourceAlphaBlendFactor(MTL::BlendFactorSourceAlpha);
        colorAttachment->setDestinationAlphaBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
        colorAttachment->setAlphaBlendOperation(MTL::BlendOperationAdd);
    }

    renderPipelineState = device->newRenderPipelineState(renderPipelineDescriptor, &err);
    if (!renderPipelineState) {
        JST_ERROR("Failed to create pipeline state:\n{}", err->description()->utf8String());
        return Result::ERROR;
    }

    renderPipelineDescriptor->release();

    for (const auto& texture : textures) {
        JST_CHECK(texture->create());
    }

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_CHECK(_draw->destroy());

    for (const auto& texture : textures) {
        JST_CHECK(texture->destroy());
    }

    renderPipelineState->release();

    return Result::SUCCESS;
}

Result Implementation::draw(MTL::RenderCommandEncoder* renderCmdEncoder) {
    renderCmdEncoder->setRenderPipelineState(renderPipelineState);

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
    JST_CHECK(_draw->encode(renderCmdEncoder));

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render
