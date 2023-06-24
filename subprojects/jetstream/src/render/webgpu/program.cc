#include "jetstream/render/webgpu/buffer.hh"
#include "jetstream/render/webgpu/draw.hh"
#include "jetstream/render/webgpu/texture.hh"
#include "jetstream/render/webgpu/program.hh"

namespace Jetstream::Render {

using Implementation = ProgramImp<Device::WebGPU>;

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

Result Implementation::create(const MTL::PixelFormat& pixelFormat) {
    JST_DEBUG("Creating Metal program.");

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
        JST_FATAL("Library error:\n{}", err->description()->utf8String());
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
    renderPipelineDescriptor->colorAttachments()->object(0)->init()->setPixelFormat(pixelFormat);
    renderPipelineState = device->newRenderPipelineState(renderPipelineDescriptor, &err);
    if (!renderPipelineState) {
        JST_FATAL("Failed to create pipeline state:\n{}", err->description()->utf8String());
        return Result::ERROR;
    }

    renderPipelineDescriptor->release();

    for (const auto& texture : textures) {
        JST_CHECK(texture->create());
    }

    for (const auto& [buffer, _] : buffers) {
        JST_CHECK(buffer->create());
    }

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
