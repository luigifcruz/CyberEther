#include "jetstream/render/metal/buffer.hh"
#include "jetstream/render/metal/draw.hh"
#include "jetstream/render/metal/texture.hh"
#include "jetstream/render/metal/program.hh"

namespace Jetstream::Render {

using Implementation = ProgramImp<Device::Metal>;

Implementation::ProgramImp(const Config& config) : Program(config) {
    JST_INFO("Greetings from the Program Metal thingy.");

    for (auto& draw : config.draws) {
        draws.push_back(
            std::dynamic_pointer_cast<DrawImp<Device::Metal>>(draw)
        );
    }

    for (auto& texture : config.textures) {
        textures.push_back(
            std::dynamic_pointer_cast<TextureImp<Device::Metal>>(texture)
        );
    }

    for (auto& buffer : config.buffers) {
        buffers.push_back(
            std::dynamic_pointer_cast<BufferImp<Device::Metal>>(buffer)
        );
    }
}

const Result Implementation::create(const MTL::PixelFormat& pixelFormat) {
    JST_DEBUG("Creating Metal program.");

    NS::Error* err = nullptr;
    const auto& shader = config.shaders[Device::Metal][0];
    auto device = Backend::State<Device::Metal>()->getDevice();

    MTL::CompileOptions* opts = MTL::CompileOptions::alloc();
    NS::String* source = NS::String::string(shader, NS::ASCIIStringEncoding);
    auto library = device->newLibrary(source, opts, &err);

    if (!library) {
        JST_FATAL("Library error:\n{}", err->description()->utf8String());
        return Result::ERROR;
    }

    MTL::Function* vertFunc = library->newFunction(
        NS::String::string("vertFunc", NS::ASCIIStringEncoding)
    );
    JST_ASSERT(vertFunc);

    MTL::Function* fragFunc = library->newFunction(
        NS::String::string("fragFunc", NS::ASCIIStringEncoding)
    );
    JST_ASSERT(fragFunc);

    auto renderPipelineDescriptor = MTL::RenderPipelineDescriptor::alloc()->init();
    JST_ASSERT(renderPipelineDescriptor);
    renderPipelineDescriptor->setVertexFunction(vertFunc);
    renderPipelineDescriptor->setFragmentFunction(fragFunc);
    renderPipelineDescriptor->colorAttachments()->object(0)->setPixelFormat(pixelFormat);

    renderPipelineState = device->newRenderPipelineState(renderPipelineDescriptor, &err);
    JST_ASSERT(renderPipelineState);

    renderPipelineDescriptor->release();

    for (const auto& draw : draws) {
        JST_CHECK(draw->create());
    }

    for (const auto& texture : textures) {
        JST_CHECK(texture->create());
    }

    for (const auto& buffer : buffers) {
        JST_CHECK(buffer->create());
    }

    return Result::SUCCESS;
}

const Result Implementation::destroy() {
    for (const auto& draw : draws) {
        JST_CHECK(draw->destroy());
    }

    for (const auto& texture : textures) {
        JST_CHECK(texture->destroy());
    }

    for (const auto& buffer : buffers) {
        JST_CHECK(buffer->destroy());
    }

    renderPipelineState->release();

    return Result::SUCCESS;
}

const Result Implementation::draw(MTL::CommandBuffer* commandBuffer,
                         MTL::RenderPassDescriptor* renderPassDescriptor) {
    auto renderCmdEncoder = commandBuffer->renderCommandEncoder(renderPassDescriptor);

    renderCmdEncoder->setRenderPipelineState(renderPipelineState);

    // Attach frame textures.
    for (U64 i = 0; i < textures.size(); i++) {
        renderCmdEncoder->setFragmentTexture(textures[i]->getHandle(), i);
    }

    // Attach frame fragment-shader buffers.
    for (U64 i = 0; i < buffers.size(); i++) {
        renderCmdEncoder->setFragmentBuffer(buffers[i]->getHandle(), 0, i);
        renderCmdEncoder->setVertexBuffer(buffers[i]->getHandle(), 0, i);
    }

    drawIndex = 0;
    for (auto& draw : draws) {
        // Attach drawIndex uniforms.
        renderCmdEncoder->setVertexBytes(&drawIndex, sizeof(drawIndex), 30);
        renderCmdEncoder->setFragmentBytes(&drawIndex, sizeof(drawIndex), 30);
        drawIndex += 1;

        // Attach frame encoder.
        JST_CHECK(draw->encode(renderCmdEncoder, buffers.size()));
    }

    renderCmdEncoder->endEncoding();
    renderCmdEncoder->release();

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render
