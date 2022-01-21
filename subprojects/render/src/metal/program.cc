#include "render/metal/program.hpp"
#include "render/metal/texture.hpp"
#include "render/metal/draw.hpp"
#include "render/metal/buffer.hpp"

namespace Render {

Metal::Program::Program(const Config& config, const Metal& instance)
         : Render::Program(config), instance(instance) {
    for (const auto& draw : config.draws) {
        draws.push_back(std::dynamic_pointer_cast<Metal::Draw>(draw));
    }

    for (const auto& texture : config.textures) {
        textures.push_back(std::dynamic_pointer_cast<Metal::Texture>(texture));
    }

    for (const auto& buffer : config.buffers) {
        buffers.push_back(std::dynamic_pointer_cast<Metal::Buffer>(buffer));
    }
}

Result Metal::Program::create(const MTL::PixelFormat& pixelFormat) {
    const auto& shader = config.shaders[instance.getBackendId()][0];

    NS::Error* err;
    MTL::CompileOptions* opts = MTL::CompileOptions::alloc();
    NS::String* source = NS::String::string(shader, NS::ASCIIStringEncoding);
    auto library = instance.getDevice()->newLibrary(source, opts, &err);

    if (!library) {
        fmt::print("Library error:\n{}\n", err->description()->utf8String());
        return Result::ERROR;
    }

    MTL::Function* vertFunc = library->newFunction(NS::String::string("vertFunc",
        NS::ASCIIStringEncoding));
    RENDER_ASSERT(vertFunc);

    MTL::Function* fragFunc = library->newFunction(NS::String::string("fragFunc",
        NS::ASCIIStringEncoding));
    RENDER_ASSERT(fragFunc);

    auto renderPipelineDesc = MTL::RenderPipelineDescriptor::alloc()->init();
    RENDER_ASSERT(renderPipelineDesc);
    renderPipelineDesc->setVertexFunction(vertFunc);
    renderPipelineDesc->setFragmentFunction(fragFunc);
    renderPipelineDesc->colorAttachments()->object(0)->setPixelFormat(pixelFormat);

    renderPipelineState = instance.getDevice()->
        newRenderPipelineState(renderPipelineDesc, &err);
    RENDER_ASSERT(renderPipelineState);

    renderPipelineDesc->release();

    for (const auto& draw : draws) {
        CHECK(draw->create());
    }

    for (const auto& texture : textures) {
        CHECK(texture->create());
    }

    for (const auto& buffer : buffers) {
        CHECK(buffer->create());
    }

    return Result::SUCCESS;
}

Result Metal::Program::destroy() {
    for (const auto& draw : draws) {
        CHECK(draw->destroy());
    }

    for (const auto& texture : textures) {
        CHECK(texture->destroy());
    }

    renderPipelineState->release();

    return Result::SUCCESS;
}

Result Metal::Program::draw(MTL::CommandBuffer* commandBuffer,
                            MTL::RenderPassDescriptor* renderPassDesc) {
    auto renderCmdEncoder = commandBuffer->renderCommandEncoder(renderPassDesc);

    renderCmdEncoder->setRenderPipelineState(renderPipelineState);

    for (std::size_t i = 0; i < textures.size(); i++) {
        renderCmdEncoder->setFragmentTexture((MTL::Texture*)textures[i]->raw(), i);
    }

    for (std::size_t i = 0; i < buffers.size(); i++) {
        renderCmdEncoder->setFragmentBuffer((MTL::Buffer*)buffers[i]->raw(), 0, i);
    }

    std::size_t index = 29;
    for (auto const& [key, data] : config.uniforms) {
        renderCmdEncoder->setVertexBytes(data.data(), data.size_bytes(), index);
        renderCmdEncoder->setFragmentBytes(data.data(), data.size_bytes(), index);
        index -= 1;
    }

    drawIndex = 0;
    for (auto& draw : draws) {
        renderCmdEncoder->setVertexBytes(&drawIndex, sizeof(drawIndex), 30);
        renderCmdEncoder->setFragmentBytes(&drawIndex, sizeof(drawIndex), 30);
        CHECK(draw->encode(renderCmdEncoder));
        drawIndex += 1;
    }

    renderCmdEncoder->endEncoding();
    renderCmdEncoder->release();

    return Result::SUCCESS;
}

}  // namespace Render
