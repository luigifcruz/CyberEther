#include "render/metal/program.hpp"
#include "render/metal/texture.hpp"
#include "render/metal/draw.hpp"

namespace Render {

Metal::Program::Program(const Config& config, const Metal& instance)
         : Render::Program(config), instance(instance) {
    for (const auto& draw : config.draws) {
        draws.push_back(std::dynamic_pointer_cast<Metal::Draw>(draw));
    }

    for (const auto& texture : config.textures) {
        textures.push_back(std::dynamic_pointer_cast<Metal::Texture>(texture));
    }
}

Result Metal::Program::create(const MTL::PixelFormat& pixelFormat) {
    NS::Error* err;
    MTL::CompileOptions* opts = MTL::CompileOptions::alloc();
    NS::String* source = NS::String::string(*config.vertexSource, NS::ASCIIStringEncoding);
    auto library = instance.device->newLibrary(source, opts, &err);

    if (!library) {
        fmt::print("Library error:\n{}\n", err->description()->utf8String());
        return Result::ERROR;
    }

    MTL::Function* vertFunc = library->newFunction(NS::String::string("vertFunc", NS::ASCIIStringEncoding));
    RENDER_ASSERT(vertFunc);

    MTL::Function* fragFunc = library->newFunction(NS::String::string("fragFunc", NS::ASCIIStringEncoding));
    RENDER_ASSERT(fragFunc);

    auto renderPipelineDesc = MTL::RenderPipelineDescriptor::alloc()->init();
    RENDER_ASSERT(renderPipelineDesc);
    renderPipelineDesc->setVertexFunction(vertFunc);
    renderPipelineDesc->setFragmentFunction(fragFunc);
    renderPipelineDesc->colorAttachments()->object(0)->setPixelFormat(pixelFormat);

    renderPipelineState = instance.device->newRenderPipelineState(renderPipelineDesc, &err);
    RENDER_ASSERT(renderPipelineState);

    renderPipelineDesc->release();

    for (const auto& draw : draws) {
        CHECK(draw->create());
    }

    for (const auto& texture : textures) {
        CHECK(texture->create());
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

    std::size_t index = 0;
    for (const auto& texture : textures) {
        renderCmdEncoder->setFragmentTexture((MTL::Texture*)texture->raw(), index);
        index += 1;
    }

    if (config.vertexUniforms) {
        renderCmdEncoder->setVertexBytes(config.vertexUniforms, config.vertexUniformsSize, 29);
    }

    if (config.fragmentUniforms) {
        renderCmdEncoder->setFragmentBytes(config.fragmentUniforms, config.fragmentUniformsSize, 30);
    }

    renderUniforms.drawIndex = 0;
    for (auto& draw : draws) {
        renderCmdEncoder->setVertexBytes(&renderUniforms, sizeof(renderUniforms), 28);
        renderCmdEncoder->setFragmentBytes(&renderUniforms, sizeof(renderUniforms), 28);
        CHECK(draw->encode(renderCmdEncoder));
        renderUniforms.drawIndex += 1;
    }

    renderCmdEncoder->endEncoding();
    renderCmdEncoder->release();

    return Result::SUCCESS;
}

} // namespace Render
