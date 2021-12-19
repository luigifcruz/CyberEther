#include "render/metal/program.hpp"
#include "render/metal/texture.hpp"
#include "render/metal/draw.hpp"

namespace Render {

Result Metal::Program::create(const MTL::PixelFormat& pixelFormat) {
    for (const auto& draw : cfg.draws) {
        draws.push_back(std::dynamic_pointer_cast<Metal::Draw>(draw));
    }

    for (const auto& texture : cfg.textures) {
        textures.push_back(std::dynamic_pointer_cast<Metal::Texture>(texture));
    }

    NS::Error* err;
    MTL::CompileOptions* opts = MTL::CompileOptions::alloc();
    NS::String* source = NS::String::string(*cfg.vertexSource, NS::ASCIIStringEncoding);
    auto library = inst.device->newLibrary(source, opts, &err);

    if (!library) {
        fmt::print("Library error:\n{}\n", err->description()->utf8String());
        return Result::ERROR;
    }

    MTL::Function* vertFunc = library->newFunction(NS::String::string("vertFunc", NS::ASCIIStringEncoding));
    assert(vertFunc);

    MTL::Function* fragFunc = library->newFunction(NS::String::string("fragFunc", NS::ASCIIStringEncoding));
    assert(fragFunc);

    auto renderPipelineDesc = MTL::RenderPipelineDescriptor::alloc()->init();
    assert(renderPipelineDesc);
    renderPipelineDesc->setVertexFunction(vertFunc);
    renderPipelineDesc->setFragmentFunction(fragFunc);
    renderPipelineDesc->colorAttachments()->object(0)->setPixelFormat(pixelFormat);

    renderPipelineState = inst.device->newRenderPipelineState(renderPipelineDesc, &err);
    assert(renderPipelineState);

    for (const auto& draw : draws) {
        CHECK(draw->create());
    }

    for (const auto& texture : textures) {
        CHECK(texture->create());
    }

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::Program::destroy() {
    for (const auto& draw : draws) {
        CHECK(draw->destroy());
    }

    for (const auto& texture : textures) {
        CHECK(texture->destroy());
    }

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
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

    for (auto& draw : draws) {
        CHECK(draw->encode(renderCmdEncoder));
    }

    renderCmdEncoder->endEncoding();

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::Program::setUniform(std::string name, const std::vector<int>& vars) {
    /*
    // optimize: this can be cached
    // optimize: are std::vector performant?
    glUseProgram(shader);
    int loc = glGetUniformLocation(shader, name.c_str());

    switch(vars.size()) {
        case 1: glUniform1i(loc, vars.at(0)); break; case 2:
            glUniform2i(loc, vars.at(0), vars.at(1));
            break;
        case 3:
            glUniform3i(loc, vars.at(0), vars.at(1), vars.at(2));
            break;
        case 4:
            glUniform4i(loc, vars.at(0), vars.at(1), vars.at(2), vars.at(3));
            break;
        default:
#ifdef RENDER_DEBUG
        std::cerr << "[RENDER:PROGRAM] Invalid number of uniforms (vars.size() > 4)." << std::endl;
#endif
            return Result::RENDER_BACKEND_ERROR;
    }
    */

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::Program::setUniform(std::string name, const std::vector<float>& vars) {
    /*
    // optimize: this can be cached
    // optimize: are std::vector performant?
    glUseProgram(shader);
    int loc = glGetUniformLocation(shader, name.c_str());

    switch(vars.size()) {
        case 1:
            glUniform1f(loc, vars.at(0));
            break;
        case 2:
            glUniform2f(loc, vars.at(0), vars.at(1));
            break;
        case 3:
            glUniform3f(loc, vars.at(0), vars.at(1), vars.at(2));
            break;
        case 4:
            glUniform4f(loc, vars.at(0), vars.at(1), vars.at(2), vars.at(3));
            break;
        default:
#ifdef RENDER_DEBUG
        std::cerr << "[RENDER:PROGRAM] Invalid number of uniforms (vars.size() > 4)." << std::endl;
#endif
            return Result::RENDER_BACKEND_ERROR;
    }
    */

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

} // namespace Render
