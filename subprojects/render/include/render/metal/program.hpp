#ifndef RENDER_METAL_PROGRAM_H
#define RENDER_METAL_PROGRAM_H

#include "render/metal/surface.hpp"

namespace Render {

class Metal::Program : public Render::Program {
public:
    Program(const Config& c, const Metal& i) : Render::Program(c), inst(i) {};

    Result setUniform(std::string, const std::vector<int>&) final;
    Result setUniform(std::string, const std::vector<float>&) final;

protected:
    Result create(const MTL::PixelFormat& pixelFormat);
    Result destroy();
    Result draw(MTL::CommandBuffer* commandBuffer,
                MTL::RenderPassDescriptor* renderPassDesc);

private:
    const Metal& inst;

    MTL::RenderPipelineState* renderPipelineState;

    std::vector<std::shared_ptr<Metal::Draw>> draws;
    std::vector<std::shared_ptr<Metal::Texture>> textures;

    static Result checkShaderCompilation(uint);
    static Result checkProgramCompilation(uint);

    friend class Metal::Surface;
};

} // namespace Render

#endif
