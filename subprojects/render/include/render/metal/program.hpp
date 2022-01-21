#ifndef RENDER_METAL_PROGRAM_H
#define RENDER_METAL_PROGRAM_H

#include <vector>
#include <memory>

#include "render/metal/surface.hpp"

namespace Render {

class Metal::Program : public Render::Program {
 public:
    explicit Program(const Config& config, const Metal& instance);

 protected:
    Result create(const MTL::PixelFormat& pixelFormat);
    Result destroy();
    Result draw(MTL::CommandBuffer* commandBuffer,
                MTL::RenderPassDescriptor* renderPassDesc);

 private:
    const Metal& instance;

    MTL::RenderPipelineState* renderPipelineState = nullptr;

    std::vector<std::shared_ptr<Metal::Draw>> draws;
    std::vector<std::shared_ptr<Metal::Texture>> textures;
    std::vector<std::shared_ptr<Metal::Buffer>> vertexBuffers;
    std::vector<std::shared_ptr<Metal::Buffer>> fragmentBuffers;

    static Result checkShaderCompilation(uint);
    static Result checkProgramCompilation(uint);

    friend class Metal::Surface;
};

}  // namespace Render

#endif
