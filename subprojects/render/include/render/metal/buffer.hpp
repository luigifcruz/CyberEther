#ifndef RENDER_METAL_BUFFER_H
#define RENDER_METAL_BUFFER_H

#include "render/metal/instance.hpp"

namespace Render {

class Metal::Buffer : public Render::Buffer {
 public:
    explicit Buffer(const Config& config, const Metal& instance);

    using Render::Buffer::size;

    Result update() final;
    Result update(const std::size_t& offset, const std::size_t& size) final;

 protected:
    Result create();
    Result destroy();

    constexpr const MTL::Buffer* getHandle() const {
        return buffer;
    }

 private:
    const Metal& instance;

    MTL::Buffer* buffer = nullptr;

    friend class Metal::Surface;
    friend class Metal::Program;
    friend class Metal::Vertex;
};

}  // namespace Render

#endif
