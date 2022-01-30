#ifndef RENDER_GLES_BUFFER_H
#define RENDER_GLES_BUFFER_H

#include "render/gles/instance.hpp"

namespace Render {

class GLES::Buffer : public Render::Buffer {
 public:
    explicit Buffer(const Config& config, const GLES& instance);

    Result update() final;
    Result update(const std::size_t& offset, const std::size_t& size) final;

 protected:
    Result begin();
    Result end();
    Result create();
    Result destroy();

    constexpr const uint* getHandle() const {
        return &id;
    }

 private:
    const GLES& instance;

    uint id, target;

    friend class GLES::Surface;
    friend class GLES::Program;
    friend class GLES::Vertex;
};

}  // namespace Render

#endif
