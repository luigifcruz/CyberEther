#ifndef RENDER_GLES_BUFFER_H
#define RENDER_GLES_BUFFER_H

#include "render/gles/instance.hpp"

namespace Render {

class GLES::Buffer : public Render::Buffer {
 public:
    explicit Buffer(const Config& config, const GLES& instance);

    using Render::Buffer::size;

    Result update() final;
    Result update(const std::size_t& offset, const std::size_t& size) final;

 protected:
    Result create();
    Result destroy();

 private:
    const GLES& instance;

    friend class GLES::Surface;
    friend class GLES::Program;
};

}  // namespace Render

#endif
