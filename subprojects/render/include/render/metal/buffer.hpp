#ifndef RENDER_METAL_BUFFER_H
#define RENDER_METAL_BUFFER_H

#include "render/metal/instance.hpp"

namespace Render {

class Metal::Buffer : public Render::Buffer {
 public:
    explicit Buffer(const Config& config, const Metal& instance);

    using Render::Buffer::size;

    void* raw() final;
    Result pour() final;
    Result fill() final;
    Result fill(const std::size_t& offset, const std::size_t& size) final;

 protected:
    Result create();
    Result destroy();

 private:
    const Metal& instance;

    MTL::Buffer* buffer = nullptr;

    friend class Metal::Surface;
    friend class Metal::Program;
};

}  // namespace Render

#endif
