#ifndef JETSTREAM_RENDER_VULKAN_DRAW_HH
#define JETSTREAM_RENDER_VULKAN_DRAW_HH

#include "jetstream/render/base/draw.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Render {

template<>
class DrawImp<Device::Vulkan> : public Draw {
 public:
    explicit DrawImp(const Config& config);

 protected:
    Result create();
    Result encode(VkCommandBuffer& commandBuffer,
                  const U64& offset);
    Result destroy();

 private:
    std::shared_ptr<VertexImp<Device::Vulkan>> buffer;

    friend class ProgramImp<Device::Vulkan>;
};

}  // namespace Jetstream::Render

#endif
