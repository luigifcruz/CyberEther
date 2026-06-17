#ifndef JETSTREAM_DOMAINS_VISUALIZATION_FRAME_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_FRAME_MODULE_IMPL_HH

#include <jetstream/domains/visualization/frame/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/surface.hh>
#include <jetstream/render/base/buffer.hh>
#include <jetstream/render/base/texture.hh>
#include <jetstream/render/base/surface.hh>
#include <jetstream/render/base/program.hh>
#include <jetstream/render/base/vertex.hh>
#include <jetstream/render/base/draw.hh>

namespace Jetstream::Modules {

struct FrameImpl : public Module::Impl, public DynamicConfig<Frame> {
 public:
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

 protected:
    Tensor input;

    U64 width = 0;
    U64 height = 0;
    U64 channels = 0;

    SurfaceInteractionState interaction;

    struct {
        int width;
        int height;
        int channels;
        int useLut;
    } frameUniforms;

    std::shared_ptr<Render::Buffer> fillScreenVerticesBuffer;
    std::shared_ptr<Render::Buffer> fillScreenTextureVerticesBuffer;
    std::shared_ptr<Render::Buffer> fillScreenIndicesBuffer;
    std::shared_ptr<Render::Buffer> frameBuffer;
    std::shared_ptr<Render::Buffer> frameUniformBuffer;

    std::shared_ptr<Render::Texture> framebufferTexture;
    std::shared_ptr<Render::Texture> lutTexture;

    std::shared_ptr<Render::Program> frameProgram;
    std::shared_ptr<Render::Surface> renderSurface;
    std::shared_ptr<Render::Vertex> vertex;
    std::shared_ptr<Render::Draw> drawVertex;

    Result createPresent();
    Result destroyPresent();
    Result present();
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_FRAME_MODULE_IMPL_HH
