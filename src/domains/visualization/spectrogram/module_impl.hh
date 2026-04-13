#ifndef JETSTREAM_DOMAINS_VISUALIZATION_SPECTROGRAM_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_SPECTROGRAM_MODULE_IMPL_HH

#include <jetstream/domains/visualization/spectrogram/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/surface.hh>
#include <jetstream/render/base/buffer.hh>
#include <jetstream/render/base/texture.hh>
#include <jetstream/render/base/surface.hh>
#include <jetstream/render/base/program.hh>
#include <jetstream/render/base/vertex.hh>
#include <jetstream/render/base/draw.hh>
#include <jetstream/render/components/axis.hh>

namespace Jetstream::Modules {

constexpr F32 kSpectrogramDecayBase = 0.999f;
constexpr F32 kSpectrogramMinTickSpacingPx = 65.0f;

struct SpectrogramImpl : public Module::Impl, public DynamicConfig<Spectrogram> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;

 protected:
    Tensor input;
    Tensor frequencyBins;

    U64 numberOfElements = 0;
    U64 numberOfBatches = 0;
    F32 decayFactor = 0.0f;

    // Surface interaction state.
    SurfaceInteractionState interaction;

    // Rendering members.

    struct {
        int width;
        int height;
        float offset;
        float zoom;
        float paddingScaleX;
        float paddingScaleY;
    } signalUniforms;

    std::shared_ptr<Render::Buffer> fillScreenVerticesBuffer;
    std::shared_ptr<Render::Buffer> fillScreenTextureVerticesBuffer;
    std::shared_ptr<Render::Buffer> fillScreenIndicesBuffer;
    std::shared_ptr<Render::Buffer> signalBuffer;
    std::shared_ptr<Render::Buffer> signalUniformBuffer;

    std::shared_ptr<Render::Texture> framebufferTexture;
    std::shared_ptr<Render::Texture> lutTexture;

    std::shared_ptr<Render::Program> signalProgram;

    std::shared_ptr<Render::Surface> renderSurface;

    std::shared_ptr<Render::Vertex> vertex;

    std::shared_ptr<Render::Draw> drawVertex;

    std::shared_ptr<Render::Components::Axis> axis;

    Result createPresent();
    Result destroyPresent();
    Result present();
    Result updateAxisState();
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_SPECTROGRAM_MODULE_IMPL_HH
