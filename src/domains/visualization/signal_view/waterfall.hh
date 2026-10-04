#ifndef JETSTREAM_DOMAINS_VISUALIZATION_SIGNAL_VIEW_WATERFALL_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_SIGNAL_VIEW_WATERFALL_HH

#include <memory>

#include <jetstream/memory/tensor.hh>
#include <jetstream/surface.hh>
#include <jetstream/render/base.hh>
#include <jetstream/render/colormap.hh>

#include "common.hh"

namespace Jetstream::Modules {

struct SignalViewWaterfall {
    struct Config {
        U64 width = 0;
        U64 height = 0;
    };

    struct Uniforms {
        int width;
        int height;
        F32 index;
        F32 offset;
        F32 zoom;
        F32 panelScaleX;
        F32 panelScaleY;
        F32 panelOffsetY;
    };

    void configure(const Config& config);
    Result create(const std::shared_ptr<Render::Window>& window,
                  Tensor& bins,
                  const std::string& colormap);
    void attach(Render::Surface::Config& surface) const;
    void layout(const Render::ScissorRect& scissor,
                F32 panelScaleX,
                F32 panelScaleY,
                F32 panelOffsetY);
    Result upload(const WaterfallFrame& frame);
    Result present(const SurfaceInteractionState& interaction, const std::string& colormap);

    Config config;
    Uniforms uniforms{};

    std::shared_ptr<Render::Buffer> fillScreenVerticesBuffer;
    std::shared_ptr<Render::Buffer> fillScreenTextureVerticesBuffer;
    std::shared_ptr<Render::Buffer> fillScreenIndicesBuffer;
    std::shared_ptr<Render::Buffer> binsBuffer;
    std::shared_ptr<Render::Buffer> uniformBuffer;
    Render::Colormap lut;
    std::shared_ptr<Render::Program> program;
    std::shared_ptr<Render::Vertex> vertex;
    std::shared_ptr<Render::Draw> draw;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_SIGNAL_VIEW_WATERFALL_HH
