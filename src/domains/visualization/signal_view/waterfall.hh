#ifndef JETSTREAM_DOMAINS_VISUALIZATION_SIGNAL_VIEW_WATERFALL_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_SIGNAL_VIEW_WATERFALL_HH

#include <array>
#include <memory>
#include <vector>

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
        int filtered;
    };

    struct FilterUniforms {
        U32 width;
        U32 height;
        U32 writeIndex;
        U32 version;
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
    Result update(const WaterfallFrame& frame);
    Result present(const SurfaceInteractionState& interaction, const std::string& colormap);

    Config config;
    Uniforms uniforms{};
    FilterUniforms filterUniforms{};
    std::vector<F32> upload;
    std::vector<std::array<U32, 2>> filterState;

    std::shared_ptr<Render::Buffer> fillScreenVerticesBuffer;
    std::shared_ptr<Render::Buffer> fillScreenTextureVerticesBuffer;
    std::shared_ptr<Render::Buffer> fillScreenIndicesBuffer;
    std::shared_ptr<Render::Buffer> binsBuffer;
    std::shared_ptr<Render::Buffer> uniformBuffer;
    std::shared_ptr<Render::Buffer> filteredBuffer;
    std::shared_ptr<Render::Buffer> filterUniformBuffer;
    std::shared_ptr<Render::Buffer> filterStateBuffer;
    std::shared_ptr<Render::Kernel> filterKernel;
    Render::Colormap lut;
    std::shared_ptr<Render::Program> program;
    std::shared_ptr<Render::Vertex> vertex;
    std::shared_ptr<Render::Draw> draw;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_SIGNAL_VIEW_WATERFALL_HH
