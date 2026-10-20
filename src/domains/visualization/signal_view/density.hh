#ifndef JETSTREAM_DOMAINS_VISUALIZATION_SIGNAL_VIEW_DENSITY_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_SIGNAL_VIEW_DENSITY_HH

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <vector>

#include <jetstream/memory/tensor.hh>
#include <jetstream/surface.hh>
#include <jetstream/render/base.hh>
#include <jetstream/render/colormap.hh>

#include "common.hh"

namespace Jetstream::Modules {

namespace detail {

constexpr U64 kDensityMaxColumns = 8192;
constexpr U64 kDensityLevels = 256;
constexpr U64 kDensityHistoryRows = 128;
constexpr const char* kDensityColormap = "density";

struct DensityColumns {
    U64 binsPerColumn = 1;
    U64 columns = 0;
};

constexpr DensityColumns PlanDensityColumns(const U64 width) {
    const U64 binsPerColumn = width > kDensityMaxColumns
        ? (width + kDensityMaxColumns - 1) / kDensityMaxColumns
        : 1;
    return {
        .binsPerColumn = binsPerColumn,
        .columns = (width + binsPerColumn - 1) / binsPerColumn,
    };
}

inline F32 DensityDecay(const F32 halfLife) {
    return std::pow(0.5f, 1.0f / std::max(halfLife, 1.0e-3f));
}

}  // namespace detail

struct SignalViewDensity {
    struct Config {
        U64 width = 0;
        U64 history = 0;
        U64 persistence = 1;
    };

    struct Uniforms {
        U32 columns;
        U32 levels;
        F32 binsPerColumn;
        F32 binSpan;
        F32 offset;
        F32 zoom;
        F32 panelScaleX;
        F32 panelScaleY;
        F32 plotWidth;
        F32 intensityScale;
        F32 panelOffsetY;
    };

    struct AccumulateUniforms {
        U32 width;
        U32 columns;
        U32 levels;
        U32 binsPerColumn;
        U32 history;
        U32 writeIndex;
        U32 version;
        U32 origin;
        U32 epoch;
        F32 decay;
        F32 padding0;
        F32 padding1;
    };

    void configure(const Config& config);
    Result create(const std::shared_ptr<Render::Window>& window, Tensor& history);
    void attach(Render::Surface::Config& surface) const;
    void layout(const Render::ScissorRect& scissor,
                F32 panelScaleX,
                F32 panelScaleY,
                F32 panelOffsetY);
    void persistence(U64 halfLife);
    void reset();
    Result update(const WaterfallFrame& frame);
    Result present(const SurfaceInteractionState& interaction);

    Config config;
    detail::DensityColumns columns;
    Uniforms uniforms{};
    AccumulateUniforms accumulateUniforms{};
    std::vector<std::array<U32, 2>> state;
    bool accumulateDirty = false;

    std::shared_ptr<Render::Buffer> fillScreenVerticesBuffer;
    std::shared_ptr<Render::Buffer> fillScreenTextureVerticesBuffer;
    std::shared_ptr<Render::Buffer> fillScreenIndicesBuffer;
    std::shared_ptr<Render::Buffer> historyBuffer;
    std::shared_ptr<Render::Buffer> densityBuffer;
    std::shared_ptr<Render::Buffer> stateBuffer;
    std::shared_ptr<Render::Buffer> uniformBuffer;
    std::shared_ptr<Render::Buffer> accumulateUniformBuffer;
    std::shared_ptr<Render::Kernel> accumulateKernel;
    Render::Colormap lut;
    std::shared_ptr<Render::Program> program;
    std::shared_ptr<Render::Vertex> vertex;
    std::shared_ptr<Render::Draw> draw;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_SIGNAL_VIEW_DENSITY_HH
