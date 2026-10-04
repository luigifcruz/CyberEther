#ifndef JETSTREAM_DOMAINS_VISUALIZATION_SIGNAL_VIEW_LINEPLOT_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_SIGNAL_VIEW_LINEPLOT_HH

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <glm/mat4x4.hpp>

#include <jetstream/memory/tensor.hh>
#include <jetstream/render/base.hh>

#include "common.hh"

namespace Jetstream::Modules {

namespace detail {

constexpr bool LineplotMaxHoldReady(const U64 completedBlocks,
                                    const U64 averaging) {
    return completedBlocks + 1 >= averaging;
}

inline std::optional<F64> LineplotAmplitudeValue(const F32 position,
                                                 const F32 min,
                                                 const F32 max) {
    if (!std::isfinite(position) || position <= -1.0f || position >= 1.0f) {
        return std::nullopt;
    }
    const F64 lower = std::min(min, max);
    const F64 upper = std::max(min, max);
    const F64 normalized = 0.5 + 0.25 * std::atanh(static_cast<F64>(position));
    return lower + normalized * (upper - lower);
}

inline std::string LineplotAmplitudeLabel(const F32 position,
                                          const F32 min,
                                          const F32 max) {
    const auto value = LineplotAmplitudeValue(position, min, max);
    if (!value) {
        return {};
    }
    const F64 rounded = std::round(*value);
    return jst::fmt::format("{:.0f}", rounded == 0.0 ? 0.0 : rounded);
}

inline void InitializeLineplotPoints(F32* signalPoints,
                                     F32* maxHoldPoints,
                                     const U64 numberOfElements) noexcept {
    for (U64 index = 0; index < numberOfElements; ++index) {
        const F32 x = index * 2.0f / (numberOfElements - 1) - 1.0f;
        signalPoints[(index * 2) + 0] = x;
        signalPoints[(index * 2) + 1] = 0.0f;
        maxHoldPoints[(index * 2) + 0] = x;
        maxHoldPoints[(index * 2) + 1] = -1.0f;
    }
}

}  // namespace detail

struct SignalViewLineplot {
    struct Config {
        U64 numberOfElements = 0;
        bool fill = true;
        bool maxHold = false;
    };

    struct Tensors {
        Tensor& signalPoints;
        Tensor& signalVertices;
        Tensor& fillVertices;
        Tensor& maxHoldPoints;
        Tensor& maxHoldVertices;
    };

    struct TraceUniforms {
        glm::mat4 transform;
        F32 thickness[2];
        F32 zoom;
        U32 numberOfPoints;
        F32 traceColor[4];
    };

    void configure(const Config& config);
    Result create(const std::shared_ptr<Render::Window>& window, const Tensors& tensors);
    void attach(Render::Surface::Config& surface) const;
    void layout(const glm::mat4& transform,
                const Extent2D<F32>& pixelSize,
                F32 panelScale,
                F32 zoom,
                const Render::ScissorRect& scissor);
    Result upload(const Tensor& signalPoints, bool& holdPending);
    Result present();
    std::optional<F32> sample(F32 position) const;

    Config config;
    std::vector<F32> displayedPoints;
    bool uniformsDirty = false;

    TraceUniforms signalUniforms{};
    TraceUniforms holdUniforms{};

    std::shared_ptr<Render::Buffer> signalPointsBuffer;
    std::shared_ptr<Render::Buffer> signalVerticesBuffer;
    std::shared_ptr<Render::Buffer> fillVerticesBuffer;
    std::shared_ptr<Render::Buffer> signalUniformBuffer;
    std::shared_ptr<Render::Buffer> maxHoldPointsBuffer;
    std::shared_ptr<Render::Buffer> maxHoldVerticesBuffer;
    std::shared_ptr<Render::Buffer> holdUniformBuffer;

    std::shared_ptr<Render::Kernel> signalKernel;
    std::shared_ptr<Render::Kernel> fillKernel;
    std::shared_ptr<Render::Kernel> maxHoldKernel;

    std::shared_ptr<Render::Program> signalProgram;
    std::shared_ptr<Render::Program> fillProgram;
    std::shared_ptr<Render::Program> maxHoldProgram;

    std::shared_ptr<Render::Vertex> signalVertex;
    std::shared_ptr<Render::Vertex> fillVertex;
    std::shared_ptr<Render::Vertex> maxHoldVertex;

    std::shared_ptr<Render::Draw> drawSignalVertex;
    std::shared_ptr<Render::Draw> drawFillVertex;
    std::shared_ptr<Render::Draw> drawMaxHoldVertex;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_SIGNAL_VIEW_LINEPLOT_HH
