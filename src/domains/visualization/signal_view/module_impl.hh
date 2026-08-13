#ifndef JETSTREAM_DOMAINS_VISUALIZATION_SIGNAL_VIEW_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_SIGNAL_VIEW_MODULE_IMPL_HH

#include <memory>

#include <glm/mat4x4.hpp>

#include <jetstream/domains/visualization/signal_view/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/memory/tensor.hh>
#include <jetstream/surface.hh>
#include <jetstream/render/base/buffer.hh>
#include <jetstream/render/base/texture.hh>
#include <jetstream/render/base/surface.hh>
#include <jetstream/render/base/program.hh>
#include <jetstream/render/base/kernel.hh>
#include <jetstream/render/base/vertex.hh>
#include <jetstream/render/base/draw.hh>
#include <jetstream/render/components/axis.hh>
#include <jetstream/render/components/text.hh>

#include "waterfall_history.hh"

namespace Jetstream::Modules {

namespace detail {

inline bool SignalViewHasLineplot(const std::string& mode) {
    return mode == "lineplot" || mode == "lineplot_waterfall";
}

inline bool SignalViewHasWaterfall(const std::string& mode) {
    return mode == "waterfall" || mode == "lineplot_waterfall";
}

constexpr U64 LineplotInputIndex(const U64 batch,
                                 const U64 index,
                                 const U64 batchStride,
                                 const U64 elementStride,
                                 const U64 decimation) {
    return (batch * batchStride) + (index * decimation * elementStride);
}

constexpr bool LineplotMaxHoldReady(const U64 completedBlocks,
                                    const U64 averaging) {
    return completedBlocks + 1 >= averaging;
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

struct SignalViewImpl : public Module::Impl,
                        public DynamicConfig<SignalView> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

 protected:
    Tensor input;

    U64 numberOfElements = 0;
    U64 numberOfBatches = 0;
    U64 inputElementCount = 0;
    U64 inputElementStride = 0;
    U64 inputBatchStride = 0;
    U64 maxHoldWarmupBlocks = 0;
    F32 normalizationFactor = 0.0f;
    bool lineplotEnabled = false;
    bool waterfallEnabled = false;

    U64 validatedNumberOfElements = 0;
    U64 validatedNumberOfBatches = 0;
    U64 validatedInputElementCount = 0;
    U64 validatedInputElementStride = 0;
    U64 validatedInputBatchStride = 0;
    F32 validatedNormalizationFactor = 0.0f;
    bool validatedLineplotEnabled = false;
    bool validatedWaterfallEnabled = false;

    // Surface interaction state.
    SurfaceInteractionState interaction;

    // Rendering state.
    Extent2D<F32> pixelSize;

    std::shared_ptr<Render::Texture> framebufferTexture;
    std::shared_ptr<Render::Surface> renderSurface;
    std::shared_ptr<Render::Components::Axis> axis;
    std::shared_ptr<Render::Components::Text> text;

    struct TraceUniforms {
        glm::mat4 transform;
        F32 thickness[2];
        F32 zoom;
        U32 numberOfPoints;
        F32 traceColor[4];
    };

    Tensor signalPoints;
    Tensor signalVertices;
    Tensor fillVertices;
    Tensor maxHoldPoints;
    Tensor maxHoldVertices;

    TraceUniforms signalUniforms{};
    TraceUniforms holdUniforms{};

    bool updateSignalPointsFlag = false;
    bool updateHoldPointsFlag = false;
    bool updateSignalUniformBufferFlag = false;

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

    Tensor waterfallBins;
    WaterfallHistory waterfallHistory;

    struct WaterfallUniforms {
        int width;
        int height;
        F32 index;
        F32 offset;
        F32 zoom;
        F32 panelScaleX;
        F32 panelScaleY;
        F32 panelOffsetY;
    } waterfallUniforms{};

    std::shared_ptr<Render::Buffer> fillScreenVerticesBuffer;
    std::shared_ptr<Render::Buffer> fillScreenTextureVerticesBuffer;
    std::shared_ptr<Render::Buffer> fillScreenIndicesBuffer;
    std::shared_ptr<Render::Buffer> waterfallBuffer;
    std::shared_ptr<Render::Buffer> waterfallUniformBuffer;
    std::shared_ptr<Render::Texture> waterfallLutTexture;
    std::shared_ptr<Render::Program> waterfallProgram;
    std::shared_ptr<Render::Vertex> waterfallVertex;
    std::shared_ptr<Render::Draw> drawWaterfallVertex;

    Result createPresent();
    Result destroyPresent();
    Result present();

    void updateState();
    void updateLabelState();
    Result resetLineplotHistory();
    Result resetHistoryState();
    virtual Buffer::Config renderStateBufferConfig() const = 0;
    virtual Result resetAveragingState() {
        return Result::SUCCESS;
    }
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_SIGNAL_VIEW_MODULE_IMPL_HH
