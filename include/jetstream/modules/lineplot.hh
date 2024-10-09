#ifndef JETSTREAM_MODULES_LINEPLOT_HH
#define JETSTREAM_MODULES_LINEPLOT_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"

#include "jetstream/memory/base.hh"
#include "jetstream/render/base.hh"
#include "jetstream/compute/graph/base.hh"

#include "jetstream/render/components/text.hh"

namespace Jetstream {

#define JST_LINEPLOT_CPU(MACRO) \
    MACRO(Lineplot, CPU, F32)

#define JST_LINEPLOT_METAL(MACRO) \
    MACRO(Lineplot, Metal, F32)

#define JST_LINEPLOT_CUDA(MACRO) \
    MACRO(Lineplot, CUDA, F32)

template<Device D, typename T = F32>
class Lineplot : public Module, public Compute, public Present {
 public:
    Lineplot();
    ~Lineplot();

    // Configuration

    struct Config {
        U64 averaging = 1;
        U64 numberOfVerticalLines = 11;
        U64 numberOfHorizontalLines = 5;
        Extent2D<U64> viewSize = {512, 384};
        F32 zoom = 1.0f;
        F32 translation = 0.0f;
        F32 thickness = 1.0f;
        F32 scale = 1.0f;

        JST_SERDES(averaging, numberOfVerticalLines, numberOfHorizontalLines, viewSize, zoom, translation, thickness, scale);
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        Tensor<D, T> buffer;

        JST_SERDES_INPUT(buffer);
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        JST_SERDES_OUTPUT();
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    // Taint & Housekeeping

    constexpr Device device() const {
        return D;
    }

    void info() const final;

    // Constructor

    Result create();

    // Miscellaneous

    constexpr const Extent2D<U64>& viewSize() const {
        return config.viewSize;
    }
    const Extent2D<U64>& viewSize(const Extent2D<U64>& viewSize);

    constexpr const F32& zoom() const {
        return config.zoom;
    }
    std::pair<F32, F32> zoom(const Extent2D<F32>& mouse_pos, const F32& zoom);

    constexpr const F32& translation() const {
        return config.translation;
    }
    const F32& translation(const F32& translation);

    constexpr const U64& averaging() const {
        return config.averaging;
    }
    const U64& averaging(const U64& averaging);

    constexpr const F32& scale() const {
        return config.scale;
    }
    const F32& scale(const F32& scale);

    constexpr const Extent2D<F32>& cursor() const {
        return cursorPos;
    }
    const Extent2D<F32>& cursor(const Extent2D<F32>& cursorPos);

    Render::Texture& getTexture();

 protected:
    Result createCompute(const Context& ctx) final;
    Result compute(const Context& ctx) final;

    Result createPresent() final;
    Result present() final;
    Result destroyPresent() final;

 private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;

    struct GImpl;
    std::unique_ptr<GImpl> gimpl;

    // TODO: Move all of this to internal implementation.
    // TODO: Abstract signal and grid with Line Component.

    Tensor<D, F32> signalPoints;
    Tensor<D, F32> signalVertices;
    Tensor<Device::CPU, F32> gridPoints;
    Tensor<Device::CPU, F32> cursorSignalPoint;
    Tensor<D, F32> gridVertices;

    std::shared_ptr<Render::Buffer> signalPointsBuffer;
    std::shared_ptr<Render::Buffer> signalVerticesBuffer;
    std::shared_ptr<Render::Buffer> signalUniformBuffer;
    std::shared_ptr<Render::Buffer> gridPointsBuffer;
    std::shared_ptr<Render::Buffer> gridVerticesBuffer;
    std::shared_ptr<Render::Buffer> gridUniformBuffer;
    std::shared_ptr<Render::Buffer> cursorVerticesBuffer;
    std::shared_ptr<Render::Buffer> cursorIndicesBuffer;
    std::shared_ptr<Render::Buffer> cursorUniformBuffer;

    std::shared_ptr<Render::Texture> framebufferTexture;
    std::shared_ptr<Render::Texture> lutTexture;

    std::shared_ptr<Render::Kernel> gridKernel;
    std::shared_ptr<Render::Kernel> signalKernel;

    std::shared_ptr<Render::Program> signalProgram;
    std::shared_ptr<Render::Program> gridProgram;
    std::shared_ptr<Render::Program> cursorProgram;

    std::shared_ptr<Render::Surface> surface;

    std::shared_ptr<Render::Vertex> gridVertex;
    std::shared_ptr<Render::Vertex> signalVertex;
    std::shared_ptr<Render::Vertex> cursorVertex;

    std::shared_ptr<Render::Draw> drawGridVertex;
    std::shared_ptr<Render::Draw> drawSignalVertex;
    std::shared_ptr<Render::Draw> drawCursorVertex;

    std::shared_ptr<Render::Components::Text> text;

    U64 numberOfElements = 0;
    U64 numberOfBatches = 0;
    F32 normalizationFactor = 0.0f;

    Extent2D<F32> cursorPos = {0.0f, 0.0f};

    bool updateGridPointsFlag = false;
    bool updateSignalPointsFlag = false;
    bool updateCursorUniformBufferFlag = false;
    bool updateSignalUniformBufferFlag = false;
    bool updateGridUniformBufferFlag = false;

    void updateState();
    void updateCursorState();
    void generateGridPoints();

    JST_DEFINE_IO()
};

#ifdef JETSTREAM_MODULE_LINEPLOT_CPU_AVAILABLE
JST_LINEPLOT_CPU(JST_SPECIALIZATION);
#endif
#ifdef JETSTREAM_MODULE_LINEPLOT_CUDA_AVAILABLE
JST_LINEPLOT_CUDA(JST_SPECIALIZATION);
#endif
#ifdef JETSTREAM_MODULE_LINEPLOT_METAL_AVAILABLE
JST_LINEPLOT_METAL(JST_SPECIALIZATION);
#endif

}  // namespace Jetstream

#endif
