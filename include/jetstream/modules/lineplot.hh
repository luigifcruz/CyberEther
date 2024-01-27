#ifndef JETSTREAM_MODULES_LINEPLOT_HH
#define JETSTREAM_MODULES_LINEPLOT_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"

#include "jetstream/memory/base.hh"
#include "jetstream/render/base.hh"
#include "jetstream/render/extras.hh"
#include "jetstream/compute/graph/base.hh"

namespace Jetstream {

#define JST_LINEPLOT_CPU(MACRO) \
    MACRO(Lineplot, CPU, F32)

#define JST_LINEPLOT_METAL(MACRO) \
    MACRO(Lineplot, Metal, F32)

template<Device D, typename T = F32>
class Lineplot : public Module, public Compute, public Present {
 public:
    // Configuration 

    struct Config {
        U64 numberOfVerticalLines = 20;
        U64 numberOfHorizontalLines = 5;
        Size2D<U64> viewSize = {512, 384};

        JST_SERDES(numberOfVerticalLines, numberOfHorizontalLines, viewSize);
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

    constexpr const Size2D<U64>& viewSize() const {
        return config.viewSize;
    }
    const Size2D<U64>& viewSize(const Size2D<U64>& viewSize);

    Render::Texture& getTexture();

 protected:
    Result createCompute(const Context& ctx) final;
    Result compute(const Context& ctx) final;

    Result createPresent() final;
    Result present() final;
    Result destroyPresent() final;

 private:
    Tensor<Device::CPU, F32> plot;
    Tensor<Device::CPU, F32> grid;

    std::shared_ptr<Render::Buffer> gridVerticesBuffer;
    std::shared_ptr<Render::Buffer> lineVerticesBuffer;

    std::shared_ptr<Render::Texture> texture;
    std::shared_ptr<Render::Texture> lutTexture;

    std::shared_ptr<Render::Program> signalProgram;
    std::shared_ptr<Render::Program> gridProgram;

    std::shared_ptr<Render::Surface> surface;

    std::shared_ptr<Render::Vertex> gridVertex;
    std::shared_ptr<Render::Vertex> lineVertex;

    std::shared_ptr<Render::Draw> drawGridVertex;
    std::shared_ptr<Render::Draw> drawLineVertex;

    // TODO: Remove backend specific code from header in favor of `pimpl->`.
#ifdef JETSTREAM_MODULE_LINEPLOT_METAL_AVAILABLE
    struct MetalConstants {
        U16 batchSize;
        U16 gridSize;
    };

    struct {
        MTL::ComputePipelineState* state;
        Tensor<Device::Metal, U8> constants;
    } metal;
#endif

    U64 numberOfElements = 0;
    U64 numberOfBatches = 0;

    JST_DEFINE_IO();
};

#ifdef JETSTREAM_MODULE_LINEPLOT_CPU_AVAILABLE
JST_LINEPLOT_CPU(JST_SPECIALIZATION);
#endif
#ifdef JETSTREAM_MODULE_LINEPLOT_METAL_AVAILABLE
JST_LINEPLOT_METAL(JST_SPECIALIZATION);
#endif

}  // namespace Jetstream

#endif
