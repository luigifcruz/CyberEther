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

template<Device D, typename T = F32>
class Lineplot : public Module, public Compute, public Present {
 public:
    // Configuration 

    struct Config {
        U64 numberOfVerticalLines = 20;
        U64 numberOfHorizontalLines = 5;
        Size2D<U64> viewSize = {512, 384};

        JST_SERDES(
            JST_SERDES_VAL("numberOfVerticalLines", numberOfVerticalLines);
            JST_SERDES_VAL("numberOfHorizontalLines", numberOfHorizontalLines);
            JST_SERDES_VAL("viewSize", viewSize);
        );
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        Vector<D, T, 2> buffer;

        JST_SERDES(
            JST_SERDES_VAL("buffer", buffer);
        );
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        JST_SERDES();
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    // Taint & Housekeeping

    constexpr Device device() const {
        return D;
    }

    constexpr std::string name() const {
        return "lineplot";
    }

    constexpr std::string prettyName() const {
        return "Lineplot";
    }

    void summary() const final; 

    // Constructor

    Result create();

    // Miscellaneous

    constexpr const Size2D<U64>& viewSize() const {
        return config.viewSize;
    }
    const Size2D<U64>& viewSize(const Size2D<U64>& viewSize);

    Render::Texture& getTexture();

 protected:
    Result createCompute(const RuntimeMetadata& meta) final;
    Result compute(const RuntimeMetadata& meta) final;

    Result createPresent() final;
    Result present() final;
    Result destroyPresent() final;

 private:
    Vector<D, F32, 2> plot;
    Vector<D, F32, 3> grid;

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

#ifdef JETSTREAM_MODULE_LINEPLOT_METAL_AVAILABLE
    struct MetalConstants {
        U16 batchSize;
        U16 gridSize;
    };

    struct {
        MTL::ComputePipelineState* state;
        Vector<Device::Metal, U8> constants;
    } metal;
#endif

    JST_DEFINE_MODULE_IO();
};

}  // namespace Jetstream

#endif
