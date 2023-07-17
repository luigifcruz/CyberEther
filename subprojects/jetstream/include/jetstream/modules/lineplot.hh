#ifndef JETSTREAM_MODULES_LINEPLOT_HH
#define JETSTREAM_MODULES_LINEPLOT_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"
#include "jetstream/memory/base.hh"
#include "jetstream/render/base.hh"
#include "jetstream/render/extras.hh"
#include "jetstream/graph/base.hh"

namespace Jetstream {

template<Device D, typename T = F32>
class Lineplot : public Module, public Compute, public Present {
 public:
    struct Config {
        U64 numberOfVerticalLines = 20;
        U64 numberOfHorizontalLines = 5;
        Render::Size2D<U64> viewSize = {1024, 512};
    };

    struct Input {
        const Vector<D, T, 2> buffer;
    };

    struct Output {
    };

    explicit Lineplot(const Config& config,
                      const Input& input);

    constexpr Device device() const {
        return D;
    }

    const std::string name() const {
        return "Lineplot";
    }

    void summary() const final;

    constexpr Config getConfig() const {
        return config;
    }

    constexpr const Render::Size2D<U64>& viewSize() const {
        return config.viewSize;
    }
    const Render::Size2D<U64>& viewSize(const Render::Size2D<U64>& viewSize);

    Render::Texture& getTexture();

    static Result Factory(std::unordered_map<std::string, std::any>& config,
                          std::unordered_map<std::string, std::any>& input,
                          std::unordered_map<std::string, std::any>& output,
                          std::shared_ptr<Lineplot<D, T>>& module,
                          const bool& castFromString = false);

 protected:
    Config config;
    const Input input;
    Output output;

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

    Result createCompute(const RuntimeMetadata& meta) final;
    Result underlyingCreateCompute(const RuntimeMetadata& meta);
    Result compute(const RuntimeMetadata& meta) final;

    Result createPresent(Render::Window& window) final;
    Result present(Render::Window& window) final;

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
};

}  // namespace Jetstream

#endif
