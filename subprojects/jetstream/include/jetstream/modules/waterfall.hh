#ifndef JETSTREAM_MODULES_WATERFALL_HH
#define JETSTREAM_MODULES_WATERFALL_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"
#include "jetstream/memory/base.hh"
#include "jetstream/render/base.hh"
#include "jetstream/render/extras.hh"
#include "jetstream/graph/base.hh"

namespace Jetstream {

template<Device D, typename T = F32>
class Waterfall : public Module, public Compute, public Present {
 public:
    struct Config {
        F32 zoom = 1.0;
        I32 offset = 0;
        U64 height = 512;
        bool interpolate = true;
        Render::Size2D<U64> viewSize = {4096, 512};
    };

    struct Input {
        const Vector<D, T, 2> buffer;
    };

    struct Output {
    };

    explicit Waterfall(const Config& config,
                       const Input& input);

    constexpr Device device() const {
        return D;
    }

    const std::string name() const {
        return "Waterfall";
    }

    void summary() const final;

    constexpr Config getConfig() const {
        return config;
    }

    constexpr const Render::Size2D<U64>& viewSize() const {
        return config.viewSize;
    }
    const Render::Size2D<U64>& viewSize(const Render::Size2D<U64>& viewSize);

    constexpr const bool& interpolate() const {
        return config.interpolate;
    }
    const bool& interpolate(const bool& interpolate);

    constexpr const F32& zoom() const {
        return config.zoom;
    }
    const F32& zoom(const F32& zoom);

    constexpr const I32& offset() const {
        return config.offset;
    }
    const I32& offset(const I32& offset);

    Render::Texture& getTexture();

    static Result Factory(std::unordered_map<std::string, std::any>& config,
                          std::unordered_map<std::string, std::any>& input,
                          std::unordered_map<std::string, std::any>& output,
                          std::shared_ptr<Waterfall<D, T>>& module,
                          const bool& castFromString = false);

 protected:
    Config config;
    const Input input;
    Output output;

    struct {
        int width;
        int height;
        int maxSize;
        float index;
        float offset;
        float zoom;
        bool interpolate;
    } shaderUniforms;

    int inc = 0, last = 0, ymax = 0;
    Vector<D, F32, 2> frequencyBins;

    std::shared_ptr<Render::Buffer> fillScreenVerticesBuffer;
    std::shared_ptr<Render::Buffer> fillScreenTextureVerticesBuffer;
    std::shared_ptr<Render::Buffer> fillScreenIndicesBuffer;

    std::shared_ptr<Render::Texture> texture;
    std::shared_ptr<Render::Buffer> binTexture;
    std::shared_ptr<Render::Buffer> uniformBuffer;
    std::shared_ptr<Render::Texture> lutTexture;
    std::shared_ptr<Render::Program> program;
    std::shared_ptr<Render::Surface> surface;
    std::shared_ptr<Render::Vertex> vertex;
    std::shared_ptr<Render::Draw> drawVertex;

    Result createCompute(const RuntimeMetadata& meta) final;
    Result compute(const RuntimeMetadata& meta) final;

    Result createPresent(Render::Window& window) final;
    Result present(Render::Window& window) final;

    Result underlyingCompute(const RuntimeMetadata& meta);
};

}  // namespace Jetstream

#endif
