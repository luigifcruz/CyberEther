#ifndef JETSTREAM_MODULES_WATERFALL_HH
#define JETSTREAM_MODULES_WATERFALL_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"

#include "jetstream/memory/base.hh"
#include "jetstream/render/base.hh"
#include "jetstream/render/extras.hh"
#include "jetstream/compute/graph/base.hh"

namespace Jetstream {

template<Device D, typename T = F32>
class Waterfall : public Module, public Compute, public Present {
 public:
    // Configuration 

    struct Config {
        F32 zoom = 1.0;
        I32 offset = 0;
        U64 height = 512;
        bool interpolate = true;
        Size2D<U64> viewSize = {512, 384};

        JST_SERDES(
            JST_SERDES_VAL("zoom", zoom);
            JST_SERDES_VAL("offset", offset);
            JST_SERDES_VAL("height", height);
            JST_SERDES_VAL("interpolate", interpolate);
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

    std::string_view name() const {
        return "waterfall";
    }

    std::string_view prettyName() const {
        return "Waterfall";
    }

    void summary() const final;

    // Constructor

    Result create();

    // Miscellaneous

    constexpr const Size2D<U64>& viewSize() const {
        return config.viewSize;
    }
    const Size2D<U64>& viewSize(const Size2D<U64>& viewSize);

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

 protected:
    Result createCompute(const RuntimeMetadata& meta) final;
    Result compute(const RuntimeMetadata& meta) final;

    Result createPresent() final;
    Result present() final;
    Result destroyPresent() final;

 private:
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

    Result underlyingCompute(const RuntimeMetadata& meta);

    JST_DEFINE_MODULE_IO();
};

}  // namespace Jetstream

#endif
