#ifndef JETSTREAM_MODULES_LINEPLOT_HH
#define JETSTREAM_MODULES_LINEPLOT_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"
#include "jetstream/memory/base.hh"
#include "jetstream/render/base.hh"
#include "jetstream/render/extras.hh"

namespace Jetstream {

template<Device D, typename IT = F32>
class Lineplot : public Module, public Compute, public Present {
 public:
    struct Config {
        Render::Size2D<U64> viewSize = {4096, 512};
    };

    struct Input {
        const Vector<D, IT>& buffer;
    };

    struct Output {
    };

    explicit Lineplot(const Config& config,
                      const Input& input);

    constexpr const Device device() const {
        return D;
    }

    constexpr const Taint taints() const {
        return Taint::None;
    }

    void summary() const final;

    constexpr const U64 getBufferSize() const {
        return input.buffer.size();
    }

    constexpr const Config getConfig() const {
        return config;
    }

    constexpr const Render::Size2D<U64>& viewSize() const {
        return config.viewSize;
    }
    const Render::Size2D<U64>& viewSize(const Render::Size2D<U64>& viewSize);

    Render::Texture& getTexture();

 protected:
    Config config;
    const Input input;
    Output output;

    Vector<Device::CPU, IT> plot;
    Vector<Device::CPU, IT> grid;

    std::shared_ptr<Render::Buffer> gridVerticesBuffer;
    std::shared_ptr<Render::Buffer> lineVerticesBuffer;

    std::shared_ptr<Render::Texture> texture;
    std::shared_ptr<Render::Texture> lutTexture;

    std::shared_ptr<Render::Program> program;

    std::shared_ptr<Render::Surface> surface;

    std::shared_ptr<Render::Vertex> gridVertex;
    std::shared_ptr<Render::Vertex> lineVertex;

    std::shared_ptr<Render::Draw> drawGridVertex;
    std::shared_ptr<Render::Draw> drawLineVertex;

    const Result createCompute(const RuntimeMetadata& meta) final;
    const Result compute(const RuntimeMetadata& meta) final;

    const Result createPresent(Render::Window& window) final;
    const Result present(Render::Window& window) final;

 private:
    //
    // Metal Shaders
    //

    const char* MetalShader = R"END(
        #include <metal_stdlib>

        using namespace metal;

        struct TexturePipelineRasterizerData {
            float4 position [[position]];
            float2 texcoord;
        };

        struct RenderUniforms {
            uint32_t drawIndex;
        };

        vertex TexturePipelineRasterizerData vertFunc(
                const device packed_float3* vertexArray [[buffer(0)]],
                unsigned int vID[[vertex_id]]) {
            float3 vert = vertexArray[vID];
            TexturePipelineRasterizerData out;

            float min_x = 0.0;
            float max_x = 1.0;
            float y = (2.0 * ((vert.y - min_x)/(max_x - min_x)) - 1.0);
            out.position = vector_float4(vert.x, y, vert.z, 1.0);

            float pos = (y + 1.0)/2.0;
            out.texcoord = vector_float2(pos, 0.0);

            return out;
        }

        fragment float4 fragFunc(
                TexturePipelineRasterizerData in [[stage_in]],
                constant uint32_t& drawIndex [[buffer(30)]],
                texture2d<float> lut [[texture(0)]]) {
            if (drawIndex == 0) {
                return vector_float4(0.2, 0.2, 0.2, 1.0);
            }

            if (drawIndex == 1) {
                sampler lutSampler(filter::linear);
                return lut.sample(lutSampler, in.texcoord);
            }

            return vector_float4(0.0, 1.0, 0.0, 1.0);
        }
    )END";
};

}  // namespace Jetstream

#endif
