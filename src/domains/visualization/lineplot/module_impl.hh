#ifndef JETSTREAM_DOMAINS_VISUALIZATION_LINEPLOT_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_LINEPLOT_MODULE_IMPL_HH

#include <glm/mat4x4.hpp>

#include <jetstream/domains/visualization/lineplot/module.hh>
#include <jetstream/detail/module_impl.hh>
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

namespace Jetstream::Modules {

struct LineplotImpl : public Module::Impl, public DynamicConfig<Lineplot> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

 protected:
    Tensor input;
    Tensor signalPoints;

    U64 numberOfElements = 0;
    U64 numberOfBatches = 0;
    F32 normalizationFactor = 0.0f;

    // Surface interaction state.
    SurfaceInteractionState interaction;

    // Rendering uniforms.

    struct {
        glm::mat4 transform;
        F32 thickness[2];
        F32 zoom;
        U32 numberOfPoints;
    } signalUniforms;

    struct {
        glm::mat4 transform;
    } cursorUniforms;

    // Rendering state.
    Extent2D<F32> pixelSize;
    Extent2D<F32> cursorPos = {0.0f, 0.0f};

    // Update flags.
    bool updateSignalPointsFlag = false;
    bool updateCursorUniformBufferFlag = false;
    bool updateSignalUniformBufferFlag = false;

    // Rendering buffers.
    Tensor signalVertices;
    Tensor cursorSignalPoint;

    std::shared_ptr<Render::Buffer> signalPointsBuffer;
    std::shared_ptr<Render::Buffer> signalVerticesBuffer;
    std::shared_ptr<Render::Buffer> signalUniformBuffer;
    std::shared_ptr<Render::Buffer> cursorVerticesBuffer;
    std::shared_ptr<Render::Buffer> cursorIndicesBuffer;
    std::shared_ptr<Render::Buffer> cursorUniformBuffer;

    std::shared_ptr<Render::Texture> framebufferTexture;
    std::shared_ptr<Render::Texture> lutTexture;

    std::shared_ptr<Render::Kernel> signalKernel;

    std::shared_ptr<Render::Program> signalProgram;
    std::shared_ptr<Render::Program> cursorProgram;

    std::shared_ptr<Render::Surface> renderSurface;

    std::shared_ptr<Render::Vertex> signalVertex;
    std::shared_ptr<Render::Vertex> cursorVertex;

    std::shared_ptr<Render::Draw> drawSignalVertex;
    std::shared_ptr<Render::Draw> drawCursorVertex;

    std::shared_ptr<Render::Components::Axis> axis;
    std::shared_ptr<Render::Components::Text> text;

    Result createPresent();
    Result destroyPresent();
    Result present();

    void updateState();
    void updateCursorState();
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_LINEPLOT_MODULE_IMPL_HH
