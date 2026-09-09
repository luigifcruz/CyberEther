#ifndef JETSTREAM_DOMAINS_VISUALIZATION_FRAME_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_FRAME_MODULE_IMPL_HH

#include <array>
#include <atomic>
#include <string>
#include <vector>

#include <jetstream/domains/visualization/frame/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/surface.hh>
#include <jetstream/render/base/buffer.hh>
#include <jetstream/render/base/texture.hh>
#include <jetstream/render/base/surface.hh>
#include <jetstream/render/base/program.hh>
#include <jetstream/render/base/vertex.hh>
#include <jetstream/render/base/draw.hh>
#include <jetstream/render/components/axis.hh>
#include <jetstream/render/components/shapes.hh>
#include <jetstream/render/components/text.hh>

namespace Jetstream::Modules {

constexpr F32 kFrameZoomSpeed = 0.15f;
constexpr F32 kFrameMaxZoom = 64.0f;
constexpr F32 kFrameDragThresholdPx = 4.0f;

struct FrameImpl : public Module::Impl, public DynamicConfig<Frame> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

 protected:
    Tensor input;

    U64 width = 0;
    U64 height = 0;
    U64 channels = 0;

    std::string frameLabel;

    std::atomic<F32> autoRangeMin{0.0f};
    std::atomic<F32> autoRangeMax{1.0f};

    SurfaceInteractionState interaction;

    struct {
        F32 zoom = 1.0f;
        Extent2D<F32> center = {0.5f, 0.5f};
        Extent2D<F32> fitScale = {1.0f, 1.0f};
        bool dragging = false;
        Extent2D<F32> dragAnchor = {0.0f, 0.0f};
        Extent2D<F32> dragCenter = {0.5f, 0.5f};
        bool selecting = false;
        Extent2D<F32> selectAnchor = {0.0f, 0.0f};
        Extent2D<F32> selectCurrent = {0.0f, 0.0f};
        bool hasCursor = false;
        Extent2D<F32> cursor = {0.0f, 0.0f};
    } view;

    std::string cursorLabel;

    bool updateLutFlag = false;

    struct {
        int width;
        int height;
        int channels;
        int useLut;
        int interpolate;
        float rangeMin;
        float rangeScale;
        float zoom;
        float centerX;
        float centerY;
        float fitScaleX;
        float fitScaleY;
        float paddingScaleX;
        float paddingScaleY;
    } frameUniforms{};

    std::array<uint8_t, 256 * 4> lutBytes{};

    std::shared_ptr<Render::Buffer> fillScreenVerticesBuffer;
    std::shared_ptr<Render::Buffer> fillScreenTextureVerticesBuffer;
    std::shared_ptr<Render::Buffer> fillScreenIndicesBuffer;
    std::shared_ptr<Render::Buffer> frameBuffer;
    std::shared_ptr<Render::Buffer> frameUniformBuffer;

    std::shared_ptr<Render::Texture> framebufferTexture;
    std::shared_ptr<Render::Texture> lutTexture;

    std::shared_ptr<Render::Program> frameProgram;
    std::shared_ptr<Render::Surface> renderSurface;
    std::shared_ptr<Render::Vertex> vertex;
    std::shared_ptr<Render::Draw> drawVertex;

    std::shared_ptr<Render::Components::Axis> axis;
    std::shared_ptr<Render::Components::Text> text;
    std::shared_ptr<Render::Components::Shapes> selection;

    Result createPresent();
    Result destroyPresent();
    Result present();

    void processMouseEvents(const std::vector<MouseEvent>& events);
    void clampViewCenter();
    Extent2D<F32> plotToView(const Extent2D<F32>& plot) const;
    F32 plotDistancePx(const Extent2D<F32>& a, const Extent2D<F32>& b) const;
    void zoomToSelection();
    Result updateSelectionState();
    Extent2D<F32> surfaceToPlot(const Extent2D<F32>& position) const;
    Extent2D<F32> plotToImage(const Extent2D<F32>& plot) const;
    Extent2D<F32> clampToFrame(const Extent2D<F32>& plot) const;
    void updateFitScale();
    Result updateViewGeometry();
    void fillLut();
    Result updateAxisState();
    Result updateTextState();
    void updateCursorReadout();
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_FRAME_MODULE_IMPL_HH
