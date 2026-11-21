#ifndef JETSTREAM_RENDER_BASE_PRESENT_HH
#define JETSTREAM_RENDER_BASE_PRESENT_HH

#include <memory>
#include <optional>
#include <unordered_set>
#include <vector>

#include "jetstream/render/base/texture.hh"
#include "jetstream/render/base/window_metrics.hh"
#include "jetstream/render/types.hh"
#include "jetstream/types.hh"

namespace Jetstream::Render {

struct JETSTREAM_API PresentList {
    enum class Blend : U8 {
        Straight,
        Premultiplied,
    };

    struct Layer {
        std::shared_ptr<const Texture> texture;
        Rect rect;
        Rect clip;
    };

    struct Quad {
        std::shared_ptr<const Texture> texture;
        U64 handle = 0;
        Rect rect;
        ScissorRect scissor;
        Blend blend = Blend::Straight;
    };

    std::vector<Quad> quads;
    std::unordered_set<const Texture*> sampled;

    static PresentList Build(const WindowMetrics& metrics,
                             const std::vector<Layer>& underlays,
                             const std::optional<Layer>& canvas,
                             Blend canvasBlend);

    void append(PresentList other);
    void clear();
    bool samples(const Texture& texture) const;
};

}  // namespace Jetstream::Render

#endif  // JETSTREAM_RENDER_BASE_PRESENT_HH
