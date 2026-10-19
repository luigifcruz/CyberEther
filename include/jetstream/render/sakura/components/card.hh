#ifndef JETSTREAM_RENDER_SAKURA_CARD_HH
#define JETSTREAM_RENDER_SAKURA_CARD_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct Card : public Component {
    enum class Layer : U8 {
        Full,
        Base,
        Frame,
    };

    struct Config {
        std::vector<Rect> rects;
        std::optional<Rect> clip;
        F32 cornerRadius = 0.0f;
        F32 softness = 0.0f;
        F32 drop = 0.0f;
        F32 borderWidth = 1.0f;
        F32 lineWidth = 1.0f;
        std::string fillKey = "tile_bg";
        F32 opacity = 1.0f;
        Layer layer = Layer::Full;
        U64 capacity = 4;

        bool operator==(const Config&) const = default;
    };

    Card();
    ~Card();

    void update(Config config);

 protected:
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_CARD_HH
