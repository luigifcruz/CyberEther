#ifndef JETSTREAM_RENDER_SAKURA_IMAGE_HH
#define JETSTREAM_RENDER_SAKURA_IMAGE_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <optional>

namespace Jetstream::Render {
class Texture;
}  // namespace Jetstream::Render

namespace Jetstream::Sakura {

struct Image : public Component {
    struct Config {
        std::shared_ptr<const Render::Texture> texture;
        Rect rect;
        std::optional<Rect> clip;
        F32 cornerRadius = 0.0f;
        F32 opacity = 1.0f;
        bool visible = true;
        bool live = false;

        bool operator==(const Config&) const = default;
    };

    Image();
    ~Image();

    void update(Config config);
    const Config& config() const;

 protected:
    Result build(Context& ctx) override;
    Result paint() override;
    void conceal() override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_IMAGE_HH
