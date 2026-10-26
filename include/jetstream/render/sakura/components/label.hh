#ifndef JETSTREAM_RENDER_SAKURA_LABEL_HH
#define JETSTREAM_RENDER_SAKURA_LABEL_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/typography.hh>
#include <jetstream/types.hh>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct Label : public Component {
    struct Instance {
        Rect rect;
        std::string str;
        bool visible = true;
        ColorRGBA<F32> color = {1.0f, 1.0f, 1.0f, 1.0f};
        F32 fontSize = Typography::FontSize;
        Extent2D<I32> alignment = {0, 0};
        F32 rotation = 0.0f;

        bool operator==(const Instance&) const = default;
    };

    struct Config {
        std::vector<Instance> instances;
        std::optional<Rect> clip;
        std::string fontName = Typography::MonoFont;
        F32 sharpness = 0.45f;
        F32 weight = 0.0f;
        U64 maxCharacters = 128;
        U64 capacity = 0;

        bool operator==(const Config&) const = default;
    };

    Label();
    ~Label();

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

#endif  // JETSTREAM_RENDER_SAKURA_LABEL_HH
