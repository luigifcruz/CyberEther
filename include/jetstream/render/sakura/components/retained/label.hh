#ifndef JETSTREAM_RENDER_SAKURA_RETAINED_LABEL_HH
#define JETSTREAM_RENDER_SAKURA_RETAINED_LABEL_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura::Retained {

struct Label : public Component {
    struct Instance {
        Rect rect;
        std::string str;
        bool visible = true;
        ColorRGBA<F32> color = {1.0f, 1.0f, 1.0f, 1.0f};
        F32 fontSize = 15.0f;
        Extent2D<I32> alignment = {0, 0};

        bool operator==(const Instance&) const = default;
    };

    struct Config {
        std::string id;
        std::vector<Instance> instances;
        std::optional<Rect> clip;
        std::string fontName = "default_mono";
        F32 sharpness = 0.45f;
        U64 maxCharacters = 128;
        U64 capacity = 0;
    };

    Label();
    ~Label();

    bool update(Config config);

 protected:
    Result build(Context& ctx) override;
    Result paint() override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura::Retained

#endif  // JETSTREAM_RENDER_SAKURA_RETAINED_LABEL_HH
