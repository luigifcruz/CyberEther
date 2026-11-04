#ifndef JETSTREAM_RENDER_SAKURA_BOX_HH
#define JETSTREAM_RENDER_SAKURA_BOX_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct Box : public Component {
    static constexpr U8 TopLeft = 1;
    static constexpr U8 TopRight = 2;
    static constexpr U8 BottomRight = 4;
    static constexpr U8 BottomLeft = 8;
    static constexpr U8 AllCorners = TopLeft | TopRight | BottomRight | BottomLeft;

    struct Instance {
        Rect rect;
        bool visible = true;
        ColorRGBA<F32> backgroundColor = {1.0f, 1.0f, 1.0f, 1.0f};
        U8 corners = AllCorners;

        bool operator==(const Instance&) const = default;
    };

    struct Config {
        std::vector<Instance> instances;
        std::optional<Rect> clip;
        F32 cornerRadius = 0.0f;
        F32 borderWidth = 0.0f;
        ColorRGBA<F32> borderColor = {0.0f, 0.0f, 0.0f, 1.0f};
        bool strokeFringe = false;
        F32 softness = 0.0f;
        bool topLit = false;
        U64 capacity = 0;

        bool operator==(const Config&) const = default;
    };

    Box();
    ~Box();

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

#endif  // JETSTREAM_RENDER_SAKURA_BOX_HH
