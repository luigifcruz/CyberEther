#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace Jetstream::Sakura {

struct Hud : public Component {
    using Child = std::function<void(const Context&)>;

    enum class Anchor {
        TopLeft,
        TopRight,
        BottomLeft,
        BottomRight,
        Center,
    };

    struct Config {
        std::string id;
        Anchor anchor = Anchor::TopLeft;
        F32 padding = 12.0f;
        bool clickable = false;
        std::optional<Extent2D<F32>> size;
        std::optional<Extent2D<F32>> windowPadding;
        std::optional<F32> borderSize;
        std::optional<F32> rounding;
        std::string backgroundColorKey = "background";
        F32 backgroundAlpha = 1.0f;
        std::string borderColorKey = "border";
        F32 borderAlpha = -1.0f;
        std::function<void()> onClick;
    };

    Hud();
    ~Hud();

    Hud(Hud&&) noexcept;
    Hud& operator=(Hud&&) noexcept;

    Hud(const Hud&) = delete;
    Hud& operator=(const Hud&) = delete;

    bool update(Config config);
    void render(const Context& ctx, Child child) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
