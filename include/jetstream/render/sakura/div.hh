#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct Div : public Component {
    using Child = std::function<void(const Context&)>;

    struct Config {
        std::string id;
        Extent2D<F32> size = {0.0f, 0.0f};
        F32 padding = 0.0f;
        F32 rounding = 0.0f;
        bool border = true;
        bool selected = false;
        bool scrollbar = true;
        bool mouseScroll = true;
        bool inputs = true;
        std::string colorKey = "card";
        std::string hoveredColorKey = "header_hovered";
        std::string selectedColorKey = "header_hovered";
        std::string borderColorKey = "border";
        std::function<void()> onClick;
        std::function<void()> onDoubleClick;
    };

    Div();
    ~Div();

    Div(Div&&) noexcept;
    Div& operator=(Div&&) noexcept;

    Div(const Div&) = delete;
    Div& operator=(const Div&) = delete;

    bool update(Config config);
    void render(const Context& ctx, Child child) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
