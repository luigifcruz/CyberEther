#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct Overlay : public Component {
    using Child = std::function<void(const Context&)>;

    enum class Anchor {
        TopLeft,
        TopCenter,
        TopRight,
        BottomLeft,
        BottomRight,
        BottomCenter,
        Center,
    };

    struct Config {
        std::string id;
        Extent2D<F32> size = {0.0f, 0.0f};
        Anchor anchor = Anchor::Center;
        Extent2D<F32> offset = {0.0f, 0.0f};
        bool inputs = false;
    };

    Overlay();
    ~Overlay();

    Overlay(Overlay&&) noexcept;
    Overlay& operator=(Overlay&&) noexcept;

    Overlay(const Overlay&) = delete;
    Overlay& operator=(const Overlay&) = delete;

    bool update(Config config);
    void render(const Context& ctx, Child child) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
