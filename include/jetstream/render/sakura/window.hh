#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace Jetstream::Sakura {

struct Window : public Component {
    using Child = std::function<void(const Context&)>;

    struct Config {
        std::string id;
        std::string title;
        Extent2D<F32> size = {500.0f, 400.0f};
        std::optional<U64> dockId;
        std::optional<Extent2D<F32>> padding;
        std::function<void()> onOpen;
        std::function<void()> onClose;
    };

    Window();
    ~Window();

    Window(Window&&) noexcept;
    Window& operator=(Window&&) noexcept;

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    bool update(Config config);
    void render(const Context& ctx, Child content);

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
