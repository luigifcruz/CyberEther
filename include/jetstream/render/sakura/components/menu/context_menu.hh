#ifndef JETSTREAM_RENDER_SAKURA_MENU_CONTEXT_MENU_HH
#define JETSTREAM_RENDER_SAKURA_MENU_CONTEXT_MENU_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace Jetstream::Sakura {

struct ContextMenu {
    using Child = std::function<void(const Context&)>;

    enum class Position {
        Mouse,
        ViewportCenter,
    };

    struct Config {
        std::string id;
        std::optional<Extent2D<F32>> size;
        std::optional<Extent2D<F32>> padding;
        std::optional<F32> rounding;
        std::optional<F32> borderSize;
        Position position = Position::Mouse;
        std::function<void()> onClose;
    };

    ContextMenu();
    ~ContextMenu();

    ContextMenu(ContextMenu&&) noexcept;
    ContextMenu& operator=(ContextMenu&&) noexcept;

    ContextMenu(const ContextMenu&) = delete;
    ContextMenu& operator=(const ContextMenu&) = delete;

    bool update(Config config);
    void render(const Context& ctx, Child child);

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_MENU_CONTEXT_MENU_HH
