#ifndef JETSTREAM_RENDER_SAKURA_DOCKSPACE_WINDOW_HH
#define JETSTREAM_RENDER_SAKURA_DOCKSPACE_WINDOW_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct DockspaceWindow {
    using Child = std::function<void(const Context&)>;

    struct DockItem {
        std::string key;
        U64 order = 0;
    };

    struct DockLayout {
        enum class Direction {
            Left,
            Right,
            Up,
            Down,
        };

        std::optional<Direction> direction;
        std::optional<F32> ratio;
        std::optional<std::vector<DockItem>> items;
        std::optional<std::vector<DockLayout>> children;
    };

    struct DockableWindow {
        std::string key;
        std::string label;
    };

    struct Config {
        std::string id;
        std::string title;
        Extent2D<F32> position = {0.0f, 0.0f};
        Extent2D<F32> size = {500.0f, 300.0f};

        std::optional<U64> parentDockId;
        bool dockIntoParent = false;

        bool restoreLayout = false;
        std::optional<DockLayout> layout;
        std::vector<DockableWindow> dockables;

        std::function<void(Extent2D<F32> position, Extent2D<F32> size)> onGeometry;
        std::function<void(std::optional<DockLayout>)> onLayout;
        std::function<void()> onClose;
    };

    DockspaceWindow();
    ~DockspaceWindow();

    DockspaceWindow(DockspaceWindow&&) noexcept;
    DockspaceWindow& operator=(DockspaceWindow&&) noexcept;

    DockspaceWindow(const DockspaceWindow&) = delete;
    DockspaceWindow& operator=(const DockspaceWindow&) = delete;

    bool update(Config config);
    void render(const Context& ctx, Child emptyContent = nullptr);

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_DOCKSPACE_WINDOW_HH
