#ifndef JETSTREAM_RENDER_SAKURA_DETAIL_DOCK_DRAG_HH
#define JETSTREAM_RENDER_SAKURA_DETAIL_DOCK_DRAG_HH

#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace Jetstream::Sakura {

struct DockTarget {
    Rect preview;
    std::function<void(const std::string& pane)> accept;
    const void* owner = nullptr;
};

struct DockDrag {
    bool active = false;
    std::string pane;
    std::string group;
    std::string title;
    std::string icon;
    const void* source = nullptr;
    std::optional<DockTarget> target;
    std::optional<DockTarget> pending;
    std::shared_ptr<bool> alive = std::make_shared<bool>(true);

    void begin() {
        pending.reset();
    }

    void commit() {
        target = pending;
    }

    void offer(DockTarget next) {
        pending = std::move(next);
    }

    void reset() {
        active = false;
        pane.clear();
        group.clear();
        title.clear();
        icon.clear();
        source = nullptr;
        target.reset();
        pending.reset();
    }
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_DETAIL_DOCK_DRAG_HH
