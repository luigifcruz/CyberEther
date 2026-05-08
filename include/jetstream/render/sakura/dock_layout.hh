#pragma once

#include <jetstream/types.hh>

#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

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

}  // namespace Jetstream::Sakura
