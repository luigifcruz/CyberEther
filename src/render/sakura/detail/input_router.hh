#ifndef JETSTREAM_RENDER_SAKURA_DETAIL_INPUT_ROUTER_HH
#define JETSTREAM_RENDER_SAKURA_DETAIL_INPUT_ROUTER_HH

#include "../input.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace Jetstream::Sakura {

struct InputRouter {
    struct Output {
        std::vector<FrameEvent> events;
        bool hovered = false;
        bool active = false;
        bool middle = false;
        bool outsideClick = false;
    };

    Output route(const InputFrame& input, const Rect& screen);

    void reset() {
        source = nullptr;
        generation = 0;
        sampled = false;
        hovered = false;
        pointer.reset();
        modifiers = {};
        position = {0.0f, 0.0f};
        buttons.fill(false);
        keys.clear();
    }

 private:
    const void* source = nullptr;
    U64 generation = 0;
    U64 routed = 0;
    bool sampled = false;
    bool hovered = false;
    std::optional<Extent2D<F32>> pointer;
    KeyModifiers modifiers;
    Extent2D<F32> position = {0.0f, 0.0f};
    std::array<bool, 3> buttons{};
    std::vector<KeyCode> keys;

    static bool valid(const Extent2D<F32>& point) {
        return std::isfinite(point.x) && std::isfinite(point.y);
    }

    bool captured() const {
        return std::any_of(buttons.begin(), buttons.end(), [](bool held) { return held; });
    }

    bool track(const KeyEvent& key) {
        const auto it = std::find(keys.begin(), keys.end(), key.key);
        if (key.type == KeyEventType::Press) {
            if (it == keys.end()) keys.push_back(key.key);
            return true;
        }
        if (it == keys.end()) return false;
        keys.erase(it);
        return true;
    }

    void cancel(Output& output, bool explicitEvent = false);

    Output& state(Output& output) const {
        output.hovered = hovered;
        output.active = buttons[0] || buttons[1];
        output.middle = buttons[2];
        return output;
    }
};

}  // namespace Jetstream::Sakura

#endif
