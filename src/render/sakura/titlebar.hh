#ifndef JETSTREAM_RENDER_SAKURA_TITLEBAR_HH
#define JETSTREAM_RENDER_SAKURA_TITLEBAR_HH

#include <jetstream/render/base/window.hh>
#include <jetstream/render/types.hh>
#include <jetstream/types.hh>

namespace Jetstream::Sakura {

struct Titlebar {
    static constexpr F32 MinimumHeight = 38.0f;

    struct Chrome {
        F32 leadingInset = 0.0f;
        F32 height = 0.0f;
        F32 controlsCenter = 0.0f;

        bool operator==(const Chrome&) const = default;
    };

    static Chrome Resolve(const Render::Window* render) {
        const Render::TitlebarChrome chrome = render ? render->titlebarChrome() : Render::TitlebarChrome{};
        if (!chrome.integrated) {
            return {};
        }
        return {.leadingInset = chrome.leadingInset, .height = chrome.height, .controlsCenter = chrome.controlsCenter};
    }

    static void Press(Render::Window* render, U8 clicks) {
        if (!render) {
            return;
        }
        if (clicks >= 2) {
            render->zoomWindow();
        } else {
            render->beginWindowDrag();
        }
    }
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_TITLEBAR_HH
