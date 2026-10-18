#ifndef JETSTREAM_TESTS_CORE_SAKURA_LEGACY_SURFACE_INPUT_HH
#define JETSTREAM_TESTS_CORE_SAKURA_LEGACY_SURFACE_INPUT_HH

#include <jetstream/render/sakura/components/code_editor.hh>
#include <jetstream/render/sakura/components/controls.hh>
#include <jetstream/render/sakura/components/scroller.hh>
#include <jetstream/render/sakura/components/text_grid.hh>
#include <jetstream/render/sakura/components/text_view.hh>

#include "component_harness.hh"
#include "harness.hh"
#include "render/sakura/detail/access.hh"
#include "render/surface_input.hh"

namespace LegacyInputTest {

using namespace Jetstream;

struct SurfaceView {
    struct Config {
        std::string id;
        U64 texture = 0;
        std::shared_ptr<const Render::Texture> textureSource;
        Extent2D<F32> size = {0.0f, 0.0f};
        std::function<void(InputEvent)> onInput;
    };

    Config config;
    mutable detail::SurfaceInputState input;

    void update(Config next) {
        if (config.id != next.id || config.textureSource != next.textureSource || !next.onInput) {
            input.reset(config.onInput);
        }
        config = std::move(next);
    }

    void render(const Sakura::Context&) const {
        if (!config.onInput || config.size.x <= 0.0f || config.size.y <= 0.0f) {
            input.reset(config.onInput);
            return;
        }
        const auto origin = ImGui::GetCursorScreenPos();
        const ImVec2 size(config.size.x, config.size.y);
        ImGui::InvisibleButton(config.id.c_str(), size,
                               ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
        detail::ForwardSurfaceInputEvents(origin, size, input, config.onInput);
    }
};

inline bool SendRetained(Sakura::Component& root, const InputEvent& input) {
    const auto* raw = std::get_if<MouseEvent>(&input);
    if (!raw) {
        if (const auto* key = std::get_if<KeyEvent>(&input)) return Sakura::Access::key(root, *key);
        return false;
    }
    auto event = *raw;
    event.position = {event.position.x * 256.0f, event.position.y * 256.0f};
    if (event.type == MouseEventType::Release && event.cancel) event.type = MouseEventType::Cancel;
    return Sakura::Access::mouseEvent(root, event);
}

struct CancellationSink : Sakura::Component {
    U64 cancellations = 0;
    bool mouseEvent(const MouseEvent& event) override {
        if (event.type == MouseEventType::Cancel) {
            ++cancellations;
            return true;
        }
        return false;
    }
};

struct RetainedInputRoot : Sakura::Component {
    Sakura::Component& widget;
    CancellationSink sink;

    explicit RetainedInputRoot(Sakura::Component& widget) : widget(widget) {
        setClipsChildren(true);
        addChild(widget);
        addChild(sink);
    }

    void layout(const Sakura::Context& ctx) override {
        layoutChild(ctx, widget, {20, 20, 200, 200});
        layoutChild(ctx, sink, {0, 0, 1, 1});
    }
};

struct RetainedSurface {
    struct Texture final : Render::Texture {
        Texture() : Render::Texture(Config{.size = {256, 256}}) {}
        Result create() override { return Result::SUCCESS; }
        Result destroy() override { return Result::SUCCESS; }
        uint64_t raw() const override { return 1; }
    };

    SakuraTest::HeadlessUi ui{1.0f, {500, 500}};
    Sakura::Component& root;
    std::vector<InputEvent> events;
    std::vector<bool> consumed;
    SurfaceView::Config config;
    SurfaceView view;

    explicit RetainedSurface(Sakura::Component& root) : root(root) {
        config.id = "retained-input";
        config.texture = 1;
        config.textureSource = std::make_shared<Texture>();
        config.size = {256, 256};
        config.onInput = [this](const InputEvent& event) {
            events.push_back(event);
            consumed.push_back(SendRetained(this->root, event));
        };
        view.update(config);
    }

    void layout() {
        auto ctx = ui.sakura();
        ctx.framebufferSize = {256, 256};
        ctx.windowFocused = true;
        Sakura::Access::layout(root, ctx, {0, 0, 256, 256});
    }

    void frame(ImVec2 point, bool down) {
        ui.setMouse(point, down);
        ui.frame([&] {
            layout();
            ImGui::SetCursorScreenPos({0, 0});
            view.render(ui.sakura());
        });
    }

    void cancel(int boundary) {
        events.clear();
        consumed.clear();
        if (boundary == 0) {
            ImGui::GetIO().AddMouseButtonEvent(1, true);
            frame({350, 350}, true);
        } else if (boundary == 1) {
            ui.frame([] {});
        } else if (boundary == 2 || boundary == 3) {
            if (boundary == 2) config.id += "-replacement";
            else config.textureSource = std::make_shared<Texture>();
            view.update(config);
        } else {
            SendRetained(root, MouseEvent{.type = MouseEventType::Cancel, .position = {-1, -1}});
        }
    }
};

}  // namespace LegacyInputTest

#endif  // JETSTREAM_TESTS_CORE_SAKURA_LEGACY_SURFACE_INPUT_HH
