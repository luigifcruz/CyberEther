#include <catch2/catch_test_macros.hpp>

#include "fonts.hh"
#include "harness.hh"
#include "jetstream/render/tools/icons_fa.hh"
#include "jetstream/viewport/adapters/generic.hh"
#include "component_harness.hh"

#include "compositor/default/presenters/file_picker.hh"
#include "compositor/default/presenters/hud/welcome.hh"
#include "compositor/default/presenters/titlebar.hh"
#include "compositor/default/presenters/workbench.hh"
#include "compositor/default/presenters/modal/container.hh"
#include "compositor/default/views/components/callout.hh"
#include "compositor/default/views/components/hint_overlay.hh"
#include "compositor/default/views/components/tooltip.hh"
#include "compositor/default/views/file_picker.hh"
#include "compositor/default/views/flowgraph/editor/picker.hh"
#include "compositor/default/views/hud/welcome.hh"
#include "compositor/default/views/titlebar.hh"
#include "compositor/default/views/workbench.hh"
#include "scene.hh"

#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

using namespace Jetstream;
using namespace SakuraTest;

namespace {


constexpr Jetstream::Rect kWide = {0.0f, 0.0f, 1600.0f, 1200.0f};

std::vector<Sakura::Button*> Buttons(Sakura::Component& root) {
    std::vector<Sakura::Button*> buttons;
    for (auto* button : Find<Sakura::Button>(root)) {
        if (!dynamic_cast<Sakura::NavigationItem*>(button)) {
            buttons.push_back(button);
        }
    }
    return buttons;
}

Extent2D<F32> Center(const Sakura::Component& component) {
    const Jetstream::Rect frame = Peek::Frame(component);
    return {frame.x + frame.width * 0.5f, frame.y + frame.height * 0.5f};
}

template<typename HostType>
bool Press(HostType& host, const Sakura::Component& target) {
    const auto point = Center(target);
    return host.click(point.x, point.y);
}

template<typename HostType>
bool DoublePress(HostType& host, const Sakura::Component& target) {
    const auto point = Center(target);
    const bool pressed = host.send(Mouse(MouseEventType::Click, point.x, point.y, MouseButton::Left, 2));
    host.send(Mouse(MouseEventType::Release, point.x, point.y));
    return pressed;
}

std::vector<Jetstream::Rect> PopupFrames(LayerHost& host) {
    std::vector<Jetstream::Rect> frames;
    for (auto* child : Peek::Children(host.popups)) {
        if (child && child->visible()) {
            frames.push_back(Peek::Frame(*child));
        }
    }
    return frames;
}

struct Stage {
    LayerSlot slot;
    LayerHost host;

    Stage(Sakura::Component& content, Jetstream::Rect rect) {
        slot.hold(content, rect);
        host.setLayers({&slot});
    }

    void place(const Sakura::Context& ctx, Jetstream::Rect bounds = kWide) {
        host.place(ctx, bounds);
    }
};

struct Mailbox {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    std::vector<Mail> mails;

    Mailbox() {
        callbacks.enqueueMail = [this](Mail&& mail) {
            mails.push_back(std::move(mail));
        };
    }

    PresenterContext context() const {
        return {state, callbacks};
    }

    template<typename T>
    const T* last() const {
        for (auto it = mails.rbegin(); it != mails.rend(); ++it) {
            if (const auto* mail = std::get_if<T>(&*it)) {
                return mail;
            }
        }
        return nullptr;
    }

    template<typename T>
    U64 count() const {
        U64 total = 0;
        for (const auto& mail : mails) {
            total += std::holds_alternative<T>(mail) ? 1 : 0;
        }
        return total;
    }
};

template<typename Update>
bool Settles(Frames& frames, Update update) {
    if (!frames.settle(update)) {
        return false;
    }
    for (U64 frame = 0; frame < 3; ++frame) {
        if (!frames.run(update).clean()) {
            return false;
        }
    }
    return true;
}

F32 EntryY(const Jetstream::Rect& popup, U64 entries, U64 separators) {
    return popup.y + 8.0f + 23.0f * static_cast<F32>(entries) + 8.0f * static_cast<F32>(separators) + 11.5f;
}

struct TitleRig {
    Scene scene{{0.0f, 0.0f, 1200.0f, 800.0f}};
    WorkspaceShellView shell;
    Probe a;
    Probe b;
    LayerSlot slot;
    LayerHost host;
    TitlebarView::Config config;
    std::vector<TitlebarView::Action> actions;
    std::vector<std::pair<TitlebarView::TabAction, std::string>> tabActions;
    std::vector<U8> presses;

    TitleRig() {
        config = {
            .id = "titlebar",
            .remoteSupported = true,
            .focusedFlowgraph = std::string("a"),
            .onAction = [this](TitlebarView::Action action) {
                actions.push_back(action);
            },
            .onTabAction = [this](TitlebarView::TabAction action, const std::string& flowgraph) {
                tabActions.emplace_back(action, flowgraph);
            },
            .onBarPress = [this](U8 clicks) {
                presses.push_back(clicks);
            },
        };
        update();
        slot.hold(shell.component(), bounds());
        host.setLayers({&slot});
        place();
    }

    static Jetstream::Rect bounds() {
        return {0.0f, 0.0f, 1200.0f, 800.0f};
    }

    void update() {
        shell.updateTitlebar(config);
        shell.updatePanes({
            {.id = "a", .title = "Alpha", .group = "main", .icon = ICON_FA_DIAGRAM_PROJECT, .content = &a},
            {.id = "b", .title = "Beta", .group = "main", .icon = ICON_FA_DIAGRAM_PROJECT, .content = &b},
        });
    }

    void place() {
        host.place(scene.ctx, bounds());
    }

    Jetstream::Rect popup() {
        Jetstream::Rect first = {};
        for (const auto& frame : PopupFrames(host)) {
            if (first.width <= 0.0f || frame.x < first.x) {
                first = frame;
            }
        }
        return first;
    }

    void open(Sakura::Component& button) {
        Press(host, button);
        place();
    }

    void choose(Sakura::Component& button, U64 entries, U64 separators) {
        open(button);
        const auto menu = popup();
        REQUIRE(menu.width > 0.0f);
        host.click(menu.x + 20.0f, EntryY(menu, entries, separators));
        place();
        if (!PopupFrames(host).empty()) {
            host.click(600.0f, 700.0f);
            place();
        }
    }

    Extent2D<F32> pill(U64 index) {
        const Jetstream::Rect leading = Peek::Frame(shell.bar().leading());
        return {leading.right() + 6.0f + 2.0f + 244.0f * static_cast<F32>(index) + 4.0f, 19.0f};
    }
};

Sakura::Component* Indicator(TitlebarView& bar, const char* icon) {
    for (auto* child : Peek::Children(bar.end())) {
        for (auto* part : Peek::Children(*child)) {
            if (auto* text = dynamic_cast<Sakura::Text*>(part); text && text->config().str == icon) {
                return child->visible() ? child : nullptr;
            }
        }
    }
    return nullptr;
}

Sakura::Component& Help(TitlebarView& bar) {
    auto* help = Indicator(bar, ICON_FA_CIRCLE_QUESTION);
    REQUIRE(help != nullptr);
    return *help;
}

std::vector<FlowgraphBlockPicker::BlockOption> Catalog() {
    return {
        {.type = "fft", .title = "FFT", .summary = "Fast Fourier transform", .category = "DSP",
         .devices = {{.device = DeviceType::CPU}, {.device = DeviceType::Metal}}},
        {.type = "filter", .title = "Filter", .summary = "FIR taps", .category = "DSP",
         .devices = {{.device = DeviceType::CPU}}},
        {.type = "file-reader", .title = "File Reader", .summary = "Reads samples from disk", .category = "IO",
         .devices = {{.device = DeviceType::CPU}}},
        {.type = "custom", .title = "Custom", .summary = "Fits spectral data", .category = "Extra",
         .devices = {{.device = DeviceType::CPU}}},
    };
}

struct PickerRig {
    struct Created {
        std::string type;
        Extent2D<F32> position;
        DeviceType device;
        RuntimeType runtime = RuntimeType::NONE;
        ProviderType provider;
    };

    static constexpr Jetstream::Rect kBounds = {0.0f, 0.0f, 1200.0f, 900.0f};

    Scene scene{kBounds};
    FlowgraphBlockPicker picker;
    Sakura::Overlay overlay;
    LayerSlot slot;
    LayerHost host;
    std::vector<Created> created;
    U64 opens = 0;

    PickerRig() {
        slot.hold(overlay, kBounds);
        host.setLayers({&slot});
    }

    FlowgraphBlockPicker::Config config() {
        return {
            .id = "picker",
            .onOpenRequest = [this] {
                ++opens;
                picker.open(Catalog(), FlowgraphBlockPicker::Mode::Pill, {40.0f, 80.0f});
            },
            .onCreateBlock = [this](const std::string& type, Extent2D<F32> position, DeviceType device,
                                    RuntimeType runtime, ProviderType provider) {
                created.push_back({type, position, device, runtime, provider});
            },
        };
    }

    void refresh(std::optional<Extent2D<F32>> mouse = std::nullopt) {
        picker.update(config());
        const Extent2D<F32> size = {FlowgraphBlockPicker::PillWidth, FlowgraphBlockPicker::PillHeight};
        const Extent2D<F32> offset = {-FlowgraphBlockPicker::PillMargin, FlowgraphBlockPicker::PillMargin};
        overlay.update({.items = {
            {.child = &picker.pillComponent(), .anchor = Sakura::Anchor::TopRight, .offset = offset, .size = size},
            {.child = &picker.popupComponent(), .anchor = Sakura::Anchor::TopRight, .offset = offset, .size = size},
        }});
        auto ctx = scene.ctx;
        ctx.mouse = mouse;
        host.place(ctx, kBounds);
    }

    std::vector<Sakura::Button*> actions() {
        return Buttons(host.popups);
    }

    void click(F32 x, F32 y, U8 clicks = 1) {
        host.send(Mouse(MouseEventType::Click, x, y, MouseButton::Left, clicks));
        host.send(Mouse(MouseEventType::Release, x, y));
    }

    void openFromPill() {
        refresh();
        click(1068.0f, 29.0f);
        refresh();
    }
};

struct KeyedPickerRig {
    MonoFontWindow window;
    FlowgraphBlockPicker picker;
    Sakura::Overlay overlay;
    LayerSlot slot;
    LayerHost host;
    Frames frames;
    std::vector<PickerRig::Created> created;
    bool handled = false;

    KeyedPickerRig() : frames(host, PickerRig::kBounds, MakeContext({1200.0f, 900.0f})) {
        frames.ctx.render = &window;
        slot.hold(overlay, PickerRig::kBounds);
        host.setLayers({&slot});
    }

    void update() {
        picker.update({
            .id = "picker",
            .onCreateBlock = [this](const std::string& type, Extent2D<F32> position, DeviceType device,
                                    RuntimeType runtime, ProviderType provider) {
                created.push_back({type, position, device, runtime, provider});
            },
        });
        const Extent2D<F32> size = {FlowgraphBlockPicker::PillWidth, FlowgraphBlockPicker::PillHeight};
        const Extent2D<F32> offset = {-FlowgraphBlockPicker::PillMargin, FlowgraphBlockPicker::PillMargin};
        overlay.update({.items = {
            {.child = &picker.pillComponent(), .anchor = Sakura::Anchor::TopRight, .offset = offset, .size = size},
            {.child = &picker.popupComponent(), .anchor = Sakura::Anchor::TopRight, .offset = offset, .size = size},
        }});
    }

    void open(FlowgraphBlockPicker::Mode mode, Extent2D<F32> position = {7.0f, 9.0f}) {
        picker.open(Catalog(), mode, position);
        frame();
        frame();
    }

    FrameStats frame() {
        FrameStats stats;
        window.frame([&] {
            stats = frames.run([&] {
                update();
            });
        });
        return stats;
    }

    FrameStats press(KeyCode key) {
        handled = Tap(host, key);
        return frame();
    }

    void type(const char* text) {
        Type(host, text);
        frame();
        frame();
    }

    bool settles() {
        bool settled = false;
        window.frame([&] {
            settled = Settles(frames, [&] {
                update();
            });
        });
        return settled;
    }
};

std::vector<FlowgraphBlockPicker::BlockOption> RoutedCatalog() {
    return {{
        .type = "fft",
        .title = "FFT",
        .summary = "Fast Fourier transform",
        .category = "DSP",
        .devices = {
            {.device = DeviceType::CPU, .runtime = RuntimeType::NATIVE, .provider = "generic"},
            {.device = DeviceType::Metal, .runtime = RuntimeType::MLIR, .provider = "iree"},
        },
        .device = DeviceType::Metal,
        .runtime = RuntimeType::PYTHON,
        .provider = "torch",
    }};
}

class ChromeWindow final : public SakuraTest::FakeWindow {
 public:
    ChromeWindow() {
        viewport.chrome = Render::TitlebarChrome{.integrated = true, .leadingInset = 78.0f, .height = 52.0f};
    }
};

bool Checked(LayerHost& host, const Jetstream::Rect& menu, F32 y) {
    for (auto* line : Find<Sakura::Polyline>(host.popups)) {
        for (const auto& instance : line->config().instances) {
            if (instance.filled || instance.points.size() != 3) {
                continue;
            }
            const auto& tip = instance.points[1];
            if (menu.contains(tip.x, tip.y) && std::abs(tip.y - y) <= 11.5f) {
                return true;
            }
        }
    }
    return false;
}

std::string Flag(bool value) {
    return value ? "true" : "false";
}

template<typename HostType>
void Choose(HostType& host, const Sakura::Context& ctx, Jetstream::Rect bounds, Sakura::Combo& combo, U64 option) {
    const Jetstream::Rect frame = Peek::Frame(combo);
    Press(host, combo);
    host.place(ctx, bounds);
    host.click(frame.x + frame.width * 0.5f, frame.bottom() + 8.0f + 23.0f * static_cast<F32>(option) + 11.5f);
    host.place(ctx, bounds);
}

struct ModalRig {
    Scene scene{kWide};
    Sakura::ModalLayer modal;
    LayerHost host;

    ModalRig() : host({&modal}) {}

    void show(Sakura::ModalLayer::Config config) {
        modal.update(std::move(config));
        place();
    }

    void place() {
        host.place(scene.ctx, kWide);
    }

    Sakura::Button& close() {
        return *Find<Sakura::Button>(modal).back();
    }
};

struct SettingsRig : ModalRig {
    SettingsView view;
    SettingsView::Config config;
    std::vector<std::string> calls;

    SettingsRig() {
        const auto log = [this](std::string entry) {
            calls.push_back(std::move(entry));
        };
        config.general = {
            .themes = {"Dark", "Light"},
            .currentThemeKey = "Dark",
            .renderer = "Metal",
            .backgroundParticles = true,
            .onThemeChange = [log](const std::string& value) { log("theme:" + value); },
            .onInterfaceScaleChange = [log](F32 value) { log("scale:" + std::to_string(value)); },
            .onBackgroundParticlesChange = [log](bool value) { log("particles:" + Flag(value)); },
        };
        config.remote = {
            .broker = {.address = "127.0.0.1", .port = 8686},
            .onCodecChange = [log](Instance::Remote::CodecType value) {
                log("codec:" + std::to_string(static_cast<U32>(value)));
            },
            .onFramerateChange = [log](U32 value) { log("fps:" + std::to_string(value)); },
            .onEncoderChange = [log](Instance::Remote::EncoderType) { log("encoder"); },
            .onAutoApproveChange = [log](bool value) { log("approve:" + Flag(value)); },
        };
        config.runtime = {
            .pythonCandidates = {{.label = "Homebrew", .path = "/opt/homebrew/bin/python3"}},
            .onPythonPathChange = [log](const std::string& value) { log("python:" + value); },
            .onDependencyPolicyChange = [log](const std::string& value) { log("policy:" + value); },
            .onOpenPythonEnvironmentCache = [log] { log("cache"); },
            .onBrowsePythonPath = [log](const std::string& current, std::function<void(std::string)> onSelect) {
                log("browse:" + current);
                onSelect("/custom/python");
            },
        };
        config.registry = {
            .domains = {{.domain = "core", .blocks = {"FFT", "Filter"}}, {.domain = "dsp", .blocks = {"Window"}}},
            .plugins = {
                {.name = "Alpha", .version = "1.0", .status = "Loaded", .path = "/plugins/alpha.cep"},
                {.name = "Beta", .version = "-", .status = "Missing", .path = "/plugins/beta.cep"},
            },
            .onNavigate = [log](RegistryPage page, const std::string& plugin) {
                log("nav:" + std::to_string(static_cast<int>(page)) + ":" + plugin);
            },
            .onBrowse = [log](const std::string& current, std::function<void(std::string)> onSelect) {
                log("browse-plugin:" + current);
                onSelect("/plugins/gamma.cep");
            },
            .onRegister = [log](const std::string& path) { log("register:" + path); },
            .onRemovePlugin = [log](const std::string& path) { log("remove:" + path); },
            .onReloadPlugin = [log](const std::string& path) { log("reload:" + path); },
        };
        config.developer = {
            .timingEnabled = true,
            .logLevel = 2,
            .onTimingEnabledChange = [log](bool value) { log("timing:" + Flag(value)); },
            .onLogLevelChange = [log](int value) { log("log:" + std::to_string(value)); },
        };
        config.about = {
            .updateSupported = true,
            .updateUpToDate = true,
            .infoTables = {{.id = "display", .title = "Display", .rows = {{"Viewport", "GLFW"}, {"Scale", "2.0"}}}},
            .onCheckForUpdates = [log] { log("check"); },
            .onDownloadUpdate = [log] { log("download"); },
            .onApplyUpdate = [log] { log("apply"); },
            .onSendFeedback = [log] { log("feedback"); },
        };
        config.legal = {.onViewFullLicenses = [log] { log("licenses"); }};
        config.onSectionChange = [log](SettingsSection section) {
            log("section:" + std::to_string(static_cast<I32>(section)));
        };
    }

    void show(SettingsSection section) {
        config.section = section;
        view.update(config);
        ModalRig::show({.content = &view.component(), .size = Extent2D<F32>{1200.0f, 900.0f}});
    }

    std::vector<Sakura::Combo*> combos() {
        return Find<Sakura::Combo>(view.component());
    }

    void choose(U64 combo, U64 option) {
        Choose(host, scene.ctx, kWide, *combos().at(combo), option);
    }
};

FilePickerView::Config PickerState(Mailbox& mailbox, FilePickerMode mode, std::string directory) {
    auto& picker = mailbox.state.filePicker;
    picker.active = true;
    picker.generation = 7;
    picker.mode = mode;
    picker.root = "/srv/flows";
    picker.directory = std::move(directory);
    picker.extensions = {"yml"};
    picker.entries = {
        {.name = "sub", .path = picker.directory + "/sub", .directory = true},
        {.name = "a.yml", .path = picker.directory + "/a.yml"},
    };
    return FilePickerPresenter(mailbox.context()).build();
}

const Sakura::Palette& Shades() {
    static const Sakura::Palette palette = [] {
        const std::vector<std::string> keys = {
            "background", "border", "button", "button_active", "button_hovered", "button_outline", "button_text",
            "action_btn", "action_btn_active", "action_btn_hovered", "action_btn_outline", "action_btn_text",
            "destructive_btn", "destructive_btn_active", "destructive_btn_hovered", "destructive_btn_outline",
            "destructive_btn_text", "card", "cyber_blue", "error_red", "frame_bg_active", "frame_bg_hovered",
            "header", "header_active", "header_hovered", "modal_dim", "popup_bg", "scrollbar_bg", "scrollbar_grab",
            "separator", "success_green", "tab", "tab_hovered", "tab_selected", "text_disabled", "text_primary",
            "text_secondary", "text_selected_bg", "warning_yellow", "accent_color", "editor_background",
            "editor_text", "editor_cursor", "editor_selection", "editor_active_line", "editor_scrollbar_track",
            "editor_scrollbar_thumb", "welcome_icon_new", "welcome_icon_open", "welcome_icon_examples",
            "tile_bg", "tile_border", "tile_header_text", "chrome_button_hover", "pill_bg", "pill_hover_bg",
            "pill_active_bg", "pill_text", "welcome_card_bg", "welcome_card_hovered", "setting_card_bg",
            "setting_card_border", "toggle_on", "toggle_on_hovered", "toggle_off", "toggle_off_hovered", "toggle_knob",
        };
        Sakura::Palette result;
        for (U64 index = 0; index < keys.size(); ++index) {
            const F32 shade = static_cast<F32>(index + 1) / static_cast<F32>(keys.size() + 1);
            result[keys[index]] = {shade, 1.0f - shade, 0.5f, 1.0f};
        }
        result["transparent"] = {0.0f, 0.0f, 0.0f, 0.0f};
        return result;
    }();
    return palette;
}

Sakura::Context Shaded(Jetstream::Rect bounds) {
    auto ctx = MakeContext({bounds.width, bounds.height});
    ctx.palette = Shades();
    return ctx;
}

struct KeyedModalRig {
    MonoFontWindow window;
    Sakura::ModalLayer modal;
    LayerHost host;
    Frames frames;
    std::function<void()> update = [] {};

    KeyedModalRig() : host({&modal}), frames(host, kWide, Shaded(kWide)) {
        frames.ctx.render = &window;
    }

    FrameStats frame() {
        FrameStats stats;
        window.frame([&] {
            stats = frames.run(update);
        });
        return stats;
    }

    bool settles() {
        bool settled = false;
        window.frame([&] {
            settled = Settles(frames, update);
        });
        return settled;
    }

    FrameStats move(Extent2D<F32> point) {
        host.send(Mouse(MouseEventType::Move, point.x, point.y));
        return frame();
    }

    FrameStats click(Extent2D<F32> point, U8 clicks = 1) {
        host.send(Mouse(MouseEventType::Click, point.x, point.y, MouseButton::Left, clicks));
        host.send(Mouse(MouseEventType::Release, point.x, point.y));
        ClickFocus(host, point);
        return frame();
    }

    FrameStats press(KeyCode key) {
        Tap(host, key);
        return frame();
    }

    FrameStats type(const char* text) {
        Type(host, text);
        return frame();
    }
};

template<typename HostType, typename Step>
void CheckHover(HostType& host, Step step, const Sakura::Component& target, Extent2D<F32> away) {
    const auto inside = Center(target);
    host.send(Mouse(MouseEventType::Move, inside.x, inside.y));
    CHECK(RepaintOnly(step()));
    CHECK(step().clean());
    host.send(Mouse(MouseEventType::Move, inside.x + 1.0f, inside.y + 1.0f));
    CHECK(step().clean());
    host.send(Mouse(MouseEventType::Move, away.x, away.y));
    CHECK(RepaintOnly(step()));
    CHECK(step().clean());
}

}  // namespace

TEST_CASE("Title row places the wordmark after the inset, the pills after it, and the plus after the last pill",
          "[core][sakura][views]") {
    TitleRig rig;
    const auto wordmark = Peek::Frame(rig.shell.bar().leading());
    CHECK(wordmark.x == 6.0f);
    CHECK(wordmark.y == 6.0f);
    CHECK(wordmark.height == 28.0f);
    CHECK(wordmark.width > 40.0f);
    const F32 plusX = wordmark.right() + 6.0f + 2.0f * 240.0f + 4.0f + 4.0f + 6.0f;
    CHECK(Peek::Frame(rig.shell.bar().trailing()) == Jetstream::Rect{plusX, 6.0f, 28.0f, 28.0f});
    CHECK(rig.b.laidFrame == Jetstream::Rect{0.0f, 32.0f, 1200.0f, 768.0f});

    rig.config.leadingInset = 81.0f;
    rig.config.height = 52.0f;
    rig.update();
    rig.place();
    CHECK(Peek::Frame(rig.shell.bar().leading()).x == 87.0f);
    CHECK(Peek::Frame(rig.shell.bar().leading()).y == 13.0f);
    CHECK(Peek::Frame(rig.shell.bar().trailing()) == Jetstream::Rect{plusX + 81.0f, 13.0f, 28.0f, 28.0f});
    CHECK(rig.b.laidFrame == Jetstream::Rect{0.0f, 46.0f, 1200.0f, 754.0f});

    rig.config.height = 32.0f;
    rig.update();
    rig.place();
    CHECK(rig.shell.bar().height() == 32.0f);
    CHECK(rig.b.laidFrame.y == 26.0f);

    rig.config.height = 0.0f;
    rig.update();
    rig.place();
    CHECK(rig.shell.bar().height() == TitlebarView::MinimumHeight);
    CHECK(rig.b.laidFrame.y == 32.0f);
}

TEST_CASE("Clicking a pill activates its flowgraph and empty title row space reports presses",
          "[core][sakura][views]") {
    TitleRig rig;
    const auto first = rig.pill(0);
    rig.host.click(first.x, first.y);
    rig.place();
    CHECK(rig.a.layouts == 1);
    CHECK(rig.presses.empty());

    rig.host.send(Mouse(MouseEventType::Click, 900.0f, 19.0f, MouseButton::Left, 1));
    rig.host.send(Mouse(MouseEventType::Release, 900.0f, 19.0f));
    rig.host.send(Mouse(MouseEventType::Click, 900.0f, 19.0f, MouseButton::Left, 2));
    rig.host.send(Mouse(MouseEventType::Release, 900.0f, 19.0f));
    CHECK(rig.presses == std::vector<U8>{1, 2});
    CHECK(rig.actions.empty());
}

TEST_CASE("Wordmark menu entries emit their actions and the update entry follows the update state",
          "[core][sakura][views]") {
    using Action = TitlebarView::Action;
    TitleRig rig;
    auto& wordmark = rig.shell.bar().leading();

    rig.choose(wordmark, 0, 0);
    rig.choose(wordmark, 1, 0);
    rig.choose(wordmark, 2, 0);
    rig.choose(wordmark, 3, 0);
    rig.choose(wordmark, 6, 0);
    rig.choose(wordmark, 7, 1);
    CHECK(rig.actions == std::vector<Action>{Action::About, Action::Settings, Action::CheckForUpdates,
                                             Action::RemoteStreaming, Action::SendFeedback, Action::Quit});

    rig.actions.clear();
    rig.config.remoteSupported = false;
    rig.config.updateAvailable = true;
    rig.update();
    rig.choose(wordmark, 3, 0);
    rig.choose(wordmark, 2, 0);
    rig.config.updateAvailable = false;
    rig.config.updateReady = true;
    rig.update();
    rig.choose(wordmark, 2, 0);
    CHECK(rig.actions == std::vector<Action>{Action::OpenUpdateModal, Action::OpenUpdateModal});
}

TEST_CASE("Wordmark help submenu opens each link", "[core][sakura][views]") {
    using Action = TitlebarView::Action;
    TitleRig rig;
    auto& wordmark = rig.shell.bar().leading();
    for (U64 entry = 0; entry < 5; ++entry) {
        rig.open(wordmark);
        const auto menu = rig.popup();
        rig.host.click(menu.x + 20.0f, EntryY(menu, 5, 0));
        rig.place();
        Jetstream::Rect submenu = {};
        for (const auto& frame : PopupFrames(rig.host)) {
            if (frame.x > submenu.x) {
                submenu = frame;
            }
        }
        REQUIRE(submenu.x > menu.x);
        rig.host.click(submenu.x + 20.0f, EntryY(submenu, entry, 0));
        rig.place();
        CHECK(PopupFrames(rig.host).empty());
    }
    CHECK(rig.actions == std::vector<Action>{Action::GettingStarted, Action::Documentation, Action::OpenRepository,
                                             Action::ReportIssue, Action::OpenTwitter});
}

TEST_CASE("Wordmark developer submenu toggles timing, opens tools, and picks a log level",
          "[core][sakura][views]") {
    using Action = TitlebarView::Action;
    TitleRig rig;
    std::vector<I32> levels;
    rig.config.onLogLevel = [&](I32 level) {
        levels.push_back(level);
    };
    rig.update();
    auto& wordmark = rig.shell.bar().leading();
    const auto rightmost = [&] {
        Jetstream::Rect found = {};
        for (const auto& frame : PopupFrames(rig.host)) {
            if (frame.x > found.x) {
                found = frame;
            }
        }
        return found;
    };
    const auto developer = [&] {
        rig.open(wordmark);
        const auto menu = rig.popup();
        rig.host.click(menu.x + 20.0f, EntryY(menu, 4, 0));
        rig.place();
        const auto submenu = rightmost();
        REQUIRE(submenu.x > menu.x);
        return submenu;
    };
    for (const auto& [entry, separators] : std::vector<std::pair<U64, U64>>{{0, 0}, {1, 0}, {2, 1}, {4, 3}}) {
        const auto submenu = developer();
        rig.host.click(submenu.x + 20.0f, EntryY(submenu, entry, separators));
        rig.place();
        CHECK(PopupFrames(rig.host).empty());
    }
    CHECK(rig.actions == std::vector<Action>{Action::ToggleTiming, Action::ShowBenchmarks, Action::ReloadPlugins,
                                             Action::OpenDeveloperSettings});

    const auto submenu = developer();
    rig.host.click(submenu.x + 20.0f, EntryY(submenu, 3, 2));
    rig.place();
    const auto logLevels = rightmost();
    REQUIRE(logLevels.x > submenu.x);
    rig.host.click(logLevels.x + 20.0f, EntryY(logLevels, 3, 0));
    rig.place();
    CHECK(levels == std::vector<I32>{3});
}

TEST_CASE("Plus button opens a new tab without a menu", "[core][sakura][views]") {
    using Action = TitlebarView::Action;
    TitleRig rig;
    rig.open(rig.shell.bar().trailing());
    CHECK(PopupFrames(rig.host).empty());
    CHECK(rig.actions == std::vector<Action>{Action::NewTab});
}

TEST_CASE("Start tab shows the welcome screen in a fixed tile and activates on request",
          "[core][sakura][views]") {
    TitleRig rig;
    StartTabView start;
    U64 closes = 0;
    U64 opened = 0;
    const auto show = [&](std::optional<std::string> activate) {
        start.update({
            .welcome = {.version = "v1", .onNewFlowgraph = [&]() {
                ++opened;
            }},
            .onClose = [&]() {
                ++closes;
            },
        });
        rig.shell.updatePanes({
            {.id = "a", .title = "Alpha", .group = "main", .content = &rig.a},
            {.id = "b", .title = "Beta", .group = "main", .content = &rig.b},
            start.pane(),
        }, std::move(activate));
        rig.place();
    };

    const auto pane = start.pane();
    CHECK(pane.id == StartTabView::PaneId);
    CHECK(pane.title == "New Tab");
    REQUIRE(pane.content);

    show(std::nullopt);
    CHECK_FALSE(rig.a.visible());
    CHECK_FALSE(rig.b.visible());
    CHECK(pane.content->visible());

    bool welcome = false;
    std::vector<Sakura::Component*> pending = {pane.content};
    while (!pending.empty()) {
        auto* component = pending.back();
        pending.pop_back();
        if (auto* text = dynamic_cast<Sakura::Text*>(component)) {
            welcome = welcome || text->config().str == "CyberEther";
        }
        for (auto* child : Peek::Children(*component)) {
            pending.push_back(child);
        }
    }
    CHECK(welcome);

    rig.host.click(rig.pill(0).x, rig.pill(0).y);
    show(std::nullopt);
    CHECK(rig.a.visible());
    CHECK_FALSE(pane.content->visible());

    show(std::string(StartTabView::PaneId));
    CHECK_FALSE(rig.a.visible());
    CHECK(pane.content->visible());

    rig.host.click(rig.pill(2).x, rig.pill(2).y, MouseButton::Right);
    rig.place();
    CHECK(PopupFrames(rig.host).empty());
    CHECK(closes == 0);
    CHECK(opened == 0);
}

TEST_CASE("Right clicking a pill opens the tab menu at the cursor acting on that flowgraph",
          "[core][sakura][views]") {
    using TabAction = TitlebarView::TabAction;
    TitleRig rig;
    const auto second = rig.pill(1);
    const auto pick = [&](U64 entries, U64 separators) {
        rig.host.click(second.x, second.y, MouseButton::Right);
        rig.place();
        REQUIRE_FALSE(PopupFrames(rig.host).empty());
        const auto menu = rig.popup();
        CHECK(menu.x == second.x);
        CHECK(menu.y == second.y);
        rig.host.click(menu.x + 20.0f, EntryY(menu, entries, separators));
        rig.place();
        CHECK(PopupFrames(rig.host).empty());
    };
    for (U64 entry = 0; entry < 5; ++entry) {
        pick(entry, 0);
    }
    pick(5, 1);
    CHECK(rig.tabActions == std::vector<std::pair<TabAction, std::string>>{
        {TabAction::Save, "b"}, {TabAction::Rename, "b"}, {TabAction::Info, "b"},
        {TabAction::ToggleMetadata, "b"}, {TabAction::ToggleEnvironment, "b"}, {TabAction::Close, "b"}});
    CHECK(rig.presses.empty());

    rig.host.click(second.x, second.y, MouseButton::Right);
    rig.place();
    REQUIRE_FALSE(PopupFrames(rig.host).empty());
    rig.host.click(600.0f, 600.0f);
    rig.place();
    CHECK(PopupFrames(rig.host).empty());
    CHECK(rig.tabActions.size() == 6);
}

TEST_CASE("Titlebar presenter actions enqueue the matching mail", "[core][sakura][views]") {
    using Action = TitlebarView::Action;
    Mailbox mailbox;
    mailbox.state.update.supported = true;
    auto config = TitlebarPresenter(mailbox.context()).build();
    CHECK(config.leadingInset == 0.0f);
    CHECK(config.height == 0.0f);

    config.onAction(Action::About);
    REQUIRE(mailbox.last<MailOpenModal>() != nullptr);
    CHECK(mailbox.last<MailOpenModal>()->content == ModalContent::About);
    config.onAction(Action::Settings);
    REQUIRE(mailbox.last<MailOpenSettings>() != nullptr);
    CHECK(mailbox.last<MailOpenSettings>()->section == SettingsSection::General);
    config.onAction(Action::CheckForUpdates);
    CHECK(mailbox.last<MailOpenSettings>()->section == SettingsSection::About);
    CHECK(mailbox.count<MailCheckForUpdates>() == 1);
    config.onAction(Action::RemoteStreaming);
    CHECK(mailbox.last<MailOpenModal>()->content == ModalContent::RemoteStreaming);
    config.onAction(Action::SendFeedback);
    CHECK(mailbox.last<MailOpenModal>()->content == ModalContent::Feedback);
    config.onAction(Action::Documentation);
    REQUIRE(mailbox.last<MailOpenUrl>() != nullptr);
    CHECK(mailbox.last<MailOpenUrl>()->url == "https://cyberether.org/docs");
    config.onAction(Action::NewFlowgraph);
    config.onAction(Action::OpenFlowgraph);
    config.onAction(Action::Quit);
    CHECK(mailbox.count<MailNewFlowgraph>() == 1);
    CHECK(mailbox.count<MailOpenFlowgraph>() == 1);
    CHECK(mailbox.count<MailQuit>() == 1);

    config.onAction(Action::SaveFlowgraph);
    config.onAction(Action::CloseFlowgraph);
    CHECK(mailbox.count<MailNotify>() == 2);
    CHECK(mailbox.count<MailSaveFlowgraph>() == 0);

    mailbox.state.interface.focusedFlowgraph = "a";
    config = TitlebarPresenter(mailbox.context()).build();
    config.onAction(Action::SaveFlowgraph);
    REQUIRE(mailbox.last<MailSaveFlowgraph>() != nullptr);
    CHECK(mailbox.last<MailSaveFlowgraph>()->flowgraph == "a");
    config.onAction(Action::ShowFlowgraphInfo);
    CHECK(mailbox.last<MailOpenModal>()->content == ModalContent::FlowgraphInfo);
    CHECK(mailbox.last<MailOpenModal>()->flowgraph == std::optional<std::string>("a"));

    config.onBarPress(1);
    config.onBarPress(2);
}

TEST_CASE("Titlebar presenter close action targets the start tab while it is the focused pane",
          "[core][sakura][views]") {
    using Action = TitlebarView::Action;
    Mailbox mailbox;
    mailbox.state.interface.focusedFlowgraph = "a";
    mailbox.state.interface.startTab = true;
    mailbox.state.interface.startTabFocused = true;
    auto config = TitlebarPresenter(mailbox.context()).build();
    config.onAction(Action::CloseFlowgraph);
    CHECK(mailbox.count<MailCloseStartTab>() == 1);
    CHECK(mailbox.count<MailCloseFlowgraph>() == 0);

    mailbox.state.interface.startTabFocused = false;
    config = TitlebarPresenter(mailbox.context()).build();
    config.onAction(Action::CloseFlowgraph);
    REQUIRE(mailbox.last<MailCloseFlowgraph>() != nullptr);
    CHECK(mailbox.last<MailCloseFlowgraph>()->flowgraph == "a");
}

TEST_CASE("Titlebar presenter tab actions target the chosen flowgraph and focus it for its views",
          "[core][sakura][views]") {
    using TabAction = TitlebarView::TabAction;
    Mailbox mailbox;
    mailbox.state.interface.focusedFlowgraph = "a";
    mailbox.state.interface.flowgraphMetadataVisible = true;
    auto config = TitlebarPresenter(mailbox.context()).build();

    config.onTabAction(TabAction::Save, "b");
    REQUIRE(mailbox.last<MailSaveFlowgraph>() != nullptr);
    CHECK(mailbox.last<MailSaveFlowgraph>()->flowgraph == "b");
    config.onTabAction(TabAction::Rename, "b");
    CHECK(mailbox.last<MailOpenModal>()->content == ModalContent::FlowgraphInfo);
    CHECK(mailbox.last<MailOpenModal>()->flowgraph == std::optional<std::string>("b"));
    config.onTabAction(TabAction::Close, "b");
    REQUIRE(mailbox.last<MailCloseFlowgraph>() != nullptr);
    CHECK(mailbox.last<MailCloseFlowgraph>()->flowgraph == "b");

    mailbox.mails.clear();
    config.onTabAction(TabAction::ToggleMetadata, "b");
    REQUIRE(mailbox.mails.size() == 2);
    REQUIRE(std::holds_alternative<MailFocusFlowgraph>(mailbox.mails[0]));
    CHECK(std::get<MailFocusFlowgraph>(mailbox.mails[0]).flowgraph == "b");
    REQUIRE(std::holds_alternative<MailSetFlowgraphMetadataVisible>(mailbox.mails[1]));
    CHECK(std::get<MailSetFlowgraphMetadataVisible>(mailbox.mails[1]).value);

    mailbox.mails.clear();
    config.onTabAction(TabAction::ToggleMetadata, "a");
    REQUIRE(mailbox.mails.size() == 1);
    CHECK_FALSE(std::get<MailSetFlowgraphMetadataVisible>(mailbox.mails[0]).value);

    mailbox.mails.clear();
    config.onTabAction(TabAction::ToggleEnvironment, "a");
    REQUIRE(mailbox.mails.size() == 1);
    CHECK(std::get<MailSetFlowgraphEnvironmentVisible>(mailbox.mails[0]).value);
}

TEST_CASE("Titlebar presenter window presses are safe without a native titlebar", "[core][sakura][views]") {
    Mailbox mailbox;
    mailbox.state.system.render = std::make_shared<SakuraTest::FakeWindow>();
    const auto config = TitlebarPresenter(mailbox.context()).build();
    CHECK(config.leadingInset == 0.0f);
    CHECK(config.height == 0.0f);
    CHECK_NOTHROW(config.onBarPress(1));
    CHECK_NOTHROW(config.onBarPress(2));
    CHECK(mailbox.mails.empty());
}

TEST_CASE("Title row settles under identical updates and while a menu is open", "[core][sakura][views]") {
    TitleRig rig;
    Frames frames(rig.host, TitleRig::bounds(), rig.scene.ctx);
    const auto update = [&] {
        rig.update();
    };
    CHECK(Settles(frames, update));

    const auto second = rig.pill(1);
    rig.host.click(second.x, second.y, MouseButton::Right);
    CHECK(Settles(frames, update));
    rig.host.click(600.0f, 600.0f);
    CHECK(Settles(frames, update));

    Press(rig.host, rig.shell.bar().leading());
    CHECK(Settles(frames, update));
}

TEST_CASE("Command shortcuts emit their titlebar actions", "[core][sakura][views]") {
    using Action = TitlebarView::Action;
    std::vector<Action> actions;
    TitlebarView bar;
    bar.update({
        .id = "titlebar",
        .onAction = [&](Action action) {
            actions.push_back(action);
        },
    });
    Sakura::Shortcuts keys;
    keys.update({.bindings = bar.shortcuts()});
    Host host(keys);

    for (const auto key : {KeyCode::N, KeyCode::O, KeyCode::S, KeyCode::W, KeyCode::I, KeyCode::G, KeyCode::Comma}) {
        CHECK(Tap(host, key, {.control = true}));
    }
    CHECK(actions == std::vector<Action>{Action::NewFlowgraph, Action::OpenFlowgraph, Action::SaveFlowgraph,
                                         Action::CloseFlowgraph, Action::ShowFlowgraphInfo, Action::SendFeedback,
                                         Action::Settings});

    actions.clear();
    CHECK(Tap(host, KeyCode::N, {.super = true}));
    CHECK(Tap(host, KeyCode::Comma, {.super = true}));
    CHECK(actions == std::vector<Action>{Action::NewFlowgraph, Action::Settings});

    actions.clear();
    CHECK_FALSE(Tap(host, KeyCode::N));
    CHECK_FALSE(Tap(host, KeyCode::S, {.alt = true}));
    CHECK(actions.empty());
}

TEST_CASE("Titlebar presenter uses integrated window chrome and routes bar presses to drag and zoom",
          "[core][sakura][views]") {
    Mailbox mailbox;
    const auto window = std::make_shared<ChromeWindow>();
    mailbox.state.system.render = window;
    auto config = TitlebarPresenter(mailbox.context()).build();
    CHECK(config.leadingInset == 78.0f);
    CHECK(config.height == 52.0f);

    config.onBarPress(1);
    CHECK(window->viewport.drags == 1);
    CHECK(window->viewport.zooms == 0);
    config.onBarPress(2);
    config.onBarPress(3);
    CHECK(window->viewport.drags == 1);
    CHECK(window->viewport.zooms == 2);
    CHECK(mailbox.mails.empty());

    window->viewport.chrome->integrated = false;
    config = TitlebarPresenter(mailbox.context()).build();
    CHECK(config.leadingInset == 0.0f);
    CHECK(config.height == 0.0f);
    CHECK_NOTHROW(Sakura::Titlebar::Press(nullptr, 2));
}

TEST_CASE("Titlebar presenter opens the about settings but checks for updates only when a check can start",
          "[core][sakura][views]") {
    using Action = TitlebarView::Action;
    Mailbox mailbox;
    const auto check = [&] {
        mailbox.mails.clear();
        TitlebarPresenter(mailbox.context()).build().onAction(Action::CheckForUpdates);
        REQUIRE(mailbox.last<MailOpenSettings>() != nullptr);
        CHECK(mailbox.last<MailOpenSettings>()->section == SettingsSection::About);
        return mailbox.count<MailCheckForUpdates>();
    };
    CHECK(check() == 0);

    auto& update = mailbox.state.update;
    update.supported = true;
    CHECK(check() == 1);
    for (bool* busy : {&update.checking, &update.available, &update.downloading, &update.ready, &update.applying}) {
        *busy = true;
        CHECK(check() == 0);
        *busy = false;
    }
    CHECK(check() == 1);
}

TEST_CASE("Titlebar presenter negates timing, forwards the log level, opens a tab, and closes the focused flowgraph",
          "[core][sakura][views]") {
    using Action = TitlebarView::Action;
    Mailbox mailbox;
    mailbox.state.debug.timingEnabled = true;
    auto config = TitlebarPresenter(mailbox.context()).build();
    config.onAction(Action::ToggleTiming);
    REQUIRE(mailbox.last<MailSetDebugTimingEnabled>() != nullptr);
    CHECK_FALSE(mailbox.last<MailSetDebugTimingEnabled>()->value);

    mailbox.state.debug.timingEnabled = false;
    config = TitlebarPresenter(mailbox.context()).build();
    config.onAction(Action::ToggleTiming);
    CHECK(mailbox.last<MailSetDebugTimingEnabled>()->value);

    config.onLogLevel(3);
    REQUIRE(mailbox.last<MailSetDebugLogLevel>() != nullptr);
    CHECK(mailbox.last<MailSetDebugLogLevel>()->value == 3);

    config.onAction(Action::NewTab);
    CHECK(mailbox.count<MailOpenStartTab>() == 1);

    config.onAction(Action::CloseFlowgraph);
    CHECK(mailbox.count<MailNotify>() == 1);
    CHECK(mailbox.count<MailCloseFlowgraph>() == 0);

    mailbox.state.interface.focusedFlowgraph = "a";
    config = TitlebarPresenter(mailbox.context()).build();
    config.onAction(Action::CloseFlowgraph);
    REQUIRE(mailbox.last<MailCloseFlowgraph>() != nullptr);
    CHECK(mailbox.last<MailCloseFlowgraph>()->flowgraph == "a");
    CHECK(mailbox.count<MailNotify>() == 1);
}

TEST_CASE("Titlebar presenter tab info opens the flowgraph info and environment focuses an unfocused tab first",
          "[core][sakura][views]") {
    using TabAction = TitlebarView::TabAction;
    Mailbox mailbox;
    mailbox.state.interface.focusedFlowgraph = "a";
    mailbox.state.interface.flowgraphEnvironmentVisible = true;
    auto config = TitlebarPresenter(mailbox.context()).build();

    config.onTabAction(TabAction::Info, "b");
    REQUIRE(mailbox.mails.size() == 1);
    REQUIRE(mailbox.last<MailOpenModal>() != nullptr);
    CHECK(mailbox.last<MailOpenModal>()->content == ModalContent::FlowgraphInfo);
    CHECK(mailbox.last<MailOpenModal>()->flowgraph == std::optional<std::string>("b"));

    mailbox.mails.clear();
    config.onTabAction(TabAction::ToggleEnvironment, "b");
    REQUIRE(mailbox.mails.size() == 2);
    REQUIRE(std::holds_alternative<MailFocusFlowgraph>(mailbox.mails[0]));
    CHECK(std::get<MailFocusFlowgraph>(mailbox.mails[0]).flowgraph == "b");
    REQUIRE(std::holds_alternative<MailSetFlowgraphEnvironmentVisible>(mailbox.mails[1]));
    CHECK(std::get<MailSetFlowgraphEnvironmentVisible>(mailbox.mails[1]).value);

    mailbox.mails.clear();
    config.onTabAction(TabAction::ToggleEnvironment, "a");
    REQUIRE(mailbox.mails.size() == 1);
    CHECK_FALSE(std::get<MailSetFlowgraphEnvironmentVisible>(mailbox.mails[0]).value);
}

TEST_CASE("Menu checkmarks follow the focused tab views, the timing toggle, and the log level",
          "[core][sakura][views]") {
    TitleRig rig;
    rig.config.flowgraphMetadataVisible = true;
    rig.config.debugTimingEnabled = true;
    rig.config.debugLogLevel = 2;
    rig.update();
    rig.place();
    const auto tabMenu = [&](U64 index) {
        const auto pill = rig.pill(index);
        rig.host.click(pill.x, pill.y, MouseButton::Right);
        rig.place();
        return rig.popup();
    };
    const auto dismiss = [&] {
        rig.host.click(600.0f, 700.0f);
        rig.place();
        REQUIRE(PopupFrames(rig.host).empty());
    };
    const auto rightmost = [&] {
        Jetstream::Rect found = {};
        for (const auto& frame : PopupFrames(rig.host)) {
            if (frame.x > found.x) {
                found = frame;
            }
        }
        return found;
    };
    const auto views = [&](U64 index) {
        const auto menu = tabMenu(index);
        REQUIRE(menu.width > 0.0f);
        std::vector<bool> checked;
        for (U64 entry = 0; entry < 5; ++entry) {
            checked.push_back(Checked(rig.host, menu, EntryY(menu, entry, 0)));
        }
        checked.push_back(Checked(rig.host, menu, EntryY(menu, 5, 1)));
        dismiss();
        return checked;
    };
    const auto developer = [&] {
        rig.open(rig.shell.bar().leading());
        const auto menu = rig.popup();
        rig.host.click(menu.x + 20.0f, EntryY(menu, 4, 0));
        rig.place();
        const auto submenu = rightmost();
        REQUIRE(submenu.x > menu.x);
        std::vector<bool> checked = {
            Checked(rig.host, submenu, EntryY(submenu, 0, 0)),
            Checked(rig.host, submenu, EntryY(submenu, 1, 0)),
            Checked(rig.host, submenu, EntryY(submenu, 2, 1)),
            Checked(rig.host, submenu, EntryY(submenu, 4, 3)),
        };
        rig.host.click(submenu.x + 20.0f, EntryY(submenu, 3, 2));
        rig.place();
        const auto levels = rightmost();
        REQUIRE(levels.x > submenu.x);
        for (U64 entry = 0; entry < 5; ++entry) {
            checked.push_back(Checked(rig.host, levels, EntryY(levels, entry, 0)));
        }
        dismiss();
        return checked;
    };

    CHECK(views(0) == std::vector<bool>{false, false, false, true, false, false});
    CHECK(views(1) == std::vector<bool>(6, false));
    CHECK(developer() == std::vector<bool>{true, false, false, false, false, false, true, false, false});

    rig.config.flowgraphMetadataVisible = false;
    rig.config.flowgraphEnvironmentVisible = true;
    rig.config.debugTimingEnabled = false;
    rig.config.debugLogLevel = 4;
    rig.update();
    rig.place();
    CHECK(views(0) == std::vector<bool>{false, false, false, false, true, false});
    CHECK(views(1) == std::vector<bool>(6, false));
    CHECK(developer() == std::vector<bool>{false, false, false, false, false, false, false, false, true});
}

TEST_CASE("Toggling checked menu state while the menus are closed settles without rebuilding",
          "[core][sakura][views]") {
    TitleRig rig;
    Frames frames(rig.host, TitleRig::bounds(), rig.scene.ctx);
    const auto update = [&] {
        rig.update();
    };
    REQUIRE(Settles(frames, update));
    const std::vector<std::function<void()>> toggles = {
        [&] { rig.config.flowgraphMetadataVisible = !rig.config.flowgraphMetadataVisible; },
        [&] { rig.config.flowgraphEnvironmentVisible = !rig.config.flowgraphEnvironmentVisible; },
        [&] { rig.config.debugTimingEnabled = !rig.config.debugTimingEnabled; },
        [&] { rig.config.debugLogLevel = (rig.config.debugLogLevel + 1) % 5; },
    };
    for (U64 index = 0; index < toggles.size(); ++index) {
        CAPTURE(index);
        toggles[index]();
        const auto stats = frames.run(update);
        CHECK(Kept(stats));
        CHECK(Settles(frames, update));
    }

    rig.host.send(Mouse(MouseEventType::Move, 900.0f, 19.0f));
    CHECK(frames.run(update).clean());
    rig.host.send(Mouse(MouseEventType::Move, 600.0f, 400.0f));
    CHECK(frames.run(update).clean());
}

TEST_CASE("New tab opens the start tab, which takes over the welcome screen and closes through mail",
          "[core][sakura][views]") {
    Mailbox mailbox;
    const auto context = mailbox.context();
    TitlebarPresenter(context).build().onAction(TitlebarView::Action::NewTab);
    CHECK(mailbox.count<MailOpenStartTab>() == 1);
    CHECK(WelcomeHudPresenter(context).build().has_value());
    CHECK_FALSE(WelcomeHudPresenter(context).buildStartTab().has_value());
    CHECK_FALSE(WorkbenchPresenter(context).build().startTab.has_value());

    mailbox.state.interface.startTab = true;
    mailbox.state.interface.startTabRequests = 3;
    CHECK_FALSE(WelcomeHudPresenter(context).build().has_value());
    const auto welcome = WelcomeHudPresenter(context).buildStartTab();
    REQUIRE(welcome.has_value());
    mailbox.mails.clear();
    welcome->onNewFlowgraph();
    CHECK(mailbox.count<MailNewFlowgraph>() == 1);

    auto config = WorkbenchPresenter(context).build();
    CHECK_FALSE(config.welcomeHud.has_value());
    REQUIRE(config.startTab.has_value());
    CHECK(config.startTabRequests == 3);
    mailbox.mails.clear();
    config.startTab->welcome.onOpenFlowgraph();
    CHECK(mailbox.count<MailOpenFlowgraph>() == 1);
    config.startTab->onClose();
    CHECK(mailbox.count<MailCloseStartTab>() == 1);
}

TEST_CASE("Workbench enables particles only while the background is exposed", "[core][sakura][views]") {
    Mailbox mailbox;
    const auto context = mailbox.context();
    mailbox.state.interface.backgroundParticles = true;
    CHECK(WorkbenchPresenter(context).build().backgroundParticles);

    mailbox.state.interface.startTab = true;
    CHECK_FALSE(WorkbenchPresenter(context).build().backgroundParticles);

    mailbox.state.interface.filePending = true;
    CHECK(WorkbenchPresenter(context).build().backgroundParticles);

    mailbox.state.interface.backgroundParticles = false;
    CHECK_FALSE(WorkbenchPresenter(context).build().backgroundParticles);
}

TEST_CASE("Start tab pill close button calls its close callback and the shell settles as the tab opens and closes",
          "[core][sakura][views]") {
    TitleRig rig;
    StartTabView start;
    U64 closes = 0;
    bool open = false;
    std::optional<std::string> activate;
    const auto update = [&] {
        rig.shell.updateTitlebar(rig.config);
        std::vector<Sakura::Dock::Pane> panes = {
            {.id = "a", .title = "Alpha", .group = "main", .content = &rig.a},
            {.id = "b", .title = "Beta", .group = "main", .content = &rig.b},
        };
        if (open) {
            start.update({
                .welcome = {.version = "v1"},
                .onClose = [&]() {
                    ++closes;
                },
            });
            panes.push_back(start.pane());
        }
        rig.shell.updatePanes(std::move(panes), std::exchange(activate, std::nullopt));
    };
    Frames frames(rig.host, TitleRig::bounds(), rig.scene.ctx);
    CHECK(Settles(frames, update));

    open = true;
    activate = StartTabView::PaneId;
    CHECK(Settles(frames, update));
    REQUIRE(start.pane().content != nullptr);
    CHECK(start.pane().content->visible());
    CHECK_FALSE(rig.b.visible());

    const auto pill = rig.pill(2);
    rig.host.click(pill.x + 221.0f, pill.y);
    CHECK(closes == 1);

    open = false;
    CHECK(Settles(frames, update));
    CHECK((rig.a.visible() || rig.b.visible()));
    CHECK(closes == 1);
}

TEST_CASE("Welcome HUD cells and links enqueue their mail", "[core][sakura][views]") {
    Scene scene{kWide};
    Mailbox mailbox;
    const auto context = mailbox.context();
    const auto config = WelcomeHudPresenter(context).build();
    REQUIRE(config.has_value());
    WelcomeHudView view;
    view.update(*config);
    Sakura::HudLayer huds;
    huds.update({.huds = {view.hud()}});
    LayerHost host({&huds});
    host.place(scene.ctx, kWide);

    const Jetstream::Rect panel = Peek::Frame(*view.hud().content);
    CHECK(panel.width == 560.0f - 4.0f);
    CHECK(panel.x + panel.width * 0.5f == 800.0f);

    auto cells = Find<WelcomeHudCell>(huds);
    REQUIRE(cells.size() == 3);
    for (auto* cell : cells) {
        CHECK(Press(host, *cell));
    }
    CHECK(mailbox.count<MailNewFlowgraph>() == 1);
    CHECK(mailbox.count<MailOpenFlowgraph>() == 1);
    REQUIRE(mailbox.last<MailOpenModal>() != nullptr);
    CHECK(mailbox.last<MailOpenModal>()->content == ModalContent::FlowgraphExamples);

    auto links = Buttons(huds);
    REQUIRE(links.size() == 4);
    mailbox.mails.clear();
    for (auto* link : links) {
        CHECK(Press(host, *link));
    }
    REQUIRE(mailbox.mails.size() == 4);
    CHECK(std::get<MailOpenUrl>(mailbox.mails[0]).url == "https://cyberether.org");
    CHECK(std::get<MailOpenUrl>(mailbox.mails[1]).url == "https://cyberether.org/docs");
    CHECK(std::get<MailOpenSettings>(mailbox.mails[2]).section == SettingsSection::General);
    CHECK(std::get<MailOpenModal>(mailbox.mails[3]).content == ModalContent::About);
}

TEST_CASE("Welcome HUD presenter hides the HUD behind flowgraphs, modals, and the file picker",
          "[core][sakura][views]") {
    Mailbox mailbox;
    const auto context = mailbox.context();
    CHECK(WelcomeHudPresenter(context).build().has_value());

    mailbox.state.modal.content = ModalContent::About;
    CHECK_FALSE(WelcomeHudPresenter(context).build().has_value());

    mailbox.state.modal.content.reset();
    mailbox.state.filePicker.active = true;
    CHECK_FALSE(WelcomeHudPresenter(context).build().has_value());
}

TEST_CASE("Welcome HUD settles under identical updates", "[core][sakura][views]") {
    Scene scene{kWide};
    Mailbox mailbox;
    const auto context = mailbox.context();
    WelcomeHudView view;
    Sakura::HudLayer huds;
    LayerHost host({&huds});
    Frames frames(host, kWide, scene.ctx);
    const auto update = [&] {
        view.update(*WelcomeHudPresenter(context).build());
        huds.update({.huds = {view.hud()}});
    };
    CHECK(Settles(frames, update));
}

TEST_CASE("Welcome HUD cards and links repaint once per hover transition and fire without rebuilding",
          "[core][sakura][views]") {
    Scene scene{kWide};
    scene.ctx.palette = Shades();
    Mailbox mailbox;
    const auto context = mailbox.context();
    WelcomeHudView view;
    Sakura::HudLayer huds;
    LayerHost host({&huds});
    Frames frames(host, kWide, scene.ctx);
    const auto update = [&] {
        view.update(*WelcomeHudPresenter(context).build());
        huds.update({.huds = {view.hud()}});
    };
    const auto step = [&] {
        return frames.run(update);
    };
    const Extent2D<F32> away = {20.0f, 20.0f};
    REQUIRE(Settles(frames, update));
    const U64 rebuilds = frames.rebuilds;

    auto cells = Find<WelcomeHudCell>(huds);
    REQUIRE(cells.size() == 3);
    auto links = Buttons(huds);
    REQUIRE(links.size() == 4);
    for (U64 index = 0; index < cells.size(); ++index) {
        CAPTURE(index);
        CheckHover(host, step, *cells[index], away);
    }
    for (U64 index = 0; index < links.size(); ++index) {
        CAPTURE(index);
        CheckHover(host, step, *links[index], away);
    }

    for (U64 index = 0; index < cells.size(); ++index) {
        CAPTURE(index);
        const auto point = Center(*cells[index]);
        host.send(Mouse(MouseEventType::Move, point.x, point.y));
        CHECK(RepaintOnly(step()));
        CHECK(Press(host, *cells[index]));
        CHECK(mailbox.mails.size() == index + 1);
        CHECK(step().clean());
        host.send(Mouse(MouseEventType::Move, away.x, away.y));
        CHECK(RepaintOnly(step()));
        CHECK(Settles(frames, update));
    }
    CHECK(mailbox.count<MailNewFlowgraph>() == 1);
    CHECK(mailbox.count<MailOpenFlowgraph>() == 1);
    CHECK(mailbox.count<MailOpenModal>() == 1);
    CHECK(frames.rebuilds == rebuilds);
}

TEST_CASE("Start tab welcome cards and links repaint once per hover transition and fire without rebuilding",
          "[core][sakura][views]") {
    TitleRig rig;
    rig.scene.ctx.palette = Shades();
    StartTabView start;
    std::vector<std::string> calls;
    std::optional<std::string> activate = std::string(StartTabView::PaneId);
    const auto update = [&] {
        start.update({
            .welcome = {
                .version = "v1",
                .onNewFlowgraph = [&] { calls.push_back("new"); },
                .onOpenFlowgraph = [&] { calls.push_back("open"); },
                .onOpenExamples = [&] { calls.push_back("examples"); },
                .onOpenWebsite = [&] { calls.push_back("website"); },
            },
            .onClose = [] {},
        });
        rig.shell.updateTitlebar(rig.config);
        rig.shell.updatePanes({
            {.id = "a", .title = "Alpha", .group = "main", .content = &rig.a},
            start.pane(),
        }, std::exchange(activate, std::nullopt));
    };
    Frames frames(rig.host, TitleRig::bounds(), rig.scene.ctx);
    const auto step = [&] {
        return frames.run(update);
    };
    const Extent2D<F32> away = {60.0f, 700.0f};
    REQUIRE(Settles(frames, update));
    REQUIRE(start.pane().content->visible());
    const U64 rebuilds = frames.rebuilds;

    auto& content = *start.pane().content;
    auto cells = Find<WelcomeHudCell>(content);
    REQUIRE(cells.size() == 3);
    auto links = Buttons(content);
    REQUIRE(links.size() == 4);
    for (U64 index = 0; index < cells.size(); ++index) {
        CAPTURE(index);
        CheckHover(rig.host, step, *cells[index], away);
    }
    for (U64 index = 0; index < links.size(); ++index) {
        CAPTURE(index);
        CheckHover(rig.host, step, *links[index], away);
    }

    for (auto* cell : cells) {
        const auto point = Center(*cell);
        rig.host.send(Mouse(MouseEventType::Move, point.x, point.y));
        CHECK(RepaintOnly(step()));
        CHECK(Press(rig.host, *cell));
        const auto pressed = step();
        CHECK(pressed.clean());
        CHECK(step().clean());
        rig.host.send(Mouse(MouseEventType::Move, away.x, away.y));
        CHECK(RepaintOnly(step()));
        CHECK(Settles(frames, update));
    }
    CHECK(Press(rig.host, *links[0]));
    CHECK(Kept(step()));
    CHECK(Settles(frames, update));
    CHECK(calls == std::vector<std::string>{"new", "open", "examples", "website"});
    CHECK(frames.rebuilds == rebuilds);
}

TEST_CASE("Remote sharing shows as a raised pill in the top right of the title row",
          "[core][sakura][views]") {
    using Action = TitlebarView::Action;
    Mailbox mailbox;
    CHECK_FALSE(TitlebarPresenter(mailbox.context()).build().remoteStarted);
    mailbox.state.remote.started = true;
    mailbox.state.remote.clientCount = 3;
    const auto built = TitlebarPresenter(mailbox.context()).build();
    CHECK(built.remoteStarted);

    TitleRig rig;
    CHECK(Indicator(rig.shell.bar(), ICON_FA_TOWER_BROADCAST) == nullptr);
    rig.config.remoteStarted = true;
    rig.update();
    rig.place();
    auto* indicator = Indicator(rig.shell.bar(), ICON_FA_TOWER_BROADCAST);
    REQUIRE(indicator != nullptr);
    auto& remote = *indicator;
    const auto frame = Peek::Frame(remote);
    const auto help = Peek::Frame(Help(rig.shell.bar()));
    CHECK(std::abs(frame.right() + TitlebarView::IndicatorGap - help.x) <= 0.5f);
    bool label = false;
    for (auto* child : Peek::Children(remote)) {
        if (auto* text = dynamic_cast<Sakura::Text*>(child)) {
            label = label || text->config().str == "Remote Sharing";
        }
    }
    CHECK(label);

    rig.config.updateAvailable = true;
    rig.update();
    rig.place();
    const auto shifted = Peek::Frame(remote);
    auto* pending = Indicator(rig.shell.bar(), ICON_FA_CIRCLE_ARROW_UP);
    REQUIRE(pending != nullptr);
    const auto update = Peek::Frame(*pending);
    CHECK(std::abs(shifted.right() + TitlebarView::IndicatorGap - update.x) <= 0.5f);
    CHECK(std::abs(update.right() + TitlebarView::IndicatorGap - help.x) <= 0.5f);
    CHECK(shifted.y == update.y);

    rig.host.click(shifted.x + shifted.width * 0.5f, shifted.y + shifted.height * 0.5f);
    rig.place();
    CHECK(rig.actions == std::vector<Action>{Action::RemoteStreaming});
    CHECK(rig.presses.empty());
}

TEST_CASE("Hint overlay action buttons call their actions", "[core][sakura][views]") {
    Scene scene{kWide};
    std::vector<std::string> calls;
    HintOverlay hint;
    hint.update({
        .id = "hint",
        .title = "Start",
        .subtitle = "Nothing here yet",
        .steps = {"One", "Two"},
        .hints = {"First tip", "Second tip"},
        .actions = {
            {.id = "new", .label = "New", .onClick = [&] { calls.push_back("new"); }},
            {.id = "open", .label = "Open", .onClick = [&] { calls.push_back("open"); }},
        },
        .size = {400.0f, 420.0f},
    });
    Stage stage(hint.component(), {100.0f, 100.0f, 400.0f, 420.0f});
    stage.place(scene.ctx);

    CHECK(hint.size() == Extent2D<F32>{400.0f, 420.0f});
    auto buttons = Buttons(hint.component());
    REQUIRE(buttons.size() == 2);
    CHECK(Press(stage.host, *buttons[1]));
    CHECK(Press(stage.host, *buttons[0]));
    CHECK(calls == std::vector<std::string>{"open", "new"});
    CHECK(Peek::Frame(*buttons[1]).y > Peek::Frame(*buttons[0]).bottom());
    CHECK(Find<Sakura::Text>(hint.component()).size() == 3 + 2 + 1 + 2);

    hint.update({.id = "hint", .title = "Start"});
    stage.place(scene.ctx);
    CHECK(Buttons(hint.component()).empty());
    CHECK(Find<Sakura::Text>(hint.component()).size() == 3);
}

TEST_CASE("Callouts wrap long text to the available width for every tone", "[core][sakura][views]") {
    Scene scene{kWide};
    for (const auto tone : {Callout::Tone::Info, Callout::Tone::Success, Callout::Tone::Warning,
                            Callout::Tone::Error}) {
        Callout callout;
        callout.update({.str = "Short", .tone = tone});
        Host host(callout);
        const auto single = host.measureOf(scene.ctx, {300.0f, kInfinite});
        callout.update({.str = std::string(200, 'x') + " " + std::string(200, 'y'), .tone = tone});
        const auto wrapped = host.measureOf(scene.ctx, {300.0f, kInfinite});
        CHECK(single.y > 0.0f);
        CHECK(wrapped.y > single.y);
        CHECK(wrapped.x <= 300.0f);
    }
}

TEST_CASE("Retained tooltips show their panel only on the frame after being requested and keep it once prepared",
          "[core][sakura][views]") {
    Probe content({100.0f, 20.0f});
    Tooltip tooltip;
    tooltip.update({.id = "tip", .content = &content});

    CHECK_FALSE(tooltip.requested());
    CHECK(tooltip.show() == nullptr);
    CHECK(tooltip.requested());
    Sakura::Component* panel = tooltip.show();
    CHECK(panel != nullptr);
    CHECK(tooltip.requested());
    CHECK(tooltip.show() == panel);
    CHECK(tooltip.requested());
    CHECK_FALSE(tooltip.requested());
    CHECK_FALSE(tooltip.requested());
    CHECK(tooltip.show() == panel);
    CHECK(tooltip.requested());
}

TEST_CASE("Clicking the search pill opens a right-aligned grouped catalog under it with docs on its left",
          "[core][sakura][views]") {
    PickerRig rig;
    rig.refresh();
    CHECK_FALSE(rig.picker.isOpen());
    CHECK(Peek::CursorAt(rig.picker.pillComponent(), {1068.0f, 29.0f}) == SurfaceCursor::Hand);

    rig.click(1068.0f, 29.0f);
    CHECK(rig.opens == 1);
    rig.refresh();
    CHECK(Find<Sakura::TextInput>(rig.picker.pillComponent()).size() == 1);
    CHECK(rig.host.popups.interactiveAt({1000.0f, 100.0f}));
    CHECK(rig.host.popups.interactiveAt({500.0f, 100.0f}));
    CHECK_FALSE(rig.host.popups.interactiveAt({300.0f, 100.0f}));
    CHECK(rig.actions().size() == 2);

    rig.click(1068.0f, 29.0f);
    rig.refresh();
    CHECK(rig.opens == 1);
    CHECK(rig.picker.isOpen());
}

TEST_CASE("Choosing a backend creates the selected block at the requested position and closes the catalog",
          "[core][sakura][views]") {
    PickerRig rig;
    rig.openFromPill();

    rig.click(1000.0f, 178.0f);
    rig.refresh();
    auto actions = rig.actions();
    REQUIRE(actions.size() == 1);
    CHECK(Press(rig.host, *actions[0]));
    REQUIRE(rig.created.size() == 1);
    CHECK(rig.created[0].type == "filter");
    CHECK(rig.created[0].device == DeviceType::CPU);
    CHECK(rig.created[0].position == Extent2D<F32>{40.0f, 80.0f});
    CHECK_FALSE(rig.picker.isOpen());
    rig.refresh();
    CHECK_FALSE(rig.host.popups.interactiveAt({1000.0f, 100.0f}));

    rig.openFromPill();
    actions = rig.actions();
    REQUIRE(actions.size() == 2);
    CHECK(Press(rig.host, *actions[1]));
    REQUIRE(rig.created.size() == 2);
    CHECK(rig.created[1].type == "fft");
    CHECK(rig.created[1].device == DeviceType::Metal);
}

TEST_CASE("Double clicking a catalog result creates it with its default backend", "[core][sakura][views]") {
    PickerRig rig;
    rig.openFromPill();
    rig.click(1000.0f, 242.0f, 2);
    REQUIRE(rig.created.size() == 1);
    CHECK(rig.created[0].type == "file-reader");
    CHECK(rig.created[0].device == DeviceType::CPU);
    CHECK_FALSE(rig.picker.isOpen());
}

TEST_CASE("Clicking outside the catalog dismisses it", "[core][sakura][views]") {
    PickerRig rig;
    rig.openFromPill();
    CHECK(rig.host.click(100.0f, 800.0f));
    CHECK_FALSE(rig.picker.isOpen());
    CHECK(rig.created.empty());
}

TEST_CASE("Opening at the cursor shows the catalog there with its own search field and docs on its right",
          "[core][sakura][views]") {
    PickerRig rig;
    rig.refresh();
    rig.picker.open(Catalog(), FlowgraphBlockPicker::Mode::Cursor, {5.0f, 5.0f});
    rig.refresh(Extent2D<F32>{200.0f, 300.0f});
    CHECK(rig.host.popups.interactiveAt({210.0f, 310.0f}));
    CHECK(rig.host.popups.interactiveAt({620.0f, 400.0f}));
    CHECK_FALSE(rig.host.popups.interactiveAt({190.0f, 310.0f}));
    const auto fields = Find<Sakura::PillField>(rig.host.popups);
    REQUIRE(fields.size() == 1);
    CHECK(Peek::Frame(*fields[0]) == Jetstream::Rect{200.0f, 300.0f, 300.0f, 34.0f});
    CHECK(Find<Sakura::TextInput>(*fields[0]).size() == 1);
    CHECK(Find<Sakura::TextInput>(rig.picker.pillComponent()).empty());

    rig.click(400.0f, 428.0f, 2);
    REQUIRE(rig.created.size() == 1);
    CHECK(rig.created[0].position == Extent2D<F32>{5.0f, 5.0f});
}

TEST_CASE("Picking a category chip narrows the catalog to that category", "[core][sakura][views]") {
    PickerRig rig;
    rig.openFromPill();
    Sakura::Label* strip = nullptr;
    for (auto* label : Find<Sakura::Label>(rig.host.popups)) {
        if (!label->config().instances.empty() && label->config().instances.front().str == "All") {
            strip = label;
        }
    }
    REQUIRE(strip);
    REQUIRE(strip->config().instances.size() == 4);
    const auto io = strip->config().instances[2].rect;
    CHECK(strip->config().instances[2].str == "IO");
    rig.click(io.x + io.width * 0.5f, io.y + io.height * 0.5f);
    rig.refresh();
    CHECK(rig.picker.isOpen());
    rig.click(1000.0f, 118.0f, 2);
    REQUIRE(rig.created.size() == 1);
    CHECK(rig.created[0].type == "file-reader");
}

TEST_CASE("Catalog with no blocks shows no actions", "[core][sakura][views]") {
    PickerRig rig;
    rig.refresh();
    rig.picker.open({}, FlowgraphBlockPicker::Mode::Pill, {0.0f, 0.0f});
    rig.refresh();
    CHECK(rig.actions().empty());
    CHECK(rig.host.popups.interactiveAt({1000.0f, 100.0f}));
}

TEST_CASE("Block picker settles under identical updates", "[core][sakura][views]") {
    PickerRig rig;
    rig.refresh();
    rig.picker.open(Catalog(), FlowgraphBlockPicker::Mode::Pill, {0.0f, 0.0f});
    Frames frames(rig.host, PickerRig::kBounds, rig.scene.ctx);
    const auto update = [&] {
        rig.refresh();
    };
    CHECK(Settles(frames, update));
}

TEST_CASE("Typing in the block search lists ranked matches and Enter creates the first one",
          "[core][sakura][views]") {
    MonoFontWindow window;
    auto ctx = MakeContext({1200.0f, 900.0f});
    ctx.render = &window;
    std::vector<std::string> created;
    FlowgraphBlockPicker picker;
    const FlowgraphBlockPicker::Config config{
        .id = "picker",
        .onCreateBlock = [&](const std::string& type, Extent2D<F32>, DeviceType, RuntimeType, ProviderType) {
            created.push_back(type);
        },
    };
    Sakura::Overlay overlay;
    LayerSlot slot;
    slot.hold(overlay, PickerRig::kBounds);
    LayerHost host({&slot});
    const auto frame = [&] {
        picker.update(config);
        overlay.update({.items = {
            {.child = &picker.pillComponent(), .anchor = Sakura::Anchor::TopRight, .offset = {-12.0f, 12.0f},
             .size = {240.0f, 34.0f}},
            {.child = &picker.popupComponent(), .anchor = Sakura::Anchor::TopRight, .offset = {-12.0f, 12.0f},
             .size = {240.0f, 34.0f}},
        }});
        window.frame([&] {
            host.place(ctx, PickerRig::kBounds);
        });
    };
    picker.open(Catalog(), FlowgraphBlockPicker::Mode::Pill, {0.0f, 0.0f});
    frame();
    frame();

    Type(host, "fi");
    frame();
    frame();
    CHECK(Tap(host, KeyCode::Enter));
    frame();
    CHECK(created == std::vector<std::string>{"filter"});
    CHECK_FALSE(picker.isOpen());
}

TEST_CASE("Arrow keys wrap through the block results and Enter creates the highlighted one",
          "[core][sakura][views]") {
    KeyedPickerRig rig;
    rig.open(FlowgraphBlockPicker::Mode::Pill);
    CHECK(rig.settles());

    CHECK(RepaintOnly(rig.press(KeyCode::UpArrow)));
    CHECK(rig.settles());
    rig.press(KeyCode::Enter);
    REQUIRE(rig.created.size() == 1);
    CHECK(rig.created[0].type == "custom");
    CHECK(rig.created[0].position == Extent2D<F32>{7.0f, 9.0f});
    CHECK_FALSE(rig.picker.isOpen());
    CHECK(rig.settles());

    rig.open(FlowgraphBlockPicker::Mode::Pill);
    for (U64 step = 0; step < 4; ++step) {
        CAPTURE(step);
        CHECK(RepaintOnly(rig.press(KeyCode::DownArrow)));
        CHECK(rig.settles());
    }
    rig.press(KeyCode::DownArrow);
    rig.press(KeyCode::Enter);
    REQUIRE(rig.created.size() == 2);
    CHECK(rig.created[1].type == "filter");
    CHECK_FALSE(rig.picker.isOpen());
    CHECK(rig.settles());
}

TEST_CASE("Escape closes the block search in both modes without creating a block",
          "[core][sakura][views]") {
    KeyedPickerRig rig;
    rig.open(FlowgraphBlockPicker::Mode::Pill);
    REQUIRE(rig.picker.isOpen());
    rig.press(KeyCode::Escape);
    CHECK_FALSE(rig.picker.isOpen());

    rig.frames.ctx.mouse = Extent2D<F32>{200.0f, 300.0f};
    rig.open(FlowgraphBlockPicker::Mode::Cursor);
    REQUIRE(rig.picker.isOpen());
    rig.press(KeyCode::Escape);
    CHECK_FALSE(rig.picker.isOpen());
    CHECK(rig.created.empty());
    CHECK(rig.settles());
}

TEST_CASE("Escape in the focused search field closes the search before any shortcut sees it",
          "[core][sakura][views]") {
    KeyedPickerRig rig;
    rig.open(FlowgraphBlockPicker::Mode::Pill);
    REQUIRE(rig.picker.isOpen());
    REQUIRE(FocusOwner(rig.host) != nullptr);
    rig.press(KeyCode::Escape);
    CHECK(rig.handled);
    CHECK_FALSE(rig.picker.isOpen());
    CHECK(rig.created.empty());
}

TEST_CASE("Enter with no matching blocks creates nothing and keeps the search open",
          "[core][sakura][views]") {
    KeyedPickerRig rig;
    rig.open(FlowgraphBlockPicker::Mode::Pill);
    rig.type("zzz");
    CHECK(rig.settles());
    rig.press(KeyCode::DownArrow);
    rig.press(KeyCode::Enter);
    CHECK(rig.created.empty());
    CHECK(rig.picker.isOpen());
    CHECK(rig.settles());
}

TEST_CASE("Typing in the cursor search field narrows the results and settles", "[core][sakura][views]") {
    KeyedPickerRig rig;
    rig.frames.ctx.mouse = Extent2D<F32>{200.0f, 300.0f};
    rig.open(FlowgraphBlockPicker::Mode::Cursor, {3.0f, 4.0f});
    CHECK(rig.settles());
    rig.type("reader");
    CHECK(rig.settles());
    rig.press(KeyCode::DownArrow);
    rig.press(KeyCode::Enter);
    REQUIRE(rig.created.size() == 1);
    CHECK(rig.created[0].type == "file-reader");
    CHECK(rig.created[0].position == Extent2D<F32>{3.0f, 4.0f});
    CHECK(rig.settles());
}

TEST_CASE("Block picker opened at the cursor settles under identical updates and idle hover",
          "[core][sakura][views]") {
    PickerRig rig;
    rig.refresh();
    rig.picker.open(Catalog(), FlowgraphBlockPicker::Mode::Cursor, {0.0f, 0.0f});
    auto ctx = rig.scene.ctx;
    ctx.mouse = Extent2D<F32>{200.0f, 300.0f};
    Frames frames(rig.host, PickerRig::kBounds, ctx);
    const auto update = [&] {
        rig.refresh(Extent2D<F32>{200.0f, 300.0f});
    };
    CHECK(Settles(frames, update));
    CHECK(rig.host.popups.interactiveAt({210.0f, 310.0f}));
    rig.host.send(Mouse(MouseEventType::Move, 1100.0f, 800.0f));
    CHECK(frames.run(update).clean());
    CHECK(rig.picker.isOpen());
}

TEST_CASE("Block backends forward the device, runtime, and provider of the chosen target",
          "[core][sakura][views]") {
    {
        KeyedPickerRig keyed;
        keyed.picker.open(RoutedCatalog(), FlowgraphBlockPicker::Mode::Pill, {0.0f, 0.0f});
        keyed.frame();
        keyed.frame();
        keyed.press(KeyCode::Enter);
        REQUIRE(keyed.created.size() == 1);
        CHECK(keyed.created[0].device == DeviceType::Metal);
        CHECK(keyed.created[0].runtime == RuntimeType::PYTHON);
        CHECK(keyed.created[0].provider == "torch");
    }

    PickerRig rig;
    const auto open = [&] {
        rig.refresh();
        rig.picker.open(RoutedCatalog(), FlowgraphBlockPicker::Mode::Pill, {1.0f, 2.0f});
        rig.refresh();
    };

    open();
    rig.click(1000.0f, 146.0f, 2);
    REQUIRE(rig.created.size() == 1);
    CHECK(rig.created[0].device == DeviceType::Metal);
    CHECK(rig.created[0].runtime == RuntimeType::PYTHON);
    CHECK(rig.created[0].provider == "torch");

    for (U64 badge = 0; badge < 2; ++badge) {
        CAPTURE(badge);
        open();
        const auto actions = rig.actions();
        REQUIRE(actions.size() == 2);
        CHECK(Press(rig.host, *actions[badge]));
    }
    REQUIRE(rig.created.size() == 3);
    CHECK(rig.created[1].device == DeviceType::CPU);
    CHECK(rig.created[1].runtime == RuntimeType::NATIVE);
    CHECK(rig.created[1].provider == "generic");
    CHECK(rig.created[2].device == DeviceType::Metal);
    CHECK(rig.created[2].runtime == RuntimeType::MLIR);
    CHECK(rig.created[2].provider == "iree");
    for (const auto& created : rig.created) {
        CHECK(created.type == "fft");
        CHECK(created.position == Extent2D<F32>{1.0f, 2.0f});
    }

    auto bare = RoutedCatalog();
    bare[0].devices.clear();
    rig.refresh();
    rig.picker.open(bare, FlowgraphBlockPicker::Mode::Pill, {0.0f, 0.0f});
    rig.refresh();
    CHECK(rig.actions().empty());
    rig.click(1000.0f, 146.0f, 2);
    REQUIRE(rig.created.size() == 4);
    CHECK(rig.created[3].device == DeviceType::Metal);
    CHECK(rig.created[3].runtime == RuntimeType::PYTHON);
    CHECK(rig.created[3].provider == "torch");
}

TEST_CASE("Modal container shows nothing without content and sizes each modal", "[core][sakura][views]") {
    ModalRig rig;
    Mailbox mailbox;
    const auto context = mailbox.context();
    ModalView view;

    view.update(ModalPresenter(context).build());
    auto layer = view.layer();
    CHECK(layer.content == nullptr);
    rig.show(layer);
    CHECK_FALSE(rig.modal.covers({800.0f, 600.0f}));

    mailbox.state.modal.content = ModalContent::FlowgraphInfo;
    view.update(ModalPresenter(context).build());
    CHECK(view.layer().content == nullptr);

    const std::vector<std::pair<ModalContent, F32>> widths = {
        {ModalContent::About, 526.0f},          {ModalContent::FlowgraphExamples, 526.0f},
        {ModalContent::Benchmark, 526.0f},      {ModalContent::RemoteStreaming, 526.0f},
        {ModalContent::Dependencies, 736.0f},
        {ModalContent::Feedback, 496.0f},
    };
    for (const auto& [content, width] : widths) {
        mailbox.state.modal.content = content;
        view.update(ModalPresenter(context).build());
        layer = view.layer();
        REQUIRE(layer.content != nullptr);
        rig.show(layer);
        CHECK(rig.modal.covers({10.0f, 10.0f}));
        CHECK(Peek::Frame(*layer.content).width == width);
        CHECK(Peek::Frame(*layer.content).x == 800.0f - width * 0.5f);
    }
}

TEST_CASE("Modal container close button closes the modal or returns from the plugin modal to settings",
          "[core][sakura][views]") {
    ModalRig rig;
    Mailbox mailbox;
    const auto context = mailbox.context();
    ModalView view;

    mailbox.state.modal.content = ModalContent::About;
    view.update(ModalPresenter(context).build());
    rig.show(view.layer());
    CHECK(rig.host.click(10.0f, 10.0f));
    CHECK(mailbox.mails.empty());
    CHECK(Press(rig.host, rig.close()));
    CHECK(mailbox.count<MailCloseModal>() == 1);
}

TEST_CASE("About modal shows the product text without buttons", "[core][sakura][views]") {
    ModalRig rig;
    AboutView view;
    view.update({});
    rig.show({.content = &view.component()});

    CHECK(Find<Sakura::Text>(view.component()).size() == 3);
    CHECK(Buttons(view.component()).empty());
}

TEST_CASE("Settings navigation reports every section in order", "[core][sakura][views]") {
    SettingsRig rig;
    rig.show(SettingsSection::General);
    CHECK(Peek::Frame(rig.view.component()).width == 1176.0f);

    auto sections = Find<Sakura::SidebarItem>(rig.view.component());
    REQUIRE(sections.size() == 8);
    for (auto* section : sections) {
        CHECK(Press(rig.host, *section));
    }
    CHECK(rig.calls == std::vector<std::string>{"section:0", "section:1", "section:2", "section:3", "section:4",
                                                 "section:5", "section:6", "section:7"});
}

TEST_CASE("Settings sections replace the editor content", "[core][sakura][views]") {
    SettingsRig rig;
    rig.show(SettingsSection::General);
    CHECK(rig.combos().size() == 3);
    CHECK(Find<Sakura::Toggle>(rig.view.component()).size() == 1);

    rig.show(SettingsSection::Remote);
    CHECK(rig.combos().size() == 4);
    CHECK(Find<Sakura::Checkbox>(rig.view.component()).empty());
    CHECK(Find<Sakura::Toggle>(rig.view.component()).size() == 1);

    rig.show(SettingsSection::Registry);
    CHECK(rig.combos().empty());
    CHECK(Find<Sakura::Table>(rig.view.component()).empty());
    CHECK(Find<Sakura::SettingCard>(rig.view.component()).size() == 2);

    rig.show(SettingsSection::Legal);
    CHECK(Find<Sakura::TabBar>(rig.view.component()).empty());
}

TEST_CASE("General settings report theme, scale, and background particles", "[core][sakura][views]") {
    SettingsRig rig;
    rig.show(SettingsSection::General);

    rig.choose(0, 1);
    rig.choose(1, 0);
    auto toggles = Find<Sakura::Toggle>(rig.view.component());
    REQUIRE(toggles.size() == 1);
    CHECK(Press(rig.host, *toggles[0]));
    CHECK(rig.calls == std::vector<std::string>{"theme:Light", "scale:" + std::to_string(0.5f),
                                                 "particles:false"});
}

TEST_CASE("Developer settings report toggles and the log level", "[core][sakura][views]") {
    SettingsRig rig;
    rig.show(SettingsSection::Developer);

    auto toggles = Find<Sakura::Toggle>(rig.view.component());
    REQUIRE(toggles.size() == 1);
    CHECK(Press(rig.host, *toggles[0]));
    rig.choose(0, 4);
    CHECK(rig.calls == std::vector<std::string>{"timing:false", "log:4"});
}

TEST_CASE("Developer settings open the benchmarks and reload plugins", "[core][sakura][views]") {
    SettingsRig rig;
    rig.config.developer.onShowBenchmarks = [&] { rig.calls.push_back("benchmarks"); };
    rig.config.developer.onReloadPlugins = [&] { rig.calls.push_back("plugins"); };
    rig.show(SettingsSection::Developer);

    std::vector<Sakura::Button*> buttons;
    for (auto* button : Buttons(rig.view.component())) {
        buttons.push_back(button);
    }
    REQUIRE(buttons.size() >= 2);
    CHECK(Press(rig.host, *buttons[buttons.size() - 2]));
    rig.place();
    CHECK(Press(rig.host, *buttons[buttons.size() - 1]));
    CHECK(rig.calls == std::vector<std::string>{"benchmarks", "plugins"});
}

TEST_CASE("Developer settings presenter opens the benchmark modal and reloads every plugin",
          "[core][sakura][views]") {
    Mailbox mailbox;
    const auto config = DeveloperSettingsPresenter(mailbox.context()).build();
    config.onShowBenchmarks();
    REQUIRE(mailbox.last<MailOpenModal>() != nullptr);
    CHECK(mailbox.last<MailOpenModal>()->content == ModalContent::Benchmark);
    config.onReloadPlugins();
    CHECK(mailbox.count<MailReloadAllPlugins>() == 1);
}

TEST_CASE("Remote settings report codec, framerate, and approval while the encoder stays disabled",
          "[core][sakura][views]") {
    SettingsRig rig;
    rig.show(SettingsSection::Remote);
    REQUIRE(rig.combos().size() == 4);
    auto toggles = Find<Sakura::Toggle>(rig.view.component());
    REQUIRE(toggles.size() == 1);

    rig.choose(1, 1);
    rig.choose(0, 2);
    rig.choose(2, 0);
    CHECK(Press(rig.host, *toggles[0]));
    CHECK(rig.calls == std::vector<std::string>{"codec:1", "fps:60", "approve:true"});
}

TEST_CASE("Runtime settings pick interpreters and browse for a custom path", "[core][sakura][views]") {
    SettingsRig rig;
    rig.show(SettingsSection::Runtime);
    REQUIRE(rig.combos().size() == 2);

    rig.choose(1, 1);
    rig.choose(0, 1);
    auto buttons = Buttons(rig.view.component());
    REQUIRE(buttons.size() == 1);
    CHECK(Press(rig.host, *buttons[0]));
    CHECK(rig.calls == std::vector<std::string>{"policy:allow", "python:/opt/homebrew/bin/python3", "cache"});

    rig.calls.clear();
    rig.choose(0, 2);
    CHECK(rig.calls.empty());
    rig.show(SettingsSection::Runtime);
    buttons = Buttons(rig.view.component());
    REQUIRE(buttons.size() == 2);
    CHECK(Press(rig.host, *buttons[0]));
    CHECK(rig.calls == std::vector<std::string>{"browse:", "python:/custom/python"});
}

TEST_CASE("Registry overview lists plugins by name and navigates to their pages", "[core][sakura][views]") {
    SettingsRig rig;
    rig.show(SettingsSection::Registry);

    auto buttons = Buttons(rig.view.component());
    REQUIRE(buttons.size() == 1);
    CHECK(Press(rig.host, *buttons[0]));
    CHECK(Find<Sakura::Badge>(rig.view.component()).empty());

    auto cards = Find<Sakura::SettingCard>(rig.view.component());
    REQUIRE(cards.size() == 2);
    const auto plugins = Peek::Frame(*cards[1]);
    CHECK(rig.host.click(plugins.x + 40.0f, plugins.y + 20.0f));
    CHECK(rig.host.click(plugins.x + 40.0f, plugins.bottom() - 20.0f));
    CHECK(rig.calls == std::vector<std::string>{"nav:1:", "nav:2:/plugins/alpha.cep", "nav:2:/plugins/beta.cep"});

    rig.calls.clear();
    rig.config.registry.plugins.clear();
    rig.config.registry.domains.clear();
    rig.show(SettingsSection::Registry);
    CHECK(Buttons(rig.view.component()).size() == 1);
    CHECK(Find<Sakura::SettingCard>(rig.view.component()).size() == 2);
}

TEST_CASE("Registry plugin page reloads, removes, and returns to the overview", "[core][sakura][views]") {
    SettingsRig rig;
    rig.config.registry.page = RegistryPage::Plugin;
    rig.config.registry.plugin = "/plugins/alpha.cep";
    rig.show(SettingsSection::Registry);

    CHECK(Find<Sakura::Badge>(rig.view.component()).size() == 1);
    auto buttons = Buttons(rig.view.component());
    REQUIRE(buttons.size() == 3);
    for (auto* button : buttons) {
        CHECK(Press(rig.host, *button));
    }
    CHECK(rig.calls == std::vector<std::string>{"nav:0:", "reload:/plugins/alpha.cep", "remove:/plugins/alpha.cep"});

    rig.config.registry.plugin = "/plugins/missing.cep";
    rig.show(SettingsSection::Registry);
    CHECK(Find<Sakura::Badge>(rig.view.component()).empty());
    CHECK(Find<Sakura::SettingCard>(rig.view.component()).size() == 2);
}

TEST_CASE("Registry add page registers a browsed path only after the source is trusted", "[core][sakura][views]") {
    SettingsRig rig;
    rig.config.registry.page = RegistryPage::AddPlugin;
    rig.show(SettingsSection::Registry);

    auto buttons = Buttons(rig.view.component());
    REQUIRE(buttons.size() == 3);
    Press(rig.host, *buttons[2]);
    CHECK(rig.calls.empty());

    CHECK(Press(rig.host, *buttons[1]));
    rig.show(SettingsSection::Registry);
    Press(rig.host, *Buttons(rig.view.component())[2]);
    CHECK(rig.calls == std::vector<std::string>{"browse-plugin:"});

    auto toggles = Find<Sakura::Toggle>(rig.view.component());
    REQUIRE(toggles.size() == 1);
    CHECK(Press(rig.host, *toggles[0]));
    rig.show(SettingsSection::Registry);
    CHECK(Press(rig.host, *Buttons(rig.view.component())[2]));
    CHECK(rig.calls == std::vector<std::string>{"browse-plugin:", "register:/plugins/gamma.cep"});

    rig.config.registry.page = RegistryPage::Overview;
    rig.show(SettingsSection::Registry);
    rig.config.registry.page = RegistryPage::AddPlugin;
    rig.show(SettingsSection::Registry);
    Press(rig.host, *Buttons(rig.view.component())[2]);
    CHECK(rig.calls.size() == 2);
}

TEST_CASE("Intelligence settings toggle features and manage providers and models", "[core][sakura][views]") {
    SettingsRig rig;
    const auto log = [&rig](std::string entry) {
        rig.calls.push_back(std::move(entry));
    };
    rig.config.intelligence = {
        .enabled = true,
        .providers = {{.name = "openai", .status = "Ready", .ready = true}},
        .models = {{.provider = "openai", .id = "gpt", .label = "GPT", .tags = {"128k context"}}},
        .voice = {.status = "Unsupported"},
        .onEnabledChange = [log](bool value) { log("enabled:" + Flag(value)); },
        .onOpenAddProvider = [log] { log("add-provider"); },
        .onEditProvider = [log](const std::string& name) { log("edit:" + name); },
        .onRemoveProvider = [log](const std::string& name) { log("remove:" + name); },
        .onOpenAddModel = [log] { log("add-model"); },
        .onRemoveModel = [log](const IntelligenceSettingsPanel::ModelRow& model) { log("remove-model:" + model.id); },
    };
    rig.show(SettingsSection::Intelligence);

    CHECK(Find<Sakura::SettingCard>(rig.view.component()).size() == 4);
    CHECK(Find<Sakura::Badge>(rig.view.component()).size() == 3);
    auto buttons = Buttons(rig.view.component());
    REQUIRE(buttons.size() == 5);
    for (auto* button : buttons) {
        CHECK(Press(rig.host, *button));
    }
    auto toggles = Find<Sakura::Toggle>(rig.view.component());
    REQUIRE(toggles.size() == 1);
    CHECK(Press(rig.host, *toggles[0]));
    CHECK(rig.calls == std::vector<std::string>{"add-provider", "edit:openai", "remove:openai", "add-model",
                                                 "remove-model:gpt", "enabled:false"});

    rig.config.intelligence.enabled = false;
    rig.config.intelligence.providers.clear();
    rig.show(SettingsSection::Intelligence);
    CHECK(Find<Sakura::SettingCard>(rig.view.component()).size() == 1);
}

TEST_CASE("Intelligence model rows keep Remove reachable when their tags overflow a narrow page",
          "[core][sakura][views]") {
    SettingsRig rig;
    std::vector<std::string> removed;
    rig.config.intelligence = {
        .enabled = true,
        .models = {{.provider = "openrouter", .id = "anthropic/claude", .label = "Claude",
                    .tags = {"1M context", "$15.00 in", "$75.00 out", "Reasoning", "Vision"}}},
        .onRemoveModel = [&](const IntelligenceSettingsPanel::ModelRow& model) { removed.push_back(model.id); },
    };
    rig.config.section = SettingsSection::Intelligence;
    rig.view.update(rig.config);
    rig.ModalRig::show({.content = &rig.view.component(), .size = Extent2D<F32>{800.0f, 900.0f}});

    std::vector<Sakura::Button*> removes;
    for (auto* button : Buttons(rig.view.component())) {
        if (Peek::Frame(*button).width > 0.0f) {
            removes.push_back(button);
        }
    }
    REQUIRE_FALSE(removes.empty());
    auto* remove = removes.back();
    const auto cards = Find<Sakura::SettingCard>(rig.view.component());
    REQUIRE(cards.size() >= 3);
    CHECK(Peek::Frame(*remove).right() <= Peek::Frame(*cards[2]).right());
    CHECK(Press(rig.host, *remove));
    CHECK(removed == std::vector<std::string>{"anthropic/claude"});
}

TEST_CASE("Intelligence provider page tests the draft and returns to the overview", "[core][sakura][views]") {
    SettingsRig rig;
    const auto log = [&rig](std::string entry) {
        rig.calls.push_back(std::move(entry));
    };
    Settings::Intelligence::Provider preset;
    preset.name = "Local";
    preset.host = "http://localhost:11434";
    rig.config.intelligence.page = IntelligencePage::Provider;
    rig.config.intelligence.provider = {
        .defaultPreset = "Custom",
        .session = 1,
        .presets = {{.label = "Custom", .description = "Any compatible endpoint.", .provider = preset}},
        .isValid = [](const Settings::Intelligence::Provider&) { return true; },
        .onTest = [log](const Settings::Intelligence::Provider& provider, const std::string&, U64) {
            log("test:" + provider.name);
        },
        .onAdd = [log](const Settings::Intelligence::Provider& provider, const std::string&) {
            log("add:" + provider.name);
        },
        .onBack = [log] { log("back"); },
    };
    rig.config.section = SettingsSection::Intelligence;
    rig.view.update(rig.config);
    rig.ModalRig::show({.content = &rig.view.component(), .size = Extent2D<F32>{1200.0f, 1180.0f}});

    CHECK(Find<Sakura::SettingCard>(rig.view.component()).size() == 3);
    auto buttons = Buttons(rig.view.component());
    REQUIRE(buttons.size() == 3);
    for (auto* button : buttons) {
        Press(rig.host, *button);
    }
    CHECK(rig.calls == std::vector<std::string>{"back", "test:Local"});
}

TEST_CASE("Intelligence model page lists models and adds the chosen one", "[core][sakura][views]") {
    SettingsRig rig;
    const auto log = [&rig](std::string entry) {
        rig.calls.push_back(std::move(entry));
    };
    rig.config.intelligence.page = IntelligencePage::AddModel;
    rig.config.intelligence.model = {
        .models = {
            {.model = {.provider = "openai", .id = "gpt-a"}, .label = "GPT A"},
            {.model = {.provider = "openai", .id = "gpt-b"}, .label = "GPT B"},
        },
        .status = "2 of 2 models shown.",
        .onCommit = [log](const Settings::Intelligence::Model& model) { log("commit:" + model.id); },
        .onBack = [log] { log("back"); },
    };
    rig.show(SettingsSection::Intelligence);

    auto buttons = Buttons(rig.view.component());
    REQUIRE(buttons.size() == 3);
    for (auto* button : buttons) {
        CHECK(Press(rig.host, *button));
    }
    CHECK(rig.calls == std::vector<std::string>{"back", "commit:gpt-a", "commit:gpt-b"});

    rig.config.intelligence.model.models.clear();
    rig.show(SettingsSection::Intelligence);
    CHECK(Buttons(rig.view.component()).size() == 1);
    CHECK(Find<Sakura::SettingCard>(rig.view.component()).size() == 1);
}

TEST_CASE("About settings check for updates, send feedback, and list info tables", "[core][sakura][views]") {
    SettingsRig rig;
    rig.show(SettingsSection::About);

    auto buttons = Buttons(rig.view.component());
    REQUIRE(buttons.size() == 2);
    CHECK(Press(rig.host, *buttons[0]));
    CHECK(Press(rig.host, *buttons[1]));
    CHECK(rig.calls == std::vector<std::string>{"check", "feedback"});
    CHECK(Find<Sakura::Table>(rig.view.component()).empty());
    CHECK(Find<AboutInfoTable>(rig.view.component()).size() == 1);
}

TEST_CASE("About update card offers the action for each update state", "[core][sakura][views]") {
    Scene scene{kWide};
    std::vector<std::string> calls;
    AboutUpdateCard card;
    Stage stage(card, {100.0f, 100.0f, 600.0f, 400.0f});
    const auto show = [&](AboutUpdateCard::Config config) {
        config.version = "CyberEther v1.0";
        config.updateVersion = "v2.0";
        config.onCheckForUpdates = [&] { calls.push_back("check"); };
        config.onDownloadUpdate = [&] { calls.push_back("download"); };
        config.onApplyUpdate = [&] { calls.push_back("apply"); };
        card.update(config);
        stage.place(scene.ctx);
        auto buttons = Buttons(card);
        for (auto* button : buttons) {
            Press(stage.host, *button);
        }
        return buttons.size();
    };

    CHECK(show({}) == 1);
    CHECK(show({.supported = true, .checking = true}) == 1);
    CHECK(calls.empty());
    CHECK(show({.supported = true}) == 1);
    CHECK(show({.supported = true, .updateAvailable = true, .releaseNotes = "Fixes"}) == 1);
    CHECK(show({.supported = true, .ready = true}) == 1);
    CHECK(show({.supported = true, .applying = true}) == 1);
    CHECK(calls == std::vector<std::string>{"check", "download", "apply"});

    CHECK(show({.supported = true, .downloading = true, .progress = 0.5f}) == 0);
    CHECK(Find<Sakura::ProgressBar>(card).size() == 1);
}

TEST_CASE("Legal settings list the license and third-party software with a full licenses button",
          "[core][sakura][views]") {
    SettingsRig rig;
    rig.show(SettingsSection::Legal);

    CHECK(Find<Sakura::TabBar>(rig.view.component()).empty());
    CHECK(Find<Sakura::SettingCard>(rig.view.component()).size() == 2);
    CHECK_FALSE(Find<Sakura::Badge>(rig.view.component()).empty());
    auto buttons = Buttons(rig.view.component());
    REQUIRE(buttons.size() == 1);
    CHECK(Press(rig.host, *buttons[0]));
    CHECK(rig.calls == std::vector<std::string>{"licenses"});
}

TEST_CASE("File picker entries select on click and navigate or open on double click",
          "[core][sakura][views]") {
    ModalRig rig;
    Mailbox mailbox;
    FilePickerView view;
    view.update(PickerState(mailbox, FilePickerMode::Open, "/srv/flows/radio"));
    rig.show(view.layer());
    CHECK(Peek::Frame(*view.layer().content).width == 720.0f - 24.0f);

    auto entries = Find<Sakura::NavigationItem>(*view.layer().content);
    REQUIRE(entries.size() == 2);
    CHECK(Press(rig.host, *entries[0]));
    REQUIRE(mailbox.last<MailFilePickerSelect>() != nullptr);
    CHECK(mailbox.last<MailFilePickerSelect>()->generation == 7);
    CHECK(mailbox.last<MailFilePickerSelect>()->path == "/srv/flows/radio/sub");

    mailbox.mails.clear();
    DoublePress(rig.host, *entries[0]);
    REQUIRE(mailbox.mails.size() == 1);
    CHECK(std::get<MailFilePickerNavigate>(mailbox.mails[0]).path == "/srv/flows/radio/sub");

    mailbox.mails.clear();
    DoublePress(rig.host, *entries[1]);
    REQUIRE(mailbox.mails.size() == 2);
    CHECK(std::get<MailFilePickerSelect>(mailbox.mails[0]).path == "/srv/flows/radio/a.yml");
    CHECK(std::get<MailFilePickerConfirm>(mailbox.mails[1]).generation == 7);
}

TEST_CASE("File picker open button needs a selected file and the close button cancels",
          "[core][sakura][views]") {
    ModalRig rig;
    Mailbox mailbox;
    FilePickerView view;
    view.update(PickerState(mailbox, FilePickerMode::Open, "/srv/flows"));
    rig.show(view.layer());

    auto buttons = Buttons(*view.layer().content);
    REQUIRE_FALSE(buttons.empty());
    Press(rig.host, *buttons.back());
    CHECK(mailbox.mails.empty());

    mailbox.state.filePicker.selectedPath = "/srv/flows/sub";
    view.update(FilePickerPresenter(mailbox.context()).build());
    rig.show(view.layer());
    Press(rig.host, *Buttons(*view.layer().content).back());
    CHECK(mailbox.mails.empty());

    mailbox.state.filePicker.selectedPath = "/srv/flows/a.yml";
    view.update(FilePickerPresenter(mailbox.context()).build());
    rig.show(view.layer());
    CHECK(Press(rig.host, *Buttons(*view.layer().content).back()));
    CHECK(mailbox.count<MailFilePickerConfirm>() == 1);

    CHECK(Press(rig.host, rig.close()));
    REQUIRE(mailbox.last<MailFilePickerCancel>() != nullptr);
    CHECK(mailbox.last<MailFilePickerCancel>()->generation == 7);
}

TEST_CASE("File picker breadcrumbs navigate to parent folders and collapse deep paths",
          "[core][sakura][views]") {
    ModalRig rig;
    Mailbox mailbox;
    FilePickerView view;
    view.update(PickerState(mailbox, FilePickerMode::Open, "/srv/flows/radio/fm"));
    rig.show(view.layer());

    auto buttons = Buttons(*view.layer().content);
    REQUIRE(buttons.size() == 3);
    CHECK(Press(rig.host, *buttons[1]));
    CHECK(Press(rig.host, *buttons[0]));
    REQUIRE(mailbox.mails.size() == 2);
    CHECK(std::get<MailFilePickerNavigate>(mailbox.mails[0]).path == "/srv/flows/radio");
    CHECK(std::get<MailFilePickerNavigate>(mailbox.mails[1]).path == "/srv/flows");

    mailbox.mails.clear();
    view.update(PickerState(mailbox, FilePickerMode::Open, "/srv/flows/a/b/c/d"));
    rig.show(view.layer());
    buttons = Buttons(*view.layer().content);
    REQUIRE(buttons.size() == 4);
    CHECK(Press(rig.host, *buttons[1]));
    CHECK(Press(rig.host, *buttons[2]));
    REQUIRE(mailbox.mails.size() == 2);
    CHECK(std::get<MailFilePickerNavigate>(mailbox.mails[0]).path == "/srv/flows/a/b");
    CHECK(std::get<MailFilePickerNavigate>(mailbox.mails[1]).path == "/srv/flows/a/b/c");
}

TEST_CASE("File picker save button follows the filename and the overwrite prompt", "[core][sakura][views]") {
    ModalRig rig;
    Mailbox mailbox;
    FilePickerView view;
    const auto confirm = [&] {
        view.update(FilePickerPresenter(mailbox.context()).build());
        rig.show(view.layer());
        CHECK(Find<Sakura::TextInput>(*view.layer().content).size() == 1);
        Press(rig.host, *Buttons(*view.layer().content).back());
        return mailbox.count<MailFilePickerConfirm>();
    };
    PickerState(mailbox, FilePickerMode::Save, "/srv/flows");

    CHECK(confirm() == 0);
    mailbox.state.filePicker.filename = "out.yml";
    CHECK(confirm() == 1);
    mailbox.state.filePicker.overwritePending = true;
    CHECK(confirm() == 2);
    mailbox.state.filePicker.overwritePending = false;
    mailbox.state.filePicker.filename.clear();
    CHECK(confirm() == 2);
    mailbox.state.filePicker.generation = 8;
    mailbox.state.filePicker.filename = "next.yml";
    CHECK(confirm() == 3);
    CHECK(mailbox.last<MailFilePickerConfirm>()->generation == 8);
}

TEST_CASE("Inactive file pickers and empty folders show no entries", "[core][sakura][views]") {
    ModalRig rig;
    Mailbox mailbox;
    FilePickerView view;
    view.update(FilePickerPresenter(mailbox.context()).build());
    CHECK(view.layer().content == nullptr);

    PickerState(mailbox, FilePickerMode::Open, "/srv/flows");
    mailbox.state.filePicker.entries.clear();
    view.update(FilePickerPresenter(mailbox.context()).build());
    rig.show(view.layer());
    CHECK(Find<Sakura::NavigationItem>(*view.layer().content).empty());
    CHECK(rig.modal.covers({10.0f, 10.0f}));
}

TEST_CASE("Benchmark modal runs and resets through the presenter", "[core][sakura][views]") {
    ModalRig rig;
    Mailbox mailbox;
    BenchmarkView view;
    const auto show = [&] {
        view.update(BenchmarkModalPresenter(mailbox.context()).build());
        rig.show({.content = &view.component()});
        return Buttons(view.component());
    };

    auto buttons = show();
    REQUIRE(buttons.size() == 2);
    CHECK(Press(rig.host, *buttons[0]));
    Press(rig.host, *buttons[1]);
    CHECK(mailbox.count<MailRunBenchmark>() == 1);
    CHECK(mailbox.count<MailResetBenchmark>() == 0);
    CHECK(Find<Sakura::Table>(view.component()).empty());

    mailbox.state.benchmark.results = {
        {"fft", {{.name = "cpu", .opsPerSec = 10.0, .msPerOp = 0.1, .error = 0.01}}},
        {"filter", {{.name = "cpu", .opsPerSec = 5.0, .msPerOp = 0.2, .error = 0.02}}},
    };
    buttons = show();
    CHECK(Press(rig.host, *buttons[1]));
    CHECK(mailbox.count<MailResetBenchmark>() == 1);
    CHECK(Find<Sakura::Table>(view.component()).size() == 2);

    mailbox.state.benchmark.running = true;
    buttons = show();
    Press(rig.host, *buttons[0]);
    Press(rig.host, *buttons[1]);
    CHECK(mailbox.count<MailRunBenchmark>() == 1);
    CHECK(mailbox.count<MailResetBenchmark>() == 1);
    CHECK(Find<Sakura::ProgressBar>(view.component()).size() == 1);
}

TEST_CASE("Benchmark module combo reports the chosen module", "[core][sakura][views]") {
    Scene scene{kWide};
    std::vector<std::string> modules;
    BenchmarkView view;
    view.update({
        .moduleOptions = {"All", "FFT", "Filter"},
        .selectedModule = "All",
        .onModuleChange = [&](const std::string& module) { modules.push_back(module); },
    });
    Stage stage(view.component(), {100.0f, 100.0f, 600.0f, 700.0f});
    stage.place(scene.ctx);

    auto combos = Find<Sakura::Combo>(view.component());
    REQUIRE(combos.size() == 1);
    Choose(stage.host, scene.ctx, kWide, *combos[0], 2);
    CHECK(modules == std::vector<std::string>{"Filter"});
}

TEST_CASE("Remote modal starts streaming or opens the remote settings", "[core][sakura][views]") {
    ModalRig rig;
    Mailbox mailbox;
    mailbox.state.remote.config.broker.address = "127.0.0.1";
    mailbox.state.remote.config.broker.port = 9000;
    mailbox.state.remote.config.video.framerate = 60;
    RemoteView view;
    view.update(RemoteStreamingModalPresenter(mailbox.context()).build());
    rig.show({.content = &view.component()});

    auto buttons = Buttons(view.component());
    REQUIRE(buttons.size() == 2);
    CHECK(Press(rig.host, *buttons[0]));
    CHECK(Press(rig.host, *buttons[1]));
    REQUIRE(mailbox.mails.size() == 3);
    const auto& start = std::get<MailStartRemote>(mailbox.mails[0]);
    CHECK(start.config.broker.address == "127.0.0.1");
    CHECK(start.config.broker.port == 9000);
    CHECK(start.config.video.framerate == 60);
    CHECK(std::holds_alternative<MailCloseModal>(mailbox.mails[1]));
    CHECK(std::get<MailOpenSettings>(mailbox.mails[2]).section == SettingsSection::Remote);
}

TEST_CASE("Remote modal shares the invite and approves only pending clients", "[core][sakura][views]") {
    ModalRig rig;
    std::vector<std::string> calls;
    RemoteView view;
    view.update({
        .started = true,
        .inviteUrl = "https://cyberether.org/remote/abc",
        .accessToken = "token-123",
        .clients = {{.sessionId = "approved-777777"}, {.sessionId = "session-abcdef12"}},
        .waitlist = {"session-abcdef12", "zz-pending9"},
        .onStop = [&] { calls.push_back("stop"); },
        .onApprove = [&](const std::string& code) { calls.push_back("approve:" + code); },
        .onOpenUrl = [&](const std::string& url) { calls.push_back("open:" + url); },
        .onCopy = [&](const std::string& label, const std::string& value) {
            calls.push_back("copy:" + label + ":" + value);
        },
    });
    rig.show({.content = &view.component()});

    auto codes = Find<Sakura::QrCode>(view.component());
    REQUIRE(codes.size() == 1);
    CHECK(Press(rig.host, *codes[0]));

    auto buttons = Buttons(view.component());
    REQUIRE(buttons.size() == 6);
    for (auto* button : buttons) {
        Press(rig.host, *button);
    }
    CHECK(calls == std::vector<std::string>{
                       "open:https://cyberether.org/remote/abc",
                       "open:https://cyberether.org/remote/abc",
                       "copy:Access token:token-123",
                       "approve:CDEF12",
                       "approve:NDING9",
                       "stop",
                   });
}

TEST_CASE("Dependency review installs approved requests through the presenter", "[core][sakura][views]") {
    ModalRig rig;
    Mailbox mailbox;
    auto& request = mailbox.state.runtime.dependencyRequest;
    request.generation = 3;
    request.state = PythonDependencyRequestState::ApprovalRequired;
    request.dependencies = {{.requirement = "numpy>=1.0", .block = "fft-python"}, {.requirement = "scipy"}};
    mailbox.state.runtime.dependencyGeneration = 3;
    DependencyReviewView view;
    const auto show = [&] {
        const auto config = DependencyReviewModalPresenter(mailbox.context()).build();
        view.update(config);
        rig.show({.content = &view.component(), .size = Extent2D<F32>{760.0f, 0.0f}});
        return config;
    };

    auto config = show();
    REQUIRE(config.dependencies.size() == 2);
    CHECK(config.dependencies[0].requestedBy == "fft");
    CHECK(config.installEnabled);
    CHECK(Find<Sakura::Table>(view.component()).size() == 1);
    auto buttons = Buttons(view.component());
    REQUIRE(buttons.size() == 1);
    CHECK(Press(rig.host, *buttons[0]));
    REQUIRE(mailbox.last<MailInstallPythonDependencies>() != nullptr);
    CHECK(mailbox.last<MailInstallPythonDependencies>()->generation == 3);

    mailbox.mails.clear();
    mailbox.state.runtime.dependencyGeneration = 4;
    config = show();
    CHECK(config.stale);
    Press(rig.host, *Buttons(view.component()).at(0));
    CHECK(mailbox.mails.empty());
}

TEST_CASE("Dependency review layout follows the install state", "[core][sakura][views]") {
    using State = DependencyReviewView::State;
    ModalRig rig;
    DependencyReviewView view;
    const auto show = [&](DependencyReviewView::Config config) {
        config.dependencies = {{.requirement = "numpy", .requestedBy = "graph / fft"}};
        view.update(std::move(config));
        rig.show({.content = &view.component(), .size = Extent2D<F32>{760.0f, 0.0f}});
    };
    const auto tables = [&] {
        return Find<Sakura::Table>(view.component()).size();
    };
    const auto logs = [&] {
        return Find<Sakura::LogView>(view.component()).size();
    };

    show({.state = State::Review, .installEnabled = true});
    CHECK(tables() == 1);
    CHECK(logs() == 0);
    CHECK(Buttons(view.component()).size() == 1);

    show({.state = State::Installing});
    CHECK(tables() == 0);
    CHECK(logs() == 1);
    CHECK(Buttons(view.component()).empty());

    show({.state = State::Installed});
    CHECK(tables() == 0);
    CHECK(Buttons(view.component()).empty());

    show({.state = State::Denied});
    CHECK(tables() == 1);
    CHECK(Buttons(view.component()).empty());

    show({.state = State::Failed, .approved = true, .installEnabled = true});
    CHECK(tables() == 0);
    CHECK(logs() == 1);
    CHECK(Buttons(view.component()).size() == 1);

    show({.state = State::Failed, .approved = false, .installEnabled = true});
    CHECK(tables() == 1);
    CHECK(Buttons(view.component()).size() == 1);

    view.update({.state = State::Review});
    rig.place();
    CHECK(tables() == 0);
}

TEST_CASE("Flowgraph info modal saves to the browsed path and resets it for another flowgraph",
          "[core][sakura][views]") {
    ModalRig rig;
    std::vector<std::string> saved;
    std::vector<std::string> browsed;
    FlowgraphInfoView view;
    FlowgraphInfoView::Config config{
        .flowgraphId = "a",
        .title = "Radio",
        .path = "/flows/a.yml",
        .onBrowse = [&](const std::string& current, std::function<void(std::string)> onSelect) {
            browsed.push_back(current);
            onSelect("/flows/b.yml");
        },
        .onSave = [&](const std::string& path) { saved.push_back(path); },
    };
    const auto show = [&] {
        view.update(config);
        rig.show({.content = &view.component()});
        return Buttons(view.component());
    };

    auto buttons = show();
    REQUIRE(buttons.size() == 2);
    CHECK(Find<Sakura::TextInput>(view.component()).size() == 6);
    CHECK(Press(rig.host, *buttons[1]));
    CHECK(Press(rig.host, *buttons[0]));
    buttons = show();
    CHECK(Press(rig.host, *buttons[1]));
    CHECK(browsed == std::vector<std::string>{"/flows/a.yml"});

    config.path = "/flows/c.yml";
    buttons = show();
    CHECK(Press(rig.host, *buttons[1]));

    config.flowgraphId = "c";
    buttons = show();
    CHECK(Press(rig.host, *buttons[1]));
    CHECK(saved == std::vector<std::string>{"/flows/a.yml", "/flows/b.yml", "/flows/b.yml", "/flows/c.yml"});
}

TEST_CASE("Flowgraph examples modal opens the clicked example", "[core][sakura][views]") {
    ModalRig rig;
    Mailbox mailbox;
    std::vector<std::string> opened;
    FlowgraphExamplesView view;
    view.update({
        .examples = {
            {.key = "fm", .title = "FM Radio", .summary = "Listen to FM"},
            {.key = "adsb", .title = "ADS-B", .summary = "Track aircraft"},
            {.key = "scope", .title = "Scope", .summary = "Plot samples"},
        },
        .onOpen = [&](const FlowgraphExamplesView::Example& example) { opened.push_back(example.key); },
    });
    rig.show({.content = &view.component()});

    auto cells = Find<FlowgraphExampleCell>(view.component());
    REQUIRE(cells.size() == 3);
    CHECK(Peek::Frame(*cells[1]).x > Peek::Frame(*cells[0]).right());
    CHECK(Peek::Frame(*cells[2]).y > Peek::Frame(*cells[0]).bottom());
    CHECK(Press(rig.host, *cells[2]));
    CHECK(Press(rig.host, *cells[0]));
    CHECK(opened == std::vector<std::string>{"scope", "fm"});

    view.update({});
    rig.place();
    CHECK(Find<FlowgraphExampleCell>(view.component()).empty());

    const auto config = FlowgraphExamplesModalPresenter(mailbox.context()).build();
    config.onOpen({.key = "fm", .content = "graph: {}"});
    REQUIRE(mailbox.mails.size() == 2);
    CHECK(std::get<MailOpenFlowgraphBlob>(mailbox.mails[0]).blob.size() == 9);
    CHECK(std::holds_alternative<MailCloseModal>(mailbox.mails[1]));
}

TEST_CASE("Feedback modal submits only within the length limits and while idle", "[core][sakura][views]") {
    ModalRig rig;
    U64 submitted = 0;
    FeedbackView view;
    const auto submit = [&](std::string text, FeedbackView::Config::Status status) {
        view.update({.text = std::move(text), .minLength = 10, .maxLength = 40, .status = status,
                     .onSubmit = [&] { ++submitted; }});
        rig.show({.content = &view.component(), .size = Extent2D<F32>{520.0f, 0.0f}});
        auto buttons = Buttons(view.component());
        REQUIRE(buttons.size() == 1);
        Press(rig.host, *buttons[0]);
        return submitted;
    };

    CHECK(submit("too short", FeedbackView::Config::Status::Idle) == 0);
    CHECK(submit("long enough feedback", FeedbackView::Config::Status::Idle) == 1);
    CHECK(submit(std::string(41, 'x'), FeedbackView::Config::Status::Idle) == 1);
    CHECK(submit("long enough feedback", FeedbackView::Config::Status::Submitting) == 1);
}

TEST_CASE("Modal views settle under identical updates", "[core][sakura][views]") {
    ModalRig rig;
    Mailbox mailbox;
    mailbox.state.remote.started = true;
    mailbox.state.remote.inviteUrl = "https://cyberether.org/remote/abc";
    mailbox.state.remote.waitlist = {"session-1"};
    mailbox.state.benchmark.results = {{"fft", {{.name = "cpu", .opsPerSec = 1.0, .msPerOp = 1.0, .error = 0.0}}}};
    mailbox.state.runtime.dependencyRequest = {
        .generation = 1,
        .state = PythonDependencyRequestState::ApprovalRequired,
        .dependencies = {{.requirement = "numpy"}},
    };
    mailbox.state.runtime.dependencyGeneration = 1;
    Frames frames(rig.host, kWide, rig.scene.ctx);
    ModalView view;

    for (const auto content : {ModalContent::About, ModalContent::RemoteStreaming, ModalContent::Benchmark,
                               ModalContent::Dependencies}) {
        mailbox.state.modal.content = content;
        const auto update = [&] {
            view.update(ModalPresenter(mailbox.context()).build());
            rig.modal.update(view.layer());
        };
        CHECK(Settles(frames, update));
    }
}

TEST_CASE("Modal layer opens, closes, and reopens through its close button without rebuilding",
          "[core][sakura][views]") {
    KeyedModalRig rig;
    Mailbox mailbox;
    ModalView view;
    rig.update = [&] {
        if (mailbox.count<MailCloseModal>() > 0) {
            mailbox.state.modal.content.reset();
        }
        mailbox.mails.clear();
        view.update(ModalPresenter(mailbox.context()).build());
        rig.modal.update(view.layer());
    };
    const auto step = [&] {
        return rig.frame();
    };
    const Extent2D<F32> away = {20.0f, 20.0f};
    REQUIRE(rig.settles());
    const U64 rebuilds = rig.frames.rebuilds;

    for (U64 round = 0; round < 2; ++round) {
        CAPTURE(round);
        mailbox.state.modal.content = ModalContent::About;
        const auto opened = rig.frame();
        CHECK_FALSE(opened.rebuilt);
        CHECK(opened.repainted);
        CHECK(rig.settles());
        CHECK(rig.move(away).clean());

        auto& close = *Find<Sakura::Button>(rig.modal).back();
        CheckHover(rig.host, step, close, away);
        CHECK(Press(rig.host, close));
        const auto closed = rig.frame();
        CHECK_FALSE(closed.rebuilt);
        CHECK(closed.repainted);
        CHECK_FALSE(mailbox.state.modal.content.has_value());
        CHECK_FALSE(rig.modal.covers({800.0f, 600.0f}));
        CHECK(rig.settles());
    }
    CHECK(rig.frames.rebuilds == rebuilds);
}

TEST_CASE("Close, rename, and feedback modal buttons and fields repaint without rebuilding and settle",
          "[core][sakura][views]") {
    KeyedModalRig rig;
    const auto step = [&] {
        return rig.frame();
    };
    const Extent2D<F32> away = {20.0f, 20.0f};
    std::vector<std::string> calls;
    U64 closes = 0;

    FlowgraphCloseView closing;
    rig.update = [&] {
        closing.update({
            .onSave = [&] { calls.push_back("save"); },
            .onDontSave = [&] { calls.push_back("discard"); },
        });
        rig.modal.update({.content = &closing.component(), .onClose = [&] { ++closes; }});
    };
    REQUIRE(rig.settles());
    const U64 rebuilds = rig.frames.rebuilds;
    auto buttons = Buttons(closing.component());
    REQUIRE(buttons.size() == 2);
    buttons.push_back(Find<Sakura::Button>(rig.modal).back());
    for (auto* button : buttons) {
        CheckHover(rig.host, step, *button, away);
        CHECK(Press(rig.host, *button));
        CHECK(Kept(rig.frame()));
        CHECK(rig.settles());
    }
    CHECK(calls == std::vector<std::string>{"save", "discard"});
    CHECK(closes == 1);

    RenameBlockView rename;
    rig.update = [&] {
        rename.update({.oldName = "fft", .onRename = [&](const std::string& name) { calls.push_back(name); }});
        rig.modal.update({.content = &rename.component(), .onClose = [&] { ++closes; }});
    };
    const auto mounted = rig.frame();
    CHECK_FALSE(mounted.rebuilt);
    CHECK(mounted.repainted);
    CHECK(rig.settles());
    auto inputs = Find<Sakura::TextInput>(rename.component());
    REQUIRE(inputs.size() == 1);
    CHECK(inputs[0]->focused());
    auto& confirm = *Buttons(rename.component()).at(0);
    CHECK(rig.move(Center(confirm)).clean());
    CHECK(rig.move(away).clean());
    CHECK(RepaintOnly(rig.click(Center(*inputs[0]))));
    CHECK(rig.settles());
    CHECK(RepaintOnly(rig.type("2")));
    CHECK(rig.settles());
    CheckHover(rig.host, step, confirm, away);
    CHECK(Press(rig.host, confirm));
    CHECK(Kept(rig.frame()));
    CHECK(rig.settles());
    CHECK(calls.back() == "fft2");

    FeedbackView feedback;
    std::string text;
    U64 submitted = 0;
    rig.update = [&] {
        feedback.update({
            .text = text,
            .minLength = 10,
            .maxLength = 40,
            .onTextChange = [&](const std::string& value) { text = value; },
            .onSubmit = [&] { ++submitted; },
        });
        rig.modal.update({.content = &feedback.component(), .size = Extent2D<F32>{520.0f, 0.0f}});
    };
    CHECK_FALSE(rig.frame().rebuilt);
    CHECK(rig.settles());
    auto editors = Find<Sakura::TextGrid>(feedback.component());
    REQUIRE_FALSE(editors.empty());
    auto& submit = *Buttons(feedback.component()).at(0);
    CHECK(rig.move(Center(submit)).clean());
    CHECK(rig.move(away).clean());
    CHECK(RepaintOnly(rig.click(Center(*editors[0]))));
    CHECK(rig.settles());
    CHECK(RepaintOnly(rig.type("long enough feedback")));
    CHECK(rig.settles());
    CHECK(text == "long enough feedback");
    CHECK(RepaintOnly(rig.click(away)));
    CHECK(rig.settles());
    CheckHover(rig.host, step, submit, away);
    CHECK(Press(rig.host, submit));
    CHECK(Kept(rig.frame()));
    CHECK(rig.settles());
    CHECK(submitted == 1);
    CHECK(rig.frames.rebuilds == rebuilds);
}

TEST_CASE("Settings modal settles under identical updates", "[core][sakura][views]") {
    SettingsRig rig;
    Frames frames(rig.host, kWide, rig.scene.ctx);
    for (const auto section : {SettingsSection::General, SettingsSection::Registry, SettingsSection::About}) {
        const auto update = [&] {
            rig.config.section = section;
            rig.view.update(rig.config);
            rig.modal.update({.content = &rig.view.component(), .size = Extent2D<F32>{1200.0f, 900.0f}});
        };
        CHECK(Settles(frames, update));
    }
}

TEST_CASE("Settings navigation, toggles, combos, and text fields repaint without rebuilding and settle",
          "[core][sakura][views]") {
    KeyedModalRig rig;
    SettingsView::Config config;
    config.general = {
        .themes = {"Dark", "Light"},
        .currentThemeKey = "Dark",
        .renderer = "Metal",
        .onThemeChange = [&](const std::string& value) { config.general.currentThemeKey = value; },
        .onBackgroundParticlesChange = [&](bool value) { config.general.backgroundParticles = value; },
    };
    config.remote = {
        .broker = {.address = "127.0.0.1", .port = 8686},
        .onBrokerChange = [&](const Instance::Remote::Config::Broker& value) { config.remote.broker = value; },
    };
    config.onSectionChange = [&](SettingsSection section) { config.section = section; };
    SettingsView view;
    rig.update = [&] {
        view.update(config);
        rig.modal.update({.content = &view.component(), .size = Extent2D<F32>{1200.0f, 900.0f}});
    };
    const auto step = [&] {
        return rig.frame();
    };
    const Extent2D<F32> away = {20.0f, 20.0f};
    REQUIRE(rig.settles());
    const U64 rebuilds = rig.frames.rebuilds;

    auto sections = Find<Sakura::SidebarItem>(view.component());
    REQUIRE(sections.size() == 8);
    CheckHover(rig.host, step, *sections[1], away);

    for (const auto section : {SettingsSection::Remote, SettingsSection::Runtime}) {
        CAPTURE(static_cast<int>(section));
        CHECK(Press(rig.host, *sections[static_cast<U64>(section)]));
        CHECK(config.section == section);
        const auto mounted = rig.frame();
        CHECK_FALSE(mounted.rebuilt);
        CHECK(mounted.repainted);
        CHECK(rig.settles());
    }
    CHECK(Press(rig.host, *sections[static_cast<U64>(SettingsSection::General)]));
    CHECK(RepaintOnly(rig.frame()));
    CHECK(rig.settles());
    CHECK(rig.move(away).clean());

    auto toggles = Find<Sakura::Toggle>(view.component());
    REQUIRE(toggles.size() == 1);
    CheckHover(rig.host, step, *toggles[0], away);
    CHECK(Press(rig.host, *toggles[0]));
    CHECK(config.general.backgroundParticles);
    CHECK(RepaintOnly(rig.frame()));
    CHECK(rig.settles());
    CHECK(rig.move(away).clean());

    auto& theme = *Find<Sakura::Combo>(view.component()).at(0);
    const Jetstream::Rect combo = Peek::Frame(theme);
    CheckHover(rig.host, step, theme, away);
    CHECK(Press(rig.host, theme));
    CHECK(RepaintOnly(rig.frame()));
    CHECK(rig.settles());
    REQUIRE_FALSE(PopupFrames(rig.host).empty());
    rig.host.click(combo.x + combo.width * 0.5f, combo.bottom() + 8.0f + 23.0f + 11.5f);
    CHECK(config.general.currentThemeKey == "Light");
    CHECK(RepaintOnly(rig.frame()));
    CHECK(rig.settles());
    CHECK(PopupFrames(rig.host).empty());

    CHECK(Press(rig.host, theme));
    CHECK(RepaintOnly(rig.frame()));
    CHECK(rig.settles());
    REQUIRE_FALSE(PopupFrames(rig.host).empty());
    CHECK(RepaintOnly(rig.click(away)));
    CHECK(rig.settles());
    CHECK(PopupFrames(rig.host).empty());
    CHECK(config.general.currentThemeKey == "Light");

    CHECK(Press(rig.host, *sections[static_cast<U64>(SettingsSection::Remote)]));
    CHECK(RepaintOnly(rig.frame()));
    CHECK(rig.settles());
    auto inputs = Find<Sakura::TextInput>(view.component());
    REQUIRE_FALSE(inputs.empty());
    auto& broker = *inputs[0];
    const auto shown = [&] {
        const auto center = Center(broker);
        return Peek::Clip(broker).contains(center.x, center.y);
    };
    const auto areas = Find<Sakura::ScrollArea>(view.component());
    REQUIRE(areas.size() == 2);
    const auto editor = Center(*areas[1]);
    for (int notch = 0; notch < 40 && !shown(); ++notch) {
        rig.host.send(Wheel(editor.x, editor.y, 0.0f, -1.0f));
        CHECK_FALSE(rig.frame().rebuilt);
    }
    REQUIRE(shown());
    CHECK(rig.settles());
    CHECK(RepaintOnly(rig.click(Center(broker))));
    CHECK(broker.focused());
    CHECK(rig.settles());
    CHECK(RepaintOnly(rig.type("1")));
    CHECK(config.remote.broker.address == "127.0.0.1");
    CHECK(rig.settles());
    CHECK(RepaintOnly(rig.click(away)));
    CHECK_FALSE(broker.focused());
    CHECK(rig.settles());
    CHECK(config.remote.broker.address == "127.0.0.11");
    CHECK(rig.frames.rebuilds == rebuilds);
}

TEST_CASE("File picker modal settles under identical updates", "[core][sakura][views]") {
    ModalRig rig;
    Mailbox mailbox;
    PickerState(mailbox, FilePickerMode::Open, "/srv/flows/radio");
    FilePickerView view;
    Frames frames(rig.host, kWide, rig.scene.ctx);
    const auto update = [&] {
        view.update(FilePickerPresenter(mailbox.context()).build());
        rig.modal.update(view.layer());
    };
    CHECK(Settles(frames, update));
}

TEST_CASE("File picker hover, selection, navigation, filename typing, and scrolling repaint without rebuilding",
          "[core][sakura][views]") {
    KeyedModalRig rig;
    Mailbox mailbox;
    auto& picker = mailbox.state.filePicker;
    const auto list = [&](const std::string& directory, U64 count) {
        picker.directory = directory;
        picker.entries = {{.name = "sub", .path = directory + "/sub", .directory = true}};
        for (U64 index = 1; index < count; ++index) {
            const std::string name = "flow" + std::to_string(index) + ".yml";
            picker.entries.push_back({.name = name, .path = directory + "/" + name});
        }
    };
    PickerState(mailbox, FilePickerMode::Save, "/srv/flows");
    list("/srv/flows", 40);
    FilePickerView view;
    rig.update = [&] {
        for (const auto& mail : mailbox.mails) {
            if (const auto* select = std::get_if<MailFilePickerSelect>(&mail)) {
                picker.selectedPath = select->path;
            } else if (const auto* navigate = std::get_if<MailFilePickerNavigate>(&mail)) {
                list(navigate->path, 44);
            } else if (const auto* filename = std::get_if<MailFilePickerSetFilename>(&mail)) {
                picker.filename = filename->value;
            }
        }
        mailbox.mails.clear();
        view.update(FilePickerPresenter(mailbox.context()).build());
        rig.modal.update(view.layer());
    };
    const auto step = [&] {
        return rig.frame();
    };
    const Extent2D<F32> away = {20.0f, 20.0f};
    REQUIRE(rig.settles());
    const U64 rebuilds = rig.frames.rebuilds;
    auto& content = *view.layer().content;

    auto entries = Find<Sakura::NavigationItem>(content);
    REQUIRE(entries.size() == 40);
    CheckHover(rig.host, step, *entries[1], away);
    CHECK(Press(rig.host, *entries[1]));
    CHECK(RepaintOnly(rig.frame()));
    CHECK(picker.selectedPath == "/srv/flows/flow1.yml");
    CHECK(rig.settles());

    const auto navigated = rig.click(Center(*entries[0]), 2);
    CHECK_FALSE(navigated.rebuilt);
    CHECK(navigated.repainted);
    CHECK(picker.directory == "/srv/flows/sub");
    CHECK(rig.settles());
    entries = Find<Sakura::NavigationItem>(content);
    REQUIRE(entries.size() == 44);
    CHECK(rig.move(away).clean());

    auto inputs = Find<Sakura::TextInput>(content);
    REQUIRE(inputs.size() == 1);
    auto& filename = *inputs[0];
    CHECK(RepaintOnly(rig.click(Center(filename))));
    CHECK(filename.focused());
    CHECK(rig.settles());
    CHECK(RepaintOnly(rig.type("out")));
    CHECK(rig.settles());
    CHECK(picker.filename == "out");
    CHECK(RepaintOnly(rig.click(away)));
    CHECK_FALSE(filename.focused());
    CHECK(rig.settles());
    CHECK(picker.filename == "out");

    auto grids = Find<Sakura::Grid>(content);
    REQUIRE(grids.size() == 1);
    const auto scroller = Center(*grids[0]);
    const F32 top = Peek::Frame(*entries[0]).y;
    rig.host.send(Wheel(scroller.x, scroller.y, -1.0f));
    CHECK(RepaintOnly(rig.frame()));
    CHECK(Peek::Frame(*entries[0]).y < top);
    CHECK(rig.settles());
    CHECK(rig.frames.rebuilds == rebuilds);
}

TEST_CASE("Workbench view accepts welcome, modal, and file pending configs without rendering",
          "[core][sakura][views]") {
    Mailbox mailbox;
    std::vector<TitlebarView::Action> actions;
    WorkbenchView view;
    WorkbenchView::Config config;
    config.titlebar = TitlebarPresenter(mailbox.context()).build();
    config.titlebar.onAction = [&actions](TitlebarView::Action action) {
        actions.push_back(action);
    };
    config.welcomeHud = WelcomeHudPresenter(mailbox.context()).build();
    CHECK_NOTHROW(view.update(config));

    mailbox.state.modal.content = ModalContent::About;
    config.welcomeHud.reset();
    config.modal = ModalPresenter(mailbox.context()).build();
    CHECK_NOTHROW(view.update(config));

    config.filePending = true;
    CHECK_NOTHROW(view.update(config));
    CHECK(actions.empty());
}

TEST_CASE("Example and info views settle under identical updates",
          "[core][sakura][views]") {
    Scene scene{kWide};
    FlowgraphExamplesView examples;
    FlowgraphInfoView info;
    LayerSlot examplesSlot;
    LayerSlot infoSlot;
    LayerHost host;
    Frames frames(host, kWide, scene.ctx);
    const auto update = [&] {
        examples.update({.examples = {{.key = "fm", .title = "FM", .summary = "Radio"}}});
        info.update({.flowgraphId = "a", .title = "Radio", .path = "/flows/a.yml"});
    };
    update();
    examplesSlot.hold(examples.component(), {800.0f, 0.0f, 800.0f, 600.0f});
    infoSlot.hold(info.component(), {0.0f, 600.0f, 800.0f, 600.0f});
    host.setLayers({&examplesSlot, &infoSlot});
    CHECK(Settles(frames, update));
}

TEST_CASE("Update button sits in the top right corner of the title row only while an update is pending",
          "[core][sakura][views]") {
    using Action = TitlebarView::Action;
    TitleRig rig;
    CHECK(Indicator(rig.shell.bar(), ICON_FA_CIRCLE_ARROW_UP) == nullptr);

    rig.config.updateAvailable = true;
    rig.update();
    rig.place();
    auto* button = Indicator(rig.shell.bar(), ICON_FA_CIRCLE_ARROW_UP);
    REQUIRE(button != nullptr);
    const auto frame = Peek::Frame(*button);
    const auto plus = Peek::Frame(rig.shell.bar().trailing());
    CHECK(std::abs(frame.right() + TitlebarView::IndicatorGap - Peek::Frame(Help(rig.shell.bar())).x) <= 0.5f);
    CHECK(frame.x > plus.right());
    CHECK(frame.height == 28.0f);
    std::optional<F32> closeCenter;
    for (auto* child : Peek::Children(rig.shell.dock())) {
        if (auto* text = dynamic_cast<Sakura::Text*>(child); text && text->config().str == ICON_FA_XMARK) {
            const auto glyph = Peek::Frame(*text);
            closeCenter = glyph.y + glyph.height * 0.5f;
        }
    }
    REQUIRE(closeCenter.has_value());
    CHECK(std::abs(frame.y + frame.height * 0.5f - *closeCenter) <= 0.5f);
    bool label = false;
    for (auto* child : Peek::Children(*button)) {
        if (auto* text = dynamic_cast<Sakura::Text*>(child)) {
            label = label || text->config().str == "Update Available";
            CHECK(text->config().colorKey == "action_btn_text");
        }
        if (auto* box = dynamic_cast<Sakura::Box*>(child); box && box->config().softness == 0.0f &&
                                                             box->config().borderWidth == 0.0f) {
            CHECK(box->config().instances.front().backgroundColor == rig.scene.ctx.color("action_btn"));
        }
    }
    CHECK(label);

    rig.host.click(frame.x + frame.width * 0.5f, frame.y + frame.height * 0.5f);
    rig.place();
    CHECK(rig.actions == std::vector<Action>{Action::OpenUpdateModal});
    CHECK(rig.presses.empty());

    rig.config.updateAvailable = false;
    rig.config.updateReady = true;
    rig.update();
    rig.place();
    REQUIRE(Indicator(rig.shell.bar(), ICON_FA_CIRCLE_ARROW_UP) == button);
    bool ready = false;
    for (auto* child : Peek::Children(*button)) {
        if (auto* text = dynamic_cast<Sakura::Text*>(child)) {
            ready = ready || text->config().str == "Update Ready";
        }
    }
    CHECK(ready);

    rig.config.updateReady = false;
    rig.update();
    rig.place();
    CHECK(Indicator(rig.shell.bar(), ICON_FA_CIRCLE_ARROW_UP) == nullptr);
    CHECK(Indicator(rig.shell.bar(), ICON_FA_CIRCLE_QUESTION) != nullptr);
}

TEST_CASE("Help button is always in the top right corner, faint, and opens feedback",
          "[core][sakura][views]") {
    using Action = TitlebarView::Action;
    TitleRig rig;
    auto& help = Help(rig.shell.bar());
    const auto frame = Peek::Frame(help);
    CHECK(frame.width == 28.0f);
    CHECK(frame.height == 28.0f);
    CHECK(frame.right() <= 1200.0f);
    CHECK(frame.right() >= 1185.0f);

    std::vector<Sakura::Box*> boxes;
    Sakura::Text* glyph = nullptr;
    for (auto* child : Peek::Children(help)) {
        if (auto* box = dynamic_cast<Sakura::Box*>(child)) {
            boxes.push_back(box);
        }
        if (auto* text = dynamic_cast<Sakura::Text*>(child); text && text->visible()) {
            glyph = text;
        }
    }
    REQUIRE(boxes.size() == 3);
    REQUIRE(glyph);
    CHECK(glyph->config().str == ICON_FA_CIRCLE_QUESTION);
    CHECK(glyph->config().colorKey == "text_secondary");
    const auto glyphFrame = Peek::Frame(*glyph);
    CHECK(std::abs(glyphFrame.x + glyphFrame.width * 0.5f - (frame.x + 14.0f)) <= 0.5f);
    CHECK(boxes[1]->config().instances.front().backgroundColor.a == 0.0f);
    CHECK_FALSE(boxes[0]->config().instances.front().visible);
    CHECK_FALSE(boxes[2]->config().instances.front().visible);

    const F32 cx = frame.x + 14.0f;
    const F32 cy = frame.y + 14.0f;
    rig.host.send(Mouse(MouseEventType::Move, cx, cy));
    rig.place();
    CHECK(boxes[1]->config().instances.front().backgroundColor == rig.scene.ctx.color("chrome_button_hover"));
    CHECK(boxes[1]->config().cornerRadius == 14.0f);

    rig.host.click(cx, cy);
    rig.place();
    CHECK(rig.actions == std::vector<Action>{Action::SendFeedback});
    CHECK(rig.presses.empty());
}
