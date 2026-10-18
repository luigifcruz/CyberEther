#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <jetstream/render/sakura/components/controls.hh>
#include <jetstream/render/sakura/components/image.hh>
#include <jetstream/render/sakura/components/node/editor.hh>
#include <jetstream/render/sakura/components/shell.hh>
#include <jetstream/render/sakura/components/shortcuts.hh>
#include <jetstream/render/sakura/components/table.hh>

#include "render/sakura/detail/input_router.hh"
#include "render/sakura/detail/root_component.hh"
#include "harness.hh"

#include <functional>
#include <string>
#include <vector>

using namespace Jetstream;

namespace {

struct InputProbe : Sakura::Component {
    Rect placed;
    U64 keys = 0;
    U64 texts = 0;
    U64 shortcuts = 0;
    U64 blurs = 0;

    bool focus() { return requestFocus(); }
    bool focused() const { return hasFocus(); }
    Extent2D<F32> measure(const Sakura::Context&, Extent2D<F32>) override { return {200, 80}; }
    void layout(const Sakura::Context&) override { placed = frame(); }
    bool keyEvent(const KeyEvent&) override { ++keys; return true; }
    bool textEvent(const TextEvent&) override { ++texts; return true; }
    bool shortcutEvent(const KeyEvent&) override { ++shortcuts; return false; }
    void focusChanged(bool focused) override { blurs += !focused; }
    bool editsText() const override { return true; }
};

struct FrameProbe : Sakura::Component {
    static Rect of(const Sakura::Component& component) {
        return (component.*(&FrameProbe::frame))();
    }
    static U64 shownFrames(const Sakura::Component& component) {
        U64 count = 0;
        for (auto* child : (component.*(&FrameProbe::children))()) {
            if (const auto* box = dynamic_cast<const Sakura::Box*>(child); box && box->visible()) {
                count += std::any_of(box->config().instances.begin(), box->config().instances.end(),
                                     [](const auto& instance) { return instance.visible; });
            }
        }
        return count;
    }
};

struct FocusProbe : InputProbe {
    U64 gains = 0;
    std::function<void(bool)> onFocus;

    void focusChanged(bool focused) override {
        InputProbe::focusChanged(focused);
        gains += focused;
        const auto callback = onFocus;
        if (callback) callback(focused);
    }
};

struct FocusGroup : InputProbe {
    void attach(Sakura::Component& component) { addChild(component); }
    void detach(Sakura::Component& component) { removeChild(component); }
    void layout(const Sakura::Context& ctx) override {
        InputProbe::layout(ctx);
        for (auto* child : children()) layoutChild(ctx, *child, frame());
    }
};

struct PointerProbe : InputProbe {
    std::vector<MouseEvent> pointerEvents;
    SurfaceCursor cursor = SurfaceCursor::Hand;
    bool held = false;
    U64 cancellations = 0;

    bool mouseEvent(const MouseEvent& event) override {
        if (event.type == MouseEventType::Cancel || event.type == MouseEventType::Leave) {
            cancellations += held;
            held = false;
            return false;
        }
        if ((event.type == MouseEventType::Click || event.type == MouseEventType::Scroll) &&
            !placed.contains(event.position.x, event.position.y)) return false;
        if (event.type == MouseEventType::Click) held = true;
        pointerEvents.push_back(event);
        return true;
    }
    std::optional<SurfaceCursor> cursorAt(const Extent2D<F32>& point) const override {
        return placed.contains(point.x, point.y) ? std::optional(cursor) : std::nullopt;
    }
    U64 count(MouseEventType type) const {
        return std::count_if(pointerEvents.begin(), pointerEvents.end(), [&](const auto& event) { return event.type == type; });
    }
};

struct PointerBody : Sakura::Component {
    Sakura::Component& content;
    Sakura::Popup popup;
    Extent2D<F32> anchor = {50, 10};

    explicit PointerBody(Sakura::Component& content) : content(content) {
        addChild(content);
        addChild(popup);
    }
    Extent2D<F32> measure(const Sakura::Context&, Extent2D<F32>) override { return {240, 120}; }
    void layout(const Sakura::Context& ctx) override {
        const auto bounds = frame();
        layoutChild(ctx, content, bounds);
        layoutChild(ctx, popup, {bounds.x + anchor.x, bounds.y + anchor.y, 20, 20});
    }
};

struct ModalBody : Sakura::Component {
    Sakura::TextInput field;
    Sakura::Popup popup;
    Sakura::Shortcuts shortcuts;

    ModalBody() {
        addChild(field);
        addChild(popup);
        addChild(shortcuts);
    }

    Extent2D<F32> measure(const Sakura::Context&, Extent2D<F32>) override { return {260, 130}; }
    void layout(const Sakura::Context& ctx) override {
        const auto bounds = frame();
        layoutChild(ctx, field, {bounds.x, bounds.y, bounds.width, 32});
        layoutChild(ctx, popup, {bounds.x + 10, bounds.y + 45, 100, 20});
        layoutChild(ctx, shortcuts, bounds);
    }
};

struct ModalFixture {
    ImGuiContext* imgui = ImGui::CreateContext();
    Sakura::Context ctx;
    Sakura::RootComponent root;
    Sakura::InputRouter router;
    Sakura::InputFrame input;

    ModalFixture() {
        ctx.pixelRatio = 1;
        ctx.framebufferSize = {800, 600};
        ctx.windowFocused = true;
    }
    ~ModalFixture() { ImGui::DestroyContext(imgui); }

    void place() {
        ++ctx.frame;
        ctx.time += 0.02;
        Sakura::Access::layout(root, ctx, {0, 0, 800, 600});
        Sakura::Access::present(root, ctx.frame);
        Sakura::Access::settleFocus(root);
    }

    void mouse(MouseEventType type, F32 x, F32 y, MouseButton button = MouseButton::Left) {
        input.advance(0.02f);
        input.pointer = Extent2D<F32>{x, y};
        if (type == MouseEventType::Click) input.buttons[static_cast<U64>(button)] = true;
        if (type == MouseEventType::Release) input.buttons[static_cast<U64>(button)] = false;
        if (type == MouseEventType::Cancel) input.buttons.fill(false);
        input.events.push_back(MouseEvent{.type = type, .button = button, .position = {x, y}});
        for (const auto& event : router.route(input, {0, 0, 800, 600}).events) {
            if (const auto* raw = std::get_if<MouseEvent>(&event)) {
                auto pointer = *raw;
                pointer.position = {pointer.position.x * 800, pointer.position.y * 600};
                Sakura::Access::mouseEvent(root, pointer);
            }
        }
        place();
    }

    void click(const Sakura::Component& component) {
        const auto bounds = FrameProbe::of(component);
        mouse(MouseEventType::Click, bounds.x + bounds.width * 0.5f, bounds.y + bounds.height * 0.5f);
        mouse(MouseEventType::Release, bounds.x + bounds.width * 0.5f, bounds.y + bounds.height * 0.5f);
    }

    bool key(KeyCode code, KeyModifiers modifiers = {}) {
        return Sakura::Access::key(root, {KeyEventType::Press, code, modifiers});
    }
    void text(char32_t value) { Sakura::Access::text(root, TextEvent{value}); }
};

struct ViewportLayer : Sakura::Layer {
    Sakura::Component* content = nullptr;
    Rect bounds{0, 0, 600, 600};
    void set(Sakura::Component& next) { content = &next; update({.content = &next}); }
    bool covers(const Extent2D<F32>& point) const override { return bounds.contains(point.x, point.y); }
    void layout(const Sakura::Context& ctx) override { if (content) layoutChild(ctx, *content, bounds); }
};

struct LinkFixture {
    ModalFixture fixture;
    SakuraTest::FakeWindow window;
    Sakura::NodeEditor editor;
    ViewportLayer layer;
    Sakura::NodeEditor::Config config;
    std::vector<std::pair<Sakura::NodeEditor::PinRef, Sakura::NodeEditor::PinRef>> links;
    Extent2D<F32> center;

    LinkFixture() {
        fixture.ctx.render = &window;
        fixture.ctx.deltaTime = 0.1f;
        config = {
            .id = "links",
            .nodes = {{.id = "src", .header = false, .pins = {{.id = "out", .label = "out"}},
                       .dimensions = {120, 0}, .gridPosition = Extent2D<F32>{40, 240}},
                      {.id = "dst", .header = false, .pins = {{.id = "in", .isInput = true, .label = "in"}},
                       .dimensions = {120, 0}, .gridPosition = Extent2D<F32>{1040, 240}}},
            .onLinkCreated = [&](auto start, auto end) { links.emplace_back(std::move(start), std::move(end)); },
            .onViewportGridCenterChange = [&](auto next) { center = next; },
        };
        editor.update(config);
        layer.set(editor);
        fixture.root.setLayers({&layer});
        fixture.place();
    }

    void start() { fixture.mouse(MouseEventType::Click, 172, 255); }
    void pan() { for (U64 frame = 0; frame < 7; ++frame) fixture.mouse(MouseEventType::Move, 700, 300); }
    Extent2D<F32> target() const { return {1040 + 300 - center.x, 255 + 300 - center.y}; }
};

struct RoutedTable : Sakura::Table {
    std::vector<MouseEvent> routed;
    bool mouseEvent(const MouseEvent& event) override {
        routed.push_back(event);
        return Sakura::Table::mouseEvent(event);
    }
    U64 count(MouseEventType type, MouseButton button) const {
        return std::count_if(routed.begin(), routed.end(), [&](const auto& event) { return event.type == type && event.button == button; });
    }
};

}  // namespace

TEST_CASE("Modal activation blocks real background edits and shortcuts and restores keyboard focus",
          "[core][sakura][modal-scope]") {
    ModalFixture fixture;
    std::string backgroundEdited;
    std::string dialogEdited;
    Sakura::TextInput background;
    Sakura::TextInput::Config backgroundConfig{
        .value = "unchanged", .submit = Sakura::TextInput::Submit::OnEdit, .focus = true,
        .onChange = [&](const std::string& value) { backgroundEdited = value; },
    };
    background.update(backgroundConfig);
    ModalBody dialog;
    dialog.field.update({.value = "dialog", .submit = Sakura::TextInput::Submit::OnEdit, .focus = true,
                         .onChange = [&](const std::string& value) { dialogEdited = value; }});
    U64 backgroundShortcut = 0;
    U64 dialogShortcut = 0;
    U64 opened = 0;
    U64 closed = 0;
    Sakura::ModalLayer modal;
    dialog.shortcuts.update({.bindings = {{.key = KeyCode::D, .onPressed = [&] { ++dialogShortcut; }}}});
    Sakura::Shortcuts workspace;
    workspace.update({.bindings = {
        {.key = KeyCode::Comma, .modifier = Sakura::Shortcuts::Modifier::Primary, .onPressed = [&] {
            ++opened;
            modal.update({.content = &dialog, .size = {400, 300}, .onClose = [&] { ++closed; modal.update({}); }});
        }},
        {.key = KeyCode::B, .onPressed = [&] { ++backgroundShortcut; }},
    }, .content = &background});
    Sakura::Layer base;
    base.update({.content = &workspace});
    fixture.root.setLayers({&base, &modal});
    fixture.place();
    fixture.place();
    REQUIRE(background.focused());

    REQUIRE(fixture.key(KeyCode::Comma, {.control = true}));
    fixture.text(U'x');
    fixture.key(KeyCode::Backspace);
    CHECK(backgroundEdited.empty());
    CHECK_FALSE(background.focused());
    fixture.place();
    fixture.place();
    REQUIRE(dialog.field.focused());
    fixture.text(U'y');
    CHECK_FALSE(dialogEdited.empty());
    CHECK(backgroundEdited.empty());
    fixture.key(KeyCode::B);
    fixture.key(KeyCode::Comma, {.control = true});
    CHECK(backgroundShortcut == 0);
    CHECK(opened == 1);
    CHECK(fixture.key(KeyCode::D));
    CHECK(dialogShortcut == 1);

    backgroundConfig.focus = false;
    background.update(backgroundConfig);
    backgroundConfig.focus = true;
    background.update(backgroundConfig);
    fixture.place();
    CHECK_FALSE(background.focused());
    fixture.key(KeyCode::Escape);
    fixture.key(KeyCode::Escape);
    fixture.place();
    CHECK(closed == 1);
    CHECK(background.focused());
    fixture.text(U'z');
    CHECK(backgroundEdited.find('z') != std::string::npos);
}

TEST_CASE("Only the top modal and its recursively owned popups receive keyboard input",
          "[core][sakura][modal-scope]") {
    ModalFixture fixture;
    InputProbe backgroundPopupContent;
    Sakura::Popup backgroundPopup;
    backgroundPopup.update({.content = &backgroundPopupContent, .open = true, .size = {150, 100}});
    Sakura::Layer base;
    base.update({.content = &backgroundPopup});
    ModalBody dialog;
    ModalBody popupContent;
    Sakura::TextInput nested;
    std::string typed;
    nested.update({.submit = Sakura::TextInput::Submit::OnEdit,
                   .onChange = [&](const std::string& value) { typed = value; }});
    popupContent.popup.update({.content = &nested, .open = true, .size = {200, 80}});
    dialog.popup.update({.content = &popupContent, .open = true, .size = {260, 150}});
    U64 popupClosed = 0;
    U64 modalClosed = 0;
    popupContent.shortcuts.update({.bindings = {{.key = KeyCode::Escape, .onPressed = [&] {
        ++popupClosed;
        popupContent.popup.update({.content = &nested, .open = false});
        popupContent.shortcuts.update({});
    }}}});
    Sakura::ModalLayer modal;
    fixture.root.setLayers({&base, &modal});
    fixture.place();
    modal.update({.content = &dialog, .size = {400, 300}, .onClose = [&] { ++modalClosed; modal.update({}); }});
    fixture.place();
    fixture.place();
    CHECK_FALSE(backgroundPopupContent.focus());
    fixture.click(nested);
    REQUIRE(nested.focused());
    fixture.text(U'p');
    CHECK(typed == "p");
    CHECK(backgroundPopupContent.keys == 0);
    CHECK(backgroundPopupContent.texts == 0);
    CHECK(backgroundPopupContent.shortcuts == 0);
    fixture.key(KeyCode::Escape);
    fixture.key(KeyCode::Escape);
    fixture.place();
    CHECK(popupClosed == 1);
    CHECK(modalClosed == 0);
    fixture.key(KeyCode::Escape);
    CHECK(modalClosed == 1);
}

TEST_CASE("Stacked modals reject background focus offers and restore the surviving owner",
          "[core][sakura][modal-scope]") {
    const bool closeLowerFirst = GENERATE(false, true);
    ModalFixture fixture;
    InputProbe background;
    InputProbe first;
    InputProbe second;
    Sakura::Layer base;
    base.update({.content = &background});
    Sakura::ModalLayer lower;
    Sakura::ModalLayer upper;
    fixture.root.setLayers({&base, &lower, &upper});
    fixture.place();
    REQUIRE(background.focus());
    lower.update({.content = &first});
    fixture.place();
    REQUIRE(first.focus());
    upper.update({.content = &second});
    fixture.place();
    CHECK_FALSE(first.focus());
    CHECK_FALSE(background.focus());
    REQUIRE(second.focus());
    fixture.key(KeyCode::Backspace);
    fixture.text(U'x');
    CHECK(second.keys == 1);
    CHECK(second.texts == 1);
    CHECK(first.keys == 0);
    CHECK(background.texts == 0);
    if (closeLowerFirst) {
        lower.update({});
        CHECK(second.focused());
    }
    upper.update({});
    fixture.place();
    if (!closeLowerFirst) {
        CHECK(first.focused());
        lower.update({});
    }
    fixture.place();
    CHECK(background.focused());
    CHECK(background.blurs == 1);
    background.setVisible(false);
    fixture.key(KeyCode::Backspace);
    CHECK(background.focused());
}

TEST_CASE("Floating chrome move and resize ignore secondary releases until their owning completion",
          "[core][sakura][secondary-owner]") {
    const auto secondary = GENERATE(MouseButton::Right, MouseButton::Middle);
    const bool resize = GENERATE(false, true);
    const bool cancel = GENERATE(false, true);
    ModalFixture fixture;
    InputProbe content;
    Sakura::FloatingLayer floating;
    floating.update({.windows = {{.id = "inspector", .title = "Inspector", .content = &content, .size = {300, 200}}}});
    fixture.root.setLayers({&floating});
    fixture.place();
    const auto before = content.placed;
    const Extent2D<F32> start = resize ? Extent2D<F32>{before.right() - 2, before.bottom() - 2}
                                      : Extent2D<F32>{before.x + 50, before.y - 16};
    fixture.mouse(MouseEventType::Click, start.x, start.y);
    fixture.mouse(MouseEventType::Move, start.x + 30, start.y + 30);
    fixture.mouse(MouseEventType::Click, start.x + 30, start.y + 30, secondary);
    fixture.mouse(MouseEventType::Release, start.x + 30, start.y + 30, secondary);
    fixture.mouse(MouseEventType::Move, start.x + 60, start.y + 60);
    if (resize) CHECK(content.placed.width == before.width + 60);
    else CHECK(content.placed.x == before.x + 60);
    fixture.mouse(cancel ? MouseEventType::Cancel : MouseEventType::Release, start.x + 60, start.y + 60);
    const auto stopped = content.placed;
    fixture.mouse(MouseEventType::Move, start.x + 90, start.y + 90);
    CHECK(content.placed == stopped);
}

TEST_CASE("Table column resizing retains the left gesture across right and middle edges",
          "[core][sakura][secondary-owner]") {
    const auto secondary = GENERATE(MouseButton::Right, MouseButton::Middle);
    const bool cancel = GENERATE(false, true);
    ModalFixture fixture;
    InputProbe left;
    InputProbe right;
    Sakura::Table table;
    table.update({.columns = {"Left", "Right"}, .cellRows = {{&left, &right}}, .fixedColumnWidths = {200},
                  .size = {Sakura::Fill, Sakura::Fill}, .resizable = true});
    Sakura::Layer base;
    base.update({.content = &table});
    fixture.root.setLayers({&base});
    fixture.place();
    const F32 before = left.placed.width;
    fixture.mouse(MouseEventType::Click, 200, 10);
    fixture.mouse(MouseEventType::Move, 230, 10);
    fixture.mouse(MouseEventType::Click, 230, 10, secondary);
    fixture.mouse(MouseEventType::Release, 230, 10, secondary);
    fixture.mouse(MouseEventType::Move, 260, 10);
    CHECK(left.placed.width == before + 60);
    fixture.mouse(cancel ? MouseEventType::Cancel : MouseEventType::Release, 260, 10);
    const F32 stopped = left.placed.width;
    fixture.mouse(MouseEventType::Move, 290, 10);
    CHECK(left.placed.width == stopped);
}

TEST_CASE("Modal transitions hide background popups without stealing modal hits or cursors and reveal them again",
          "[core][sakura][modal-scope][modal-pointer]") {
    ModalFixture fixture;
    PointerProbe workspace;
    PointerProbe menu;
    PointerBody background(workspace);
    background.anchor = {240, 200};
    U64 dismissed = 0;
    const Sakura::Popup::Config config{.content = &menu, .open = true, .panel = true, .size = {140, 80},
                                       .onDismiss = [&] { ++dismissed; }};
    background.popup.update(config);
    Sakura::Layer base;
    base.update({.content = &background});
    PointerProbe dialog;
    dialog.cursor = SurfaceCursor::ResizeEW;
    Sakura::ModalLayer modal;
    fixture.root.setLayers({&base, &modal});
    fixture.place();
    const auto bounds = menu.placed;
    const Extent2D<F32> point{bounds.x + 15, bounds.y + 15};
    REQUIRE(fixture.root.popups.interactiveAt(point));
    REQUIRE(Sakura::Access::cursor(fixture.root, point) == SurfaceCursor::Hand);
    modal.update({.content = &dialog, .size = {400, 300}});
    CHECK_FALSE(fixture.root.popups.covers(point));
    CHECK(Sakura::Access::cursor(fixture.root, point) != SurfaceCursor::Hand);
    fixture.place();
    CHECK_FALSE(menu.visible());
    REQUIRE(dialog.placed.contains(point.x, point.y));
    CHECK(Sakura::Access::cursor(fixture.root, point) == SurfaceCursor::ResizeEW);
    fixture.mouse(MouseEventType::Click, point.x, point.y);
    fixture.mouse(MouseEventType::Release, point.x, point.y);
    fixture.mouse(MouseEventType::Scroll, point.x, point.y);
    CHECK(menu.pointerEvents.empty());
    CHECK(dialog.count(MouseEventType::Click) == 1);
    CHECK(dialog.count(MouseEventType::Scroll) == 1);
    CHECK(dismissed == 0);
    modal.update({});
    fixture.place();
    fixture.place();
    REQUIRE(menu.visible());
    fixture.click(menu);
    CHECK(menu.count(MouseEventType::Click) == 1);
    CHECK(menu.count(MouseEventType::Release) == 1);
    background.popup.update({.content = &menu, .open = false});
    fixture.place();
    CHECK_FALSE(menu.visible());
    CHECK_FALSE(fixture.root.popups.interactiveAt(point));
    CHECK(dismissed == 0);
}

TEST_CASE("Opening a modal cancels a background button chord even when the modal closes before release",
          "[core][sakura][modal-scope][modal-pointer]") {
    const auto secondary = GENERATE(MouseButton::Right, MouseButton::Middle);
    const bool closeBeforeRelease = GENERATE(false, true);
    ModalFixture fixture;
    U64 actions = 0;
    Sakura::Button button;
    button.update({.str = "Background", .onClick = [&] { ++actions; }});
    Sakura::Layer base;
    base.update({.content = &button});
    PointerProbe dialog;
    Sakura::ModalLayer modal;
    fixture.root.setLayers({&base, &modal});
    fixture.place();
    fixture.mouse(MouseEventType::Click, 20, 20);
    fixture.mouse(MouseEventType::Click, 20, 20, secondary);
    REQUIRE(fixture.root.captured == &base);
    modal.update({.content = &dialog, .size = {400, 300}});
    CHECK(fixture.root.captured == nullptr);
    CHECK(std::none_of(fixture.root.capturedButtons.begin(), fixture.root.capturedButtons.end(), [](bool held) { return held; }));
    if (closeBeforeRelease) modal.update({});
    fixture.place();
    fixture.mouse(MouseEventType::Release, 20, 20, secondary);
    fixture.mouse(MouseEventType::Release, 20, 20);
    CHECK(actions == 0);
    CHECK(dialog.count(MouseEventType::Release) == 0);
    modal.update({});
    fixture.place();
    fixture.mouse(MouseEventType::Click, 20, 20);
    fixture.mouse(MouseEventType::Release, 20, 20);
    CHECK(actions == 1);
}

TEST_CASE("Modal activation cancels surface input and dock state captured in either a background layer or popup",
          "[core][sakura][modal-scope][modal-pointer]") {
    const bool popupCapture = GENERATE(false, true);
    ModalFixture fixture;
    SakuraTest::FakeWindow window;
    fixture.ctx.render = &window;
    std::vector<InputEvent> inputs;
    Sakura::SurfaceImage image;
    image.update({.onInput = [&](InputEvent event) { inputs.push_back(event); }});
    PointerProbe workspace;
    PointerBody background(popupCapture ? static_cast<Sakura::Component&>(workspace) : image);
    background.popup.update({.content = &image, .open = popupCapture, .size = {200, 100}});
    Sakura::Layer base;
    base.update({.content = &background});
    PointerProbe dialog;
    Sakura::ModalLayer modal;
    fixture.root.setLayers({&base, &modal});
    fixture.place();
    const auto bounds = FrameProbe::of(image);
    const Extent2D<F32> point{bounds.x + 10, bounds.y + 10};
    for (auto button : {MouseButton::Left, MouseButton::Right, MouseButton::Middle}) fixture.mouse(MouseEventType::Click, point.x, point.y, button);
    fixture.root.drag.active = true;
    fixture.root.drag.pane = "background";
    modal.update({.content = &dialog});
    CHECK(fixture.root.captured == nullptr);
    CHECK_FALSE(fixture.root.drag.active);
    CHECK(fixture.root.drag.pane.empty());
    for (auto button : {MouseButton::Right, MouseButton::Left, MouseButton::Middle}) fixture.mouse(MouseEventType::Release, 400, 250, button);
    const auto count = [&](MouseEventType type) {
        return std::count_if(inputs.begin(), inputs.end(), [&](const auto& input) {
            const auto* mouse = std::get_if<MouseEvent>(&input);
            return mouse && mouse->type == type;
        });
    };
    CHECK(count(MouseEventType::Click) == 3);
    CHECK(count(MouseEventType::Cancel) == 1);
    CHECK(count(MouseEventType::Release) == 0);
    CHECK(dialog.count(MouseEventType::Release) == 0);
}

TEST_CASE("Active modal and recursively owned popup captures survive polling and unrelated lower-modal changes",
          "[core][sakura][modal-scope][modal-pointer]") {
    const int depth = GENERATE(0, 1, 2);
    const auto secondary = GENERATE(MouseButton::Right, MouseButton::Middle);
    ModalFixture fixture;
    PointerProbe lowerBody;
    PointerProbe replacement;
    PointerProbe dialog;
    PointerProbe menu;
    PointerProbe nested;
    dialog.cursor = SurfaceCursor::ResizeEW;
    PointerBody menuBody(menu);
    PointerBody dialogBody(dialog);
    menuBody.popup.update({.content = &nested, .open = depth == 2, .panel = true, .size = {80, 30}});
    dialogBody.popup.update({.content = &menuBody, .open = depth > 0, .panel = true, .size = {200, 120}});
    Sakura::ModalLayer lower;
    Sakura::ModalLayer upper;
    lower.update({.content = &lowerBody});
    upper.update({.content = &dialogBody, .size = {400, 300}});
    fixture.root.setLayers({&lower, &upper});
    fixture.place();
    fixture.place();
    auto& target = depth == 0 ? dialog : depth == 1 ? menu : nested;
    if (depth > 0) {
        const auto bounds = FrameProbe::of(menuBody);
        CHECK(Sakura::Access::cursor(fixture.root, {bounds.x - 5, bounds.y - 5}) == SurfaceCursor::Default);
    }
    const Extent2D<F32> point{target.placed.x + 5, target.placed.y + 5};
    fixture.mouse(MouseEventType::Click, point.x, point.y);
    fixture.mouse(MouseEventType::Click, point.x, point.y, secondary);
    REQUIRE(fixture.root.captured == (depth == 0 ? static_cast<Sakura::Component*>(&upper) : &fixture.root.popups));
    for (U64 frame = 0; frame < 4; ++frame) fixture.place();
    lower.update({.content = &replacement});
    fixture.place();
    fixture.mouse(MouseEventType::Move, 750, 550);
    fixture.mouse(MouseEventType::Release, 750, 550, secondary);
    CHECK(fixture.root.captured != nullptr);
    fixture.mouse(MouseEventType::Release, 750, 550);
    CHECK(target.count(MouseEventType::Click) == 2);
    CHECK(target.count(MouseEventType::Release) == 2);
    CHECK(target.cancellations == 0);
    CHECK(fixture.root.captured == nullptr);
}

TEST_CASE("Modal capture withdrawal balances cancellation without transferring releases to replacement content",
          "[core][sakura][modal-scope][modal-pointer]") {
    const int boundary = GENERATE(0, 1, 2, 3);
    ModalFixture fixture;
    PointerProbe background;
    PointerProbe original;
    PointerProbe replacement;
    Sakura::Layer base;
    base.update({.content = &background});
    Sakura::ModalLayer lower;
    Sakura::ModalLayer upper;
    lower.update({.content = &original, .size = {400, 300}});
    fixture.root.setLayers({&base, &lower, &upper});
    fixture.place();
    const auto bounds = original.placed;
    const Extent2D<F32> point{bounds.x + 10, bounds.y + 10};
    fixture.mouse(MouseEventType::Click, point.x, point.y);
    fixture.mouse(MouseEventType::Click, point.x, point.y, MouseButton::Right);
    if (boundary == 0) upper.update({.content = &replacement, .size = {400, 300}});
    if (boundary == 1) lower.update({.content = &replacement, .size = {400, 300}});
    if (boundary == 2) fixture.root.setLayers({&base, &upper});
    if (boundary == 3) {
        fixture.input.windowFocused = false;
        fixture.mouse(MouseEventType::Move, point.x, point.y);
        fixture.input.windowFocused = true;
    }
    fixture.place();
    fixture.mouse(MouseEventType::Release, point.x, point.y, MouseButton::Right);
    fixture.mouse(MouseEventType::Release, point.x, point.y);
    CHECK(original.cancellations == 1);
    CHECK(original.count(MouseEventType::Release) == 0);
    CHECK(replacement.count(MouseEventType::Release) == 0);
    CHECK(background.count(MouseEventType::Release) == 0);
    CHECK(fixture.root.captured == nullptr);
}

TEST_CASE("A modal interrupts floating move and resize without resuming the old gesture when it closes",
          "[core][sakura][modal-scope][modal-pointer]") {
    const bool resize = GENERATE(false, true);
    ModalFixture fixture;
    InputProbe content;
    InputProbe dialog;
    Sakura::FloatingLayer floating;
    floating.update({.windows = {{.id = "inspector", .title = "Inspector", .content = &content, .size = {300, 200}}}});
    Sakura::ModalLayer modal;
    fixture.root.setLayers({&floating, &modal});
    fixture.place();
    const auto before = content.placed;
    const Extent2D<F32> start = resize ? Extent2D<F32>{before.right() - 2, before.bottom() - 2}
                                      : Extent2D<F32>{before.x + 50, before.y - 16};
    fixture.mouse(MouseEventType::Click, start.x, start.y);
    fixture.mouse(MouseEventType::Move, start.x + 30, start.y + 30);
    const auto stopped = content.placed;
    modal.update({.content = &dialog});
    modal.update({});
    fixture.mouse(MouseEventType::Move, start.x + 90, start.y + 90);
    fixture.mouse(MouseEventType::Release, start.x + 90, start.y + 90);
    CHECK(content.placed == stopped);
}

TEST_CASE("A cancelled background button release cannot end a new modal-owned secondary capture",
          "[core][sakura][modal-scope][modal-pointer]") {
    const auto secondary = GENERATE(MouseButton::Right, MouseButton::Middle);
    ModalFixture fixture;
    PointerProbe background;
    PointerProbe dialog;
    Sakura::Layer base;
    base.update({.content = &background});
    Sakura::ModalLayer modal;
    fixture.root.setLayers({&base, &modal});
    fixture.place();
    fixture.mouse(MouseEventType::Click, 20, 20);
    modal.update({.content = &dialog, .size = {400, 300}});
    fixture.place();
    const Extent2D<F32> point{dialog.placed.x + 10, dialog.placed.y + 10};
    fixture.mouse(MouseEventType::Click, point.x, point.y, secondary);
    fixture.mouse(MouseEventType::Release, point.x, point.y);
    CHECK(fixture.root.captured == &modal);
    CHECK(dialog.count(MouseEventType::Release) == 0);
    fixture.mouse(MouseEventType::Release, point.x, point.y, secondary);
    CHECK(dialog.count(MouseEventType::Release) == 1);
    CHECK(background.cancellations == 1);
    CHECK(background.count(MouseEventType::Release) == 0);
    CHECK(fixture.root.captured == nullptr);
}

TEST_CASE("Nested popup capture is cancelled by a higher modal and can be reopened after that modal closes",
          "[core][sakura][modal-scope][modal-pointer]") {
    ModalFixture fixture;
    PointerProbe body;
    PointerProbe menu;
    PointerProbe nested;
    PointerProbe upperBody;
    PointerBody menuBody(menu);
    PointerBody dialogBody(body);
    menuBody.popup.update({.content = &nested, .open = true, .size = {80, 30}});
    dialogBody.popup.update({.content = &menuBody, .open = true, .size = {200, 120}});
    Sakura::ModalLayer lower;
    Sakura::ModalLayer upper;
    lower.update({.content = &dialogBody, .size = {400, 300}});
    fixture.root.setLayers({&lower, &upper});
    fixture.place();
    fixture.place();
    fixture.mouse(MouseEventType::Click, nested.placed.x + 5, nested.placed.y + 5);
    upper.update({.content = &upperBody, .size = {400, 300}});
    fixture.place();
    CHECK(nested.cancellations == 1);
    CHECK_FALSE(nested.visible());
    fixture.mouse(MouseEventType::Release, 400, 250);
    CHECK(nested.count(MouseEventType::Release) == 0);
    CHECK(upperBody.count(MouseEventType::Release) == 0);
    upper.update({});
    fixture.place();
    fixture.place();
    REQUIRE(nested.visible());
    REQUIRE(fixture.root.acceptsInput(nested));
    fixture.click(nested);
    CHECK(nested.count(MouseEventType::Click) == 2);
    CHECK(nested.count(MouseEventType::Release) == 1);
}

TEST_CASE("A modal cancels a table resize without allowing the old left gesture to resume",
          "[core][sakura][modal-scope][modal-pointer]") {
    ModalFixture fixture;
    InputProbe left;
    InputProbe right;
    InputProbe dialog;
    Sakura::Table table;
    table.update({.columns = {"Left", "Right"}, .cellRows = {{&left, &right}}, .fixedColumnWidths = {200},
                  .size = {Sakura::Fill, Sakura::Fill}, .resizable = true});
    Sakura::Layer base;
    base.update({.content = &table});
    Sakura::ModalLayer modal;
    fixture.root.setLayers({&base, &modal});
    fixture.place();
    fixture.mouse(MouseEventType::Click, 200, 10);
    fixture.mouse(MouseEventType::Move, 230, 10);
    const auto stopped = left.placed;
    modal.update({.content = &dialog});
    modal.update({});
    fixture.mouse(MouseEventType::Move, 290, 10);
    fixture.mouse(MouseEventType::Release, 290, 10);
    CHECK(left.placed == stopped);
}

TEST_CASE("Closing a modal restores focus to a surviving background popup after it is revealed",
          "[core][sakura][modal-scope][modal-pointer]") {
    ModalFixture fixture;
    InputProbe workspace;
    InputProbe editor;
    PointerBody background(workspace);
    background.popup.update({.content = &editor, .open = true, .size = {200, 100}});
    Sakura::Layer base;
    base.update({.content = &background});
    InputProbe dialog;
    Sakura::ModalLayer modal;
    fixture.root.setLayers({&base, &modal});
    fixture.place();
    REQUIRE(editor.focus());
    modal.update({.content = &dialog});
    fixture.place();
    REQUIRE(dialog.focus());
    CHECK_FALSE(editor.focused());
    modal.update({});
    fixture.place();
    fixture.place();
    CHECK(editor.visible());
    CHECK(editor.focused());
    CHECK(editor.blurs == 1);
}

TEST_CASE("A blur callback opening a modal cannot grant background focus even temporarily",
          "[core][sakura][modal-scope][modal-admission]") {
    const bool focusDialog = GENERATE(false, true);
    ModalFixture fixture;
    FocusProbe first;
    FocusProbe second;
    FocusProbe dialog;
    FocusGroup group;
    group.attach(first);
    group.attach(second);
    Sakura::Layer base;
    base.update({.content = &group});
    Sakura::ModalLayer modal;
    fixture.root.setLayers({&base, &modal});
    fixture.place();
    REQUIRE(first.focus());
    first.onFocus = [&](bool focused) {
        if (focused) return;
        modal.update({.content = &dialog});
        if (focusDialog) REQUIRE(dialog.focus());
    };
    CHECK_FALSE(second.focus());
    CHECK_FALSE(second.focused());
    CHECK(second.gains == 0);
    CHECK(first.blurs == 1);
    CHECK(dialog.focused() == focusDialog);
    fixture.place();
    CHECK_FALSE(second.focused());
    first.onFocus = {};
}

TEST_CASE("Reentrant focus callbacks preserve their new owner and same-field focus has no duplicate side effects",
          "[core][sakura][modal-scope][modal-admission]") {
    ModalFixture fixture;
    FocusProbe first;
    FocusProbe second;
    FocusProbe third;
    FocusGroup group;
    for (auto* field : {&first, &second, &third}) group.attach(*field);
    Sakura::Layer base;
    base.update({.content = &group});
    fixture.root.setLayers({&base});
    fixture.place();
    REQUIRE(first.focus());
    first.onFocus = [&](bool focused) { if (!focused) REQUIRE(third.focus()); };
    second.focus();
    CHECK(third.focused());
    CHECK_FALSE(second.focused());
    CHECK(second.gains == 0);
    CHECK(first.blurs == 1);
    CHECK(third.gains == 1);
    REQUIRE(third.focus());
    CHECK(third.gains == 1);
    first.onFocus = {};
}

TEST_CASE("Focus handoff rejects candidates whose blur callback changes their root or destroys a participant",
          "[core][sakura][modal-scope][modal-admission]") {
    const int boundary = GENERATE(0, 1, 2, 3);
    ModalFixture fixture;
    auto inputRoot = std::make_unique<Sakura::RootComponent>();
    FocusProbe first;
    auto second = std::make_unique<FocusProbe>();
    FocusGroup group;
    group.attach(first);
    group.attach(*second);
    Sakura::Layer base;
    base.update({.content = &group});
    inputRoot->setLayers({&base});
    Sakura::Access::layout(*inputRoot, fixture.ctx, {0, 0, 800, 600});
    FocusGroup foreignGroup;
    Sakura::Layer foreignBase;
    foreignBase.update({.content = &foreignGroup});
    Sakura::RootComponent foreignRoot;
    foreignRoot.setLayers({&foreignBase});
    U64 forbiddenGains = 0;
    second->onFocus = [&](bool focused) { forbiddenGains += focused; };
    first.onFocus = [&](bool focused) {
        if (focused) return;
        if (boundary == 0) second.reset();
        if (boundary == 1) inputRoot.reset();
        if (boundary == 2) group.detach(*second);
        if (boundary == 3) foreignGroup.attach(*second);
    };
    REQUIRE(first.focus());
    auto* candidate = second.get();
    CHECK_FALSE(candidate->focus());
    CHECK(forbiddenGains == 0);
    if (second) CHECK_FALSE(second->focused());
    first.onFocus = {};
}

TEST_CASE("Destroying the previous focus owner on blur still permits an otherwise valid handoff",
          "[core][sakura][modal-scope][modal-admission]") {
    ModalFixture fixture;
    auto first = std::make_unique<FocusProbe>();
    FocusProbe second;
    FocusGroup group;
    group.attach(*first);
    group.attach(second);
    Sakura::Layer base;
    base.update({.content = &group});
    fixture.root.setLayers({&base});
    fixture.place();
    REQUIRE(first->focus());
    first->onFocus = [&](bool focused) { if (!focused) first.reset(); };
    REQUIRE(second.focus());
    CHECK(second.focused());
    CHECK(second.gains == 1);
    CHECK_FALSE(first);
}

TEST_CASE("A first-show background popup never mounts or presents above an existing modal",
          "[core][sakura][modal-scope][modal-admission]") {
    struct PresentedPopup : PointerProbe {
        U64 presentations = 0;
        void wentOnscreen(U64) override { ++presentations; }
    } menu;
    ModalFixture fixture;
    PointerProbe workspace;
    PointerBody background(workspace);
    background.anchor = {240, 200};
    Sakura::Layer base;
    base.update({.content = &background});
    PointerProbe dialog;
    dialog.cursor = SurfaceCursor::ResizeEW;
    Sakura::ModalLayer modal;
    fixture.root.setLayers({&base, &modal});
    fixture.place();
    modal.update({.content = &dialog, .size = {400, 300}});
    fixture.place();
    U64 dismissed = 0;
    background.popup.update({.content = &menu, .open = true, .panel = true, .size = {160, 80},
                             .onDismiss = [&] { ++dismissed; }});
    const Extent2D<F32> point{260, 240};
    for (U64 frame = 0; frame < 3; ++frame) {
        fixture.place();
        CHECK_FALSE(Sakura::Access::attached(menu, fixture.root));
        CHECK(menu.presentations == 0);
        CHECK(FrameProbe::shownFrames(fixture.root.popups) == 0);
        CHECK_FALSE(fixture.root.popups.covers(point));
        CHECK(Sakura::Access::cursor(fixture.root, point) == SurfaceCursor::ResizeEW);
    }
    fixture.mouse(MouseEventType::Click, point.x, point.y);
    fixture.mouse(MouseEventType::Release, point.x, point.y);
    CHECK(menu.count(MouseEventType::Click) == 0);
    CHECK(dialog.count(MouseEventType::Click) == 1);
    CHECK(dismissed == 0);
    modal.update({});
    fixture.place();
    REQUIRE(Sakura::Access::attached(menu, fixture.root));
    CHECK(menu.visible());
    fixture.click(menu);
    CHECK(menu.count(MouseEventType::Click) == 1);
}

TEST_CASE("First-show modal popups and their newly requested nested content are admitted in the hosting root",
          "[core][sakura][modal-scope][modal-admission]") {
    const bool nestedPopup = GENERATE(false, true);
    ModalFixture fixture;
    PointerProbe dialog;
    PointerProbe menu;
    PointerProbe nested;
    PointerBody menuBody(menu);
    PointerBody dialogBody(dialog);
    Sakura::ModalLayer modal;
    modal.update({.content = &dialogBody, .size = {400, 300}});
    fixture.root.setLayers({&modal});
    fixture.place();
    menuBody.popup.update({.content = &nested, .open = nestedPopup, .panel = true, .size = {80, 40}});
    dialogBody.popup.update({.content = &menuBody, .open = true, .panel = true, .size = {200, 120}});
    fixture.place();
    CHECK(menuBody.visible());
    CHECK(Sakura::Access::attached(menuBody, fixture.root));
    CHECK(FrameProbe::shownFrames(fixture.root.popups) == (nestedPopup ? 2 : 1));
    auto& target = nestedPopup ? nested : menu;
    REQUIRE(Sakura::Access::attached(target, fixture.root));
    REQUIRE(fixture.root.acceptsInput(target));
    fixture.click(target);
    CHECK(target.count(MouseEventType::Click) == 1);
}

TEST_CASE("A hosting root resolves fully unmounted multi-popup ownership chains before the first mount",
          "[core][sakura][modal-scope][modal-admission]") {
    ModalFixture fixture;
    FocusProbe owner;
    FocusProbe secondOwner;
    FocusProbe thirdOwner;
    FocusGroup outer;
    FocusGroup middle;
    PointerProbe leaf;
    outer.attach(secondOwner);
    middle.attach(thirdOwner);
    Sakura::ModalLayer modal;
    modal.update({.content = &owner});
    fixture.root.setLayers({&modal});
    fixture.place();
    fixture.root.popups.replace(thirdOwner, Sakura::Access::alive(thirdOwner),
                               {{.content = &leaf, .anchor = {650, 350, 20, 20}, .panel = true, .size = {100, 60}}});
    fixture.root.popups.replace(secondOwner, Sakura::Access::alive(secondOwner),
                               {{.content = &middle, .anchor = {450, 250, 20, 20}, .panel = true, .size = {140, 80}}});
    fixture.root.popups.replace(owner, Sakura::Access::alive(owner),
                               {{.content = &outer, .anchor = {250, 150, 20, 20}, .panel = true, .size = {160, 80}}});
    REQUIRE_FALSE(Sakura::Access::attached(outer, fixture.root));
    REQUIRE_FALSE(Sakura::Access::attached(middle, fixture.root));
    REQUIRE_FALSE(Sakura::Access::attached(leaf, fixture.root));
    fixture.place();
    CHECK(FrameProbe::shownFrames(fixture.root.popups) == 3);
    REQUIRE(Sakura::Access::attached(leaf, fixture.root));
    REQUIRE(fixture.root.acceptsInput(leaf));
    fixture.click(leaf);
    CHECK(leaf.count(MouseEventType::Click) == 1);
}

TEST_CASE("Popup hosting rejects foreign, expired and cyclic unrooted owners before mounting content",
          "[core][sakura][modal-scope][modal-admission]") {
    const int boundary = GENERATE(0, 1, 2);
    const bool modalOpen = GENERATE(false, true);
    ModalFixture fixture;
    FocusProbe dialog;
    Sakura::ModalLayer modal;
    if (modalOpen) modal.update({.content = &dialog});
    fixture.root.setLayers({&modal});
    fixture.place();
    auto owner = std::make_unique<FocusProbe>();
    FocusProbe secondOwner;
    FocusGroup outer;
    FocusGroup inner;
    outer.attach(*owner);
    inner.attach(secondOwner);
    Sakura::Layer foreignBase;
    foreignBase.update({.content = &outer});
    Sakura::RootComponent foreignRoot;
    if (boundary == 0) foreignRoot.setLayers({&foreignBase});
    fixture.root.popups.replace(*owner, Sakura::Access::alive(*owner),
                               {{.content = &inner, .anchor = {300, 200, 20, 20}, .panel = true, .size = {160, 80}}});
    if (boundary == 1) owner.reset();
    if (boundary == 2) fixture.root.popups.replace(secondOwner, Sakura::Access::alive(secondOwner),
                                                 {{.content = &outer, .anchor = {500, 300, 20, 20}, .panel = true, .size = {160, 80}}});
    for (U64 frame = 0; frame < 3; ++frame) {
        fixture.place();
        CHECK_FALSE(Sakura::Access::attached(inner, fixture.root));
        CHECK(FrameProbe::shownFrames(fixture.root.popups) == 0);
    }
}

TEST_CASE("A popup request cannot take content or nested owners from another mounted hosting root",
          "[core][sakura][modal-scope][modal-admission]") {
    ModalFixture fixture;
    FocusProbe hostOwner;
    Sakura::ModalLayer modal;
    modal.update({.content = &hostOwner});
    fixture.root.setLayers({&modal});
    fixture.place();
    FocusProbe foreignOwner;
    FocusGroup foreignContent;
    foreignContent.attach(foreignOwner);
    Sakura::Layer foreignBase;
    foreignBase.update({.content = &foreignContent});
    Sakura::RootComponent foreignRoot;
    foreignRoot.setLayers({&foreignBase});
    PointerProbe nested;
    fixture.root.popups.replace(foreignOwner, Sakura::Access::alive(foreignOwner),
                               {{.content = &nested, .anchor = {500, 300, 20, 20}, .panel = true, .size = {160, 80}}});
    fixture.root.popups.replace(hostOwner, Sakura::Access::alive(hostOwner),
                               {{.content = &foreignContent, .anchor = {300, 200, 20, 20}, .panel = true, .size = {160, 80}}});
    fixture.place();
    CHECK(Sakura::Access::attached(foreignContent, foreignRoot));
    CHECK_FALSE(Sakura::Access::attached(nested, fixture.root));
    CHECK(FrameProbe::shownFrames(fixture.root.popups) == 0);
}

TEST_CASE("Link creation keeps its source alive through root capture while auto-pan brings a distant target onscreen",
          "[core][sakura][opus-followup][link-capture]") {
    LinkFixture rig;
    const auto* source = Sakura::Access::pointerTarget(rig.fixture.root, {172, 255});
    REQUIRE(source != nullptr);
    REQUIRE(source != &rig.editor);
    rig.start();
    REQUIRE(rig.fixture.root.captured == &rig.layer);
    REQUIRE(rig.fixture.root.capturedTarget == source);
    rig.pan();
    CHECK(rig.center.x > 800);
    CHECK(source->visible());
    CHECK(rig.fixture.root.captured == &rig.layer);
    const auto target = rig.target();
    CHECK(target.x > 0);
    CHECK(target.x < 600);
    rig.fixture.mouse(MouseEventType::Move, target.x, target.y);
    rig.fixture.mouse(MouseEventType::Release, target.x, target.y);
    REQUIRE(rig.links.size() == 1);
    CHECK(rig.links.front().first == Sakura::NodeEditor::PinRef{"src", "out", false});
    CHECK(rig.links.front().second == Sakura::NodeEditor::PinRef{"dst", "in", true});
    CHECK(rig.fixture.root.captured == nullptr);
    CHECK_FALSE(source->visible());
}

TEST_CASE("Offscreen link-source retention does not prevent explicit cancellation or replacement cancellation",
          "[core][sakura][opus-followup][link-capture]") {
    const int boundary = GENERATE(0, 1, 2);
    LinkFixture rig;
    PointerProbe replacement;
    rig.start();
    rig.pan();
    const auto target = rig.target();
    if (boundary == 0) rig.fixture.mouse(MouseEventType::Cancel, target.x, target.y);
    if (boundary == 1) rig.layer.set(replacement);
    if (boundary == 2) {
        rig.config.nodes.erase(rig.config.nodes.begin());
        rig.editor.update(rig.config);
    }
    rig.fixture.place();
    rig.fixture.mouse(MouseEventType::Release, target.x, target.y);
    CHECK(rig.links.empty());
    CHECK(rig.fixture.root.captured == nullptr);
    CHECK(replacement.count(MouseEventType::Release) == 0);
}

TEST_CASE("Floating table resizes retain their content window across right and middle release orders outside its bounds",
          "[core][sakura][opus-followup][secondary-owner]") {
    const auto secondary = GENERATE(MouseButton::Right, MouseButton::Middle);
    const bool primaryFirst = GENERATE(false, true);
    ModalFixture fixture;
    InputProbe left;
    InputProbe right;
    RoutedTable table;
    table.update({.columns = {"Left", "Right"}, .cellRows = {{&left, &right}}, .fixedColumnWidths = {140},
                  .size = {Sakura::Fill, Sakura::Fill}, .resizable = true});
    Sakura::FloatingLayer floating;
    floating.update({.windows = {{.id = "metadata", .title = "Metadata", .content = &table, .size = {300, 220}}}});
    fixture.root.setLayers({&floating});
    fixture.place();
    const auto bounds = FrameProbe::of(table);
    const F32 x = bounds.x + 140;
    const F32 y = bounds.y + 10;
    const F32 before = left.placed.width;
    fixture.mouse(MouseEventType::Click, x, y);
    fixture.mouse(MouseEventType::Move, x + 30, y);
    REQUIRE(left.placed.width == before + 30);
    fixture.mouse(MouseEventType::Click, x + 30, y, secondary);
    if (!primaryFirst) fixture.mouse(MouseEventType::Release, x + 30, y, secondary);
    fixture.mouse(MouseEventType::Move, 750, y);
    CHECK(left.placed.width > before + 30);
    fixture.mouse(MouseEventType::Release, 750, y);
    if (primaryFirst) fixture.mouse(MouseEventType::Release, 750, y, secondary);
    const F32 stopped = left.placed.width;
    fixture.mouse(MouseEventType::Move, bounds.x + 70, y);
    CHECK(left.placed.width == stopped);
    CHECK(table.count(MouseEventType::Click, MouseButton::Left) == 1);
    CHECK(table.count(MouseEventType::Click, secondary) == 1);
    CHECK(table.count(MouseEventType::Release, MouseButton::Left) == 1);
    CHECK(table.count(MouseEventType::Release, secondary) == 1);
    CHECK(fixture.root.captured == nullptr);
}

TEST_CASE("Floating content capture clears on cancellation or withdrawal without delivering completion to a replacement",
          "[core][sakura][opus-followup][secondary-owner]") {
    const int boundary = GENERATE(0, 1, 2);
    ModalFixture fixture;
    PointerProbe original;
    PointerProbe replacement;
    Sakura::FloatingLayer floating;
    floating.update({.windows = {{.id = "metadata", .title = "Metadata", .content = &original, .size = {300, 220}}}});
    fixture.root.setLayers({&floating});
    fixture.place();
    const Extent2D<F32> point{original.placed.x + 40, original.placed.y + 20};
    fixture.mouse(MouseEventType::Click, point.x, point.y);
    fixture.mouse(MouseEventType::Click, point.x, point.y, MouseButton::Right);
    if (boundary == 0) fixture.mouse(MouseEventType::Cancel, point.x, point.y);
    if (boundary == 1) floating.update({});
    if (boundary == 2) floating.update({.windows = {{.id = "metadata", .title = "Metadata", .content = &replacement, .size = {300, 220}}}});
    fixture.place();
    fixture.mouse(MouseEventType::Release, point.x, point.y, MouseButton::Right);
    fixture.mouse(MouseEventType::Release, point.x, point.y);
    CHECK(original.cancellations == 1);
    CHECK(original.count(MouseEventType::Release) == 0);
    CHECK(replacement.count(MouseEventType::Release) == 0);
    CHECK(fixture.root.captured == nullptr);
}
