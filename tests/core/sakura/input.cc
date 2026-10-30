#include <catch2/catch_test_macros.hpp>

#include "harness.hh"
#include "component_harness.hh"

#include <jetstream/render/sakura/clipboard.hh>
#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/shortcuts.hh>
#include <jetstream/render/sakura/components/text_grid.hh>
#include <jetstream/render/sakura/components/text_input.hh>

#include "render/sakura/input.hh"
#include "render/sakura/detail/input_router.hh"

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>

using namespace Jetstream;
using namespace SakuraTest;

namespace {

using Jetstream::Sakura::FrameEvent;
using Jetstream::Sakura::InputFrame;

struct Node : Sakura::Component {
    std::string name;
    std::vector<std::string>& log;
    Rect area;
    bool takesKeys = false;
    bool takesText = false;
    bool edits = false;
    bool claims = false;
    std::function<void(bool)> onFocus;
    std::function<void()> onKey;
    std::function<void()> onShortcut;
    std::vector<KeyCode> shortcuts;
    std::vector<bool> focusChanges;

    Node(std::string name, std::vector<std::string>& log, Rect area = {})
        : name(std::move(name)), log(log), area(area) {}

    void adopt(Sakura::Component& child) {
        addChild(child);
    }

    void drop(Sakura::Component& child) {
        removeChild(child);
    }

    void show(bool shown) {
        setVisible(shown);
    }

    void focus() {
        requestFocus();
    }

    void offer(bool offered) {
        offerFocus(offered);
    }

    bool send(const MouseEvent& event) {
        return mouseEventChildren(event);
    }

    void blur() {
        releaseFocus();
    }

    bool focused() const {
        return hasFocus();
    }

    void place(const Sakura::Context& ctx) {
        invalidate(Dirty::Layout);
        layoutRoot(ctx, area);
    }

 protected:
    void layout(const Sakura::Context& ctx) override {
        for (auto* child : children()) {
            const auto* node = dynamic_cast<Node*>(child);
            layoutChild(ctx, *child, node ? node->area : frame());
        }
    }

    std::optional<SurfaceCursor> cursorAt(const Extent2D<F32>& point) const override {
        if (frame().contains(point.x, point.y)) {
            return SurfaceCursor::Default;
        }
        return std::nullopt;
    }

    bool keyEvent(const KeyEvent&) override {
        log.push_back(name + ":key");
        if (onKey) {
            onKey();
            return false;
        }
        return takesKeys;
    }

    bool textEvent(const TextEvent&) override {
        log.push_back(name + ":text");
        return takesText;
    }

    bool shortcutEvent(const KeyEvent& event) override {
        log.push_back(name + ":shortcut");
        if (onShortcut) {
            onShortcut();
            return false;
        }
        return std::find(shortcuts.begin(), shortcuts.end(), event.key) != shortcuts.end();
    }

    void focusChanged(bool focused) override {
        focusChanges.push_back(focused);
        if (onFocus) {
            onFocus(focused);
        }
    }

    bool editsText() const override {
        return edits;
    }

    bool focusesOnClick() const override {
        return claims;
    }
};

KeyEvent Press(KeyCode key, KeyModifiers modifiers = {}, bool repeat = false) {
    return {KeyEventType::Press, key, modifiers, repeat};
}

struct Tree {
    std::vector<std::string> log;
    Node root{"root", log, {0.0f, 0.0f, 200.0f, 100.0f}};
    Node a{"a", log, {0.0f, 0.0f, 100.0f, 100.0f}};
    Node a1{"a1", log, {10.0f, 10.0f, 50.0f, 50.0f}};
    Node b{"b", log, {100.0f, 0.0f, 100.0f, 100.0f}};

    Tree() {
        root.adopt(a);
        a.adopt(a1);
        root.adopt(b);
        root.place(MakeContext({200.0f, 100.0f}));
    }
};

InputFrame Frame(U64 serial, std::optional<Extent2D<F32>> pointer, std::vector<FrameEvent> events = {}) {
    static const U8 source = 0;
    InputFrame frame;
    frame.source = &source;
    frame.serial = serial;
    frame.pointer = pointer;
    frame.events = std::move(events);
    for (auto& event : frame.events) {
        if (auto* mouse = std::get_if<MouseEvent>(&event); mouse && pointer.has_value()) {
            mouse->position = *pointer;
        }
    }
    return frame;
}

MouseEvent Button(MouseEventType type, MouseButton button = MouseButton::Left) {
    return {.type = type, .button = button};
}

template<typename Type>
std::vector<Type> Only(const std::vector<FrameEvent>& events) {
    std::vector<Type> result;
    for (const auto& event : events) {
        if (const auto* typed = std::get_if<Type>(&event)) {
            result.push_back(*typed);
        }
    }
    return result;
}

}  // namespace

TEST_CASE("Focus has a single owner per tree and notifies both sides when it moves",
          "[core][sakura][input][focus]") {
    Tree tree;
    CHECK(FocusOwner(tree.root) == nullptr);

    tree.a1.focus();
    CHECK(FocusOwner(tree.root) == &tree.a1);
    CHECK(tree.a1.focused());
    CHECK(tree.a1.focusChanges == std::vector<bool>{true});

    tree.b.focus();
    CHECK(FocusOwner(tree.root) == &tree.b);
    CHECK_FALSE(tree.a1.focused());
    CHECK(tree.a1.focusChanges == std::vector<bool>{true, false});
    CHECK(tree.b.focusChanges == std::vector<bool>{true});

    tree.b.focus();
    CHECK(tree.b.focusChanges == std::vector<bool>{true});

    tree.b.blur();
    tree.b.blur();
    CHECK(FocusOwner(tree.root) == nullptr);
    CHECK(tree.b.focusChanges == std::vector<bool>{true, false});
}

TEST_CASE("Focus survives a remount and leaves with a destroyed owner",
          "[core][sakura][input][focus]") {
    Tree tree;
    tree.a1.focus();

    tree.root.drop(tree.a);
    CHECK(FocusOwner(tree.root) == nullptr);
    CHECK(tree.a1.focused());
    tree.root.adopt(tree.a);
    CHECK(FocusOwner(tree.root) == &tree.a1);
    CHECK(tree.a1.focusChanges == std::vector<bool>{true});

    auto doomed = std::make_unique<Node>("doomed", tree.log);
    tree.root.adopt(*doomed);
    doomed->focus();
    CHECK(tree.a1.focusChanges == std::vector<bool>{true, false});
    doomed.reset();
    CHECK(FocusOwner(tree.root) == nullptr);
    tree.a1.focus();
    CHECK(FocusOwner(tree.root) == &tree.a1);
}

TEST_CASE("A subtree focused before it joins a tree brings its focus along unless the tree has an owner",
          "[core][sakura][input][focus]") {
    std::vector<std::string> log;
    Node root("root", log);
    Node first("first", log);
    Node firstChild("first-child", log);
    first.adopt(firstChild);
    firstChild.focus();

    root.adopt(first);
    CHECK(FocusOwner(root) == &firstChild);
    CHECK(firstChild.focusChanges == std::vector<bool>{true});

    Node second("second", log);
    Node secondChild("second-child", log);
    second.adopt(secondChild);
    secondChild.focus();
    root.adopt(second);
    CHECK(FocusOwner(root) == &firstChild);
    CHECK_FALSE(secondChild.focused());
    CHECK(secondChild.focusChanges == std::vector<bool>{true, false});
}

TEST_CASE("Keys bubble from the focus owner through its ancestors, then shortcuts run topmost first",
          "[core][sakura][input][keys]") {
    Tree tree;
    tree.a1.focus();

    CHECK_FALSE(Sakura::Access::key(tree.root, Press(KeyCode::Enter)));
    CHECK(tree.log == std::vector<std::string>{"a1:key", "a:key", "root:key", "b:shortcut", "a1:shortcut",
                                               "a:shortcut", "root:shortcut"});

    tree.log.clear();
    tree.a.takesKeys = true;
    CHECK(Sakura::Access::key(tree.root, Press(KeyCode::Enter)));
    CHECK(tree.log == std::vector<std::string>{"a1:key", "a:key"});

    tree.log.clear();
    tree.a.takesKeys = false;
    tree.b.shortcuts = {KeyCode::Escape};
    CHECK(Sakura::Access::key(tree.root, Press(KeyCode::Escape)));
    CHECK(tree.log == std::vector<std::string>{"a1:key", "a:key", "root:key", "b:shortcut"});

    tree.log.clear();
    tree.a1.blur();
    CHECK(Sakura::Access::key(tree.root, Press(KeyCode::Escape)));
    CHECK(tree.log == std::vector<std::string>{"b:shortcut"});
}

TEST_CASE("Hidden owners and hidden layers receive nothing, and text only climbs the owner chain",
          "[core][sakura][input][keys]") {
    Tree tree;
    tree.a1.focus();
    tree.a.show(false);

    CHECK_FALSE(Sakura::Access::key(tree.root, Press(KeyCode::A)));
    CHECK(tree.log == std::vector<std::string>{"b:shortcut", "root:shortcut"});
    tree.log.clear();
    CHECK_FALSE(Sakura::Access::text(tree.root, {U'x'}));
    CHECK(tree.log.empty());

    tree.a.show(true);
    tree.a.takesText = true;
    CHECK(Sakura::Access::text(tree.root, {U'x'}));
    CHECK(tree.log == std::vector<std::string>{"a1:text", "a:text"});
}

TEST_CASE("Clicks blur the owner unless they land inside it or a handler moved focus",
          "[core][sakura][input][focus]") {
    Tree tree;
    tree.a1.focus();

    ClickFocus(tree.root, Extent2D<F32>{20.0f, 20.0f});
    CHECK(tree.a1.focused());
    ClickFocus(tree.root, Extent2D<F32>{80.0f, 80.0f});
    CHECK_FALSE(tree.a1.focused());

    tree.a1.focus();
    const U64 generation = Sakura::Access::focusGeneration(tree.root);
    tree.b.focus();
    Sakura::Access::clickFocus(tree.root, Extent2D<F32>{20.0f, 20.0f}, true, generation);
    CHECK(tree.b.focused());

    ClickFocus(tree.root, std::nullopt);
    CHECK(FocusOwner(tree.root) == nullptr);
}

TEST_CASE("Focus offers settle after layout, never during it, and never take focus from a text editor",
          "[core][sakura][input][focus]") {
    Tree tree;
    const auto ctx = MakeContext({200.0f, 100.0f});
    Node field("field", tree.log, {0.0f, 0.0f, 100.0f, 20.0f});
    tree.root.adopt(field);
    const auto settle = [&] {
        tree.root.place(ctx);
        return Sakura::Access::settleFocus(tree.root);
    };

    field.offer(true);
    tree.root.place(ctx);
    CHECK_FALSE(field.focused());
    CHECK(Sakura::Access::settleFocus(tree.root));
    CHECK(field.focused());
    CHECK_FALSE(settle());

    field.blur();
    tree.b.edits = true;
    tree.b.focus();
    field.offer(true);
    CHECK_FALSE(settle());
    CHECK_FALSE(field.focused());
    CHECK(tree.b.focused());

    tree.b.edits = false;
    field.offer(true);
    CHECK(settle());
    CHECK(field.focused());

    field.blur();
    field.offer(true);
    field.offer(false);
    CHECK_FALSE(settle());
    CHECK_FALSE(field.focused());

    field.area = {};
    field.offer(true);
    CHECK_FALSE(settle());
    CHECK_FALSE(field.focused());
    field.area = {0.0f, 0.0f, 100.0f, 20.0f};
    CHECK(settle());
    CHECK(field.focused());

    field.blur();
    Node detached("detached", tree.log, {120.0f, 0.0f, 50.0f, 20.0f});
    detached.offer(true);
    tree.root.adopt(detached);
    CHECK(settle());
    CHECK(detached.focused());
}

TEST_CASE("A focus handover stops when the old owner's blur moves focus or destroys the new owner",
          "[core][sakura][input][focus]") {
    Tree tree;
    tree.a1.focus();
    tree.a1.onFocus = [&](bool focused) {
        if (!focused) {
            tree.b.focus();
        }
    };
    tree.a.focus();
    CHECK(FocusOwner(tree.root) == &tree.b);
    CHECK(tree.a.focusChanges.empty());
    CHECK(tree.b.focusChanges == std::vector<bool>{true});

    auto doomed = std::make_unique<Node>("doomed", tree.log);
    tree.root.adopt(*doomed);
    tree.a1.onFocus = nullptr;
    tree.b.onFocus = [&](bool focused) {
        if (!focused) {
            doomed.reset();
        }
    };
    doomed->focus();
    CHECK(doomed == nullptr);
    CHECK(FocusOwner(tree.root) == nullptr);
}

TEST_CASE("Key dispatch stops cleanly when a handler destroys the component that received the key",
          "[core][sakura][input][keys]") {
    Tree tree;
    auto doomed = std::make_unique<Node>("doomed", tree.log);
    tree.a.adopt(*doomed);
    doomed->focus();
    doomed->onKey = [&] { doomed.reset(); };

    CHECK(Sakura::Access::key(tree.root, Press(KeyCode::A)));
    CHECK(doomed == nullptr);
    CHECK(tree.log == std::vector<std::string>{"doomed:key"});

    tree.log.clear();
    CHECK_FALSE(Sakura::Access::key(tree.root, Press(KeyCode::A)));
    CHECK(tree.log == std::vector<std::string>{"b:shortcut", "a1:shortcut", "a:shortcut", "root:shortcut"});
}

TEST_CASE("A text grid destroyed while it takes focus stops before touching itself",
          "[core][sakura][input][focus]") {
    Tree tree;
    auto grid = std::make_unique<Sakura::TextGrid>();
    tree.b.adopt(*grid);
    grid->update({.id = "doomed-grid", .value = "abc", .editable = true, .fontSize = 15.0f});
    tree.root.place(MakeContext({200.0f, 100.0f}));
    tree.a1.focus();
    tree.a1.onFocus = [&](bool focused) {
        if (!focused) {
            grid.reset();
        }
    };

    Sakura::TextGrid* raw = grid.get();
    raw->focus(true);
    CHECK(grid == nullptr);
    CHECK(FocusOwner(tree.root) == nullptr);

    grid = std::make_unique<Sakura::TextGrid>();
    tree.b.adopt(*grid);
    grid->update({.id = "clicked-grid", .value = "abc", .editable = true, .fontSize = 15.0f});
    tree.root.place(MakeContext({200.0f, 100.0f}));
    tree.a1.focus();
    CHECK(tree.root.send(Mouse(MouseEventType::Click, 150.0f, 50.0f)));
    CHECK(grid == nullptr);
    CHECK(FocusOwner(tree.root) == nullptr);
}

TEST_CASE("Shortcut dispatch stops when a handler destroys itself",
          "[core][sakura][input][keys]") {
    Tree tree;
    auto doomed = std::make_unique<Node>("doomed", tree.log);
    tree.root.adopt(*doomed);
    doomed->onShortcut = [&] { doomed.reset(); };

    CHECK(Sakura::Access::key(tree.root, Press(KeyCode::A)));
    CHECK(doomed == nullptr);
    CHECK(tree.log == std::vector<std::string>{"doomed:shortcut"});
}

TEST_CASE("Settling focus offers stops when a blur destroys the root being settled",
          "[core][sakura][input][focus]") {
    std::vector<std::string> log;
    auto root = std::make_unique<Node>("root", log, Rect{0.0f, 0.0f, 200.0f, 100.0f});
    Node owner("owner", log, {0.0f, 0.0f, 50.0f, 50.0f});
    Node field("field", log, {60.0f, 0.0f, 50.0f, 50.0f});
    Node later("later", log, {120.0f, 0.0f, 50.0f, 50.0f});
    root->adopt(owner);
    root->adopt(field);
    root->adopt(later);
    root->place(MakeContext({200.0f, 100.0f}));
    owner.focus();
    owner.onFocus = [&](bool focused) {
        if (!focused) {
            root.reset();
        }
    };
    field.offer(true);
    later.offer(true);

    Node* raw = root.get();
    CHECK(Sakura::Access::settleFocus(*raw));
    CHECK(root == nullptr);
    CHECK(later.focusChanges.empty());
}

TEST_CASE("Clicks keep focus inside the owner and otherwise hand it to the nearest click claimant",
          "[core][sakura][input][focus]") {
    Tree tree;
    tree.a.claims = true;
    tree.a1.focus();

    ClickFocus(tree.root, Extent2D<F32>{20.0f, 20.0f});
    CHECK(tree.a1.focused());
    ClickFocus(tree.root, Extent2D<F32>{80.0f, 80.0f});
    CHECK(tree.a.focused());
    CHECK(tree.a1.focusChanges == std::vector<bool>{true, false});

    ClickFocus(tree.root, Extent2D<F32>{150.0f, 50.0f});
    CHECK(FocusOwner(tree.root) == nullptr);
    ClickFocus(tree.root, Extent2D<F32>{20.0f, 20.0f});
    CHECK(tree.a.focused());
    ClickFocus(tree.root, std::nullopt);
    CHECK(FocusOwner(tree.root) == nullptr);
}

TEST_CASE("Text inputs commit and blur in the same event that changes their focus",
          "[core][sakura][input][focus]") {
    std::vector<std::string> log;
    std::vector<std::string> changes;
    Node root("root", log, {0.0f, 0.0f, 300.0f, 25.0f});
    Sakura::TextInput input;
    root.adopt(input);
    Sakura::TextInput::Config config{
        .value = "alpha",
        .submit = Sakura::TextInput::Submit::OnCommit,
        .onChange = [&](const std::string& value) { changes.push_back(value); },
    };
    input.update(config);
    const auto ctx = MakeContext({300.0f, 25.0f});
    root.place(ctx);
    Sakura::Access::settleFocus(root);

    root.send(Mouse(MouseEventType::Click, 290.0f, 12.0f));
    root.send(Mouse(MouseEventType::Release, 290.0f, 12.0f));
    REQUIRE(input.focused());
    Type(root, "!");
    CHECK(changes.empty());
    ClickFocus(root, std::nullopt);
    CHECK_FALSE(input.focused());
    CHECK(changes == std::vector<std::string>{"alpha!"});

    config.value = "alpha!";
    config.submit = Sakura::TextInput::Submit::OnEnter;
    config.focus = true;
    input.update(config);
    root.place(ctx);
    CHECK_FALSE(input.focused());
    Sakura::Access::settleFocus(root);
    REQUIRE(input.focused());
    Type(root, "?");
    CHECK(Tap(root, KeyCode::Enter));
    CHECK_FALSE(input.focused());
    CHECK(changes == std::vector<std::string>{"alpha!", "alpha!?"});
}

TEST_CASE("The input router forwards clicks over the canvas and keeps capture through a release outside",
          "[core][sakura][input][router]") {
    Sakura::InputRouter router;
    const Rect screen = {100.0f, 50.0f, 200.0f, 100.0f};

    auto frame = Frame(1, Extent2D<F32>{150.0f, 100.0f}, {Button(MouseEventType::Click)});
    frame.buttons[0] = true;
    auto output = router.route(frame, screen);
    auto mouse = Only<MouseEvent>(output.events);
    REQUIRE(mouse.size() == 1);
    CHECK(mouse[0].type == MouseEventType::Click);
    CHECK(mouse[0].position == Extent2D<F32>{0.25f, 0.5f});
    CHECK(output.hovered);
    CHECK(output.active);

    frame = Frame(2, Extent2D<F32>{400.0f, 300.0f});
    frame.buttons[0] = true;
    output = router.route(frame, screen);
    mouse = Only<MouseEvent>(output.events);
    REQUIRE(mouse.size() == 1);
    CHECK(mouse[0].type == MouseEventType::Move);
    CHECK(mouse[0].position == Extent2D<F32>{1.5f, 2.5f});
    CHECK_FALSE(output.hovered);
    CHECK(output.active);

    output = router.route(Frame(3, Extent2D<F32>{400.0f, 300.0f}, {Button(MouseEventType::Release)}), screen);
    mouse = Only<MouseEvent>(output.events);
    REQUIRE(mouse.size() == 1);
    CHECK(mouse[0].type == MouseEventType::Release);
    CHECK_FALSE(mouse[0].cancel);
    CHECK_FALSE(output.active);

    output = router.route(Frame(4, Extent2D<F32>{400.0f, 300.0f}, {Button(MouseEventType::Click)}), screen);
    CHECK(output.events.empty());
    CHECK(output.outsideClick);
    output = router.route(Frame(4, Extent2D<F32>{150.0f, 100.0f}, {Button(MouseEventType::Click)}), screen);
    CHECK(output.events.empty());

    frame = Frame(5, Extent2D<F32>{150.0f, 100.0f}, {Button(MouseEventType::Click, MouseButton::Middle)});
    frame.buttons[2] = true;
    output = router.route(frame, screen);
    CHECK_FALSE(output.outsideClick);
    CHECK(output.middle);
    CHECK_FALSE(output.active);
}

TEST_CASE("A missed release cancels the drag and losing window focus releases held keys",
          "[core][sakura][input][router]") {
    Sakura::InputRouter router;
    const Rect screen = {0.0f, 0.0f, 100.0f, 100.0f};

    auto frame = Frame(1, Extent2D<F32>{50.0f, 50.0f}, {Button(MouseEventType::Click), Press(KeyCode::A)});
    frame.buttons[0] = true;
    auto output = router.route(frame, screen);
    CHECK(Only<KeyEvent>(output.events).size() == 1);

    output = router.route(Frame(2, Extent2D<F32>{50.0f, 50.0f}), screen);
    auto mouse = Only<MouseEvent>(output.events);
    REQUIRE(mouse.size() == 2);
    CHECK(mouse[0].type == MouseEventType::Release);
    CHECK(mouse[0].cancel);
    CHECK(mouse[1].type == MouseEventType::Cancel);
    CHECK_FALSE(output.active);
    const auto cancelledKeys = Only<KeyEvent>(output.events);
    REQUIRE(cancelledKeys.size() == 1);
    CHECK(cancelledKeys[0].type == KeyEventType::Release);
    CHECK(cancelledKeys[0].key == KeyCode::A);

    frame = Frame(3, Extent2D<F32>{50.0f, 50.0f}, {Press(KeyCode::A)});
    REQUIRE(Only<KeyEvent>(router.route(frame, screen).events).size() == 1);
    frame = Frame(4, Extent2D<F32>{50.0f, 50.0f}, {TextEvent{U'q'}});
    frame.windowFocused = false;
    output = router.route(frame, screen);
    const auto keys = Only<KeyEvent>(output.events);
    REQUIRE(keys.size() == 1);
    CHECK(keys[0].type == KeyEventType::Release);
    CHECK(keys[0].key == KeyCode::A);
    CHECK(Only<TextEvent>(output.events).empty());

    output = router.route(Frame(5, Extent2D<F32>{50.0f, 50.0f},
                                {KeyEvent{KeyEventType::Release, KeyCode::A, {}}, TextEvent{U'q'}}), screen);
    CHECK(Only<KeyEvent>(output.events).empty());
    CHECK(Only<TextEvent>(output.events).size() == 1);
}

TEST_CASE("Shortcuts match keys by modifier and repeat, pass layout through, and stay quiet when disabled",
          "[core][sakura][input][shortcuts]") {
    std::vector<std::string> fired;
    const auto record = [&](std::string name) {
        return [&fired, name]() { fired.push_back(name); };
    };
    Probe content({40.0f, 20.0f});
    Sakura::Shortcuts keys;
    Sakura::Shortcuts::Config config{
        .bindings = {
            {.key = KeyCode::A, .onPressed = record("any")},
            {.key = KeyCode::B, .modifier = Sakura::Shortcuts::Modifier::Primary, .onPressed = record("primary")},
            {.key = KeyCode::C, .modifier = Sakura::Shortcuts::Modifier::PrimaryAlt, .onPressed = record("alt")},
            {.key = KeyCode::D, .repeat = true, .onPressed = record("repeat")},
        },
        .content = &content,
    };
    keys.update(config);
    Host host(keys);
    const auto ctx = MakeContext();
    CHECK(host.measureOf(ctx, {300.0f, 300.0f}) == Extent2D<F32>{40.0f, 20.0f});
    host.place(ctx, {10.0f, 20.0f, 120.0f, 60.0f});
    CHECK(content.laidFrame == Rect{10.0f, 20.0f, 120.0f, 60.0f});

    CHECK(Tap(host, KeyCode::A, {.shift = true}));
    CHECK_FALSE(Tap(host, KeyCode::B));
    CHECK(Tap(host, KeyCode::B, {.super = true}));
    CHECK_FALSE(Tap(host, KeyCode::C, {.control = true}));
    CHECK(Tap(host, KeyCode::C, {.control = true, .alt = true}));
    CHECK_FALSE(Sakura::Access::key(host, Press(KeyCode::A, {}, true)));
    CHECK(Sakura::Access::key(host, Press(KeyCode::D, {}, true)));
    CHECK(fired == std::vector<std::string>{"any", "primary", "alt", "repeat"});

    config.enabled = false;
    keys.update(config);
    CHECK_FALSE(Tap(host, KeyCode::A));
    CHECK(fired.size() == 4);
}

TEST_CASE("Shortcuts settle with content and binding or enabled changes cause no work",
          "[core][sakura][input][shortcuts][retention]") {
    U64 fired = 0;
    Sakura::Box content;
    Sakura::Shortcuts keys;
    Host host(keys);
    Frames frames(host, {10.0f, 10.0f, 200.0f, 100.0f});
    KeyCode key = KeyCode::A;
    bool enabled = true;
    const auto update = [&] {
        content.update({.instances = {{.rect = {10.0f, 10.0f, 40.0f, 20.0f}}}});
        keys.update({.bindings = {{.key = key, .onPressed = [&] { ++fired; }}}, .enabled = enabled, .content = &content});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));

    key = KeyCode::B;
    CHECK(frames.run(update).clean());
    CHECK(Tap(host, KeyCode::B));
    CHECK(fired == 1);
    CHECK(frames.run(update).clean());

    enabled = false;
    CHECK(frames.run(update).clean());
    CHECK_FALSE(Tap(host, KeyCode::B));
    CHECK(Retains(frames, update));
    CHECK(fired == 1);
    CHECK(frames.rebuilds == 1);
    CHECK(frames.restructures == 0);
}

TEST_CASE("A single line grid keeps editing keys and hands Up, Down, unclaimed Enter and Escape to its ancestors",
          "[core][sakura][input][text-grid]") {
    std::vector<std::string> log;
    Node root("root", log, {0.0f, 0.0f, 200.0f, 30.0f});
    Sakura::TextGrid grid;
    root.adopt(grid);
    grid.update({.id = "single", .value = "abc", .editable = true, .fontSize = 15.0f, .singleLine = true});
    root.place(MakeContext({200.0f, 30.0f}));
    grid.focus();
    REQUIRE(FocusOwner(root) == &grid);
    const auto press = [&](KeyCode key) {
        return Sakura::Access::key(root, Press(key));
    };

    CHECK(press(KeyCode::LeftArrow));
    CHECK(press(KeyCode::Backspace));
    CHECK(log.empty());

    CHECK_FALSE(press(KeyCode::UpArrow));
    CHECK_FALSE(press(KeyCode::DownArrow));
    CHECK_FALSE(press(KeyCode::Enter));
    CHECK(std::count(log.begin(), log.end(), "root:key") == 3);
    CHECK(grid.focused());

    log.clear();
    CHECK_FALSE(press(KeyCode::Escape));
    CHECK(log == std::vector<std::string>{"root:key", "root:shortcut"});
    CHECK_FALSE(grid.focused());
    CHECK(FocusOwner(root) == nullptr);
}

TEST_CASE("Text grid caret moves by line with Command and by word with Control or Option on every platform",
          "[core][sakura][input][text-grid]") {
    Sakura::TextGrid grid;
    Host host(grid);
    grid.update({.id = "words", .value = "alpha beta gamma", .editable = true, .fontSize = 15.0f});
    host.place(MakeContext({300.0f, 60.0f}), {0.0f, 0.0f, 300.0f, 60.0f});
    grid.focus();
    REQUIRE(FocusOwner(host) == &grid);

    grid.setCursor({0, 16});
    CHECK(Tap(host, KeyCode::LeftArrow, {.super = true}));
    CHECK(grid.cursor() == Sakura::TextGrid::Position{0, 0});
    CHECK(Tap(host, KeyCode::RightArrow, {.control = true}));
    const U64 word = grid.cursor().column;
    CHECK(word > 0);
    CHECK(word < 16);
    CHECK(Tap(host, KeyCode::RightArrow, {.alt = true}));
    CHECK(grid.cursor().column > word);
    CHECK(grid.cursor().column < 16);
    CHECK(Tap(host, KeyCode::RightArrow, {.super = true}));
    CHECK(grid.cursor() == Sakura::TextGrid::Position{0, 16});

    Type(host, "!");
    CHECK(grid.cursor().column == 17);
    CHECK_FALSE(Sakura::Access::text(host, {U'?', {.control = true}}));
    CHECK_FALSE(Sakura::Access::text(host, {U'?', {.super = true}}));
    CHECK(grid.cursor().column == 17);
}

TEST_CASE("Pasting follows the same single line and character rules as typing",
          "[core][sakura][input][text-grid]") {
    std::string value;
    Sakura::TextGrid grid;
    Host host(grid);
    Sakura::TextGrid::Config config{
        .id = "paste",
        .value = "",
        .editable = true,
        .fontSize = 15.0f,
        .singleLine = true,
        .onChange = [&](const std::string& next) { value = next; },
    };
    grid.update(config);
    SakuraTest::FakeWindow window;
    auto ctx = MakeContext({200.0f, 30.0f});
    ctx.render = &window;
    host.place(ctx, {0.0f, 0.0f, 200.0f, 30.0f});
    grid.focus();
    REQUIRE(FocusOwner(host) == &grid);

    Sakura::SetClipboardText(&window, "alpha\r\nbeta\ngamma");
    CHECK(Tap(host, KeyCode::V, {.super = true}));
    CHECK(value == "alpha beta gamma");

    config.characterFilter = [](U32 codepoint) { return codepoint >= '0' && codepoint <= '9'; };
    grid.update(config);
    REQUIRE(Tap(host, KeyCode::A, {.super = true}));
    Sakura::SetClipboardText(&window, "1a2\n3");
    CHECK(Tap(host, KeyCode::V, {.control = true}));
    CHECK(value == "123");
}

TEST_CASE("Copying and cutting write the selection to the window clipboard",
          "[core][sakura][input][text-grid]") {
    std::string value = "alpha beta";
    Sakura::TextGrid grid;
    Host host(grid);
    Sakura::TextGrid::Config config{
        .id = "copy",
        .value = value,
        .editable = true,
        .fontSize = 15.0f,
        .singleLine = true,
        .onChange = [&](const std::string& next) { value = next; },
    };
    grid.update(config);
    SakuraTest::FakeWindow window;
    auto ctx = MakeContext({200.0f, 30.0f});
    ctx.render = &window;
    host.place(ctx, {0.0f, 0.0f, 200.0f, 30.0f});
    grid.focus();
    REQUIRE(FocusOwner(host) == &grid);

    REQUIRE(Tap(host, KeyCode::A, {.super = true}));
    CHECK(Tap(host, KeyCode::C, {.super = true}));
    CHECK(Sakura::GetClipboardText(&window) == "alpha beta");
    CHECK(value == "alpha beta");

    Sakura::SetClipboardText(&window, "");
    REQUIRE(Tap(host, KeyCode::A, {.control = true}));
    CHECK(Tap(host, KeyCode::X, {.control = true}));
    CHECK(Sakura::GetClipboardText(&window) == "alpha beta");
    CHECK(value.empty());
}

TEST_CASE("Clicks without a pointer position count as outside clicks and are not routed",
          "[core][sakura][input][router]") {
    Sakura::InputFrame frame;
    Sakura::InputRouter router;
    frame.advance(0.1f);
    frame.pointer.reset();
    frame.buttons[0] = true;
    frame.events = {MouseEvent{.type = MouseEventType::Click, .button = MouseButton::Left, .positioned = false}};
    const auto left = router.route(frame, {0.0f, 0.0f, 100.0f, 100.0f});
    CHECK(left.outsideClick);
    CHECK_FALSE(left.active);
    CHECK(Only<MouseEvent>(left.events).empty());

    frame.advance(0.1f);
    frame.buttons = {false, false, true};
    frame.events = {MouseEvent{.type = MouseEventType::Click, .button = MouseButton::Middle, .positioned = false}};
    const auto middle = router.route(frame, {0.0f, 0.0f, 100.0f, 100.0f});
    CHECK_FALSE(middle.outsideClick);
    CHECK(Only<MouseEvent>(middle.events).empty());
}

TEST_CASE("A press, move and release in one frame route at their own positions",
          "[core][sakura][input][router]") {
    using Queue = Jetstream::Render::InputQueue;
    const auto run = [](const Rect& screen) {
        Jetstream::Render::InputState state;
        Sakura::InputFrame frame;
        Sakura::InputRouter router;
        std::vector<MouseEvent> mouse;
        Sakura::InputRouter::Output output;
        bool outsideClick = false;
        F64 clock = 1.0;
        std::vector<Queue::Event> batch = {
            Queue::Pointer{Extent2D<F32>{10.0f, 10.0f}},
            Queue::Button{MouseButton::Left, true},
            Queue::Pointer{Extent2D<F32>{90.0f, 90.0f}},
            Queue::Button{MouseButton::Left, false},
        };
        do {
            state.advance(clock, batch);
            batch.clear();
            clock += 1.0 / 60.0;
            Sakura::Private::SampleInput(frame, state);
            output = router.route(frame, screen);
            outsideClick |= output.outsideClick;
            const auto routed = Only<MouseEvent>(output.events);
            mouse.insert(mouse.end(), routed.begin(), routed.end());
        } while (state.hasPending());
        return std::tuple{mouse, output, outsideClick};
    };

    const auto [mouse, output, outsideClick] = run({0.0f, 0.0f, 100.0f, 100.0f});
    REQUIRE(mouse.size() == 4);
    CHECK(mouse[0].type == MouseEventType::Move);
    CHECK(mouse[0].position == Extent2D<F32>{0.1f, 0.1f});
    CHECK(mouse[1].type == MouseEventType::Click);
    CHECK(mouse[1].position == Extent2D<F32>{0.1f, 0.1f});
    CHECK(mouse[2].type == MouseEventType::Move);
    CHECK(mouse[2].position == Extent2D<F32>{0.9f, 0.9f});
    CHECK(mouse[3].type == MouseEventType::Release);
    CHECK(mouse[3].position == Extent2D<F32>{0.9f, 0.9f});
    CHECK(output.hovered);
    CHECK_FALSE(output.active);
    CHECK_FALSE(outsideClick);

    const auto [away, awayOutput, awayOutside] = run({50.0f, 50.0f, 100.0f, 100.0f});
    CHECK(awayOutside);
    CHECK(std::none_of(away.begin(), away.end(), [](const auto& event) {
        return event.type == MouseEventType::Click || event.type == MouseEventType::Release;
    }));
    CHECK(awayOutput.hovered);
}

TEST_CASE("Pointer validity follows input from the state to the router", "[core][sakura][input][router]") {
    using Queue = Jetstream::Render::InputQueue;
    const Rect screen = {0.0f, 0.0f, 100.0f, 100.0f};
    const auto run = [&](const std::vector<Queue::Event>& batch) {
        Jetstream::Render::InputState state;
        Sakura::InputFrame frame;
        Sakura::InputRouter router;
        std::vector<MouseEvent> routed;
        F64 clock = 1.0;
        auto queued = batch;
        do {
            state.advance(clock, queued);
            queued.clear();
            clock += 1.0 / 60.0;
            Sakura::Private::SampleInput(frame, state);
            const auto output = router.route(frame, screen);
            const auto mouse = Only<MouseEvent>(output.events);
            routed.insert(routed.end(), mouse.begin(), mouse.end());
        } while (state.hasPending());
        return routed;
    };

    const auto wheel = run({
        Queue::Pointer{Extent2D<F32>{50.0f, 50.0f}},
        Queue::Pointer{},
        Queue::Wheel{{0.0f, 1.0f}},
    });
    CHECK(std::none_of(wheel.begin(), wheel.end(), [](const auto& event) {
        return event.type == MouseEventType::Scroll;
    }));

    const auto release = run({
        Queue::Pointer{Extent2D<F32>{50.0f, 50.0f}},
        Queue::Button{MouseButton::Left, true},
        Queue::Pointer{},
        Queue::Button{MouseButton::Left, false},
    });
    const auto click = std::find_if(release.begin(), release.end(), [](const auto& event) {
        return event.type == MouseEventType::Click;
    });
    const auto released = std::find_if(release.begin(), release.end(), [](const auto& event) {
        return event.type == MouseEventType::Release;
    });
    REQUIRE(click != release.end());
    REQUIRE(released != release.end());
    CHECK(click->position == Extent2D<F32>{0.5f, 0.5f});
    CHECK(released->cancel);
    CHECK(std::none_of(release.begin(), release.end(), [](const auto& event) {
        return event.type == MouseEventType::Release && !event.cancel;
    }));
    CHECK(std::any_of(release.begin(), release.end(), [](const auto& event) {
        return event.type == MouseEventType::Cancel;
    }));
}

TEST_CASE("Masked text grids never write their text to the clipboard",
          "[core][sakura][input][text-grid]") {
    std::string value = "secret";
    Sakura::TextGrid grid;
    Host host(grid);
    grid.update({
        .id = "masked",
        .value = value,
        .editable = true,
        .fontSize = 15.0f,
        .singleLine = true,
        .masked = true,
        .onChange = [&](const std::string& next) { value = next; },
    });
    SakuraTest::FakeWindow window;
    auto ctx = MakeContext({200.0f, 30.0f});
    ctx.render = &window;
    host.place(ctx, {0.0f, 0.0f, 200.0f, 30.0f});
    grid.focus();
    REQUIRE(FocusOwner(host) == &grid);

    Sakura::SetClipboardText(&window, "untouched");
    REQUIRE(Tap(host, KeyCode::A, {.control = true}));
    Tap(host, KeyCode::C, {.control = true});
    CHECK(Sakura::GetClipboardText(&window) == "untouched");
    Tap(host, KeyCode::X, {.control = true});
    CHECK(Sakura::GetClipboardText(&window) == "untouched");
    CHECK(value == "secret");

    Sakura::SetClipboardText(&window, "typed");
    CHECK(Tap(host, KeyCode::V, {.control = true}));
    CHECK(value == "typed");

    Type(host, "(\"[");
    CHECK(value == "typed(\"[");
    Type(host, ")");
    CHECK(value == "typed(\"[)");

    Tap(host, KeyCode::LeftArrow);
    Tap(host, KeyCode::Backspace);
    CHECK(value == "typed(\")");

    grid.update({.id = "masked", .value = "    ", .editable = true, .fontSize = 15.0f, .singleLine = true,
                 .masked = true, .onChange = [&](const std::string& next) { value = next; }});
    host.place(ctx, {0.0f, 0.0f, 200.0f, 30.0f});
    Tap(host, KeyCode::End);
    Tap(host, KeyCode::Backspace);
    CHECK(value == "   ");
}
