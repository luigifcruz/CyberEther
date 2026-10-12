#include <catch2/catch_test_macros.hpp>

#include "jetstream/render/sakura/components/retained/canvas.hh"
#include "jetstream/render/sakura/components/retained/chat.hh"
#include "jetstream/render/sakura/components/retained/dropdown.hh"
#include "harness.hh"

#include <imgui_internal.h>
#include <type_traits>

using namespace Jetstream;

namespace {

struct ChatProbe : Sakura::Retained::Chat {
    using Component::event;
    using Component::frame;
    using Component::layoutRoot;
};

struct DropdownProbe : Sakura::Retained::Dropdown {
    using Dropdown::event;
};

struct DropdownRoot : Sakura::Component {
    DropdownProbe dropdown;
    Jetstream::Rect field{20.0f, 20.0f, 200.0f, 30.0f};
    F32 ratio = 1.0f;

    DropdownRoot() {
        add(dropdown);
    }

    void layout(const Sakura::Context& ctx) override {
        ratio = ctx.pixelRatio;
        layoutChild(ctx, dropdown, field);
    }
};

void Click(Extent2D<F32> position,
           const std::function<bool(const MouseEvent&)>& dispatch) {
    dispatch({.type = MouseEventType::Click, .button = MouseButton::Left,
              .position = position});
    dispatch({.type = MouseEventType::Release, .button = MouseButton::Left,
              .position = position});
}

}  // namespace

TEST_CASE("Retained chat mounts in a canvas and routes controls through callbacks",
          "[core][sakura][chat]") {
    static_assert(std::is_base_of_v<Sakura::Component, Sakura::Retained::Chat>);
    static_assert(!std::is_move_constructible_v<Sakura::Retained::Chat>);
    SakuraTest::HeadlessUi ui;
    ChatProbe chat;
    Sakura::Retained::Canvas canvas;
    canvas.mount(chat);
    canvas.update({.id = "chat-contract", .size = {400.0f, 300.0f}});
    I32 cancellations = 0;
    Sakura::Retained::Chat::Config config{
        .id = "chat-content",
        .snapshot = {.streaming = true},
        .onCancel = [&] { cancellations++; },
    };
    REQUIRE(chat.update(config));
    ui.frame([&] { canvas.render(ui.sakura()); });
    const auto bounds = chat.frame();
    REQUIRE_FALSE(bounds.empty());
    const F32 ratio = bounds.width / 400.0f;
    const Extent2D<F32> stop{bounds.right() - 40.0f * ratio,
                            bounds.bottom() - 20.0f * ratio};
    Click(stop, [&](const auto& event) { return chat.event(event); });
    REQUIRE(cancellations == 1);

    config.snapshot.streaming = false;
    config.snapshot.messages.push_back({
        Sakura::Retained::Chat::Role::Assistant, "", {
            {.kind = Sakura::Retained::Chat::PartKind::Markdown, .key = 0, .revision = 1,
             .text = "Partial response"},
            {.kind = Sakura::Retained::Chat::PartKind::Markdown, .key = 1, .revision = 1,
             .text = "> [!ERROR]\n> HTTP 429: Retry later"},
        },
    });
    REQUIRE(chat.update(config));
    ui.frame([&] { canvas.render(ui.sakura()); });
    Click(stop, [&](const auto& event) { return chat.event(event); });
    CHECK(cancellations == 1);
    CHECK(ImGui::FindWindowByName("Agent Chat###agent-chat:window") == nullptr);
}

TEST_CASE("Retained chat remeasures content when its identity changes with reused revisions",
          "[core][sakura][chat]") {
    using Chat = Sakura::Retained::Chat;
    SakuraTest::HeadlessUi ui;
    ChatProbe chat;
    Chat::Part part{.key = 1, .revision = 1};
    SECTION("markdown") {
        part.kind = Chat::PartKind::Markdown;
    }
    SECTION("disclosure summary") {
        part.kind = Chat::PartKind::Disclosure;
        part.label = "Details";
    }
    const auto show = [&](const std::string& id, U64 lines) {
        std::string text = "Line";
        for (U64 i = 1; i < lines; ++i) {
            text += "\nLine";
        }
        if (part.kind == Chat::PartKind::Markdown) {
            part.text = std::move(text);
        } else {
            part.summary = std::move(text);
        }
        Chat::Config config{.id = id, .snapshot = {.generation = 1}};
        config.snapshot.messages.push_back({Chat::Role::Assistant, "", {part}});
        chat.update(std::move(config));
        chat.layoutRoot(ui.sakura(), {0.0f, 0.0f, 400.0f, 300.0f});
    };
    const auto scroll = [&] {
        // The transcript's scrollbar gutter avoids scrolling an individual text view.
        return chat.event({.type = MouseEventType::Scroll, .position = {391.0f, 20.0f},
                           .scroll = {0.0f, 1.0f}});
    };
    show("short-chat", 1);
    REQUIRE_FALSE(scroll());
    show("long-chat", 40);
    CHECK(scroll());
    show("another-short-chat", 1);
    CHECK_FALSE(scroll());
}

TEST_CASE("Retained dropdowns preserve identities and close when disabled",
          "[core][sakura][dropdown]") {
    SakuraTest::HeadlessUi ui;
    DropdownRoot root;
    Sakura::Retained::Canvas canvas;
    canvas.mount(root);
    canvas.update({.id = "dropdown-contract", .size = {400.0f, 300.0f}});
    std::string selected;
    Sakura::Retained::Dropdown::Config config{
        .id = "models",
        .options = {{"provider/model-a", "Same label"},
                    {"provider/model-b", "Same label"}},
        .onSelect = [&](const std::string& value) { selected = value; },
    };
    const auto frame = [&] {
        root.dropdown.update(config);
        ui.frame([&] { canvas.render(ui.sakura()); });
    };
    const auto click = [&](Extent2D<F32> position) {
        Click(position,
              [&](const auto& event) { return root.dropdown.event(event); });
    };
    frame();
    click(root.field.center());
    frame();
    const F32 rowHeight = config.fontSize + 8.0f * root.ratio;
    const Extent2D<F32> secondRow{
        root.field.x + 20.0f,
        root.field.bottom() + 8.0f * root.ratio + 1.5f * rowHeight,
    };
    click(secondRow);
    REQUIRE(selected == "provider/model-b");
    selected.clear();

    click(root.field.center());
    frame();
    config.disabled = true;
    frame();
    config.disabled = false;
    frame();
    click(secondRow);
    CHECK(selected.empty());
}
