#include <catch2/catch_test_macros.hpp>

#include "harness.hh"
#include "fonts.hh"
#include "component_harness.hh"
#include "scene.hh"

#include <jetstream/render/sakura/components/dropdown.hh>
#include <jetstream/render/sakura/components/menu.hh>
#include <jetstream/render/sakura/components/polyline.hh>
#include <jetstream/render/sakura/components/shell.hh>
#include <jetstream/render/sakura/toast.hh>
#include <jetstream/render/tools/imgui_icons_ext.hh>

#include "render/sakura/toast_queue.hh"

#include <algorithm>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

using namespace Jetstream;
using namespace SakuraTest;

namespace {


struct NestedRequester : Sakura::Component {
    Sakura::Component* child = nullptr;
    Sakura::PopupRequest request;
    bool requesting = true;

    void hold(Sakura::Component& next) {
        child = &next;
        addChild(next);
    }

    void withdraw() {
        requesting = false;
        invalidate(Dirty::Layout);
    }

 protected:
    void layout(const Sakura::Context& ctx) override {
        if (requesting) {
            Sakura::RequestPopup(ctx, request);
        }
        if (child) {
            layoutChild(ctx, *child, frame());
        }
    }
};

struct ToastProbe : Sakura::ToastLayer {
    using Component::children;

    template<typename T>
    std::vector<T*> parts() const {
        std::vector<T*> found;
        for (auto* child : children()) {
            if (auto* match = dynamic_cast<T*>(child); match && child->visible()) {
                found.push_back(match);
            }
        }
        return found;
    }

    U64 shown() const {
        U64 total = 0;
        for (const auto* child : children()) {
            total += child->visible() ? 1 : 0;
        }
        return total;
    }
};

struct FloatingProbe : Sakura::FloatingLayer {
    using Component::children;
    using Sakura::FloatingLayer::cursorAt;

    template<typename T>
    std::vector<T*> parts() const {
        std::vector<T*> found;
        for (auto* child : children()) {
            if (auto* match = dynamic_cast<T*>(child); match && child->visible()) {
                found.push_back(match);
            }
        }
        return found;
    }
};

struct TextCursorProbe : Probe {
 protected:
    std::optional<SurfaceCursor> cursorAt(const Extent2D<F32>&) const override {
        return SurfaceCursor::TextInput;
    }
};

Sakura::Context WindowContext(SakuraTest::FakeWindow& window, Extent2D<F32> size = {800.0f, 600.0f}) {
    auto ctx = MakeContext(size);
    ctx.render = &window;
    return ctx;
}

Rect PlacePopup(Probe& content, Sakura::PopupRequest request, Rect bounds = kScreen) {
    request.content = &content;
    LayerSlot slot;
    slot.request({request});
    LayerHost host({&slot});
    host.place(MakeContext({bounds.width, bounds.height}), bounds);
    return content.laidFrame;
}

}  // namespace

TEST_CASE("Below popups open under the anchor and are at least as wide as it",
          "[core][sakura][shell]") {
    Probe narrow({100.0f, 50.0f});
    CHECK(PlacePopup(narrow, {.anchor = {200.0f, 100.0f, 80.0f, 20.0f}}) == Rect{200.0f, 120.0f, 100.0f, 50.0f});

    Probe wide({100.0f, 50.0f});
    CHECK(PlacePopup(wide, {.anchor = {200.0f, 100.0f, 150.0f, 20.0f}}) == Rect{200.0f, 120.0f, 150.0f, 50.0f});
}

TEST_CASE("Fixed width popups measure their height at that width", "[core][sakura][shell]") {
    Flow content(300.0f, 15.0f);
    const Rect placed = PlacePopup(content, {.anchor = {200.0f, 100.0f, 80.0f, 20.0f}, .panel = false, .size = {100.0f, 0.0f}});
    CHECK(placed.width == 100.0f);
    CHECK(placed.height == 45.0f);
}

TEST_CASE("Below popups flip above the anchor when there is no room below", "[core][sakura][shell]") {
    Probe flipped({100.0f, 50.0f});
    CHECK(PlacePopup(flipped, {.anchor = {200.0f, 570.0f, 80.0f, 20.0f}}) == Rect{200.0f, 520.0f, 100.0f, 50.0f});

    Probe clamped({100.0f, 50.0f});
    CHECK(PlacePopup(clamped, {.anchor = {200.0f, 30.0f, 80.0f, 560.0f}}) == Rect{200.0f, 550.0f, 100.0f, 50.0f});
}

TEST_CASE("Point popups open at the anchor and flip away from the right and bottom edges",
          "[core][sakura][shell]") {
    const auto point = [](Rect anchor) {
        Probe content({100.0f, 50.0f});
        return PlacePopup(content, {.anchor = anchor, .placement = Sakura::PopupPlacement::Point});
    };

    CHECK(point({300.0f, 200.0f, 1.0f, 1.0f}) == Rect{300.0f, 200.0f, 100.0f, 50.0f});
    CHECK(point({750.0f, 200.0f, 1.0f, 1.0f}) == Rect{650.0f, 200.0f, 100.0f, 50.0f});
    CHECK(point({300.0f, 580.0f, 1.0f, 1.0f}) == Rect{300.0f, 530.0f, 100.0f, 50.0f});
    CHECK(point({750.0f, 580.0f, 1.0f, 1.0f}) == Rect{650.0f, 530.0f, 100.0f, 50.0f});
}

TEST_CASE("Side popups open to the right of the anchor and flip left at the right edge",
          "[core][sakura][shell]") {
    const auto side = [](Rect anchor) {
        Probe content({100.0f, 50.0f});
        return PlacePopup(content, {.anchor = anchor, .placement = Sakura::PopupPlacement::Side});
    };

    CHECK(side({200.0f, 100.0f, 80.0f, 20.0f}) == Rect{280.0f, 92.0f, 100.0f, 50.0f});
    CHECK(side({700.0f, 100.0f, 80.0f, 20.0f}) == Rect{600.0f, 92.0f, 100.0f, 50.0f});
    CHECK(side({200.0f, 2.0f, 80.0f, 20.0f}) == Rect{280.0f, 0.0f, 100.0f, 50.0f});
}

TEST_CASE("Tooltip popups sit below the cursor and move above it near the bottom edge",
          "[core][sakura][shell]") {
    const auto tooltip = [](Rect anchor) {
        Probe content({100.0f, 50.0f});
        return PlacePopup(content, {.anchor = anchor, .placement = Sakura::PopupPlacement::Tooltip});
    };

    CHECK(tooltip({300.0f, 200.0f, 1.0f, 1.0f}) == Rect{300.0f, 224.0f, 100.0f, 50.0f});
    CHECK(tooltip({300.0f, 560.0f, 1.0f, 1.0f}) == Rect{300.0f, 502.0f, 100.0f, 50.0f});
    CHECK(tooltip({750.0f, 200.0f, 1.0f, 1.0f}) == Rect{700.0f, 224.0f, 100.0f, 50.0f});
}

TEST_CASE("Center popups ignore the anchor and center on the layer", "[core][sakura][shell]") {
    Probe content({100.0f, 50.0f});
    CHECK(PlacePopup(content, {.anchor = {10.0f, 10.0f, 1.0f, 1.0f}, .placement = Sakura::PopupPlacement::Center}) ==
          Rect{350.0f, 275.0f, 100.0f, 50.0f});

    Probe offset({100.0f, 50.0f});
    CHECK(PlacePopup(offset, {.placement = Sakura::PopupPlacement::Center}, {40.0f, 20.0f, 400.0f, 300.0f}) ==
          Rect{190.0f, 145.0f, 100.0f, 50.0f});
}

TEST_CASE("Popups with an explicit size skip measurement and stay within the layer",
          "[core][sakura][shell]") {
    Probe sized({100.0f, 50.0f});
    CHECK(PlacePopup(sized, {.anchor = {200.0f, 100.0f, 80.0f, 20.0f}, .size = {240.0f, 120.0f}}) ==
          Rect{200.0f, 120.0f, 240.0f, 120.0f});
    CHECK(sized.measures == 0);

    Probe oversized({100.0f, 50.0f});
    CHECK(PlacePopup(oversized, {.placement = Sakura::PopupPlacement::Center, .size = {1200.0f, 900.0f}}) ==
          Rect{0.0f, 0.0f, 800.0f, 600.0f});
}

TEST_CASE("Popup max width caps the measured content width", "[core][sakura][shell]") {
    Probe content({500.0f, 50.0f});
    CHECK(PlacePopup(content, {.anchor = {100.0f, 100.0f, 1.0f, 1.0f}, .placement = Sakura::PopupPlacement::Point,
                               .maxWidth = 300.0f}) == Rect{100.0f, 100.0f, 300.0f, 50.0f});

    Probe paneled({500.0f, 50.0f});
    CHECK(PlacePopup(paneled, {.anchor = {100.0f, 100.0f, 1.0f, 1.0f}, .placement = Sakura::PopupPlacement::Point,
                               .panel = true, .maxWidth = 300.0f}) == Rect{112.0f, 112.0f, 276.0f, 50.0f});
}

TEST_CASE("Panel popups inset their content by the panel padding", "[core][sakura][shell]") {
    Probe content({100.0f, 50.0f});
    CHECK(PlacePopup(content, {.anchor = {200.0f, 100.0f, 80.0f, 20.0f}, .panel = true, .padding = 10.0f}) ==
          Rect{210.0f, 130.0f, 100.0f, 50.0f});

    Probe flush({100.0f, 50.0f});
    CHECK(PlacePopup(flush, {.anchor = {200.0f, 100.0f, 80.0f, 20.0f}, .panel = false, .padding = 10.0f}) ==
          Rect{200.0f, 120.0f, 100.0f, 50.0f});
}

TEST_CASE("Popup requests for the same content keep only the first placement", "[core][sakura][shell]") {
    Probe content({100.0f, 50.0f});
    LayerSlot slot;
    slot.request({
        {.content = &content, .anchor = {100.0f, 100.0f, 1.0f, 1.0f}, .placement = Sakura::PopupPlacement::Point},
        {.content = &content, .anchor = {400.0f, 300.0f, 1.0f, 1.0f}, .placement = Sakura::PopupPlacement::Point},
    });
    LayerHost host({&slot});
    host.place(MakeContext(), kScreen);

    CHECK(content.laidFrame == Rect{100.0f, 100.0f, 100.0f, 50.0f});
    CHECK(content.layouts == 1);
}

TEST_CASE("Popup layer covers only the popup unless the popup is dismissible", "[core][sakura][shell]") {
    Probe content({100.0f, 50.0f});
    LayerSlot slot;
    slot.request({{.content = &content, .anchor = {100.0f, 100.0f, 1.0f, 1.0f},
                      .placement = Sakura::PopupPlacement::Point}});
    LayerHost host({&slot});
    host.place(MakeContext(), kScreen);

    CHECK(host.popups.covers({150.0f, 120.0f}));
    CHECK(host.popups.interactiveAt({150.0f, 120.0f}));
    CHECK_FALSE(host.popups.covers({500.0f, 500.0f}));
    CHECK_FALSE(host.popups.covers({99.0f, 120.0f}));
    CHECK_FALSE(host.popups.covers({150.0f, 150.0f}));

    auto dismissible = slot.requests;
    dismissible[0].onDismiss = [] {};
    slot.request(dismissible);
    host.place(MakeContext(), kScreen);

    CHECK(host.popups.covers({500.0f, 500.0f}));
    CHECK_FALSE(host.popups.interactiveAt({500.0f, 500.0f}));
}

TEST_CASE("Clicking outside every popup dismisses all of them", "[core][sakura][shell]") {
    Probe first({100.0f, 50.0f});
    Probe second({100.0f, 50.0f});
    U64 firstDismissed = 0;
    U64 secondDismissed = 0;
    LayerSlot slot;
    slot.requests = {
        {.content = &first, .anchor = {100.0f, 100.0f, 1.0f, 1.0f}, .placement = Sakura::PopupPlacement::Point,
         .onDismiss = [&] { ++firstDismissed; }},
        {.content = &second, .anchor = {400.0f, 100.0f, 1.0f, 1.0f}, .placement = Sakura::PopupPlacement::Point,
         .onDismiss = [&] { ++secondDismissed; }},
    };
    LayerHost host({&slot});
    host.place(MakeContext(), kScreen);

    CHECK(host.click(120.0f, 120.0f));
    CHECK(first.count(MouseEventType::Click) + second.count(MouseEventType::Click) == 1);
    CHECK(firstDismissed == 0);
    CHECK(secondDismissed == 0);

    CHECK(host.click(600.0f, 500.0f));
    CHECK(firstDismissed == 1);
    CHECK(secondDismissed == 1);
    CHECK(first.count(MouseEventType::Click) + second.count(MouseEventType::Click) == 1);
}

TEST_CASE("Clicks inside a popup passthrough rect reach the layer below without dismissing",
          "[core][sakura][shell]") {
    Probe content({100.0f, 50.0f});
    Probe below({200.0f, 30.0f});
    U64 dismissed = 0;
    LayerSlot slot;
    slot.hold(below, {0.0f, 0.0f, 200.0f, 30.0f});
    slot.requests = {{.content = &content, .anchor = {100.0f, 100.0f, 1.0f, 1.0f},
                      .placement = Sakura::PopupPlacement::Point, .onDismiss = [&] { ++dismissed; },
                      .passthrough = Rect{0.0f, 0.0f, 200.0f, 30.0f}}};
    LayerHost host({&slot});
    host.place(MakeContext(), kScreen);

    CHECK_FALSE(host.popups.covers({50.0f, 10.0f}));
    CHECK(host.click(50.0f, 10.0f));
    CHECK(below.count(MouseEventType::Click) == 1);
    CHECK(dismissed == 0);

    CHECK(host.click(50.0f, 300.0f));
    CHECK(dismissed == 1);
}

TEST_CASE("Non-interactive popups neither cover nor capture clicks", "[core][sakura][shell]") {
    Probe tooltip({100.0f, 50.0f});
    Probe below({800.0f, 600.0f});
    LayerSlot slot;
    slot.hold(below, kScreen);
    slot.request({{.content = &tooltip, .anchor = {100.0f, 100.0f, 1.0f, 1.0f},
                      .placement = Sakura::PopupPlacement::Point, .interactive = false}});
    LayerHost host({&slot});
    host.place(MakeContext(), kScreen);

    CHECK(tooltip.laidFrame == Rect{100.0f, 100.0f, 100.0f, 50.0f});
    CHECK_FALSE(host.popups.covers({120.0f, 120.0f}));
    CHECK_FALSE(host.popups.interactiveAt({120.0f, 120.0f}));
    CHECK(host.click(120.0f, 120.0f));
    CHECK(below.count(MouseEventType::Click) == 1);
    CHECK(tooltip.count(MouseEventType::Click) == 0);
}

TEST_CASE("Popup requests last a single frame", "[core][sakura][shell]") {
    Probe content({100.0f, 50.0f});
    LayerSlot slot;
    slot.request({{.content = &content, .anchor = {100.0f, 100.0f, 1.0f, 1.0f},
                      .placement = Sakura::PopupPlacement::Point, .onDismiss = [] {}}});
    LayerHost host({&slot});
    host.place(MakeContext(), kScreen);
    REQUIRE(host.popups.interactiveAt({120.0f, 120.0f}));

    slot.request({});
    host.place(MakeContext(), kScreen);

    CHECK_FALSE(host.popups.interactiveAt({120.0f, 120.0f}));
    CHECK_FALSE(host.popups.covers({500.0f, 500.0f}));
    CHECK(content.layouts == 1);
}

TEST_CASE("Popup component requests its content only while open", "[core][sakura][shell]") {
    Probe content({100.0f, 50.0f});
    Sakura::Popup popup;
    U64 dismissed = 0;
    Sakura::Popup::Config config{
        .content = &content,
        .open = true,
        .placement = Sakura::PopupPlacement::Below,
        .onDismiss = [&] { ++dismissed; },
    };
    popup.update(config);
    LayerSlot slot;
    slot.hold(popup, {100.0f, 100.0f, 80.0f, 20.0f});
    LayerHost host({&slot});
    host.place(MakeContext(), kScreen);

    CHECK(content.laidFrame == Rect{100.0f, 120.0f, 100.0f, 50.0f});
    CHECK(host.popups.interactiveAt({150.0f, 150.0f}));

    CHECK(host.click(600.0f, 500.0f));
    CHECK(dismissed == 1);

    config.open = false;
    popup.update(config);
    host.place(MakeContext(), kScreen);

    CHECK_FALSE(host.popups.interactiveAt({150.0f, 150.0f}));
    CHECK_FALSE(host.popups.covers({600.0f, 500.0f}));
    CHECK(content.layouts == 1);
}

TEST_CASE("Popup component anchored at the cursor keeps the position from when it opened",
          "[core][sakura][shell]") {
    Probe content({100.0f, 50.0f});
    Sakura::Popup popup;
    popup.update({
        .content = &content,
        .open = true,
        .placement = Sakura::PopupPlacement::Point,
        .atCursor = true,
        .panel = true,
        .padding = 6.0f,
    });
    LayerSlot slot;
    slot.hold(popup, {0.0f, 0.0f, 50.0f, 50.0f});
    LayerHost host({&slot});

    auto ctx = MakeContext();
    ctx.mouse = Extent2D<F32>{300.0f, 200.0f};
    host.place(ctx, kScreen);
    CHECK(content.laidFrame == Rect{306.0f, 206.0f, 100.0f, 50.0f});

    ctx.mouse = Extent2D<F32>{500.0f, 400.0f};
    host.place(ctx, kScreen);
    CHECK(content.laidFrame == Rect{306.0f, 206.0f, 100.0f, 50.0f});
}

TEST_CASE("Popup requests are anchored in the coordinates of a translated layout",
          "[core][sakura][shell]") {
    Probe content({100.0f, 50.0f});
    LayerSlot slot;
    slot.onLayout = [&](const Sakura::Context& ctx) {
        Sakura::RequestPopup(ctx.translated({40.0f, 30.0f}),
                               {.content = &content, .anchor = {100.0f, 100.0f, 1.0f, 1.0f},
                                .placement = Sakura::PopupPlacement::Point});
    };
    LayerHost host({&slot});
    host.place(MakeContext(), kScreen);

    CHECK(host.popups.interactiveAt({145.0f, 135.0f}));
    CHECK_FALSE(host.popups.interactiveAt({105.0f, 105.0f}));
}

TEST_CASE("HUD anchors place the padded content inside the margin", "[core][sakura][shell]") {
    const auto hud = [](Sakura::Anchor anchor) {
        Probe content({100.0f, 40.0f});
        Sakura::HudLayer layer;
        layer.update({.huds = {{.content = &content, .anchor = anchor}}});
        LayerHost host({&layer});
        host.place(MakeContext(), kScreen);
        return content.laidFrame;
    };

    CHECK(hud(Sakura::Anchor::TopLeft) == Rect{24.0f, 24.0f, 100.0f, 40.0f});
    CHECK(hud(Sakura::Anchor::TopCenter) == Rect{350.0f, 24.0f, 100.0f, 40.0f});
    CHECK(hud(Sakura::Anchor::TopRight) == Rect{676.0f, 24.0f, 100.0f, 40.0f});
    CHECK(hud(Sakura::Anchor::Center) == Rect{350.0f, 280.0f, 100.0f, 40.0f});
    CHECK(hud(Sakura::Anchor::BottomLeft) == Rect{24.0f, 536.0f, 100.0f, 40.0f});
    CHECK(hud(Sakura::Anchor::BottomCenter) == Rect{350.0f, 536.0f, 100.0f, 40.0f});
    CHECK(hud(Sakura::Anchor::BottomRight) == Rect{676.0f, 536.0f, 100.0f, 40.0f});
}

TEST_CASE("HUD margin, padding, and explicit size shape the HUD rect", "[core][sakura][shell]") {
    Probe content({100.0f, 40.0f});
    Sakura::HudLayer layer;
    layer.update({.huds = {{.content = &content, .anchor = Sakura::Anchor::BottomRight, .margin = 20.0f,
                            .padding = {6.0f, 4.0f}}}});
    LayerHost host({&layer});
    host.place(MakeContext(), kScreen);
    CHECK(content.laidFrame == Rect{674.0f, 536.0f, 100.0f, 40.0f});

    layer.update({.huds = {{.content = &content, .anchor = Sakura::Anchor::TopLeft,
                            .size = Extent2D<F32>{200.0f, 80.0f}}}});
    const U64 measured = content.measures;
    host.place(MakeContext(), kScreen);
    CHECK(content.laidFrame == Rect{24.0f, 24.0f, 176.0f, 56.0f});
    CHECK(content.measures == measured);
    CHECK(layer.covers({205.0f, 85.0f}));
    CHECK_FALSE(layer.covers({215.0f, 85.0f}));
}

TEST_CASE("HUD layer covers only its HUDs", "[core][sakura][shell]") {
    Probe left({100.0f, 40.0f});
    Probe right({100.0f, 40.0f});
    Sakura::HudLayer layer;
    layer.update({.huds = {{.content = &left, .anchor = Sakura::Anchor::TopLeft},
                           {.content = &right, .anchor = Sakura::Anchor::BottomRight}}});
    LayerHost host({&layer});
    host.place(MakeContext(), kScreen);

    CHECK(layer.covers({12.0f, 12.0f}));
    CHECK(layer.covers({135.0f, 75.0f}));
    CHECK_FALSE(layer.covers({136.0f, 75.0f}));
    CHECK_FALSE(layer.covers({11.0f, 12.0f}));
    CHECK(layer.covers({700.0f, 560.0f}));
    CHECK_FALSE(layer.covers({400.0f, 300.0f}));
}

TEST_CASE("Clickable HUDs call onClick for clicks their content does not handle", "[core][sakura][shell]") {
    Probe content({100.0f, 40.0f});
    Probe plain({100.0f, 40.0f});
    U64 clicked = 0;
    U64 plainClicked = 0;
    Sakura::HudLayer layer;
    layer.update({.huds = {
        {.content = &content, .anchor = Sakura::Anchor::TopLeft, .clickable = true, .onClick = [&] { ++clicked; }},
        {.content = &plain, .anchor = Sakura::Anchor::TopRight, .onClick = [&] { ++plainClicked; }},
    }});
    content.consume = false;
    plain.consume = false;
    Sakura::Layer base;
    LayerHost host({&base, &layer});
    host.place(MakeContext(), kScreen);

    CHECK(host.click(14.0f, 14.0f));
    CHECK(clicked == 1);
    CHECK(content.count(MouseEventType::Click) == 1);

    content.consume = true;
    CHECK(host.click(50.0f, 40.0f));
    CHECK(content.count(MouseEventType::Click) == 2);
    CHECK(clicked == 1);

    content.consume = false;
    host.click(670.0f, 14.0f);
    CHECK(plainClicked == 0);
    CHECK(clicked == 1);

    host.click(14.0f, 14.0f, MouseButton::Right);
    CHECK(clicked == 1);
}

TEST_CASE("Modal layer centers its content at the minimum width", "[core][sakura][shell]") {
    const ImGuiContextGuard imguiContext;
    SakuraTest::FakeWindow window;
    Probe content({100.0f, 50.0f});
    Sakura::ModalLayer modal;
    modal.update({.content = &content});
    LayerHost host({&modal});
    host.place(WindowContext(window), kScreen);

    const Rect frame = content.laidFrame;
    CHECK(frame.x == 137.0f);
    CHECK(frame.width == 526.0f);
    CHECK(frame.height == 50.0f);
    CHECK(frame.y > 150.0f);
    CHECK(frame.y < 300.0f - 25.0f);
}

TEST_CASE("Modal width is clamped to the layer and an explicit size is honoured",
          "[core][sakura][shell]") {
    const ImGuiContextGuard imguiContext;
    SakuraTest::FakeWindow window;
    Probe narrow({100.0f, 50.0f});
    Sakura::ModalLayer clamped;
    clamped.update({.content = &narrow});
    LayerHost small({&clamped});
    small.place(WindowContext(window, {400.0f, 600.0f}), {0.0f, 0.0f, 400.0f, 600.0f});
    CHECK(narrow.laidFrame.x == 31.0f);
    CHECK(narrow.laidFrame.width == 338.0f);

    Probe sized({100.0f, 50.0f});
    Sakura::ModalLayer fixed;
    fixed.update({.content = &sized, .size = Extent2D<F32>{400.0f, 300.0f}});
    LayerHost host({&fixed});
    host.place(WindowContext(window), kScreen);
    CHECK(sized.laidFrame.x == 212.0f);
    CHECK(sized.laidFrame.y == 162.0f);
    CHECK(sized.laidFrame.width == 376.0f);
    CHECK(sized.laidFrame.height > 50.0f);
    CHECK(sized.laidFrame.height < 276.0f - 48.0f);

    Probe wide({100.0f, 50.0f});
    Sakura::ModalLayer custom;
    custom.update({.content = &wide, .size = {300.0f, 0.0f}});
    LayerHost other({&custom});
    other.place(WindowContext(window), kScreen);
    CHECK(wide.laidFrame.x == 262.0f);
    CHECK(wide.laidFrame.width == 276.0f);
}

TEST_CASE("Modal layer covers everything only while it has content", "[core][sakura][shell]") {
    Probe content({100.0f, 50.0f});
    Sakura::ModalLayer modal;
    CHECK_FALSE(modal.covers({10.0f, 10.0f}));

    modal.update({.content = &content});
    CHECK(modal.covers({10.0f, 10.0f}));
    CHECK(modal.covers({790.0f, 590.0f}));

    modal.update({});
    CHECK_FALSE(modal.covers({400.0f, 300.0f}));
}

TEST_CASE("Modal footer close button calls onClose while dim clicks are swallowed",
          "[core][sakura][shell]") {
    const ImGuiContextGuard imguiContext;
    SakuraTest::FakeWindow window;
    Probe content({100.0f, 50.0f});
    Probe below({800.0f, 600.0f});
    U64 closed = 0;
    LayerSlot base;
    base.hold(below, kScreen);
    Sakura::ModalLayer modal;
    modal.update({.content = &content, .onClose = [&] { ++closed; }});
    LayerHost host({&base, &modal});
    host.place(WindowContext(window), kScreen);

    const F32 top = content.laidFrame.y - 12.0f;
    const F32 bottom = 600.0f - top;
    CHECK(host.click(20.0f, 20.0f));
    CHECK(closed == 0);
    CHECK(below.count(MouseEventType::Click) == 0);

    CHECK(host.click(400.0f, bottom - 12.0f - 20.0f));
    CHECK(closed == 1);

    const U64 before = content.count(MouseEventType::Click);
    CHECK(host.click(400.0f, content.laidFrame.y + 10.0f));
    CHECK(content.count(MouseEventType::Click) == before + 1);
    CHECK(closed == 1);
}

TEST_CASE("Floating windows are centered and cascade, with content below the title bar",
          "[core][sakura][shell]") {
    Probe first;
    Probe second;
    Sakura::FloatingLayer layer;
    layer.update({.windows = {
        {.id = "a", .title = "A", .content = &first, .size = {300.0f, 200.0f}},
        {.id = "b", .title = "B", .content = &second, .size = {300.0f, 200.0f}, .padding = 4.0f},
    }});
    LayerHost host({&layer});
    host.place(MakeContext(), kScreen);

    CHECK(first.laidFrame == Rect{250.0f, 232.0f, 300.0f, 168.0f});
    CHECK(second.laidFrame == Rect{278.0f, 260.0f, 292.0f, 160.0f});
    CHECK(layer.covers({250.0f, 200.0f}));
    CHECK(layer.covers({573.0f, 423.0f}));
    CHECK(layer.covers({576.0f, 426.0f}));
    CHECK_FALSE(layer.covers({576.0f, 300.0f}));
    CHECK_FALSE(layer.covers({249.0f, 200.0f}));
    CHECK_FALSE(layer.covers({100.0f, 100.0f}));
}

TEST_CASE("Floating window close button calls onClose", "[core][sakura][shell]") {
    Probe content;
    U64 closed = 0;
    U64 focused = 0;
    Sakura::FloatingLayer layer;
    layer.update({.windows = {{.id = "a", .title = "A", .content = &content, .size = {300.0f, 200.0f},
                               .onClose = [&] { ++closed; }, .onFocus = [&] { ++focused; }}}});
    Sakura::Layer base;
    LayerHost host({&base, &layer});
    host.place(MakeContext(), kScreen);

    CHECK(host.click(537.0f, 212.0f));
    CHECK(closed == 1);
    CHECK(focused == 0);

    CHECK(host.click(500.0f, 212.0f));
    CHECK(closed == 1);
}

TEST_CASE("Clicking a floating window focuses and raises it", "[core][sakura][shell]") {
    Probe first;
    Probe second;
    U64 firstFocused = 0;
    U64 secondFocused = 0;
    Sakura::FloatingLayer layer;
    layer.update({.windows = {
        {.id = "a", .title = "A", .content = &first, .size = {300.0f, 200.0f},
         .onFocus = [&] { ++firstFocused; }},
        {.id = "b", .title = "B", .content = &second, .size = {300.0f, 200.0f},
         .onFocus = [&] { ++secondFocused; }},
    }});
    Sakura::Layer base;
    LayerHost host({&base, &layer});
    host.place(MakeContext(), kScreen);

    CHECK(host.click(300.0f, 300.0f));
    CHECK(second.count(MouseEventType::Click) == 1);
    CHECK(first.count(MouseEventType::Click) == 0);
    CHECK(secondFocused == 0);

    CHECK(host.click(260.0f, 300.0f));
    CHECK(firstFocused == 1);
    host.place(MakeContext(), kScreen);

    const U64 firstBefore = first.count(MouseEventType::Click);
    const U64 secondBefore = second.count(MouseEventType::Click);
    CHECK(host.click(300.0f, 300.0f));
    CHECK(first.count(MouseEventType::Click) == firstBefore + 1);
    CHECK(second.count(MouseEventType::Click) == secondBefore);
    CHECK(firstFocused == 1);

    CHECK(host.click(560.0f, 400.0f));
    CHECK(secondFocused == 1);
}

TEST_CASE("Dragging a floating window title moves the window", "[core][sakura][shell]") {
    Probe content;
    Sakura::FloatingLayer layer;
    layer.update({.windows = {{.id = "a", .title = "A", .content = &content, .size = {300.0f, 200.0f}}}});
    Sakura::Layer base;
    LayerHost host({&base, &layer});
    host.place(MakeContext(), kScreen);

    CHECK(host.send(Mouse(MouseEventType::Click, 300.0f, 210.0f)));
    CHECK(layer.covers({10.0f, 10.0f}));
    CHECK(host.send(Mouse(MouseEventType::Move, 350.0f, 260.0f)));
    CHECK(host.send(Mouse(MouseEventType::Release, 350.0f, 260.0f)));
    host.place(MakeContext(), kScreen);

    CHECK(content.laidFrame == Rect{300.0f, 282.0f, 300.0f, 168.0f});
    CHECK(content.count(MouseEventType::Click) == 0);
    CHECK_FALSE(layer.covers({10.0f, 10.0f}));

    host.send(Mouse(MouseEventType::Click, 400.0f, 260.0f));
    host.send(Mouse(MouseEventType::Move, 2400.0f, 2260.0f));
    host.send(Mouse(MouseEventType::Release, 2400.0f, 2260.0f));
    host.place(MakeContext(), kScreen);

    CHECK(content.laidFrame.x == 704.0f);
    CHECK(content.laidFrame.y == 600.0f);
}

TEST_CASE("Dragging a floating window grip resizes the window", "[core][sakura][shell]") {
    Probe content;
    FloatingProbe layer;
    layer.update({.windows = {{.id = "a", .title = "A", .content = &content, .size = {300.0f, 200.0f}}}});
    Sakura::Layer base;
    LayerHost host({&base, &layer});
    host.place(MakeContext(), kScreen);

    host.send(Mouse(MouseEventType::Click, 545.0f, 395.0f));
    host.send(Mouse(MouseEventType::Move, 595.0f, 425.0f));
    host.send(Mouse(MouseEventType::Release, 595.0f, 425.0f));
    host.place(MakeContext(), kScreen);

    CHECK(content.laidFrame == Rect{250.0f, 232.0f, 350.0f, 198.0f});
    CHECK(content.count(MouseEventType::Click) == 0);

    CHECK(layer.cursorAt({602.0f, 432.0f}) == SurfaceCursor::ResizeNWSE);
    host.send(Mouse(MouseEventType::Click, 602.0f, 432.0f));
    host.send(Mouse(MouseEventType::Move, 552.0f, 402.0f));
    host.send(Mouse(MouseEventType::Release, 552.0f, 402.0f));
    host.place(MakeContext(), kScreen);

    CHECK(content.laidFrame == Rect{250.0f, 232.0f, 300.0f, 168.0f});
    CHECK(content.count(MouseEventType::Click) == 0);
}

TEST_CASE("Floating window chrome matches the tile outline with no visible grip",
          "[core][sakura][shell]") {
    Probe content;
    FloatingProbe layer;
    layer.update({.windows = {{.id = "a", .title = "A", .content = &content, .size = {300.0f, 200.0f}}}});
    LayerHost host({&layer});
    host.place(MakeContext(), kScreen);

    U64 strokes = 0;
    U64 topLit = 0;
    for (auto* box : layer.parts<Sakura::Box>()) {
        CHECK(box->config().cornerRadius >= Sakura::FloatingLayer::Rounding - 1.0f);
        if (box->config().borderWidth > 0.0f) {
            ++strokes;
            topLit += box->config().topLit ? 1 : 0;
        }
    }
    CHECK(strokes == 3);
    CHECK(topLit == 2);
    CHECK(layer.parts<Sakura::Polyline>().empty());
    CHECK(content.laidFrame == Rect{250.0f, 232.0f, 300.0f, 168.0f});
}

TEST_CASE("Toast layer drains the queue and stacks titled cards upward from the bottom left",
          "[core][sakura][shell]") {
    const ImGuiContextGuard imguiContext;
    SakuraTest::FakeWindow window;
    (void)Sakura::Private::TakeToasts();
    ToastProbe layer;
    CHECK_FALSE(layer.covers({400.0f, 300.0f}));

    Sakura::PushToast(Sakura::ToastType::Error, 1000, "Signaller connection rejected.", "Connection Error");
    Sakura::PushToast(Sakura::ToastType::Success, 1000, "");
    layer.update();
    CHECK(Sakura::Private::TakeToasts().empty());

    LayerHost host({&layer});
    auto ctx = WindowContext(window);
    ctx.time = 10.0;
    host.place(ctx, kScreen);
    ctx.time = 10.2;
    host.place(ctx, kScreen);

    std::vector<std::string> headings;
    for (auto* text : layer.parts<Sakura::Text>()) {
        if (text->config().font == Sakura::Text::Font::DisplaySemiBold) {
            headings.push_back(text->config().str);
        }
    }
    CHECK(headings == std::vector<std::string>{"Connection Error", "Success"});

    std::vector<Rect> cards;
    for (auto* box : layer.parts<Sakura::Box>()) {
        if (box->config().borderWidth > 0.0f && !box->config().topLit && box->config().cornerRadius == 12.0f) {
            cards.push_back(box->config().instances.front().rect);
        }
    }
    REQUIRE(cards.size() == 2);
    CHECK(cards[0].x == 20.0f);
    CHECK(cards[0].bottom() == 580.0f);
    CHECK(cards[1].x == 20.0f);
    CHECK(cards[1].bottom() == cards[0].y - 10.0f);

    CHECK(layer.covers({cards[0].x + 10.0f, cards[0].y + 10.0f}));
    CHECK(layer.covers({cards[1].x + 10.0f, cards[1].y + 10.0f}));
    CHECK_FALSE(layer.covers({780.0f, 580.0f}));
    CHECK_FALSE(layer.covers({cards[0].right() + 5.0f, cards[0].y + 10.0f}));

    ctx.time = 11.31;
    host.place(ctx, kScreen);
    CHECK(layer.shown() == 0);
    CHECK_FALSE(layer.covers({cards[0].x + 10.0f, cards[0].y + 10.0f}));
    layer.update();
    CHECK(layer.children().empty());
}

TEST_CASE("Toasts slide in from the left while they fade in", "[core][sakura][shell]") {
    const ImGuiContextGuard imguiContext;
    SakuraTest::FakeWindow window;
    (void)Sakura::Private::TakeToasts();
    ToastProbe layer;
    Sakura::PushToast(Sakura::ToastType::Info, 1000, "Saved the flowgraph");
    layer.update();
    LayerHost host({&layer});
    auto ctx = WindowContext(window);
    const auto cardX = [&] {
        for (auto* box : layer.parts<Sakura::Box>()) {
            if (box->config().borderWidth > 0.0f && !box->config().topLit && box->config().cornerRadius == 12.0f) {
                return box->config().instances.front().rect.x;
            }
        }
        return -1.0f;
    };
    ctx.time = 10.0;
    host.place(ctx, kScreen);
    CHECK(cardX() == 4.0f);
    ctx.time = 10.075;
    host.place(ctx, kScreen);
    CHECK(cardX() > 4.0f);
    CHECK(cardX() < 20.0f);
    ctx.time = 10.2;
    host.place(ctx, kScreen);
    CHECK(cardX() == 20.0f);
}

TEST_CASE("Hovering a toast shows its close badge and holds it, and the badge dismisses it across updates",
          "[core][sakura][shell]") {
    const ImGuiContextGuard imguiContext;
    SakuraTest::FakeWindow window;
    (void)Sakura::Private::TakeToasts();
    ToastProbe layer;
    Sakura::PushToast(Sakura::ToastType::Warning, 1000, "Python dependencies need approval.");
    layer.update();
    LayerHost host({&layer});
    auto ctx = WindowContext(window);
    ctx.time = 10.0;
    host.place(ctx, kScreen);
    ctx.time = 10.2;
    host.place(ctx, kScreen);
    const U64 shown = layer.shown();
    REQUIRE(shown > 0);

    const auto glyph = [&]() -> const Sakura::Text* {
        for (auto* text : layer.parts<Sakura::Text>()) {
            if (text->config().str == ICON_FA_XMARK) {
                return text;
            }
        }
        return nullptr;
    };
    REQUIRE(glyph());
    CHECK(glyph()->config().opacity == 0.0f);
    CHECK_FALSE(Peek::CursorAt(layer, {14.0f, 545.0f}).has_value());

    REQUIRE(layer.covers({40.0f, 570.0f}));
    host.send(Mouse(MouseEventType::Move, 40.0f, 570.0f));
    ctx.time = 10.3;
    host.place(ctx, kScreen);
    CHECK(glyph()->config().opacity == 1.0f);

    for (const F64 time : {11.0, 12.0, 20.0}) {
        ctx.time = time;
        host.place(ctx, kScreen);
        CHECK(layer.shown() == shown);
    }

    const Rect badge = Peek::Frame(*glyph());
    const Extent2D<F32> center = {badge.x + badge.width * 0.5f, badge.y + badge.height * 0.5f};
    host.send(Mouse(MouseEventType::Move, center.x, center.y));
    CHECK(Peek::CursorAt(layer, center) == SurfaceCursor::Hand);
    CHECK(host.send(Mouse(MouseEventType::Click, center.x, center.y)));
    layer.update();
    host.place(ctx, kScreen);
    CHECK(host.send(Mouse(MouseEventType::Release, center.x, center.y)));
    ctx.time = 20.1;
    host.place(ctx, kScreen);
    CHECK(layer.shown() == shown);
    ctx.time = 20.2;
    host.place(ctx, kScreen);
    CHECK(layer.shown() == shown);
    ctx.time = 20.3;
    host.place(ctx, kScreen);
    CHECK(layer.shown() == 0);
}

TEST_CASE("Background layers never cover while plain layers cover with content",
          "[core][sakura][shell]") {
    Sakura::BackgroundLayer background;
    background.update({.particleCount = 8});
    LayerHost host({&background});
    host.place(MakeContext(), kScreen);

    CHECK_FALSE(background.covers({400.0f, 300.0f}));

    Sakura::Layer plain;
    CHECK_FALSE(plain.covers({400.0f, 300.0f}));
    Probe content;
    plain.update({.content = &content});
    CHECK(plain.covers({400.0f, 300.0f}));
    plain.update({.content = nullptr});
    CHECK_FALSE(plain.covers({400.0f, 300.0f}));
}

TEST_CASE("Overlay items are anchored inside the overlay with their offsets", "[core][sakura][shell]") {
    Probe fill;
    Probe corner({60.0f, 20.0f});
    Probe sized({10.0f, 10.0f});
    Sakura::Overlay overlay;
    overlay.update({.items = {
        {.child = &fill},
        {.child = &corner, .anchor = Sakura::Anchor::BottomRight, .offset = {-5.0f, -5.0f}},
        {.child = &sized, .anchor = Sakura::Anchor::Center, .size = {100.0f, 50.0f}},
    }});
    Host host(overlay);
    host.place(MakeContext(), {10.0f, 20.0f, 400.0f, 300.0f});

    CHECK(fill.laidFrame == Rect{10.0f, 20.0f, 400.0f, 300.0f});
    CHECK(corner.laidFrame == Rect{345.0f, 295.0f, 60.0f, 20.0f});
    CHECK(sized.laidFrame == Rect{160.0f, 145.0f, 100.0f, 50.0f});
    CHECK(sized.measures == 0);
}

TEST_CASE("Open menus request a popup at their anchor and close on outside clicks",
          "[core][sakura][shell]") {
    U64 closed = 0;
    Sakura::Menu menu;
    Sakura::Menu::Config config{
        .entries = {{.label = "Open"}, {.label = "Save"}, {.label = "Quit"}},
        .open = false,
        .atCursor = false,
        .onClose = [&] { ++closed; },
    };
    menu.update(config);
    LayerSlot slot;
    slot.hold(menu, {100.0f, 100.0f, 1.0f, 1.0f});
    LayerHost host({&slot});
    host.place(MakeContext(), kScreen);
    CHECK_FALSE(host.popups.interactiveAt({120.0f, 120.0f}));

    config.open = true;
    menu.update(config);
    host.place(MakeContext(), kScreen);
    CHECK(host.popups.interactiveAt({100.0f, 100.0f}));
    CHECK(host.popups.interactiveAt({149.0f, 184.0f}));
    CHECK_FALSE(host.popups.interactiveAt({150.0f, 184.0f}));
    CHECK_FALSE(host.popups.interactiveAt({149.0f, 185.0f}));

    CHECK(host.click(600.0f, 500.0f));
    CHECK(closed == 1);
    host.place(MakeContext(), kScreen);
    CHECK_FALSE(host.popups.interactiveAt({120.0f, 120.0f}));
}

TEST_CASE("Clicking a menu entry closes the menu before running the entry", "[core][sakura][shell]") {
    std::vector<std::string> calls;
    Sakura::Menu menu;
    menu.update({
        .entries = {
            {.label = "Open", .onClick = [&] { calls.push_back("open"); }},
            {.label = "Save", .onClick = [&] { calls.push_back("save"); }},
        },
        .open = true,
        .atCursor = false,
        .onClose = [&] { calls.push_back("close"); },
    });
    LayerSlot slot;
    slot.hold(menu, {100.0f, 100.0f, 1.0f, 1.0f});
    LayerHost host({&slot});
    host.place(MakeContext(), kScreen);

    CHECK(host.click(120.0f, 142.0f));
    CHECK(calls == std::vector<std::string>{"close", "save"});

    host.place(MakeContext(), kScreen);
    CHECK_FALSE(host.popups.interactiveAt({120.0f, 142.0f}));
}

TEST_CASE("Menu entries run on release over an entry, not on release outside the menu",
          "[core][sakura][shell]") {
    std::vector<std::string> calls;
    Sakura::Menu menu;
    menu.update({
        .entries = {
            {.label = "Open", .onClick = [&] { calls.push_back("open"); }},
            {.label = "Save", .onClick = [&] { calls.push_back("save"); }},
        },
        .open = true,
        .atCursor = false,
        .onClose = [&] { calls.push_back("close"); },
    });
    LayerSlot slot;
    slot.hold(menu, {100.0f, 100.0f, 1.0f, 1.0f});
    LayerHost host({&slot});
    host.place(MakeContext(), kScreen);

    host.send(Mouse(MouseEventType::Click, 120.0f, 119.0f));
    host.send(Mouse(MouseEventType::Release, 300.0f, 300.0f));
    CHECK(calls.empty());

    host.send(Mouse(MouseEventType::Click, 120.0f, 119.0f));
    host.send(Mouse(MouseEventType::Release, 120.0f, 142.0f));
    CHECK(calls == std::vector<std::string>{"close", "save"});
}

TEST_CASE("Disabled menu entries and separators ignore clicks", "[core][sakura][shell]") {
    U64 clicked = 0;
    U64 closed = 0;
    Sakura::Menu menu;
    menu.update({
        .entries = {
            {.label = "Open", .enabled = false, .onClick = [&] { ++clicked; }},
            {.separator = true},
            {.label = "Save", .onClick = [&] { ++clicked; }},
        },
        .open = true,
        .atCursor = false,
        .onClose = [&] { ++closed; },
    });
    LayerSlot slot;
    slot.hold(menu, {100.0f, 100.0f, 1.0f, 1.0f});
    LayerHost host({&slot});
    host.place(MakeContext(), kScreen);

    CHECK(host.popups.interactiveAt({149.0f, 170.0f}));
    CHECK_FALSE(host.popups.interactiveAt({149.0f, 171.0f}));

    host.click(120.0f, 119.0f);
    CHECK(clicked == 0);
    CHECK(closed == 0);

    host.click(120.0f, 135.5f);
    CHECK(clicked == 0);
    CHECK(closed == 0);

    host.click(120.0f, 151.0f);
    CHECK(clicked == 1);
    CHECK(closed == 1);
}

TEST_CASE("Menu submenus open to the side and their entries close the whole menu",
          "[core][sakura][shell]") {
    U64 recent = 0;
    U64 closed = 0;
    Sakura::Menu menu;
    menu.update({
        .entries = {
            {.label = "Open", .children = {{.label = "Recent", .onClick = [&] { ++recent; }}}},
            {.label = "Quit"},
        },
        .open = true,
        .atCursor = false,
        .onClose = [&] { ++closed; },
    });
    LayerSlot slot;
    slot.hold(menu, {100.0f, 100.0f, 1.0f, 1.0f});
    LayerHost host({&slot});
    host.place(MakeContext(), kScreen);
    CHECK_FALSE(host.popups.interactiveAt({160.0f, 119.0f}));

    host.click(120.0f, 119.0f);
    CHECK(closed == 0);
    host.place(MakeContext(), kScreen);

    CHECK(host.popups.interactiveAt({151.0f, 100.0f}));
    CHECK(host.popups.interactiveAt({191.0f, 138.0f}));
    CHECK_FALSE(host.popups.interactiveAt({192.0f, 119.0f}));

    CHECK(host.click(160.0f, 119.0f));
    CHECK(recent == 1);
    CHECK(closed == 1);
}

TEST_CASE("Hovering a menu entry with children opens its submenu until a plain entry is hovered",
          "[core][sakura][shell]") {
    Sakura::Menu menu;
    menu.update({
        .entries = {{.label = "Quit"}, {.label = "Open", .children = {{.label = "Recent"}}}},
        .open = true,
        .atCursor = false,
    });
    LayerSlot slot;
    slot.hold(menu, {100.0f, 100.0f, 1.0f, 1.0f});
    LayerHost host({&slot});
    host.place(MakeContext(), kScreen);

    host.send(Mouse(MouseEventType::Move, 120.0f, 142.0f));
    host.place(MakeContext(), kScreen);
    CHECK(host.popups.interactiveAt({160.0f, 140.0f}));

    host.send(Mouse(MouseEventType::Move, 400.0f, 400.0f));
    host.place(MakeContext(), kScreen);
    CHECK(host.popups.interactiveAt({160.0f, 140.0f}));

    host.send(Mouse(MouseEventType::Move, 120.0f, 119.0f));
    host.place(MakeContext(), kScreen);
    CHECK_FALSE(host.popups.interactiveAt({160.0f, 140.0f}));
}

TEST_CASE("Dropdown requests a popup only while open and selects the clicked row",
          "[core][sakura][shell]") {
    std::vector<U64> selected;
    Sakura::Dropdown dropdown;
    dropdown.update({.options = {"alpha", "beta", "gamma"}, .onSelect = [&](U64 row) { selected.push_back(row); }});
    LayerSlot slot;
    slot.onLayout = [&](const Sakura::Context& ctx) {
        dropdown.request(ctx, {100.0f, 100.0f, 200.0f, 25.0f});
    };
    LayerHost host({&slot});
    host.place(MakeContext(), kScreen);
    CHECK_FALSE(host.popups.interactiveAt({150.0f, 150.0f}));

    dropdown.open();
    CHECK(dropdown.isOpen());
    host.place(MakeContext(), kScreen);
    CHECK(host.popups.interactiveAt({100.0f, 125.0f}));
    CHECK(host.popups.interactiveAt({299.0f, 223.0f}));
    CHECK_FALSE(host.popups.interactiveAt({300.0f, 125.0f}));
    CHECK_FALSE(host.popups.interactiveAt({150.0f, 224.0f}));

    CHECK(host.click(150.0f, 199.0f));
    CHECK(selected == std::vector<U64>{2});
    CHECK_FALSE(dropdown.isOpen());
    host.place(MakeContext(), kScreen);
    CHECK_FALSE(host.popups.interactiveAt({150.0f, 150.0f}));
}

TEST_CASE("Dropdown ignores drags between rows and closes on outside clicks", "[core][sakura][shell]") {
    std::vector<U64> selected;
    Sakura::Dropdown dropdown;
    dropdown.update({.options = {"alpha", "beta", "gamma"}, .onSelect = [&](U64 row) { selected.push_back(row); }});
    LayerSlot slot;
    slot.onLayout = [&](const Sakura::Context& ctx) {
        dropdown.request(ctx, {100.0f, 100.0f, 200.0f, 25.0f});
    };
    LayerHost host({&slot});
    dropdown.open();
    host.place(MakeContext(), kScreen);

    host.send(Mouse(MouseEventType::Click, 150.0f, 149.0f));
    host.send(Mouse(MouseEventType::Release, 150.0f, 174.0f));
    CHECK(selected.empty());
    CHECK(dropdown.isOpen());

    host.click(600.0f, 500.0f);
    CHECK(selected.empty());
    CHECK_FALSE(dropdown.isOpen());

    dropdown.open();
    dropdown.close();
    CHECK_FALSE(dropdown.isOpen());
}

TEST_CASE("Layers below a held layer receive no hover or activity until release", "[core][sakura][shell]") {
    Probe below;
    below.consume = false;
    Probe above;
    LayerSlot bottom;
    LayerSlot top;
    bottom.hold(below, kScreen);
    top.hold(above, {100.0f, 100.0f, 200.0f, 40.0f});
    LayerHost host({&bottom, &top});
    host.place(MakeContext(), kScreen);
    const auto reached = [&] {
        return !below.events.empty() && below.events.back().position.x > -1.0e8f;
    };

    host.send(Mouse(MouseEventType::Move, 400.0f, 400.0f));
    CHECK(reached());

    host.send(Mouse(MouseEventType::Move, 150.0f, 110.0f));
    CHECK_FALSE(reached());

    host.send(Mouse(MouseEventType::Click, 150.0f, 110.0f));
    host.send(Mouse(MouseEventType::Move, 400.0f, 400.0f));
    CHECK_FALSE(reached());

    host.send(Mouse(MouseEventType::Release, 400.0f, 400.0f));
    host.send(Mouse(MouseEventType::Move, 400.0f, 400.0f));
    CHECK(reached());
}

TEST_CASE("Covering layers stop cursor hits from reaching layers below", "[core][sakura][shell]") {
    TextCursorProbe below;
    Probe above;
    LayerSlot bottom;
    LayerSlot top;
    bottom.hold(below, kScreen);
    top.hold(above, {100.0f, 100.0f, 200.0f, 40.0f});
    LayerHost host({&bottom, &top});
    host.place(MakeContext(), kScreen);

    CHECK(Frames::Access::cursor(host, {400.0f, 400.0f}) == SurfaceCursor::TextInput);
    CHECK(Frames::Access::cursor(host, {150.0f, 110.0f}) == SurfaceCursor::Default);
}

TEST_CASE("The drag ghost is a pill with the pane icon and title that follows the cursor only while dragging",
          "[core][sakura][shell]") {
    Probe content;
    LayerSlot slot;
    slot.hold(content, kScreen);
    LayerHost host({&slot});
    host.place(MakeContext(), kScreen);
    CHECK_FALSE(host.ghost.visible());
    CHECK_FALSE(host.ghostIcon.visible());

    host.drag.active = true;
    host.drag.pane = "scope";
    host.drag.title = "Scope";
    host.drag.icon = ICON_FA_CHART_LINE;
    auto ctx = MakeContext();
    ctx.mouse = Extent2D<F32>{300.0f, 200.0f};
    host.send(Mouse(MouseEventType::Move, 300.0f, 200.0f));
    host.place(ctx, kScreen);
    CHECK(host.ghost.visible());
    CHECK(host.ghostIcon.visible());
    CHECK(Peek::Frame(host.ghostIcon) == Rect{326.0f, 220.0f, 15.0f, 15.0f});
    CHECK(Peek::Frame(host.ghost).x == 347.0f);
    CHECK(Peek::Frame(host.ghost).y == 220.0f);

    ctx.mouse = Extent2D<F32>{320.0f, 210.0f};
    host.send(Mouse(MouseEventType::Move, 320.0f, 210.0f));
    host.place(ctx, kScreen);
    CHECK(Peek::Frame(host.ghostIcon) == Rect{346.0f, 230.0f, 15.0f, 15.0f});

    host.drag.icon.clear();
    Sakura::Access::invalidate(host);
    host.place(ctx, kScreen);
    CHECK_FALSE(host.ghostIcon.visible());
    CHECK(Peek::Frame(host.ghost).x == 346.0f);

    host.drag.reset();
    Sakura::Access::invalidate(host);
    ctx.mouse = Extent2D<F32>{500.0f, 500.0f};
    host.place(ctx, kScreen);
    CHECK_FALSE(host.ghost.visible());
    const Rect parked = Peek::Frame(host.ghost);
    ctx.mouse = Extent2D<F32>{100.0f, 100.0f};
    host.send(Mouse(MouseEventType::Move, 100.0f, 100.0f));
    host.place(ctx, kScreen);
    CHECK(Peek::Frame(host.ghost) == parked);
}

TEST_CASE("The drop preview fades in on its first target, eases between targets and fades out in place",
          "[core][sakura][shell]") {
    Probe content;
    LayerSlot slot;
    slot.hold(content, kScreen);
    std::optional<Rect> offered;
    slot.onLayout = [&](const Sakura::Context& ctx) {
        if (offered.has_value() && ctx.dockDrag) {
            ctx.dockDrag->offer({.preview = *offered});
        }
    };
    LayerHost host({&slot});
    auto ctx = MakeContext();
    ctx.mouse = Extent2D<F32>{300.0f, 300.0f};
    const auto at = [&](F64 time) {
        ctx.time = time;
        host.place(ctx, kScreen);
    };
    const Rect first = {10.0f, 10.0f, 200.0f, 100.0f};
    const Rect second = {410.0f, 210.0f, 300.0f, 200.0f};

    host.drag.active = true;
    offered = first;
    at(1.0);
    CHECK(host.motion.rect == first);
    CHECK(host.motion.alpha == 0.0f);
    at(1.06);
    CHECK(host.motion.alpha > 0.5f);
    CHECK(host.motion.alpha < 1.0f);
    CHECK(host.motion.rect == first);
    CHECK_FALSE(host.motion.settled());
    at(1.2);
    CHECK(host.motion.alpha == 1.0f);
    CHECK(host.motion.settled());

    offered = second;
    host.send(Mouse(MouseEventType::Move, 301.0f, 300.0f));
    at(2.0);
    CHECK(host.motion.rect == first);
    at(2.06);
    CHECK(host.motion.rect.x > first.x);
    CHECK(host.motion.rect.x < second.x);
    CHECK(host.motion.rect.width > first.width);
    CHECK(host.motion.rect.width < second.width);
    CHECK(host.motion.alpha == 1.0f);
    at(2.2);
    CHECK(host.motion.rect == second);

    host.drag.reset();
    offered.reset();
    Sakura::Access::invalidate(host);
    at(3.0);
    CHECK(host.motion.alpha == 1.0f);
    at(3.06);
    CHECK(host.motion.alpha > 0.0f);
    CHECK(host.motion.alpha < 0.5f);
    CHECK(host.motion.rect == second);
    at(3.2);
    CHECK(host.motion.alpha == 0.0f);
    CHECK(host.motion.settled());

    host.drag.active = true;
    offered = first;
    Sakura::Access::invalidate(host);
    at(4.0);
    CHECK(host.motion.rect == first);
    CHECK(host.motion.alpha == 0.0f);
}

TEST_CASE("The drop preview border is a solid band, not a soft fringe",
          "[core][sakura][shell]") {
    Probe content;
    LayerSlot slot;
    slot.hold(content, kScreen);
    const Rect target = {10.0f, 10.0f, 200.0f, 100.0f};
    slot.onLayout = [&](const Sakura::Context& ctx) {
        if (ctx.dockDrag) {
            ctx.dockDrag->offer({.preview = target});
        }
    };
    LayerHost host({&slot});
    auto ctx = MakeContext();
    ctx.mouse = Extent2D<F32>{300.0f, 300.0f};
    host.drag.active = true;
    ctx.time = 1.0;
    host.place(ctx, kScreen);
    ctx.time = 1.2;
    host.place(ctx, kScreen);

    const auto& config = host.preview.config();
    REQUIRE(config.instances.size() == 1);
    CHECK(config.instances.front().rect == target);
    CHECK(config.instances.front().visible);
    CHECK(config.borderWidth == 2.0f);
    CHECK_FALSE(config.strokeFringe);
    CHECK(config.borderColor == ctx.color("drop_border"));
}

namespace {

const Sakura::Palette& Tones() {
    static const Sakura::Palette palette = [] {
        const std::vector<std::string> keys = {
            "text_primary", "text_secondary", "warning_yellow", "chrome_button_hover", "popup_bg", "border",
            "header", "header_hovered", "header_active", "separator", "tile_header_text", "tile_button_hover",
        };
        Sakura::Palette result;
        for (U64 index = 0; index < keys.size(); ++index) {
            const F32 value = static_cast<F32>(index + 1) / static_cast<F32>(keys.size() + 1);
            result[keys[index]] = {value, 1.0f - value, 0.5f, 1.0f};
        }
        return result;
    }();
    return palette;
}

Sakura::Text* Named(Sakura::Component& root, const std::string& str) {
    for (auto* text : Find<Sakura::Text>(root)) {
        if (text->config().str == str) {
            return text;
        }
    }
    return nullptr;
}

}  // namespace

TEST_CASE("Below end popups align to the anchor end and flip above it when there is no room below",
          "[core][sakura][shell]") {
    const auto belowEnd = [](Rect anchor) {
        Probe content({100.0f, 50.0f});
        return PlacePopup(content, {.anchor = anchor, .placement = Sakura::PopupPlacement::BelowEnd});
    };

    CHECK(belowEnd({600.0f, 100.0f, 80.0f, 20.0f}) == Rect{580.0f, 126.0f, 100.0f, 50.0f});
    CHECK(belowEnd({600.0f, 570.0f, 80.0f, 20.0f}) == Rect{580.0f, 514.0f, 100.0f, 50.0f});
    CHECK(belowEnd({200.0f, 30.0f, 80.0f, 560.0f}) == Rect{180.0f, 550.0f, 100.0f, 50.0f});
}

TEST_CASE("HUD offsets shift the HUD away from its anchor", "[core][sakura][shell]") {
    Probe left({100.0f, 40.0f});
    Probe right({100.0f, 40.0f});
    Sakura::HudLayer layer;
    layer.update({.huds = {
        {.content = &left, .anchor = Sakura::Anchor::TopLeft, .offset = {10.0f, -5.0f}},
        {.content = &right, .anchor = Sakura::Anchor::BottomRight, .offset = {-20.0f, -10.0f}},
    }});
    LayerHost host({&layer});
    host.place(MakeContext(), kScreen);

    CHECK(left.laidFrame == Rect{34.0f, 19.0f, 100.0f, 40.0f});
    CHECK(right.laidFrame == Rect{656.0f, 526.0f, 100.0f, 40.0f});
    CHECK(layer.covers({22.0f, 7.0f}));
    CHECK_FALSE(layer.covers({21.0f, 7.0f}));
    CHECK(layer.covers({145.0f, 70.0f}));
    CHECK_FALSE(layer.covers({146.0f, 70.0f}));
    CHECK(layer.covers({767.0f, 577.0f}));
    CHECK_FALSE(layer.covers({769.0f, 579.0f}));
}

TEST_CASE("Menu entries draw a checkmark when selected, a shortcut column, and their label color",
          "[core][sakura][shell]") {
    const ImGuiContextGuard imguiContext;
    MonoFontWindow window;
    auto ctx = WindowContext(window);
    ctx.palette = Tones();
    Sakura::Menu menu;
    menu.update({
        .entries = {
            {.label = "Open", .shortcut = "Ctrl+O", .selected = true},
            {.label = "Save As", .shortcut = "Ctrl+Shift+S", .selected = true, .colorKey = "warning_yellow"},
            {.label = "Recent", .children = {{.label = "Flowgraph"}}},
            {.label = "Quit", .enabled = false},
        },
        .open = true,
        .atCursor = false,
    });
    LayerSlot slot;
    slot.hold(menu, {100.0f, 100.0f, 1.0f, 1.0f});
    LayerHost host({&slot});
    host.place(ctx, kScreen);

    auto* open = Named(host, "Open");
    auto* save = Named(host, "Save As");
    auto* recent = Named(host, "Recent");
    auto* quit = Named(host, "Quit");
    auto* openKeys = Named(host, "Ctrl+O");
    auto* saveKeys = Named(host, "Ctrl+Shift+S");
    REQUIRE(open);
    REQUIRE(save);
    REQUIRE(recent);
    REQUIRE(quit);
    REQUIRE(openKeys);
    REQUIRE(saveKeys);

    CHECK(open->config().colorKey == "text_primary");
    CHECK(save->config().colorKey == "warning_yellow");
    CHECK(quit->config().opacity < 1.0f);
    CHECK(openKeys->config().colorKey == "text_secondary");
    CHECK(saveKeys->config().colorKey == "text_secondary");

    const Rect label = Peek::Frame(*open);
    const Rect keys = Peek::Frame(*openKeys);
    const Rect saveRow = Peek::Frame(*save);
    CHECK(label.width > 0.0f);
    CHECK(keys.x == label.x + label.width + 8.0f);
    CHECK(Peek::Frame(*saveKeys).x == keys.x);
    CHECK(keys.y == label.y);
    CHECK(Peek::Frame(*saveKeys).y == saveRow.y);
    CHECK(Peek::Frame(*saveKeys).width == keys.width);
    CHECK_FALSE(Named(host, "Flowgraph"));

    const auto lines = Find<Sakura::Polyline>(host);
    REQUIRE(lines.size() == 1);
    const auto& marks = lines.front()->config().instances;
    REQUIRE(marks.size() == 3);
    const F32 markColumn = keys.right() + 8.0f;
    const auto within = [&](const Sakura::Polyline::Instance& mark, const Rect& row) {
        for (const auto& point : mark.points) {
            if (point.x < markColumn || point.y < row.y || point.y > row.bottom()) {
                return false;
            }
        }
        return true;
    };
    CHECK_FALSE(marks[0].closed);
    CHECK(marks[0].points.size() == 3);
    CHECK(marks[0].color == ctx.color("text_primary"));
    CHECK(within(marks[0], label));
    CHECK_FALSE(marks[1].closed);
    CHECK(marks[1].color == ctx.color("warning_yellow"));
    CHECK(within(marks[1], saveRow));
    CHECK(marks[2].closed);
    CHECK(marks[2].filled);
    CHECK(within(marks[2], Peek::Frame(*recent)));
}

TEST_CASE("Updating an open menu resyncs its open submenu and closes it when its entry goes away",
          "[core][sakura][shell]") {
    Sakura::Menu menu;
    Sakura::Menu::Config config{
        .entries = {
            {.label = "Open", .children = {{.label = "Recent"}, {.label = "Pinned"}}},
            {.label = "Quit"},
        },
        .open = true,
        .atCursor = false,
    };
    LayerSlot slot;
    slot.hold(menu, {100.0f, 100.0f, 1.0f, 1.0f});
    LayerHost host({&slot});
    Frames frames(host, kScreen, MakeContext());
    const auto update = [&] {
        menu.update(config);
    };
    REQUIRE(frames.settle(update));

    host.click(120.0f, 119.0f);
    REQUIRE(frames.settle(update));
    CHECK(host.popups.interactiveAt({160.0f, 150.0f}));
    CHECK(Named(host, "Recent"));
    CHECK(Named(host, "Pinned"));

    config.entries[0].children = {{.label = "Starred"}};
    const auto resync = frames.run(update);
    CHECK_FALSE(resync.rebuilt);
    CHECK(resync.repainted);
    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK(host.popups.interactiveAt({160.0f, 119.0f}));
    CHECK_FALSE(host.popups.interactiveAt({160.0f, 150.0f}));
    CHECK(Named(host, "Starred"));
    CHECK_FALSE(Named(host, "Recent"));
    CHECK_FALSE(Named(host, "Pinned"));

    config.entries = {{.label = "Quit"}};
    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK_FALSE(host.popups.interactiveAt({160.0f, 119.0f}));
    CHECK_FALSE(Named(host, "Starred"));
    CHECK(Named(host, "Quit"));
}

TEST_CASE("Chrome buttons with entries toggle their menu from the button and highlight on hover",
          "[core][sakura][shell]") {
    Sakura::ChromeButton button;
    U64 clicks = 0;
    Sakura::ChromeButton::Config config{
        .label = "CyberEther",
        .entries = {{.label = "About"}, {.label = "Quit"}},
        .onClick = [&] { ++clicks; },
    };
    LayerSlot slot;
    slot.hold(button, {100.0f, 10.0f, 60.0f, 28.0f});
    LayerHost host({&slot});
    auto ctx = MakeContext();
    ctx.palette = Tones();
    Frames frames(host, kScreen, ctx);
    const auto update = [&] {
        button.update(config);
    };
    const auto highlighted = [&] {
        for (auto* child : Peek::Children(button)) {
            if (auto* box = dynamic_cast<Sakura::Box*>(child)) {
                const auto& instance = box->config().instances.front();
                return instance.visible && instance.backgroundColor == ctx.color("chrome_button_hover");
            }
        }
        return false;
    };
    REQUIRE(frames.settle(update));
    const U64 rebuilds = frames.rebuilds;
    CHECK_FALSE(highlighted());

    host.send(Mouse(MouseEventType::Move, 130.0f, 24.0f));
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(highlighted());
    CHECK(DirtyFrames(frames, update) == 0);
    host.send(Mouse(MouseEventType::Move, 131.0f, 25.0f));
    CHECK(frames.run(update).clean());

    host.send(Mouse(MouseEventType::Leave, -10.0f, -10.0f));
    CHECK(RepaintOnly(frames.run(update)));
    CHECK_FALSE(highlighted());
    CHECK(DirtyFrames(frames, update) == 0);
    host.send(Mouse(MouseEventType::Move, 400.0f, 400.0f));
    CHECK(frames.run(update).clean());

    CHECK(host.click(130.0f, 24.0f));
    CHECK(button.menuOpen());
    CHECK_FALSE(frames.run(update).rebuilt);
    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK(host.popups.interactiveAt({110.0f, 60.0f}));
    CHECK(highlighted());

    CHECK_FALSE(host.popups.covers({130.0f, 24.0f}));
    CHECK(host.click(130.0f, 24.0f));
    CHECK_FALSE(button.menuOpen());
    CHECK_FALSE(frames.run(update).rebuilt);
    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK_FALSE(host.popups.interactiveAt({110.0f, 60.0f}));
    CHECK_FALSE(highlighted());

    host.click(130.0f, 24.0f);
    REQUIRE(frames.settle(update));
    REQUIRE(button.menuOpen());
    CHECK(host.click(600.0f, 500.0f));
    CHECK_FALSE(button.menuOpen());
    REQUIRE(frames.settle(update));
    CHECK_FALSE(host.popups.interactiveAt({110.0f, 60.0f}));
    CHECK(frames.rebuilds == rebuilds);
    CHECK(clicks == 0);
}

TEST_CASE("Chrome buttons close their menu when the entries are cleared and click through without entries",
          "[core][sakura][shell]") {
    Sakura::ChromeButton button;
    U64 clicks = 0;
    Sakura::ChromeButton::Config config{
        .glyph = Sakura::ChromeButton::Glyph::Plus,
        .entries = {{.label = "Flowgraph"}, {.label = "Block"}},
        .onClick = [&] { ++clicks; },
    };
    LayerSlot slot;
    slot.hold(button, {100.0f, 10.0f, 28.0f, 28.0f});
    LayerHost host({&slot});
    Frames frames(host, kScreen, MakeContext());
    const auto update = [&] {
        button.update(config);
    };
    REQUIRE(frames.settle(update));

    host.click(114.0f, 24.0f);
    REQUIRE(frames.settle(update));
    REQUIRE(button.menuOpen());
    REQUIRE(host.popups.interactiveAt({110.0f, 60.0f}));

    config.entries.clear();
    update();
    CHECK_FALSE(button.menuOpen());
    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK_FALSE(host.popups.interactiveAt({110.0f, 60.0f}));
    CHECK_FALSE(host.popups.covers({600.0f, 500.0f}));
    CHECK(clicks == 0);

    host.click(114.0f, 24.0f);
    CHECK(clicks == 1);
    CHECK_FALSE(button.menuOpen());
    REQUIRE(frames.settle(update));
    CHECK_FALSE(host.popups.covers({600.0f, 500.0f}));

    host.click(114.0f, 24.0f, MouseButton::Right);
    host.click(300.0f, 24.0f);
    CHECK(clicks == 1);
}

TEST_CASE("Floating window grip resizing stops at three title heights and dragging clamps to the left and top",
          "[core][sakura][shell]") {
    Probe content;
    FloatingProbe layer;
    layer.update({.windows = {{.id = "a", .title = "A", .content = &content, .size = {300.0f, 200.0f}}}});
    Sakura::Layer base;
    LayerHost host({&base, &layer});
    host.place(MakeContext(), kScreen);

    host.send(Mouse(MouseEventType::Click, 545.0f, 395.0f));
    host.send(Mouse(MouseEventType::Move, 300.0f, 250.0f));
    host.send(Mouse(MouseEventType::Release, 300.0f, 250.0f));
    host.place(MakeContext(), kScreen);
    CHECK(content.laidFrame == Rect{250.0f, 232.0f, 96.0f, 64.0f});

    host.send(Mouse(MouseEventType::Click, 260.0f, 210.0f));
    host.send(Mouse(MouseEventType::Move, -2000.0f, -2000.0f));
    host.send(Mouse(MouseEventType::Release, -2000.0f, -2000.0f));
    host.place(MakeContext(), kScreen);
    CHECK(content.laidFrame == Rect{0.0f, 32.0f, 96.0f, 64.0f});

    host.send(Mouse(MouseEventType::Click, 80.0f, 90.0f));
    host.send(Mouse(MouseEventType::Move, 680.0f, 390.0f));
    host.send(Mouse(MouseEventType::Release, 680.0f, 390.0f));
    host.place(MakeContext(), kScreen);
    CHECK(content.laidFrame == Rect{0.0f, 32.0f, 696.0f, 364.0f});

    host.send(Mouse(MouseEventType::Click, 10.0f, 10.0f));
    host.send(Mouse(MouseEventType::Move, -1000.0f, 10.0f));
    host.send(Mouse(MouseEventType::Release, -1000.0f, 10.0f));
    host.place(MakeContext(), kScreen);
    CHECK(content.laidFrame.x == 96.0f - 696.0f);
    CHECK(content.laidFrame.y == 32.0f);
}

TEST_CASE("Floating window focus colors the title and a focus change repaints without a rebuild",
          "[core][sakura][shell]") {
    Probe first;
    Probe second;
    U64 firstFocused = 0;
    FloatingProbe layer;
    Sakura::FloatingLayer::Config config{.windows = {
        {.id = "a", .title = "A", .content = &first, .size = {300.0f, 200.0f}, .onFocus = [&] { ++firstFocused; }},
        {.id = "b", .title = "B", .content = &second, .size = {300.0f, 200.0f}},
    }};
    Sakura::Layer base;
    LayerHost host({&base, &layer});
    Frames frames(host, kScreen, MakeContext());
    const auto update = [&] {
        layer.update(config);
    };
    const auto title = [&](const std::string& str) {
        auto* text = Named(layer, str);
        return text ? text->config().colorKey : std::string();
    };
    REQUIRE(frames.settle(update));
    CHECK(title("A") == "tile_header_text");
    CHECK(title("B") == "text_primary");
    const U64 rebuilds = frames.rebuilds;

    host.click(260.0f, 300.0f);
    const auto focus = frames.run(update);
    CHECK(focus.repainted);
    CHECK_FALSE(focus.rebuilt);
    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK(frames.rebuilds == rebuilds);
    CHECK(firstFocused == 1);
    CHECK(title("A") == "text_primary");
    CHECK(title("B") == "tile_header_text");

    host.click(270.0f, 300.0f);
    CHECK(frames.run(update).clean());
    CHECK(firstFocused == 1);
}

TEST_CASE("Floating window close glyph lights up on hover with a single repaint",
          "[core][sakura][shell]") {
    Probe content;
    FloatingProbe layer;
    Sakura::FloatingLayer::Config config{
        .windows = {{.id = "a", .title = "A", .content = &content, .size = {300.0f, 200.0f}}}};
    Sakura::Layer base;
    LayerHost host({&base, &layer});
    auto ctx = MakeContext();
    ctx.palette = DistinctPalette();
    Frames frames(host, kScreen, ctx);
    const auto update = [&] {
        layer.update(config);
    };
    const auto glyph = [&] {
        auto* text = Named(layer, ICON_FA_XMARK);
        return text ? text->config().colorKey : std::string();
    };
    REQUIRE(frames.settle(update));
    host.send(Mouse(MouseEventType::Move, 400.0f, 300.0f));
    frames.run(update);
    REQUIRE(frames.settle(update));
    CHECK(glyph() == "tile_header_text");

    host.send(Mouse(MouseEventType::Move, 532.0f, 216.0f));
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(glyph() == "tile_button_hover");
    CHECK(DirtyFrames(frames, update) == 0);
    host.send(Mouse(MouseEventType::Move, 533.0f, 217.0f));
    CHECK(frames.run(update).clean());

    host.send(Mouse(MouseEventType::Move, 400.0f, 300.0f));
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(glyph() == "tile_header_text");
    CHECK(DirtyFrames(frames, update) == 0);
}

TEST_CASE("Floating window drags end on leave, cancel, and removal of the dragged window",
          "[core][sakura][shell]") {
    Probe first;
    Probe second;
    FloatingProbe layer;
    Sakura::FloatingLayer::Config config{.windows = {
        {.id = "a", .title = "A", .content = &first, .size = {300.0f, 200.0f}},
        {.id = "b", .title = "B", .content = &second, .size = {300.0f, 200.0f}},
    }};
    Sakura::Layer base;
    LayerHost host({&base, &layer});
    Frames frames(host, kScreen, MakeContext());
    const auto update = [&] {
        layer.update(config);
    };
    REQUIRE(frames.settle(update));
    const Rect firstFrame = first.laidFrame;
    const Rect secondFrame = second.laidFrame;
    CHECK(secondFrame == Rect{274.0f, 256.0f, 300.0f, 168.0f});

    for (const auto type : {MouseEventType::Leave, MouseEventType::Cancel}) {
        host.send(Mouse(MouseEventType::Click, 450.0f, 230.0f));
        REQUIRE(layer.covers({10.0f, 10.0f}));
        host.send(Mouse(type, -10.0f, -10.0f));
        CHECK_FALSE(layer.covers({10.0f, 10.0f}));
        host.send(Mouse(MouseEventType::Move, 500.0f, 400.0f));
        host.send(Mouse(MouseEventType::Release, 500.0f, 400.0f));
        REQUIRE(frames.settle(update));
        CHECK(DirtyFrames(frames, update) == 0);
        CHECK(second.laidFrame == secondFrame);
    }

    host.send(Mouse(MouseEventType::Click, 450.0f, 230.0f));
    REQUIRE(layer.covers({10.0f, 10.0f}));
    config.windows.pop_back();
    update();
    CHECK_FALSE(layer.covers({10.0f, 10.0f}));
    host.send(Mouse(MouseEventType::Move, 100.0f, 100.0f));
    host.send(Mouse(MouseEventType::Release, 100.0f, 100.0f));
    REQUIRE(frames.settle(update));
    CHECK(DirtyFrames(frames, update) == 0);
    CHECK(first.laidFrame == firstFrame);
    CHECK_FALSE(layer.covers({450.0f, 410.0f}));
}

TEST_CASE("A newly added floating window takes focus and a right click raises a window without dragging it",
          "[core][sakura][shell]") {
    Probe first;
    Probe second;
    U64 firstFocused = 0;
    FloatingProbe layer;
    Sakura::FloatingLayer::Config config{.windows = {
        {.id = "a", .title = "A", .content = &first, .size = {300.0f, 200.0f}, .onFocus = [&] { ++firstFocused; }},
    }};
    Sakura::Layer base;
    LayerHost host({&base, &layer});
    layer.update(config);
    host.place(MakeContext(), kScreen);
    CHECK(Named(layer, "A")->config().colorKey == "text_primary");

    config.windows.push_back({.id = "b", .title = "B", .content = &second, .size = {300.0f, 200.0f}});
    layer.update(config);
    host.place(MakeContext(), kScreen);
    CHECK(Named(layer, "A")->config().colorKey == "tile_header_text");
    CHECK(Named(layer, "B")->config().colorKey == "text_primary");
    CHECK(firstFocused == 0);
    const Rect firstFrame = first.laidFrame;

    host.send(Mouse(MouseEventType::Click, 260.0f, 210.0f, MouseButton::Right));
    CHECK(firstFocused == 1);
    CHECK_FALSE(layer.covers({10.0f, 10.0f}));
    host.send(Mouse(MouseEventType::Move, 400.0f, 500.0f));
    host.send(Mouse(MouseEventType::Release, 400.0f, 500.0f, MouseButton::Right));
    host.place(MakeContext(), kScreen);
    CHECK(first.laidFrame == firstFrame);
    CHECK(Named(layer, "A")->config().colorKey == "text_primary");
    CHECK(Named(layer, "B")->config().colorKey == "tile_header_text");

    const U64 before = second.count(MouseEventType::Click);
    host.click(400.0f, 300.0f);
    CHECK(second.count(MouseEventType::Click) == before);
    CHECK(first.count(MouseEventType::Click) == 1);
}

TEST_CASE("The top floating window blocks pointer input to windows it covers, even where its content ignores it",
          "[core][sakura][shell]") {
    Probe back;
    Probe front;
    front.consume = false;
    FloatingProbe layer;
    layer.update({.windows = {
        {.id = "back", .title = "Back", .content = &back, .size = {300.0f, 200.0f}},
        {.id = "front", .title = "Front", .content = &front, .size = {300.0f, 200.0f}, .padding = 40.0f},
    }});
    Sakura::Layer base;
    LayerHost host({&base, &layer});
    host.place(MakeContext(), kScreen);

    const Extent2D<F32> covered = {front.laidFrame.x - 20.0f, front.laidFrame.y + 10.0f};
    REQUIRE(back.laidFrame.contains(covered.x, covered.y));
    REQUIRE_FALSE(front.laidFrame.contains(covered.x, covered.y));

    host.send(Mouse(MouseEventType::Move, covered.x, covered.y));
    host.click(covered.x, covered.y);
    CHECK(back.count(MouseEventType::Move) == 0);
    CHECK(back.count(MouseEventType::Click) == 0);

    const Extent2D<F32> exposed = {back.laidFrame.x + 4.0f, back.laidFrame.y + 4.0f};
    host.send(Mouse(MouseEventType::Move, exposed.x, exposed.y));
    CHECK(back.count(MouseEventType::Move) == 1);
    host.send(Mouse(MouseEventType::Move, covered.x, covered.y));
    CHECK(back.count(MouseEventType::Leave) == 1);

    const Extent2D<F32> outside = {front.laidFrame.right() + 20.0f, front.laidFrame.bottom() + 20.0f};
    REQUIRE_FALSE(back.laidFrame.contains(outside.x, outside.y));
    REQUIRE(layer.covers(outside));
    host.send(Mouse(MouseEventType::Click, exposed.x, exposed.y));
    host.send(Mouse(MouseEventType::Move, outside.x, outside.y));
    host.send(Wheel(outside.x, outside.y, 0.0f, 1.0f));
    CHECK(back.count(MouseEventType::Leave) == 1);
    CHECK(front.count(MouseEventType::Scroll) == 0);
    host.send(Mouse(MouseEventType::Release, outside.x, outside.y));
    CHECK(back.count(MouseEventType::Release) == 1);
}

TEST_CASE("Popup sizes fill the layer or stay fixed per axis", "[core][sakura][shell][sizing]") {
    Flow wide(300.0f, 15.0f);
    CHECK(PlacePopup(wide, {.anchor = {200.0f, 100.0f, 80.0f, 20.0f}, .size = {Sakura::Fill, 0.0f}}) ==
          Rect{0.0f, 120.0f, 800.0f, 15.0f});

    Probe tall({100.0f, 50.0f});
    CHECK(PlacePopup(tall, {.anchor = {200.0f, 100.0f, 80.0f, 20.0f}, .size = {0.0f, Sakura::Fill}}) ==
          Rect{200.0f, 0.0f, 100.0f, 600.0f});

    Probe fixed({100.0f, 50.0f});
    CHECK(PlacePopup(fixed, {.anchor = {200.0f, 100.0f, 80.0f, 20.0f}, .size = {0.0f, 80.0f}}) ==
          Rect{200.0f, 120.0f, 100.0f, 80.0f});
    CHECK(fixed.measures == 1);

    Probe framed({100.0f, 50.0f});
    CHECK(PlacePopup(framed, {.anchor = {200.0f, 100.0f, 80.0f, 20.0f}, .panel = true, .size = {0.0f, 80.0f}}) ==
          Rect{212.0f, 132.0f, 100.0f, 56.0f});

    Probe both({100.0f, 50.0f});
    CHECK(PlacePopup(both, {.placement = Sakura::PopupPlacement::Center, .size = {Sakura::Fill, Sakura::Fill}}) ==
          Rect{0.0f, 0.0f, 800.0f, 600.0f});
    CHECK(both.measures == 0);
}

TEST_CASE("Popup requests scale their panel padding, max width and fixed size at a pixel ratio of two",
          "[core][sakura][shell][pixel-ratio]") {
    auto ctx = MakeContext();
    ctx.pixelRatio = 2.0f;
    const auto place = [&](Probe& content, Sakura::PopupRequest request) {
        request.content = &content;
        LayerSlot slot;
        slot.request({request});
        LayerHost host({&slot});
        host.place(ctx, kScreen);
        return content.laidFrame;
    };

    Probe padded({100.0f, 50.0f});
    CHECK(place(padded, {.anchor = {200.0f, 100.0f, 80.0f, 20.0f}, .panel = true, .padding = 10.0f}) ==
          Rect{220.0f, 140.0f, 100.0f, 50.0f});

    Probe capped({900.0f, 50.0f});
    CHECK(place(capped, {.anchor = {100.0f, 100.0f, 1.0f, 1.0f}, .placement = Sakura::PopupPlacement::Point,
                         .panel = true, .maxWidth = 300.0f}) == Rect{124.0f, 124.0f, 552.0f, 50.0f});

    Probe sized({100.0f, 50.0f});
    CHECK(place(sized, {.anchor = {200.0f, 100.0f, 80.0f, 20.0f}, .size = {240.0f, 120.0f}}) ==
          Rect{200.0f, 120.0f, 480.0f, 240.0f});

    Probe framed({100.0f, 50.0f});
    CHECK(place(framed, {.anchor = {200.0f, 100.0f, 80.0f, 20.0f}, .panel = true, .size = {240.0f, 120.0f}}) ==
          Rect{224.0f, 144.0f, 432.0f, 192.0f});
}

TEST_CASE("Popup components scale their padding, max width and size at a pixel ratio of two",
          "[core][sakura][shell][pixel-ratio]") {
    auto ctx = MakeContext();
    ctx.pixelRatio = 2.0f;
    ctx.mouse = Extent2D<F32>{300.0f, 200.0f};

    Probe cursor({100.0f, 50.0f});
    Sakura::Popup atCursor;
    atCursor.update({
        .content = &cursor,
        .open = true,
        .placement = Sakura::PopupPlacement::Point,
        .atCursor = true,
        .panel = true,
        .padding = 6.0f,
    });
    Probe capped({500.0f, 50.0f});
    Sakura::Popup limited;
    limited.update({
        .content = &capped,
        .open = true,
        .placement = Sakura::PopupPlacement::Point,
        .maxWidth = 100.0f,
    });
    Flow flow(300.0f, 15.0f);
    Sakura::Popup sized;
    sized.update({.content = &flow, .open = true, .size = {120.0f, 0.0f}});

    LayerSlot first;
    first.hold(atCursor, {0.0f, 0.0f, 50.0f, 50.0f});
    LayerSlot second;
    second.hold(limited, {100.0f, 300.0f, 80.0f, 20.0f});
    LayerSlot third;
    third.hold(sized, {100.0f, 100.0f, 80.0f, 20.0f});
    LayerHost host({&first, &second, &third});
    host.place(ctx, kScreen);

    CHECK(cursor.laidFrame == Rect{312.0f, 212.0f, 100.0f, 50.0f});
    CHECK(capped.laidFrame == Rect{100.0f, 300.0f, 200.0f, 50.0f});
    CHECK(flow.laidFrame == Rect{100.0f, 120.0f, 240.0f, 19.0f});
}

TEST_CASE("HUDs scale their margin, padding, rounding, border, offset and size at a pixel ratio of two",
          "[core][sakura][shell][pixel-ratio]") {
    auto ctx = MakeContext({1600.0f, 1200.0f});
    ctx.pixelRatio = 2.0f;
    const Rect bounds = {0.0f, 0.0f, 1600.0f, 1200.0f};
    Probe content({100.0f, 40.0f});
    Probe corner({100.0f, 40.0f});
    Sakura::HudLayer layer;
    layer.update({.huds = {
        {.content = &content, .anchor = Sakura::Anchor::TopLeft},
        {.content = &corner, .anchor = Sakura::Anchor::BottomRight, .margin = 20.0f, .padding = {6.0f, 4.0f},
         .offset = {-10.0f, -5.0f}},
    }});
    LayerHost host({&layer});
    host.place(ctx, bounds);

    CHECK(content.laidFrame == Rect{48.0f, 48.0f, 100.0f, 40.0f});
    CHECK(corner.laidFrame == Rect{1428.0f, 1102.0f, 100.0f, 40.0f});
    const auto boxes = Find<Sakura::Box>(layer);
    REQUIRE(boxes.size() == 2);
    CHECK(boxes[0]->config().instances.front().rect == Rect{24.0f, 24.0f, 148.0f, 88.0f});
    CHECK(boxes[0]->config().cornerRadius == 24.0f);
    CHECK(boxes[0]->config().borderWidth == 2.0f);
    CHECK(boxes[1]->config().instances.front().rect == Rect{1416.0f, 1094.0f, 124.0f, 56.0f});

    Probe sized({100.0f, 40.0f});
    Sakura::HudLayer fixed;
    fixed.update({.huds = {{.content = &sized, .anchor = Sakura::Anchor::TopLeft, .size = {200.0f, 80.0f}}}});
    LayerHost other({&fixed});
    other.place(ctx, bounds);
    CHECK(sized.laidFrame == Rect{48.0f, 48.0f, 352.0f, 112.0f});
}

TEST_CASE("HUD sizes fill the room inside the margin, stay fixed, or measure their content per axis",
          "[core][sakura][shell][sizing]") {
    const auto place = [](Probe& content, Extent2D<F32> size) {
        Sakura::HudLayer layer;
        layer.update({.huds = {{.content = &content, .anchor = Sakura::Anchor::TopLeft, .size = size}}});
        LayerHost host({&layer});
        host.place(MakeContext(), kScreen);
        return content.laidFrame;
    };

    Flow wide(352.0f, 10.0f);
    CHECK(place(wide, {Sakura::Fill, 0.0f}) == Rect{24.0f, 24.0f, 752.0f, 10.0f});

    Flow narrow(352.0f, 10.0f);
    CHECK(place(narrow, {200.0f, 0.0f}) == Rect{24.0f, 24.0f, 176.0f, 20.0f});

    Probe tall({100.0f, 40.0f});
    CHECK(place(tall, {0.0f, Sakura::Fill}) == Rect{24.0f, 24.0f, 100.0f, 552.0f});

    Probe fixedHeight({100.0f, 40.0f});
    CHECK(place(fixedHeight, {0.0f, 60.0f}) == Rect{24.0f, 24.0f, 100.0f, 36.0f});

    Probe filled({100.0f, 40.0f});
    CHECK(place(filled, {Sakura::Fill, Sakura::Fill}) == Rect{24.0f, 24.0f, 752.0f, 552.0f});
    CHECK(filled.measures == 0);
}

TEST_CASE("Cancelling or leaving over a clickable HUD reaches its content without clicking",
          "[core][sakura][shell][interaction]") {
    Probe content({100.0f, 40.0f});
    content.consume = false;
    U64 clicked = 0;
    Sakura::HudLayer layer;
    layer.update({.huds = {{.content = &content, .anchor = Sakura::Anchor::TopLeft, .clickable = true,
                            .onClick = [&] { ++clicked; }}}});
    Sakura::Layer base;
    LayerHost host({&base, &layer});
    host.place(MakeContext(), kScreen);

    host.send(Mouse(MouseEventType::Cancel, 50.0f, 40.0f));
    host.send(Mouse(MouseEventType::Leave, 50.0f, 40.0f));
    CHECK(clicked == 0);
    CHECK(content.count(MouseEventType::Cancel) == 1);
    CHECK(content.count(MouseEventType::Leave) == 1);

    host.send(Mouse(MouseEventType::Click, 50.0f, 40.0f));
    CHECK(clicked == 1);
    host.send(Mouse(MouseEventType::Cancel, 50.0f, 40.0f));
    host.send(Mouse(MouseEventType::Release, 50.0f, 40.0f));
    CHECK(clicked == 1);
    CHECK(content.count(MouseEventType::Cancel) == 2);
    CHECK(content.count(MouseEventType::Release) == 1);
}

TEST_CASE("Modal layers scale their margin, padding, fixed size and close button at a pixel ratio of two",
          "[core][sakura][shell][pixel-ratio]") {
    const ImGuiContextGuard imguiContext;
    SakuraTest::FakeWindow window;
    auto ctx = WindowContext(window, {1600.0f, 1200.0f});
    ctx.pixelRatio = 2.0f;

    Probe natural({100.0f, 50.0f});
    Sakura::ModalLayer wide;
    wide.update({.content = &natural});
    LayerHost host({&wide});
    host.place(ctx, {0.0f, 0.0f, 1600.0f, 1200.0f});
    CHECK(natural.laidFrame.x == 274.0f);
    CHECK(natural.laidFrame.width == 1052.0f);
    CHECK(natural.laidFrame.height == 50.0f);

    auto narrowCtx = WindowContext(window, {800.0f, 1200.0f});
    narrowCtx.pixelRatio = 2.0f;
    Probe clamped({100.0f, 50.0f});
    Sakura::ModalLayer narrow;
    narrow.update({.content = &clamped});
    LayerHost small({&narrow});
    small.place(narrowCtx, {0.0f, 0.0f, 800.0f, 1200.0f});
    CHECK(clamped.laidFrame.x == 62.0f);
    CHECK(clamped.laidFrame.width == 676.0f);

    U64 closed = 0;
    Probe sized({100.0f, 50.0f});
    Sakura::ModalLayer fixed;
    fixed.update({.content = &sized, .size = {400.0f, 300.0f}, .onClose = [&] { ++closed; }});
    LayerHost other({&fixed});
    other.place(ctx, {0.0f, 0.0f, 1600.0f, 1200.0f});
    CHECK(sized.laidFrame.x == 424.0f);
    CHECK(sized.laidFrame.y == 324.0f);
    CHECK(sized.laidFrame.width == 752.0f);
    CHECK(sized.laidFrame.height < 552.0f - 80.0f);

    CHECK(other.click(800.0f, 806.0f));
    CHECK(closed == 1);
}

TEST_CASE("Modal sizes fill the room inside the margin or measure the content width when natural",
          "[core][sakura][shell][sizing]") {
    const ImGuiContextGuard imguiContext;
    SakuraTest::FakeWindow window;
    const auto place = [&](Probe& content, Extent2D<F32> size, U64* closed = nullptr) {
        auto modal = std::make_unique<Sakura::ModalLayer>();
        modal->update({.content = &content, .size = size, .onClose = [closed] {
            if (closed) {
                ++*closed;
            }
        }});
        auto host = std::make_unique<LayerHost>(std::vector<Sakura::Layer*>{modal.get()});
        host->place(WindowContext(window), kScreen);
        return std::make_pair(std::move(modal), std::move(host));
    };

    Probe fillWidth({100.0f, 50.0f});
    place(fillWidth, {Sakura::Fill, 0.0f});
    CHECK(fillWidth.laidFrame.x == 31.0f);
    CHECK(fillWidth.laidFrame.width == 738.0f);
    CHECK(fillWidth.laidFrame.height == 50.0f);

    Probe natural({100.0f, 50.0f});
    place(natural, {0.0f, 0.0f});
    CHECK(natural.laidFrame.x == 350.0f);
    CHECK(natural.laidFrame.width == 100.0f);

    Probe oversized({900.0f, 50.0f});
    place(oversized, {0.0f, 0.0f});
    CHECK(oversized.laidFrame.x == 31.0f);
    CHECK(oversized.laidFrame.width == 738.0f);

    Probe column({100.0f, 50.0f});
    place(column, {300.0f, Sakura::Fill});
    CHECK(column.laidFrame.x == 262.0f);
    CHECK(column.laidFrame.y == 31.0f);
    CHECK(column.laidFrame.width == 276.0f);

    U64 closed = 0;
    Probe filled({100.0f, 50.0f});
    auto layers = place(filled, {Sakura::Fill, Sakura::Fill}, &closed);
    CHECK(filled.laidFrame.x == 31.0f);
    CHECK(filled.laidFrame.y == 31.0f);
    CHECK(filled.laidFrame.width == 738.0f);
    CHECK(filled.laidFrame.height > 50.0f);
    CHECK(layers.second->click(400.0f, 560.0f));
    CHECK(closed == 1);
}

TEST_CASE("Floating windows scale their initial size, title bar, padding, close button and grip at a pixel ratio of two",
          "[core][sakura][shell][pixel-ratio]") {
    Probe first;
    Probe second;
    U64 closed = 0;
    FloatingProbe layer;
    layer.update({.windows = {
        {.id = "a", .title = "A", .content = &first, .size = {300.0f, 200.0f}, .onClose = [&] { ++closed; }},
        {.id = "b", .title = "B", .content = &second, .size = {300.0f, 200.0f}, .padding = 4.0f},
    }});
    Sakura::Layer base;
    LayerHost host({&base, &layer});
    auto ctx = MakeContext({1600.0f, 1200.0f});
    ctx.pixelRatio = 2.0f;
    host.place(ctx, {0.0f, 0.0f, 1600.0f, 1200.0f});

    CHECK(first.laidFrame == Rect{500.0f, 464.0f, 600.0f, 336.0f});
    CHECK(second.laidFrame == Rect{556.0f, 520.0f, 584.0f, 320.0f});

    CHECK(layer.cursorAt({1117.0f, 817.0f}) == SurfaceCursor::ResizeNWSE);
    CHECK(layer.cursorAt({1154.0f, 854.0f}) == SurfaceCursor::ResizeNWSE);
    CHECK_FALSE(layer.cursorAt({1114.0f, 814.0f}).has_value());

    CHECK(host.click(1046.0f, 415.0f));
    CHECK(closed == 1);
}

TEST_CASE("Toasts scale their margin, spacing, padding and card metrics at a pixel ratio of two",
          "[core][sakura][shell][pixel-ratio]") {
    const ImGuiContextGuard imguiContext;
    SakuraTest::FakeWindow window;
    struct Cards {
        std::vector<Rect> rects;
        F32 border = 0.0f;
        F32 softness = 0.0f;
    };
    const auto cards = [&](F32 ratio) {
        (void)Sakura::Private::TakeToasts();
        ToastProbe layer;
        Sakura::PushToast(Sakura::ToastType::Success, 1000, "");
        Sakura::PushToast(Sakura::ToastType::Info, 1000, "");
        layer.update();
        LayerHost host({&layer});
        const Rect bounds = {0.0f, 0.0f, 800.0f * ratio, 600.0f * ratio};
        auto ctx = WindowContext(window, {bounds.width, bounds.height});
        ctx.pixelRatio = ratio;
        ctx.time = 10.0;
        host.place(ctx, bounds);
        ctx.time = 10.2;
        host.place(ctx, bounds);
        Cards result;
        for (auto* box : layer.parts<Sakura::Box>()) {
            const auto& config = box->config();
            if (config.borderWidth > 0.0f && !config.topLit && config.cornerRadius == 12.0f * ratio) {
                result.rects.push_back(config.instances.front().rect);
                result.border = config.borderWidth;
            }
            if (config.softness > 0.0f) {
                result.softness = config.softness;
            }
        }
        return result;
    };

    const Cards single = cards(1.0f);
    const Cards doubled = cards(2.0f);
    REQUIRE(single.rects.size() == 2);
    REQUIRE(doubled.rects.size() == 2);
    CHECK(doubled.rects[0].x == 40.0f);
    CHECK(doubled.rects[0].bottom() == 1160.0f);
    CHECK(doubled.rects[0].height == 108.0f);
    CHECK(doubled.rects[0].width == 2.0f * single.rects[0].width);
    CHECK(doubled.rects[1].bottom() == doubled.rects[0].y - 20.0f);
    CHECK(doubled.border == 2.0f);
    CHECK(doubled.softness == 16.0f);
    CHECK(single.softness == 8.0f);
}

TEST_CASE("Menus scale their rows, padding and separator hairline at a pixel ratio of two",
          "[core][sakura][shell][pixel-ratio]") {
    Sakura::Menu menu;
    menu.update({
        .entries = {{.label = "Open"}, {.separator = true}, {.label = "Save"}},
        .open = true,
        .atCursor = false,
    });
    LayerSlot slot;
    slot.hold(menu, {100.0f, 100.0f, 1.0f, 1.0f});
    LayerHost host({&slot});
    auto ctx = MakeContext();
    ctx.pixelRatio = 2.0f;
    host.place(ctx, kScreen);

    CHECK(host.popups.interactiveAt({199.0f, 241.0f}));
    CHECK_FALSE(host.popups.interactiveAt({200.0f, 241.0f}));
    CHECK_FALSE(host.popups.interactiveAt({199.0f, 242.0f}));

    auto* open = Named(host, "Open");
    auto* save = Named(host, "Save");
    REQUIRE(open);
    REQUIRE(save);
    CHECK(Peek::Frame(*open).x == 124.0f);
    CHECK(Peek::Frame(*open).y == 124.0f);
    CHECK(Peek::Frame(*open).height == 30.0f);
    CHECK(Peek::Frame(*save).y == 188.0f);
    CHECK(Peek::Frame(*save).height == 30.0f);

    std::optional<Rect> separator;
    for (auto* box : Find<Sakura::Box>(host)) {
        const auto& instances = box->config().instances;
        if (instances.size() == 1 && instances.front().rect.height == 2.0f) {
            separator = instances.front().rect;
        }
    }
    REQUIRE(separator.has_value());
    CHECK(*separator == Rect{102.0f, 170.0f, 96.0f, 2.0f});
}

TEST_CASE("Chrome buttons measure a natural, fixed or filling height", "[core][sakura][shell][sizing]") {
    const auto measure = [](Sakura::ChromeButton::Config config, Extent2D<F32> available) {
        Sakura::ChromeButton button;
        button.update(std::move(config));
        Host host(button);
        return host.measureOf(MakeContext(), available);
    };
    const Extent2D<F32> unbounded = {kInfinite, kInfinite};

    CHECK(measure({.label = "File"}, unbounded) == Extent2D<F32>{20.0f, 28.0f});
    CHECK(measure({.label = "File", .height = 40.0f}, {200.0f, 30.0f}) == Extent2D<F32>{20.0f, 30.0f});
    CHECK(measure({.label = "File", .height = 0.0f, .padding = {10.0f, 4.0f}}, unbounded) == Extent2D<F32>{20.0f, 23.0f});
    CHECK(measure({.label = "File", .height = Sakura::Fill, .padding = {10.0f, 4.0f}}, {200.0f, 40.0f}) ==
          Extent2D<F32>{20.0f, 40.0f});
    CHECK(measure({.label = "File", .height = Sakura::Fill, .padding = {10.0f, 4.0f}}, unbounded) ==
          Extent2D<F32>{20.0f, 23.0f});
    CHECK(measure({.glyph = Sakura::ChromeButton::Glyph::Plus, .height = 0.0f, .padding = {10.0f, 4.0f}}, unbounded) ==
          Extent2D<F32>{23.0f, 23.0f});
}

TEST_CASE("Chrome buttons scale their height, padding, rounding and glyph at a pixel ratio of two",
          "[core][sakura][shell][pixel-ratio]") {
    auto ctx = MakeContext();
    ctx.pixelRatio = 2.0f;

    Sakura::ChromeButton label;
    label.update({.label = "File"});
    Host labelHost(label);
    CHECK(labelHost.measureOf(ctx, {kInfinite, kInfinite}) == Extent2D<F32>{40.0f, 56.0f});

    Sakura::ChromeButton natural;
    natural.update({.label = "File", .height = 0.0f, .padding = {10.0f, 4.0f}});
    Host naturalHost(natural);
    CHECK(naturalHost.measureOf(ctx, {kInfinite, kInfinite}) == Extent2D<F32>{40.0f, 46.0f});

    Sakura::ChromeButton plus;
    plus.update({.glyph = Sakura::ChromeButton::Glyph::Plus});
    Host plusHost(plus);
    CHECK(plusHost.measureOf(ctx, {kInfinite, kInfinite}) == Extent2D<F32>{56.0f, 56.0f});
    plusHost.place(ctx, {0.0f, 0.0f, 56.0f, 56.0f});

    const auto boxes = Find<Sakura::Box>(plus);
    REQUIRE_FALSE(boxes.empty());
    CHECK(boxes.front()->config().cornerRadius == 14.0f);
    const auto lines = Find<Sakura::Polyline>(plus);
    REQUIRE(lines.size() == 1);
    const auto& marks = lines.front()->config().instances;
    REQUIRE(marks.size() == 2);
    CHECK(marks[0].thickness == 3.0f);
    CHECK(marks[0].points.front() == Extent2D<F32>{16.0f, 28.0f});
    CHECK(marks[0].points.back() == Extent2D<F32>{40.0f, 28.0f});
}

TEST_CASE("Dropdowns scale their rows, padding and wheel step at a pixel ratio of two",
          "[core][sakura][shell][pixel-ratio]") {
    auto ctx = MakeContext();
    ctx.pixelRatio = 2.0f;

    std::vector<U64> selected;
    Sakura::Dropdown dropdown;
    dropdown.update({.options = {"alpha", "beta", "gamma"}, .onSelect = [&](U64 row) { selected.push_back(row); }});
    LayerSlot slot;
    slot.onLayout = [&](const Sakura::Context& layoutCtx) {
        dropdown.request(layoutCtx, {100.0f, 100.0f, 200.0f, 25.0f});
    };
    LayerHost host({&slot});
    dropdown.open();
    host.place(ctx, kScreen);

    CHECK(host.popups.interactiveAt({100.0f, 125.0f}));
    CHECK(host.popups.interactiveAt({299.0f, 322.0f}));
    CHECK_FALSE(host.popups.interactiveAt({300.0f, 125.0f}));
    CHECK_FALSE(host.popups.interactiveAt({150.0f, 323.0f}));
    CHECK(host.click(150.0f, 260.0f));
    CHECK(selected == std::vector<U64>{2});

    std::vector<std::string> options;
    for (U64 index = 0; index < 20; ++index) {
        options.push_back("o" + std::to_string(index));
    }
    Sakura::Dropdown scrolling;
    scrolling.update({.options = options});
    LayerSlot scrollSlot;
    scrollSlot.onLayout = [&](const Sakura::Context& layoutCtx) {
        scrolling.request(layoutCtx, {100.0f, 100.0f, 200.0f, 25.0f});
    };
    LayerHost scrollHost({&scrollSlot});
    scrolling.open();
    scrollHost.place(ctx, kScreen);
    CHECK(scrollHost.popups.interactiveAt({299.0f, 572.0f}));
    CHECK_FALSE(scrollHost.popups.interactiveAt({150.0f, 573.0f}));

    const auto firstRow = [&]() -> std::optional<Sakura::Label::Instance> {
        for (auto* label : Find<Sakura::Label>(scrollHost)) {
            const auto& instances = label->config().instances;
            if (!instances.empty() && instances.front().str.starts_with("o")) {
                return instances.front();
            }
        }
        return std::nullopt;
    };
    REQUIRE(firstRow().has_value());
    CHECK(firstRow()->str == "o0");
    CHECK(firstRow()->rect.y == 159.0f);
    CHECK(firstRow()->fontSize == 30.0f);

    CHECK(scrollHost.send(Wheel(150.0f, 300.0f, -1.0f)));
    scrollHost.place(ctx, kScreen);
    REQUIRE(firstRow().has_value());
    CHECK(firstRow()->str == "o1");
    CHECK(firstRow()->rect.y == 113.0f);
}

TEST_CASE("A popup owner hidden while open drops its popup and requests it again when shown",
          "[core][sakura][shell]") {
    Probe content({100.0f, 40.0f});
    Sakura::Popup popup;
    popup.update({.content = &content, .open = true});
    LayerSlot slot;
    slot.hold(popup, {100.0f, 100.0f, 50.0f, 20.0f});
    LayerHost host({&slot});
    const auto ctx = MakeContext();

    host.place(ctx, kScreen);
    host.place(ctx, kScreen);
    REQUIRE(content.visible());
    REQUIRE(host.popups.interactiveAt({110.0f, 130.0f}));

    popup.setVisible(false);
    for (int frame = 0; frame < 2; ++frame) {
        Frames::Access::present(host, static_cast<U64>(frame));
        host.place(ctx, kScreen);
    }
    CHECK_FALSE(host.popups.interactiveAt({110.0f, 130.0f}));

    popup.setVisible(true);
    host.place(ctx, kScreen);
    host.place(ctx, kScreen);
    CHECK(host.popups.interactiveAt({110.0f, 130.0f}));
}

TEST_CASE("Popup content destroyed while its owner lives is dropped before the popup layer touches it",
          "[core][sakura][shell]") {
    auto content = std::make_unique<Probe>(Extent2D<F32>{100.0f, 40.0f});
    U64 dismissed = 0;
    LayerSlot slot;
    slot.request({{.content = content.get(), .anchor = {100.0f, 100.0f, 50.0f, 20.0f},
                   .onDismiss = [&] { ++dismissed; }}});
    LayerHost host({&slot});
    Frames frames(host);
    frames.verify = false;
    REQUIRE(frames.settle());
    REQUIRE(host.popups.interactiveAt({110.0f, 130.0f}));

    content.reset();
    frames.run();
    CHECK_FALSE(host.popups.interactiveAt({110.0f, 130.0f}));
    CHECK_FALSE(host.popups.covers({600.0f, 500.0f}));
    CHECK_FALSE(host.click(600.0f, 500.0f));
    CHECK(dismissed == 0);
    CHECK(frames.settle());
}

TEST_CASE("A popup whose owner and content are destroyed between frames is pruned before it is mounted",
          "[core][sakura][shell]") {
    auto content = std::make_unique<Probe>(Extent2D<F32>{100.0f, 40.0f});
    auto popup = std::make_unique<Sakura::Popup>();
    popup->update({.content = content.get(), .open = true});
    LayerSlot slot;
    slot.hold(*popup, {100.0f, 100.0f, 50.0f, 20.0f});
    LayerHost host({&slot});
    Frames frames(host);
    frames.verify = false;
    REQUIRE(frames.settle());
    REQUIRE(host.popups.interactiveAt({110.0f, 130.0f}));

    slot.child = nullptr;
    slot.update({.content = nullptr});
    popup.reset();
    content.reset();
    frames.run();
    CHECK_FALSE(host.popups.interactiveAt({110.0f, 130.0f}));
    CHECK(frames.settle());
}

TEST_CASE("Content requested by two popup owners shows once at the first owner's anchor",
          "[core][sakura][shell]") {
    Probe content({100.0f, 40.0f});
    LayerSlot first;
    LayerSlot second;
    first.request({{.content = &content, .anchor = {100.0f, 100.0f, 50.0f, 20.0f}}});
    second.request({{.content = &content, .anchor = {400.0f, 300.0f, 50.0f, 20.0f}}});
    LayerHost host({&first, &second});
    Frames frames(host);
    frames.verify = false;
    REQUIRE(frames.settle());

    CHECK(content.currentFrame() == Rect{100.0f, 120.0f, 100.0f, 40.0f});
    CHECK(host.popups.interactiveAt({110.0f, 130.0f}));
    CHECK_FALSE(host.popups.interactiveAt({410.0f, 330.0f}));

    first.request({});
    frames.run();
    CHECK(content.currentFrame() == Rect{400.0f, 320.0f, 100.0f, 40.0f});
    CHECK_FALSE(host.popups.interactiveAt({110.0f, 130.0f}));
    CHECK(host.popups.interactiveAt({410.0f, 330.0f}));
}

TEST_CASE("An outside click after the first of two owners sharing popup content is destroyed dismisses the survivor",
          "[core][sakura][shell]") {
    Probe content({100.0f, 40.0f});
    U64 firstDismissed = 0;
    U64 secondDismissed = 0;
    auto first = std::make_unique<Sakura::Popup>();
    first->update({.content = &content, .open = true, .onDismiss = [&] { ++firstDismissed; }});
    Sakura::Popup second;
    second.update({.content = &content, .open = true, .onDismiss = [&] { ++secondDismissed; }});
    LayerSlot firstSlot;
    LayerSlot secondSlot;
    firstSlot.hold(*first, {100.0f, 100.0f, 50.0f, 20.0f});
    secondSlot.hold(second, {400.0f, 300.0f, 50.0f, 20.0f});
    LayerHost host({&firstSlot, &secondSlot});
    Frames frames(host);
    frames.verify = false;
    REQUIRE(frames.settle());
    REQUIRE(content.currentFrame() == Rect{100.0f, 120.0f, 100.0f, 40.0f});

    firstSlot.child = nullptr;
    firstSlot.update({.content = nullptr});
    first.reset();
    frames.run();
    CHECK(host.click(700.0f, 550.0f));
    CHECK(firstDismissed == 0);
    CHECK(secondDismissed == 1);
}

TEST_CASE("A dropdown reopened at the same anchor after scrolling draws and selects from its first row",
          "[core][sakura][shell]") {
    const Scene scene;
    std::vector<std::string> options;
    for (U64 index = 0; index < 30; ++index) {
        options.push_back("o" + std::to_string(index));
    }
    std::vector<U64> selections;
    Sakura::Dropdown dropdown;
    LayerSlot slot;
    slot.onLayout = [&](const Sakura::Context& ctx) {
        dropdown.request(ctx, {100.0f, 100.0f, 240.0f, 25.0f});
    };
    LayerHost host({&slot});
    Frames frames(host, kScreen, scene.context());
    frames.verify = false;
    const auto update = [&] {
        dropdown.update({.options = options, .onSelect = [&](U64 index) { selections.push_back(index); }});
    };
    const auto firstRow = [&]() -> std::optional<Sakura::Label::Instance> {
        for (const auto& leaf : Drawn(host)) {
            if (const auto* label = std::get_if<Sakura::Label::Config>(&leaf.config)) {
                if (!label->instances.empty() && label->instances.front().str.starts_with("o")) {
                    return label->instances.front();
                }
            }
        }
        return std::nullopt;
    };
    dropdown.open();
    slot.relayout();
    REQUIRE(frames.settle(update));
    REQUIRE(firstRow().has_value());
    REQUIRE(firstRow()->str == "o0");

    Hover(host, frames, 150.0f, 160.0f);
    host.send(Wheel(150.0f, 160.0f, -10.0f));
    REQUIRE(frames.settle(update));
    REQUIRE(firstRow()->str != "o0");

    dropdown.close();
    REQUIRE(frames.settle(update));
    dropdown.open();
    REQUIRE(frames.settle(update));
    const auto row = firstRow();
    REQUIRE(row.has_value());
    CHECK(row->str == "o0");
    const F32 x = row->rect.x + 4.0f;
    const F32 y = row->rect.y + row->rect.height * 0.5f;
    Hover(host, frames, x, y);
    frames.run(update);
    Press(host, frames, x, y);
    Release(host, frames, x, y);
    frames.run(update);
    const U64 shown = static_cast<U64>(std::stoull(row->str.substr(1)));
    CHECK(selections == std::vector<U64>{shown});
}

TEST_CASE("An inside click after a popup owner is destroyed never reaches the content it left behind",
          "[core][sakura][shell]") {
    Probe content({100.0f, 40.0f});
    auto popup = std::make_unique<Sakura::Popup>();
    popup->update({.content = &content, .open = true});
    LayerSlot slot;
    slot.hold(*popup, {100.0f, 100.0f, 50.0f, 20.0f});
    LayerHost host({&slot});
    Frames frames(host);
    frames.verify = false;
    REQUIRE(frames.settle());
    REQUIRE(host.popups.interactiveAt({110.0f, 130.0f}));
    const U64 clicks = content.count(MouseEventType::Click);

    slot.child = nullptr;
    slot.update({.content = nullptr});
    popup.reset();
    CHECK_FALSE(host.popups.interactiveAt({110.0f, 130.0f}));
    frames.run();
    CHECK_FALSE(content.visible());
    host.click(110.0f, 130.0f);
    CHECK(content.count(MouseEventType::Click) == clicks);
}

TEST_CASE("A child keeps showing popup content it shares with its parent after the parent withdraws",
          "[core][sakura][shell]") {
    Probe content({100.0f, 40.0f});
    Sakura::Popup popup;
    popup.update({.content = &content, .open = true});
    NestedRequester parent;
    parent.request = {.content = &content, .anchor = {100.0f, 100.0f, 50.0f, 20.0f}};
    parent.hold(popup);
    LayerSlot slot;
    slot.hold(parent, {400.0f, 300.0f, 50.0f, 20.0f});
    LayerHost host({&slot});
    Frames frames(host);
    frames.verify = false;
    const auto shown = [&] {
        const auto probes = Find<Probe>(host);
        return std::find(probes.begin(), probes.end(), &content) != probes.end();
    };
    REQUIRE(frames.settle());
    REQUIRE(shown());

    parent.withdraw();
    REQUIRE(frames.settle());
    CHECK(shown());
    CHECK(content.currentFrame() == Rect{400.0f, 320.0f, 100.0f, 40.0f});
    CHECK(host.popups.interactiveAt({410.0f, 330.0f}));
}

TEST_CASE("Open dropdowns follow their anchor width without any other body change",
          "[core][sakura][shell]") {
    Sakura::Dropdown dropdown;
    dropdown.update({.options = {"a", "b"}});
    F32 width = 240.0f;
    LayerSlot slot;
    slot.onLayout = [&](const Sakura::Context& ctx) {
        dropdown.request(ctx, {100.0f, 100.0f, width, 25.0f});
    };
    LayerHost host({&slot});
    Frames frames(host);
    frames.verify = false;
    dropdown.open();
    REQUIRE(frames.settle());
    CHECK(host.popups.interactiveAt({339.0f, 140.0f}));
    CHECK_FALSE(host.popups.interactiveAt({341.0f, 140.0f}));
    Hover(host, frames, 150.0f, 140.0f);
    frames.run();

    width = 120.0f;
    slot.relayout();
    frames.run();
    CHECK(host.popups.interactiveAt({219.0f, 140.0f}));
    CHECK_FALSE(host.popups.interactiveAt({221.0f, 140.0f}));
}

TEST_CASE("An open chrome button menu runs the newest entry callback after a callback-only update",
          "[core][sakura][shell]") {
    const ImGuiContextGuard imguiContext;
    MonoFontWindow window;
    Sakura::ChromeButton button;
    U64 stale = 0;
    U64 fresh = 0;
    Sakura::ChromeButton::Config config{
        .label = "File",
        .entries = {{.label = "Quit", .onClick = [&] { ++stale; }}},
    };
    LayerSlot slot;
    slot.hold(button, {100.0f, 10.0f, 60.0f, 28.0f});
    LayerHost host({&slot});
    Frames frames(host, kScreen, WindowContext(window));
    frames.verify = false;
    const auto update = [&] {
        button.update(config);
    };
    REQUIRE(frames.settle(update));
    host.click(130.0f, 24.0f);
    REQUIRE(frames.settle(update));
    REQUIRE(button.menuOpen());
    auto* quit = Named(host, "Quit");
    REQUIRE(quit);
    const Rect row = Peek::Frame(*quit);
    const Extent2D<F32> center = {row.x + row.width * 0.5f, row.y + row.height * 0.5f};

    config.entries[0].onClick = [&] { ++fresh; };
    CHECK(frames.run(update).clean());
    host.send(Mouse(MouseEventType::Move, center.x, center.y));
    frames.run(update);
    CHECK(host.click(center.x, center.y));
    CHECK(fresh == 1);
    CHECK(stale == 0);
    CHECK_FALSE(button.menuOpen());
}

TEST_CASE("Closing a toast after a long clean hover starts its fade instead of expiring it at once",
          "[core][sakura][shell]") {
    const ImGuiContextGuard imguiContext;
    SakuraTest::FakeWindow window;
    (void)Sakura::Private::TakeToasts();
    ToastProbe layer;
    Sakura::PushToast(Sakura::ToastType::Info, 1000, "Saved the flowgraph");
    LayerHost host({&layer});
    Frames frames(host, kScreen, WindowContext(window));
    frames.verify = false;
    const auto at = [&](F64 time) {
        frames.ctx.time = time;
        return frames.run([&] { layer.update(); });
    };
    at(10.0);
    at(10.2);
    const U64 shown = layer.shown();
    REQUIRE(shown > 0);

    const auto glyph = [&]() -> const Sakura::Text* {
        for (auto* text : layer.parts<Sakura::Text>()) {
            if (text->config().str == ICON_FA_XMARK) {
                return text;
            }
        }
        return nullptr;
    };
    host.send(Mouse(MouseEventType::Move, 40.0f, 570.0f));
    at(10.3);
    const Rect badge = Peek::Frame(*glyph());
    const Extent2D<F32> center = {badge.x + badge.width * 0.5f, badge.y + badge.height * 0.5f};
    host.send(Mouse(MouseEventType::Move, center.x, center.y));
    at(10.4);
    for (const F64 time : {12.0, 15.0, 20.0}) {
        CHECK(at(time).clean());
    }

    CHECK(host.send(Mouse(MouseEventType::Click, center.x, center.y)));
    CHECK(host.send(Mouse(MouseEventType::Release, center.x, center.y)));
    at(20.0);
    CHECK(layer.shown() == shown);
    at(20.1);
    CHECK(layer.shown() == shown);
    CHECK(glyph()->config().opacity < 1.0f);
    at(20.2);
    CHECK(layer.shown() == 0);
}

TEST_CASE("Hovering a toast late in its life pauses only from the hover onward",
          "[core][sakura][shell]") {
    const ImGuiContextGuard imguiContext;
    SakuraTest::FakeWindow window;
    (void)Sakura::Private::TakeToasts();
    ToastProbe layer;
    Sakura::PushToast(Sakura::ToastType::Info, 1000, "Saved the flowgraph");
    LayerHost host({&layer});
    Frames frames(host, kScreen, WindowContext(window));
    frames.verify = false;
    const auto at = [&](F64 time) {
        frames.ctx.time = time;
        return frames.run([&] { layer.update(); });
    };
    at(10.0);
    at(10.2);
    const U64 shown = layer.shown();
    REQUIRE(shown > 0);
    CHECK(at(10.5).clean());

    host.send(Mouse(MouseEventType::Move, 40.0f, 570.0f));
    at(11.0);
    CHECK(at(11.5).clean());
    host.send(Mouse(MouseEventType::Leave, -10.0f, -10.0f));
    at(11.5);
    CHECK(layer.shown() == shown);
    at(11.7);
    CHECK(layer.shown() == shown);
    at(11.75);
    at(11.85);
    CHECK(layer.shown() == 0);
}

TEST_CASE("An outside click after a popup owner was destroyed never runs that owner's dismiss callback",
          "[core][sakura][shell]") {
    Probe content({100.0f, 40.0f});
    U64 dismissed = 0;
    auto popup = std::make_unique<Sakura::Popup>();
    popup->update({.content = &content, .open = true, .onDismiss = [&] { ++dismissed; }});
    LayerSlot slot;
    slot.hold(*popup, {100.0f, 100.0f, 50.0f, 20.0f});
    LayerHost host({&slot});
    Frames frames(host);
    frames.verify = false;
    REQUIRE(frames.settle());
    REQUIRE(content.visible());

    slot.child = nullptr;
    slot.update({.content = nullptr});
    popup.reset();
    host.click(600.0f, 500.0f);
    CHECK(dismissed == 0);
    frames.run();
    CHECK_FALSE(content.visible());
    CHECK_FALSE(host.popups.covers({600.0f, 500.0f}));
}
