#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <vector>

#include "harness.hh"
#include "superluminal/surface_interaction.hh"

using namespace Jetstream;

TEST_CASE("Superluminal forwards captured moves and release outside the surface",
          "[core][sakura][superluminal][surface][capture]") {
    SakuraTest::HeadlessUi ui;
    const ImVec2 origin{20.0f, 20.0f};
    const ImVec2 size{200.0f, 100.0f};
    std::vector<MouseEvent> events;
    SurfaceInteractionState interaction;
    interaction.zoom = 2.0f;
    const auto frame = [&](const ImVec2& position, bool down) {
        events.clear();
        ui.setMouse(position, down);
        ui.frame([&] {
            ImGui::SetCursorScreenPos(origin);
            ImGui::InvisibleButton("plot", size);
            detail::ForwardSuperluminalSurfaceMouseEvents(origin, size,
                [&](const MouseEvent& event) { events.push_back(event); });
        });
        auto pending = events;
        interaction = ProcessSurfaceInteraction(interaction, {}, std::move(pending));
    };
    const auto count = [&](MouseEventType type) {
        return std::count_if(events.begin(), events.end(),
            [type](const auto& event) { return event.type == type; });
    };

    frame({120.0f, 70.0f}, false);
    frame({120.0f, 70.0f}, false);
    frame({120.0f, 70.0f}, true);
    REQUIRE(count(MouseEventType::Click) == 1);
    REQUIRE(interaction.dragging);

    frame({260.0f, 90.0f}, true);
    CHECK(count(MouseEventType::Move) == 1);
    if (!events.empty()) {
        CHECK(events.back().position.x == Catch::Approx(1.2f));
        CHECK(events.back().position.y == Catch::Approx(0.7f));
    }
    frame({280.0f, 100.0f}, false);
    CHECK(count(MouseEventType::Release) == 1);
    const auto released = std::find_if(events.begin(), events.end(), [](const auto& event) {
        return event.type == MouseEventType::Release;
    });
    if (released != events.end()) {
        CHECK(released->button == MouseButton::Left);
        CHECK(released->position.x == Catch::Approx(1.3f));
        CHECK(released->position.y == Catch::Approx(0.8f));
    }
    CHECK_FALSE(interaction.dragging);
    const auto offsetAfterRelease = interaction.offset;

    frame({80.0f, 70.0f}, false);
    CHECK(count(MouseEventType::Release) == 0);
    CHECK(count(MouseEventType::Click) == 0);
    CHECK_FALSE(interaction.dragging);
    CHECK(interaction.offset == offsetAfterRelease);
}

TEST_CASE("Superluminal does not start gestures or scroll outside the surface",
          "[core][sakura][superluminal][surface][capture]") {
    SakuraTest::HeadlessUi ui;
    const ImVec2 origin{20.0f, 20.0f};
    const ImVec2 size{200.0f, 100.0f};
    std::vector<MouseEvent> events;
    const auto frame = [&](const ImVec2& position, bool down, F32 scroll = 0.0f) {
        events.clear();
        ui.setMouse(position, down);
        ImGui::GetIO().MouseWheel = scroll;
        ui.frame([&] {
            ImGui::SetCursorScreenPos(origin);
            ImGui::InvisibleButton("plot", size);
            detail::ForwardSuperluminalSurfaceMouseEvents(origin, size,
                [&](const MouseEvent& event) { events.push_back(event); });
        });
    };

    frame({260.0f, 90.0f}, false);
    frame({260.0f, 90.0f}, true, 1.0f);
    REQUIRE(events.empty());
    frame({260.0f, 90.0f}, false);
    REQUIRE(events.empty());

    frame({120.0f, 70.0f}, false, 1.0f);
    REQUIRE(std::any_of(events.begin(), events.end(), [](const auto& event) {
        return event.type == MouseEventType::Scroll && event.scroll.y == 1.0f;
    }));
    frame({120.0f, 70.0f}, true);
    frame({260.0f, 90.0f}, true, 1.0f);
    REQUIRE(std::none_of(events.begin(), events.end(), [](const auto& event) {
        return event.type == MouseEventType::Scroll || event.type == MouseEventType::Click;
    }));
    frame({260.0f, 90.0f}, false);
}
