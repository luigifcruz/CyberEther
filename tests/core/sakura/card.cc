#include <catch2/catch_test_macros.hpp>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/card.hh>

#include "harness.hh"
#include "component_harness.hh"
#include "scene.hh"
#include "render/sakura/metrics.hh"

#include <vector>

using namespace Jetstream;
using namespace SakuraTest;

namespace {

constexpr Rect kCard = {20.0f, 30.0f, 200.0f, 100.0f};

Sakura::Card::Config TileCard(Sakura::Card::Layer layer = Sakura::Card::Layer::Full) {
    return {
        .rects = {kCard},
        .cornerRadius = 12.0f,
        .softness = 6.0f,
        .drop = 2.0f,
        .borderWidth = 1.5f,
        .lineWidth = 1.0f,
        .fillKey = "tile_bg",
        .layer = layer,
    };
}

std::vector<const Sakura::Box::Config*> Boxes(Sakura::Card& card) {
    std::vector<const Sakura::Box::Config*> boxes;
    for (auto* box : Find<Sakura::Box>(card)) {
        boxes.push_back(&box->config());
    }
    return boxes;
}

}  // namespace

TEST_CASE("Full cards draw a dropped shadow, a bordered fill and a tonal rim on the outline",
          "[core][sakura][card]") {
    Scene scene;
    const auto ctx = scene.context();
    Sakura::Card card;
    Host host(card);
    card.update(TileCard());
    host.place(ctx, kScreen);

    const auto boxes = Boxes(card);
    REQUIRE(boxes.size() == 3);
    const auto& shadow = *boxes[0];
    REQUIRE(shadow.instances.size() == 1);
    CHECK(shadow.instances[0].rect == Rect{14.0f, 26.0f, 212.0f, 112.0f});
    CHECK(shadow.instances[0].backgroundColor == ctx.color("tile_shadow"));
    CHECK(shadow.softness == 6.0f);
    const auto& fill = *boxes[1];
    CHECK(fill.instances[0].rect == kCard);
    CHECK(fill.instances[0].backgroundColor == ctx.color("tile_bg"));
    CHECK(fill.borderWidth == 1.5f);
    CHECK(fill.borderColor == ctx.color("tile_border"));
    const auto& rim = *boxes[2];
    CHECK(rim.topLit);
    CHECK(rim.borderWidth == 1.0f);
    CHECK(rim.instances[0].rect == kCard);
    CHECK(rim.cornerRadius == 12.0f);
    CHECK(rim.borderColor == Sakura::RimColor(ctx.color("tile_bg")));
}

TEST_CASE("Base and frame cards split the tile into the parts below and above content",
          "[core][sakura][card]") {
    Scene scene;
    const auto ctx = scene.context();
    Sakura::Card base;
    Sakura::Card frame;
    Host baseHost(base);
    Host frameHost(frame);
    base.update(TileCard(Sakura::Card::Layer::Base));
    frame.update(TileCard(Sakura::Card::Layer::Frame));
    baseHost.place(ctx, kScreen);
    frameHost.place(ctx, kScreen);

    const auto below = Boxes(base);
    REQUIRE(below.size() == 2);
    CHECK(below[0]->softness == 6.0f);
    CHECK(below[1]->instances[0].backgroundColor == ctx.color("tile_bg"));
    CHECK(below[1]->borderWidth == 0.0f);

    const auto above = Boxes(frame);
    REQUIRE(above.size() == 2);
    CHECK(above[0]->borderWidth == 1.5f);
    CHECK(above[0]->borderColor == ctx.color("tile_border"));
    CHECK_FALSE(above[0]->topLit);
    CHECK(above[1]->topLit);
    for (const auto* box : above) {
        CHECK(box->instances[0].backgroundColor.a == 0.0f);
    }
}

TEST_CASE("Card opacity fades every color and hides the card at zero", "[core][sakura][card]") {
    Scene scene;
    const auto ctx = scene.context();
    Sakura::Card card;
    Host host(card);
    auto config = TileCard();
    config.opacity = 0.5f;
    card.update(config);
    host.place(ctx, kScreen);
    auto boxes = Boxes(card);
    REQUIRE(boxes.size() == 3);
    CHECK(boxes[1]->instances[0].backgroundColor.a == ctx.color("tile_bg").a * 0.5f);
    CHECK(boxes[1]->borderColor.a == ctx.color("tile_border").a * 0.5f);

    config.opacity = 0.0f;
    card.update(config);
    host.place(ctx, kScreen);
    for (const auto* box : Boxes(card)) {
        CHECK_FALSE(box->instances[0].visible);
    }
}

TEST_CASE("Cards settle and repaint without restructuring when their rects or layer change",
          "[core][sakura][card][retention]") {
    Scene scene;
    Sakura::Card card;
    Host host(card);
    Frames frames(host, kScreen, scene.context());
    std::vector<Rect> rects = {kCard};
    auto layer = Sakura::Card::Layer::Full;
    const auto update = [&] {
        auto config = TileCard(layer);
        config.rects = rects;
        card.update(std::move(config));
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    const U64 restructures = frames.restructures;

    rects.push_back({240.0f, 30.0f, 200.0f, 100.0f});
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));

    layer = Sakura::Card::Layer::Frame;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));

    layer = Sakura::Card::Layer::Base;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.restructures == restructures);
}
