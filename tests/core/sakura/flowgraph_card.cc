#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/button.hh>
#include <jetstream/render/sakura/components/flowgraph_card.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/components/polyline.hh>
#include <jetstream/render/sakura/components/text.hh>

#include "component_harness.hh"
#include "harness.hh"
#include "scene.hh"

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

using namespace Jetstream;
using namespace SakuraTest;

TEST_CASE("Flowgraph cards draw the diagram, open on release anywhere on the card, and settle", "[core][sakura][flowgraph-card]") {
    Scene scene;
    U64 opened = 0;
    Sakura::FlowgraphCard card;
    Place place(card, {0.0f, 0.0f, 260.0f, 320.0f});
    Frames frames(place, kScreen, scene.context());
    const auto update = [&] {
        card.update({
            .diagram = "tone[Tone] --> add[Add] --> spectrum[Spectrum]\nnoise[Noise] --> add",
            .category = "Basics",
            .title = "First Steps",
            .description = "Build a spectrum analyzer.",
            .onOpen = [&] { ++opened; },
        });
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));

    std::vector<std::string> labels;
    for (const auto* label : Find<Sakura::Label>(card)) {
        for (const auto& instance : label->config().instances) {
            if (instance.str == "Tone" || instance.str == "Noise" || instance.str == "Add" ||
                instance.str == "Spectrum") {
                labels.push_back(instance.str);
                CHECK(instance.rect.right() <= 260.0f);
            }
        }
    }
    CHECK(labels.size() == 4);

    CHECK(Find<Sakura::Button>(card).empty());
    place.send(Mouse(MouseEventType::Click, 130.0f, 250.0f));
    CHECK(opened == 0);
    place.send(Mouse(MouseEventType::Release, 130.0f, 250.0f));
    CHECK(opened == 1);
    place.send(Mouse(MouseEventType::Click, 130.0f, 250.0f));
    place.send(Mouse(MouseEventType::Release, 600.0f, 250.0f));
    CHECK(opened == 1);
}

TEST_CASE("Flowgraph cards drop a held press when they are rebound to another flowgraph",
          "[core][sakura][flowgraph-card]") {
    Scene scene;
    std::vector<std::string> opened;
    Sakura::FlowgraphCard card;
    Place place(card, {0.0f, 0.0f, 260.0f, 320.0f});
    Frames frames(place, kScreen, scene.context());
    frames.verify = false;
    std::string key = "alpha";
    const auto update = [&] {
        card.update({
            .id = key,
            .diagram = "tone[Tone] --> add[Add]",
            .category = "Basics",
            .title = key,
            .description = "Build a spectrum analyzer.",
            .onOpen = [&, bound = key] { opened.push_back(bound); },
        });
    };
    REQUIRE(frames.settle(update));

    place.send(Mouse(MouseEventType::Click, 130.0f, 250.0f));
    key = "beta";
    frames.run(update);
    place.send(Mouse(MouseEventType::Release, 130.0f, 250.0f));
    CHECK(opened.empty());

    place.send(Mouse(MouseEventType::Click, 130.0f, 250.0f));
    frames.run(update);
    place.send(Mouse(MouseEventType::Release, 130.0f, 250.0f));
    CHECK(opened == std::vector<std::string>{"beta"});
}

TEST_CASE("Flowgraph cards resolve their accent through the palette", "[core][sakura][flowgraph-card]") {
    Scene scene;
    const auto ctx = scene.context();
    Sakura::FlowgraphCard card;
    Host host(card);
    const auto accent = [&](std::string key) {
        Sakura::FlowgraphCard::Config config{.diagram = "a[A] --> b[B]", .title = "Card"};
        if (!key.empty()) {
            config.accentColorKey = std::move(key);
        }
        card.update(std::move(config));
        host.place(ctx, {0.0f, 0.0f, 260.0f, 320.0f});
        const auto labels = Find<Sakura::Label>(card);
        for (const auto* label : labels) {
            for (const auto& instance : label->config().instances) {
                if (instance.str == "A") {
                    return instance.color;
                }
            }
        }
        return ColorRGBA<F32>{};
    };
    CHECK(accent("") == ctx.color("cyber_blue"));
    CHECK(accent("gallery_purple") == ctx.color("gallery_purple"));
    CHECK(ctx.color("gallery_purple") != ctx.color("cyber_blue"));
}

TEST_CASE("Flowgraph card diagrams cap long chains and never wire a dropped node", "[core][sakura][flowgraph-card]") {
    Scene scene;
    Sakura::FlowgraphCard card;
    std::string chain = "n0[N0]";
    for (U64 index = 1; index < 24; ++index) {
        chain += " --> n" + std::to_string(index) + "[N" + std::to_string(index) + "]";
    }
    chain += "\nn0 --> n20[N20]\nn3 --> n5";
    card.update({.diagram = chain, .title = "Chain"});
    Host host(card);
    host.place(scene.context(), {0.0f, 0.0f, 600.0f, 320.0f});

    std::vector<std::string> labels;
    for (const auto* label : Find<Sakura::Label>(card)) {
        for (const auto& instance : label->config().instances) {
            if (!instance.str.empty() && instance.str.front() == 'N') {
                labels.push_back(instance.str);
            }
        }
    }
    CHECK(labels.size() == 16);
    CHECK(std::find(labels.begin(), labels.end(), "N15") != labels.end());
    CHECK(std::find(labels.begin(), labels.end(), "N16") == labels.end());
    U64 wires = 0;
    for (const auto* line : Find<Sakura::Polyline>(card)) {
        wires += line->config().instances.size();
    }
    CHECK(wires == 16);
}

TEST_CASE("Flowgraph card wires fit their budget for every edge a capped diagram keeps", "[core][sakura][flowgraph-card]") {
    Scene scene;
    Sakura::FlowgraphCard card;
    std::string fan;
    for (U64 from = 0; from < 8; ++from) {
        for (U64 to = 8; to < 16; ++to) {
            fan += "n" + std::to_string(from) + " --> n" + std::to_string(to) + "\n";
        }
    }
    card.update({.diagram = fan, .title = "Fan"});
    Host host(card);
    host.place(scene.context(), {0.0f, 0.0f, 600.0f, 320.0f});

    const auto lines = Find<Sakura::Polyline>(card);
    REQUIRE(lines.size() == 1);
    const auto& config = lines.front()->config();
    CHECK(config.instances.size() == 32);
    U64 vertices = 0;
    for (const auto& instance : config.instances) {
        vertices += 4 * instance.points.size();
    }
    CHECK(vertices <= config.capacity);
}

TEST_CASE("Flowgraph cards ease a hover lift in and out, ignore cancelled presses, and repaint on content changes",
          "[core][sakura][flowgraph-card][retention]") {
    Scene scene;
    U64 opened = 0;
    Sakura::FlowgraphCard card;
    Place place(card, {0.0f, 0.0f, 260.0f, 320.0f});
    Frames frames(place, kScreen, scene.context());
    frames.timeStep = 1.0 / 60.0;
    std::string title = "First Steps";
    std::string diagram = "tone[Tone] --> add[Add]";
    const auto update = [&] {
        card.update({
            .diagram = diagram,
            .category = "Basics",
            .title = title,
            .description = "Build a spectrum analyzer.",
            .onOpen = [&] { ++opened; },
        });
    };
    const auto send = [&](const MouseEvent& event) {
        return place.send(event);
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CHECK(QuietMoves(frames, update, {{400.0f, 400.0f}, {500.0f, 400.0f}}, send));
    const auto top = [&] {
        return Direct<Sakura::Box>(card).front()->config().instances.at(0).rect.y;
    };
    const auto accentOpacity = [&] {
        F32 opacity = 0.0f;
        for (auto* text : Direct<Sakura::Text>(card)) {
            if (text->config().str == title && !text->config().colorKey.empty()) {
                opacity = text->config().opacity;
            }
        }
        return opacity;
    };
    REQUIRE(top() == 0.0f);
    const U64 rebuilds = frames.rebuilds;

    place.send(Mouse(MouseEventType::Move, 130.0f, 250.0f));
    U64 animated = 0;
    for (U64 frame = 0; frame < 30; ++frame) {
        const auto stats = frames.run(update);
        animated += stats.repainted ? 1 : 0;
        CHECK(Kept(stats));
        if (frame == 6) {
            CHECK(top() < 0.0f);
            CHECK(top() > -3.0f);
            CHECK(accentOpacity() > 0.0f);
            CHECK(accentOpacity() < 1.0f);
        }
    }
    CHECK(animated > 3);
    CHECK(top() == -3.0f);
    CHECK(accentOpacity() == 1.0f);
    CHECK(frames.run(update).clean());
    CHECK(QuietMoves(frames, update, {{131.0f, 250.0f}, {140.0f, 260.0f}}, send));

    place.send(Mouse(MouseEventType::Move, 400.0f, 400.0f));
    REQUIRE(frames.settle(update, 40));
    CHECK(top() == 0.0f);
    CHECK(accentOpacity() == 0.0f);
    CHECK(frames.rebuilds == rebuilds);

    place.send(Mouse(MouseEventType::Move, 130.0f, 250.0f));
    REQUIRE(frames.settle(update, 40));
    place.send(Mouse(MouseEventType::Leave, 130.0f, 250.0f));
    REQUIRE(frames.settle(update, 40));

    CHECK(place.send(Mouse(MouseEventType::Click, 130.0f, 250.0f)));
    place.send(Mouse(MouseEventType::Cancel, 130.0f, 250.0f));
    CHECK_FALSE(place.send(Mouse(MouseEventType::Release, 130.0f, 250.0f)));
    CHECK(opened == 0);
    REQUIRE(frames.settle(update));

    title = "Second Steps";
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    diagram = "tone[Tone] --> add[Add] --> sink[Sink]";
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.restructures == 0);
}

TEST_CASE("Flowgraph cards keep a 16:9 preview, upper-case the category, scale paddings and draw a fixed dot grid",
          "[core][sakura][flowgraph-card]") {
    const Scene scene;
    Sakura::FlowgraphCard card;
    card.update({.diagram = "a[A] --> b[B]", .category = "Basics", .title = "Card"});
    Host host(card);
    auto doubled = scene.ctx;
    doubled.pixelRatio = 2.0f;
    const auto category = [&] {
        for (auto* text : Find<Sakura::Text>(card)) {
            if (text->config().str == "BASICS") {
                return Peek::Frame(*text);
            }
        }
        return Rect{};
    };
    host.place(scene.ctx, {0.0f, 0.0f, 260.0f, 400.0f});
    for (auto* text : Find<Sakura::Text>(card)) {
        const auto& config = text->config();
        if (config.str == "BASICS") {
            CHECK(config.font == Sakura::Text::Font::SansBold);
        } else if (config.str == "Card") {
            CHECK(config.font == Sakura::Text::Font::DisplaySemiBold);
        }
    }
    CHECK(category().x == 18.0f);
    CHECK(category().y == std::round(260.0f * 9.0f / 16.0f) + 18.0f);

    host.place(doubled, {0.0f, 0.0f, 520.0f, 800.0f});
    CHECK(category().x == 36.0f);
    CHECK(category().y == std::round(520.0f * 9.0f / 16.0f) + 36.0f);

    host.place(scene.ctx, {0.0f, 0.0f, 10.0f, 400.0f});
    for (const auto* label : Find<Sakura::Label>(card)) {
        if (label->config().maxCharacters != 24) {
            continue;
        }
        for (const auto& instance : label->config().instances) {
            CHECK((!instance.visible || !instance.rect.empty()));
            CHECK(instance.fontSize >= 0.0f);
        }
    }
    for (const auto* box : Find<Sakura::Box>(card)) {
        CHECK(box->config().cornerRadius >= 0.0f);
    }

    const auto dotGrid = [&]() -> U64 {
        for (const auto* box : Find<Sakura::Box>(card)) {
            if (box->config().capacity > 100) {
                CHECK(box->config().instances.size() == box->config().capacity);
                return box->config().capacity;
            }
        }
        return 0;
    };
    host.place(scene.ctx, {0.0f, 0.0f, 260.0f, 400.0f});
    const U64 grid = dotGrid();
    CHECK(grid > 0);
    auto narrow = scene.ctx;
    narrow.framebufferSize = {32, 600};
    host.place(narrow, {0.0f, 0.0f, 260.0f, 400.0f});
    CHECK(dotGrid() == grid);
    host.place(doubled, {0.0f, 0.0f, 520.0f, 800.0f});
    CHECK(dotGrid() == grid);
}

TEST_CASE("Flowgraph card diagrams center merges between their inputs and fan outputs around them",
          "[core][sakura][flowgraph-card]") {
    const Scene scene;
    Sakura::FlowgraphCard card;
    card.update({.diagram = "tone[Tone] --> add[Add] --> spectrum[Spectrum]\nnoise[Noise] --> add\nadd --> iq[IQ Plot]",
                 .category = "Basics",
                 .title = "First Steps"});
    Host host(card);
    host.place(scene.ctx, {0.0f, 0.0f, 320.0f, 400.0f});

    std::map<std::string, Rect> nodes;
    for (const auto* label : Find<Sakura::Label>(card)) {
        if (label->config().maxCharacters != 24) {
            continue;
        }
        for (const auto& instance : label->config().instances) {
            nodes[instance.str] = instance.rect;
        }
    }
    REQUIRE(nodes.size() == 5);
    const auto center = [&](const std::string& name) {
        return nodes.at(name).y + nodes.at(name).height * 0.5f;
    };
    CHECK(nodes.at("Tone").x == nodes.at("Noise").x);
    CHECK(nodes.at("Add").x > nodes.at("Tone").right());
    CHECK(nodes.at("Spectrum").x > nodes.at("Add").right());
    CHECK(center("Add") == Catch::Approx((center("Tone") + center("Noise")) * 0.5f).margin(0.01f));
    CHECK(center("Spectrum") == Catch::Approx(center("Tone")).margin(0.01f));
    CHECK(center("IQ Plot") == Catch::Approx(center("Noise")).margin(0.01f));
    const F32 middle = (nodes.at("Tone").x + nodes.at("Spectrum").right()) * 0.5f;
    CHECK(middle == Catch::Approx(160.0f).margin(0.5f));
}

TEST_CASE("Flowgraph card wires coarsen their curves so long skip edges stay inside the budget",
          "[core][sakura][flowgraph-card]") {
    Scene scene;
    Sakura::FlowgraphCard card;
    std::string diagram = "n0";
    for (U64 index = 1; index < 16; ++index) {
        diagram += " --> n" + std::to_string(index);
    }
    for (U64 to = 15; to >= 2; --to) {
        diagram += "\nn0 --> n" + std::to_string(to);
    }
    for (U64 to = 15; to >= 13; --to) {
        diagram += "\nn1 --> n" + std::to_string(to);
    }
    card.update({.diagram = diagram, .title = "Skips"});
    Host host(card);
    host.place(scene.context(), {0.0f, 0.0f, 600.0f, 320.0f});

    const auto lines = Find<Sakura::Polyline>(card);
    REQUIRE(lines.size() == 1);
    const auto& config = lines.front()->config();
    CHECK(config.instances.size() == 32);
    U64 vertices = 0;
    for (const auto& instance : config.instances) {
        CHECK(instance.points.size() >= 2);
        vertices += 4 * instance.points.size();
    }
    CHECK(vertices <= config.capacity);
}

TEST_CASE("Flowgraph card node labels ellipsize to the node width on whole characters",
          "[core][sakura][flowgraph-card]") {
    Scene scene;
    Sakura::FlowgraphCard card;
    const std::string ascii = "Polyphase Channelizer Bank";
    const std::string accented = "R\xC3\xA9sum\xC3\xA9 R\xC3\xA9sum\xC3\xA9 R\xC3\xA9sum\xC3\xA9";
    card.update({.diagram = "a[" + ascii + "] --> b[" + accented + "] --> c[FM]", .title = "Labels"});
    Host host(card);
    host.place(scene.context(), {0.0f, 0.0f, 320.0f, 400.0f});

    std::vector<Sakura::Label::Instance> nodes;
    for (const auto* label : Find<Sakura::Label>(card)) {
        if (label->config().maxCharacters != 24) {
            continue;
        }
        for (const auto& instance : label->config().instances) {
            nodes.push_back(instance);
        }
    }
    REQUIRE(nodes.size() == 3);
    for (const auto& node : nodes) {
        if (node.str == "FM") {
            continue;
        }
        REQUIRE(node.str.size() > 3);
        CHECK(node.str.ends_with("..."));
        const std::string kept = node.str.substr(0, node.str.size() - 3);
        const std::string& source = node.str.starts_with("Poly") ? ascii : accented;
        REQUIRE(kept.size() < source.size());
        CHECK(source.starts_with(kept));
        CHECK((static_cast<unsigned char>(source[kept.size()]) & 0xC0) != 0x80);
    }
}

TEST_CASE("Height constrained flowgraph cards clip their text to the card", "[core][sakura][flowgraph-card]") {
    Scene scene;
    Sakura::FlowgraphCard card;
    const Rect bounds = {0.0f, 0.0f, 260.0f, 170.0f};
    card.update({
        .diagram = "a[A] --> b[B]",
        .category = "Basics",
        .title = "First Steps",
        .description = "Build a spectrum analyzer that keeps going on and on across several wrapped lines.",
    });
    Place place(card, bounds);
    place.place(scene.context(), kScreen);

    U64 checked = 0;
    for (const auto* text : Find<Sakura::Text>(card)) {
        CAPTURE(text->config().str);
        const Rect clip = Peek::Clip(*text);
        CHECK(clip.x >= bounds.x);
        CHECK(clip.y >= bounds.y);
        CHECK(clip.right() <= bounds.right());
        CHECK(clip.bottom() <= bounds.bottom());
        ++checked;
    }
    CHECK(checked >= 3);
}
