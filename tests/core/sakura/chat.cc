#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/chat.hh>
#include <jetstream/render/sakura/components/chat_composer.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/components/polyline.hh>
#include <jetstream/render/sakura/components/text_grid.hh>
#include <jetstream/render/sakura/components/text_markdown.hh>
#include <jetstream/render/sakura/components/text_view.hh>

#include "render/sakura/detail/chat_transcript.hh"

#include "component_harness.hh"
#include "fonts.hh"
#include "harness.hh"
#include "scene.hh"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

using namespace Jetstream;
using namespace SakuraTest;

namespace {

using Chat = Sakura::Chat;
using Transcript = Sakura::ChatTranscript;
using Catch::Approx;

constexpr Rect kRect = {10.0f, 10.0f, 400.0f, 300.0f};
constexpr Rect kChatRect = {0.0f, 0.0f, 400.0f, 300.0f};
constexpr F32 kSingleBubble = 15.0f * 1.15f + 18.0f;
constexpr F32 kToggleHeight = 15.0f + 8.0f;

Chat::Message UserMessage(std::string text) {
    return {.role = Chat::Role::User, .text = std::move(text)};
}

Chat::Message AssistantMessage(std::vector<Chat::Part> parts) {
    return {.role = Chat::Role::Assistant, .parts = std::move(parts)};
}

Chat::Part MarkdownPart(U64 key, U64 revision, std::string text) {
    return {.kind = Chat::PartKind::Markdown, .key = key, .revision = revision, .text = std::move(text)};
}

Chat::Part DisclosurePart(U64 key,
                          U64 revision,
                          std::string label,
                          std::string summary = {},
                          std::string text = {},
                          bool busy = false,
                          bool expandable = true) {
    return {
        .kind = Chat::PartKind::Disclosure,
        .key = key,
        .revision = revision,
        .text = std::move(text),
        .label = std::move(label),
        .summary = std::move(summary),
        .busy = busy,
        .expandable = expandable,
    };
}

Chat::Snapshot Conversation(U64 count) {
    Chat::Snapshot snapshot{.generation = 1};
    for (U64 i = 0; i < count; ++i) {
        if (i % 2 == 0) {
            snapshot.messages.push_back(UserMessage("Question " + std::to_string(i)));
        } else {
            snapshot.messages.push_back(AssistantMessage({MarkdownPart(0, 1, "Answer " + std::to_string(i))}));
        }
    }
    return snapshot;
}

Chat::Snapshot Questions(U64 count) {
    Chat::Snapshot snapshot{.generation = 1};
    for (U64 i = 0; i < count; ++i) {
        snapshot.messages.push_back(UserMessage("Message " + std::to_string(i)));
    }
    return snapshot;
}

template<typename T>
std::vector<Rect> Visible(Sakura::Component& root) {
    std::vector<Rect> rects;
    for (auto* view : Direct<T>(root)) {
        rects.push_back(Peek::Frame(*view));
    }
    std::sort(rects.begin(), rects.end(), [](const Rect& a, const Rect& b) {
        return a.y < b.y;
    });
    return rects;
}

template<typename T>
U64 Pooled(const Sakura::Component& root) {
    U64 count = 0;
    for (const auto* child : Peek::Children(root)) {
        count += dynamic_cast<const T*>(child) ? 1 : 0;
    }
    return count;
}

const Sakura::Box& Bubbles(Sakura::Component& transcript) {
    return *Direct<Sakura::Box>(transcript).at(0);
}

const Sakura::Box& Highlights(Sakura::Component& transcript) {
    return *Direct<Sakura::Box>(transcript).at(1);
}

const Sakura::Box& Shimmer(Sakura::Component& transcript) {
    return *Direct<Sakura::Box>(transcript).at(2);
}

const Sakura::Label& Toggles(Sakura::Component& transcript) {
    return *Direct<Sakura::Label>(transcript).at(0);
}

const Sakura::Polyline& Carets(Sakura::Component& transcript) {
    return *Direct<Sakura::Polyline>(transcript).at(0);
}

Extent2D<F32> TogglePoint(Sakura::Component& transcript) {
    const auto rect = Toggles(transcript).config().instances.at(0).rect;
    return {rect.x - 5.0f, rect.y + rect.height * 0.5f};
}

F32 Bottom(Sakura::Component& transcript) {
    F32 bottom = 0.0f;
    for (const auto& rect : Visible<Sakura::TextView>(transcript)) {
        bottom = std::max(bottom, rect.bottom());
    }
    return bottom;
}

bool PointsRight(const Sakura::Polyline::Instance& caret) {
    return caret.points.size() == 3 && caret.points[0].x == caret.points[2].x &&
           caret.points[1].x > caret.points[0].x;
}

bool PointsDown(const Sakura::Polyline::Instance& caret) {
    return caret.points.size() == 3 && caret.points[0].y == caret.points[1].y &&
           caret.points[2].y > caret.points[0].y;
}

bool Expanded(Sakura::Component& transcript) {
    return PointsDown(Carets(transcript).config().instances.at(0));
}

struct Twin {
    Transcript live;
    Transcript reference;
    Place livePlace{live, kRect};
    Place referencePlace{reference, kRect};

    void update(const Transcript::Config& config) {
        live.update(config);
        reference.update(config);
    }

    bool send(const MouseEvent& event) {
        const bool handled = livePlace.send(event);
        referencePlace.send(event);
        return handled;
    }

    void click(Extent2D<F32> point) {
        send(Mouse(MouseEventType::Click, point.x, point.y));
        send(Mouse(MouseEventType::Release, point.x, point.y));
    }
};

}  // namespace

TEST_CASE("Chats measure their composer column before the first layout", "[core][sakura][chat][sizing]") {
    Scene scene;
    Sakura::Chat chat;
    chat.update({.id = "chat"});
    Host host(chat);
    const auto measured = host.measureOf(scene.ctx, {kChatRect.width, kChatRect.height});

    Sakura::ChatComposer composer;
    composer.update({.id = "chat:composer", .style = Chat::ComposerStyle{.minLines = 2.0f}});
    Host composerHost(composer);
    auto surface = scene.ctx;
    surface.backdropColorKey = "background";
    const auto expected = composerHost.measureOf(surface, {kChatRect.width - 16.0f, kInfinite});
    REQUIRE(expected.y > 0.0f);

    CHECK(measured.x == kChatRect.width);
    CHECK(measured.y == Approx(std::min(kChatRect.height, expected.y + 16.0f)));

    host.place(scene.ctx, kChatRect);
    CHECK(Peek::Frame(*Find<Sakura::ChatComposer>(chat).at(0)).height == Approx(expected.y));
}

TEST_CASE("Chats place the transcript above a centered composer column", "[core][sakura][chat]") {
    Scene scene;
    Sakura::Chat chat;
    Host host(chat);
    chat.update({.id = "chat", .snapshot = Conversation(2)});
    host.place(scene.ctx, kChatRect);

    const auto composer = Peek::Frame(*Find<Sakura::ChatComposer>(chat).at(0));
    const auto transcript = Peek::Frame(*Find<Transcript>(chat).at(0));
    CHECK(composer.x == 8.0f);
    CHECK(composer.width == kChatRect.width - 16.0f);
    CHECK(composer.bottom() == Approx(kChatRect.bottom() - 8.0f));
    CHECK(transcript.x == 8.0f);
    CHECK(transcript.y == 8.0f);
    CHECK(transcript.width == composer.width);
    CHECK(transcript.bottom() == Approx(composer.y - 8.0f));
    CHECK(Direct<Sakura::Box>(chat).at(0)->config().instances.at(0).rect == kChatRect);

    const Rect wide = {0.0f, 0.0f, 800.0f, 400.0f};
    chat.update({.id = "chat", .snapshot = Conversation(2), .maxContentWidth = 300.0f});
    host.place(scene.ctx, wide);
    const auto column = Peek::Frame(*Find<Sakura::ChatComposer>(chat).at(0));
    const auto content = Peek::Frame(*Find<Transcript>(chat).at(0));
    CHECK(column.x == 250.0f);
    CHECK(column.width == 300.0f);
    CHECK(content.x == 250.0f);
    CHECK(content.width == 300.0f);
    CHECK(content.bottom() == Approx(column.y - 8.0f));
}

TEST_CASE("Chats scale their surface padding and column width with the pixel ratio", "[core][sakura][chat]") {
    Scene scene;
    Sakura::Chat chat;
    Host host(chat);
    auto ctx = scene.ctx;
    ctx.pixelRatio = 2.0f;
    const Rect bounds = {0.0f, 0.0f, 800.0f, 600.0f};

    chat.update({.id = "chat", .snapshot = Conversation(2)});
    host.place(ctx, bounds);
    auto composer = Peek::Frame(*Find<Sakura::ChatComposer>(chat).at(0));
    auto transcript = Peek::Frame(*Find<Transcript>(chat).at(0));
    CHECK(composer.x == 16.0f);
    CHECK(composer.width == bounds.width - 32.0f);
    CHECK(composer.bottom() == Approx(bounds.bottom() - 16.0f));
    CHECK(transcript.y == 16.0f);
    CHECK(transcript.bottom() == Approx(composer.y - 16.0f));

    chat.update({.id = "chat", .snapshot = Conversation(2), .maxContentWidth = 300.0f});
    host.place(ctx, bounds);
    composer = Peek::Frame(*Find<Sakura::ChatComposer>(chat).at(0));
    CHECK(composer.width == 600.0f);
    CHECK(composer.x == 100.0f);

    chat.update({.id = "chat", .snapshot = Conversation(2), .fontSize = 30.0f});
    host.place(scene.ctx, bounds);
    transcript = Peek::Frame(*Find<Transcript>(chat).at(0));
    CHECK(transcript.x == 16.0f);
    CHECK(transcript.y == 16.0f);
}

TEST_CASE("Chats route composer controls through callbacks", "[core][sakura][chat]") {
    static_assert(std::is_base_of_v<Sakura::Component, Sakura::Chat>);
    static_assert(!std::is_move_constructible_v<Sakura::Chat>);
    Scene scene;
    Sakura::Chat chat;
    Host host(chat);
    U64 cancellations = 0;
    Chat::Config config{
        .id = "chat-content",
        .snapshot = {.streaming = true},
        .onCancel = [&] { ++cancellations; },
    };
    chat.update(config);
    host.place(scene.ctx, kChatRect);
    const Extent2D<F32> stop = {kChatRect.right() - 40.0f, kChatRect.bottom() - 20.0f};
    host.click(stop.x, stop.y);
    REQUIRE(cancellations == 1);

    config.snapshot.streaming = false;
    config.snapshot.messages.push_back(AssistantMessage({
        MarkdownPart(0, 1, "Partial response"),
        MarkdownPart(1, 1, "> [!ERROR]\n> HTTP 429: Retry later"),
    }));
    chat.update(config);
    host.place(scene.ctx, kChatRect);
    host.click(stop.x, stop.y);
    CHECK(cancellations == 1);
}

TEST_CASE("Chats remeasure content when their identity changes with reused revisions", "[core][sakura][chat]") {
    Scene scene;
    Sakura::Chat chat;
    Host host(chat);
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
        config.snapshot.messages.push_back(AssistantMessage({part}));
        chat.update(std::move(config));
        host.place(scene.ctx, kChatRect);
    };
    const auto scroll = [&] {
        return host.send(Wheel(391.0f, 20.0f, 1.0f));
    };
    show("short-chat", 1);
    REQUIRE_FALSE(scroll());
    show("long-chat", 40);
    CHECK(scroll());
    show("another-short-chat", 1);
    CHECK_FALSE(scroll());
}

TEST_CASE("Chats settle under a persistent snapshot and fade the transcript with a cover",
          "[core][sakura][chat][retention]") {
    Scene scene;
    Sakura::Chat chat;
    Place place(chat, kChatRect);
    Frames frames(place, kScreen, scene.context());
    const Chat::Snapshot snapshot = Conversation(4);
    F32 opacity = 1.0f;
    const auto update = [&] {
        chat.update({.id = "chat", .snapshot = snapshot, .transcriptOpacity = opacity, .surfaceColorKey = "agent_panel"});
    };
    const auto cover = [&] {
        return Direct<Sakura::Box>(chat).at(1)->config().instances.at(0);
    };
    REQUIRE(frames.settle(update));
    const Transcript* transcript = Find<Transcript>(chat).at(0);
    CHECK(Retains(frames, update));
    CHECK(QuietMoves(frames, update, {{60.0f, 30.0f}, {200.0f, 60.0f}, {380.0f, 20.0f}}, [&](const MouseEvent& event) {
        return place.send(event);
    }));
    CHECK_FALSE(cover().visible);
    const U64 restructures = frames.restructures;

    opacity = 0.25f;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    const auto panel = frames.ctx.palette.get().at("agent_panel");
    CHECK(cover().visible);
    CHECK(cover().rect == Peek::Frame(*Find<Transcript>(chat).at(0)));
    CHECK(cover().backgroundColor.r == panel.r);
    CHECK(cover().backgroundColor.a == Approx(panel.a * 0.75f));

    opacity = 0.0f;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK_FALSE(cover().visible);
    CHECK_FALSE(transcript->visible());

    opacity = 1.0f;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK_FALSE(cover().visible);
    CHECK(transcript->visible());
    CHECK(Retains(frames, update));
    CHECK(frames.restructures == restructures);
}

TEST_CASE("Chat clear requests make the transcript stick to the tail again", "[core][sakura][chat]") {
    Scene scene;
    Sakura::Chat chat;
    Host host(chat);
    Chat::Config config{.id = "chat", .snapshot = Questions(30)};
    chat.update(config);
    host.place(scene.ctx, kChatRect);
    auto& transcript = *Find<Transcript>(chat).at(0);
    const F32 bottom = Peek::Frame(transcript).bottom();
    CHECK(Bottom(transcript) == Approx(bottom).margin(1.0f));

    REQUIRE(host.send(Wheel(200.0f, 100.0f, 1.0f)));
    host.place(scene.ctx, kChatRect);
    CHECK(Bottom(transcript) > bottom + 40.0f);

    config.snapshot.messages.push_back(UserMessage("Late message"));
    chat.update(config);
    host.place(scene.ctx, kChatRect);
    CHECK(Bottom(transcript) > bottom + 40.0f);

    ++config.clearRequest;
    chat.update(config);
    host.place(scene.ctx, kChatRect);
    CHECK(Bottom(transcript) == Approx(bottom).margin(1.0f));
}

TEST_CASE("Chat transcripts right-align user bubbles within the bubble width ratio", "[core][sakura][chat]") {
    Scene scene;
    Transcript transcript;
    Place place(transcript, kRect);
    Frames frames(place, kScreen, scene.context());
    std::string words;
    for (int i = 0; i < 40; ++i) {
        words += "word ";
    }
    Chat::Snapshot snapshot{.generation = 1};
    snapshot.messages.push_back(UserMessage("Hello there"));
    snapshot.messages.push_back(UserMessage(words));
    const auto update = [&] {
        transcript.update({.id = "chat", .snapshot = snapshot});
    };
    REQUIRE(frames.settle(update));

    const F32 content = kRect.width - 14.0f;
    const auto views = Visible<Sakura::TextView>(transcript);
    REQUIRE(views.size() == 2);
    for (const auto& view : views) {
        CHECK(view.right() == Approx(kRect.right() - 14.0f));
        CHECK(view.width <= Approx(content * 0.72f));
    }
    CHECK(views[0].y == kRect.y);
    CHECK(views[0].height == Approx(kSingleBubble));
    CHECK(views[0].width < views[1].width);
    CHECK(views[1].height > kSingleBubble + 10.0f);

    const auto& bubbles = Bubbles(transcript).config();
    const auto& palette = frames.ctx.palette.get();
    REQUIRE(bubbles.instances.size() == 2);
    CHECK(bubbles.instances[0].rect == views[0]);
    CHECK(bubbles.instances[1].rect == views[1]);
    CHECK(bubbles.instances[0].backgroundColor == palette.at("chat_user_bubble"));
    CHECK(bubbles.borderColor == palette.at("chat_user_bubble_outline"));
    CHECK(bubbles.cornerRadius == Approx(15.0f * 1.15f * 0.5f + 9.0f));
    CHECK(bubbles.borderWidth == 1.0f);
    CHECK(bubbles.strokeFringe);
    CHECK(bubbles.capacity == Transcript::UserSlots);
}

TEST_CASE("Chat transcripts keep turn, bubble and part gaps", "[core][sakura][chat]") {
    Scene scene;
    Transcript transcript;
    Place place(transcript, kRect);
    Frames frames(place, kScreen, scene.context());
    Chat::Snapshot snapshot{.generation = 1};
    snapshot.messages.push_back(UserMessage("First"));
    snapshot.messages.push_back(UserMessage("Second"));
    snapshot.messages.push_back(AssistantMessage({MarkdownPart(0, 1, "One"), MarkdownPart(1, 1, "Two")}));
    snapshot.messages.push_back(UserMessage("Third"));
    const auto update = [&] {
        transcript.update({.id = "chat", .snapshot = snapshot});
    };
    REQUIRE(frames.settle(update));

    const auto users = Visible<Sakura::TextView>(transcript);
    const auto markdown = Visible<Sakura::TextMarkdown>(transcript);
    REQUIRE(users.size() == 3);
    REQUIRE(markdown.size() == 2);
    CHECK(users[1].y == Approx(users[0].bottom() + 10.0f));
    CHECK(markdown[0].y == Approx(users[1].bottom() + 16.0f));
    CHECK(markdown[1].y == Approx(markdown[0].bottom() + 6.0f));
    CHECK(users[2].y == Approx(markdown[1].bottom() + 16.0f));
    CHECK(markdown[0].x == kRect.x);
    CHECK(markdown[0].width == Approx(kRect.width - 14.0f));
}

TEST_CASE("Chat transcripts scale bubbles, gaps and toggles with the pixel ratio", "[core][sakura][chat]") {
    Scene scene;
    Transcript transcript;
    Host host(transcript);
    auto ctx = scene.context();
    ctx.pixelRatio = 2.0f;
    Chat::Snapshot snapshot{.generation = 1};
    snapshot.messages.push_back(UserMessage("First"));
    snapshot.messages.push_back(UserMessage("Second"));
    snapshot.messages.push_back(AssistantMessage({DisclosurePart(2, 1, "Tool"), MarkdownPart(3, 1, "Done")}));
    transcript.update({.id = "chat", .snapshot = snapshot});
    const Rect bounds = {20.0f, 20.0f, 800.0f, 560.0f};
    host.place(ctx, bounds);

    const auto users = Visible<Sakura::TextView>(transcript);
    REQUIRE(users.size() == 2);
    CHECK(users[0].right() == Approx(bounds.right() - 28.0f));
    CHECK(users[0].height == Approx(30.0f * 1.15f + 36.0f));
    CHECK(users[1].y == Approx(users[0].bottom() + 20.0f));

    const auto& bubbles = Bubbles(transcript).config();
    CHECK(bubbles.cornerRadius == Approx(30.0f * 1.15f * 0.5f + 18.0f));
    CHECK(bubbles.borderWidth == 2.0f);

    const auto label = Toggles(transcript).config().instances.at(0);
    CHECK(label.fontSize == 30.0f);
    CHECK(label.rect.x == Approx(bounds.x + 30.0f + 16.0f));
    CHECK(label.rect.y == Approx(users[1].bottom() + 32.0f));
    CHECK(label.rect.height == Approx(30.0f + 16.0f));

    const auto caret = Carets(transcript).config().instances.at(0);
    REQUIRE(PointsRight(caret));
    CHECK(caret.points[1].x - caret.points[0].x == Approx(8.0f));
    CHECK(caret.points[2].y - caret.points[0].y == Approx(14.0f));
    CHECK((caret.points[0].x + caret.points[1].x) * 0.5f == Approx(bounds.x + 16.0f + 7.0f));

    const auto markdown = Visible<Sakura::TextMarkdown>(transcript);
    REQUIRE(markdown.size() == 1);
    CHECK(markdown[0].y == Approx(label.rect.bottom() + 12.0f));
}

TEST_CASE("Chat transcripts follow the tail until the user scrolls up", "[core][sakura][chat]") {
    Scene scene;
    Transcript transcript;
    Place place(transcript, kRect);
    Frames frames(place, kScreen, scene.context());
    Chat::Snapshot snapshot = Questions(30);
    U64 tail = 0;
    const auto update = [&] {
        transcript.update({.id = "chat", .snapshot = snapshot, .tailRequest = tail});
    };
    const auto wheel = [&](F32 dy) {
        return place.send(Wheel(200.0f, 150.0f, dy));
    };
    REQUIRE(frames.settle(update));
    CHECK(Bottom(transcript) == Approx(kRect.bottom()).margin(1.0f));
    CHECK(Visible<Sakura::TextView>(transcript).size() <= Transcript::UserSlots);

    REQUIRE(wheel(1.0f));
    CHECK(RepaintOnly(frames.run(update)));
    const F32 scrolled = Bottom(transcript);
    CHECK(scrolled > kRect.bottom() + 40.0f);

    snapshot.messages.push_back(UserMessage("Late message"));
    frames.run(update);
    CHECK(Bottom(transcript) == Approx(scrolled));

    REQUIRE(wheel(-100.0f));
    frames.run(update);
    CHECK(Bottom(transcript) == Approx(kRect.bottom()).margin(1.0f));
    snapshot.messages.push_back(UserMessage("Another message"));
    frames.run(update);
    CHECK(Bottom(transcript) == Approx(kRect.bottom()).margin(1.0f));

    REQUIRE(wheel(1.0f));
    frames.run(update);
    CHECK(Bottom(transcript) > kRect.bottom() + 40.0f);
    ++tail;
    frames.run(update);
    CHECK(Bottom(transcript) == Approx(kRect.bottom()).margin(1.0f));
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
}

TEST_CASE("Chat transcripts reset scroll and disclosures when the generation changes", "[core][sakura][chat]") {
    Scene scene;
    Twin twin;
    Frames frames(twin.livePlace, twin.referencePlace, kScreen, scene.context());
    Chat::Snapshot snapshot = Questions(20);
    snapshot.messages.push_back(AssistantMessage({DisclosurePart(5, 1, "Plan", "", "Step one\n\nStep two\n\nStep three")}));
    const auto update = [&] {
        twin.update({.id = "chat", .snapshot = snapshot});
    };
    const auto toggleBottom = [&] {
        return Toggles(twin.live).config().instances.at(0).rect.bottom();
    };
    REQUIRE(frames.settle(update));
    CHECK(toggleBottom() == Approx(kRect.bottom()).margin(1.0f));
    CHECK_FALSE(Expanded(twin.live));

    twin.click(TogglePoint(twin.live));
    REQUIRE(frames.settle(update));
    CHECK(Expanded(twin.live));
    CHECK(Visible<Sakura::TextMarkdown>(twin.live).size() == 1);
    CHECK(Visible<Sakura::TextMarkdown>(twin.live).at(0).bottom() == Approx(kRect.bottom()).margin(1.0f));

    const F32 expanded = toggleBottom();
    REQUIRE(twin.send(Wheel(200.0f, 150.0f, 1.0f)));
    REQUIRE(frames.settle(update));
    CHECK(toggleBottom() > expanded + 40.0f);
    CHECK(Visible<Sakura::TextMarkdown>(twin.live).at(0).bottom() > kRect.bottom() + 40.0f);

    snapshot.generation = 2;
    REQUIRE(frames.settle(update));
    CHECK_FALSE(Expanded(twin.live));
    CHECK(Visible<Sakura::TextMarkdown>(twin.live).empty());
    CHECK(toggleBottom() == Approx(kRect.bottom()).margin(1.0f));
}

TEST_CASE("Chat transcript disclosures expand and collapse per message part", "[core][sakura][chat]") {
    Scene scene;
    Twin twin;
    Frames frames(twin.livePlace, twin.referencePlace, kScreen, scene.context());
    Chat::Snapshot snapshot{.generation = 1};
    snapshot.messages.push_back(AssistantMessage({
        DisclosurePart(7, 1, "Read file", "Opened the file", "Body line one\nBody line two"),
        MarkdownPart(8, 1, "Answer"),
    }));
    const auto update = [&] {
        twin.update({.id = "chat", .snapshot = snapshot});
    };
    REQUIRE(frames.settle(update));

    auto& live = twin.live;
    const auto summary = Visible<Sakura::TextView>(live);
    REQUIRE(summary.size() == 1);
    CHECK(summary[0].x == Approx(kRect.x + 17.0f));
    CHECK(summary[0].y == Approx(kRect.y + kToggleHeight + 2.0f));
    CHECK(summary[0].width == Approx(kRect.width - 14.0f - 17.0f));
    auto markdown = Visible<Sakura::TextMarkdown>(live);
    REQUIRE(markdown.size() == 1);
    CHECK(markdown[0].y == Approx(summary[0].bottom() + 6.0f));
    CHECK_FALSE(Expanded(live));
    CHECK(PointsRight(Carets(live).config().instances.at(0)));

    const auto toggle = TogglePoint(live);
    twin.click(toggle);
    REQUIRE(frames.settle(update));
    markdown = Visible<Sakura::TextMarkdown>(live);
    REQUIRE(markdown.size() == 2);
    CHECK(markdown[0].y == Approx(summary[0].bottom() + 6.0f));
    CHECK(markdown[1].y == Approx(markdown[0].bottom() + 6.0f));
    CHECK(Expanded(live));

    twin.click(toggle);
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Visible<Sakura::TextMarkdown>(live).size() == 1);
    CHECK_FALSE(Expanded(live));
    CHECK(Retains(frames, update));

    twin.click(toggle);
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Visible<Sakura::TextMarkdown>(live).size() == 2);
    CHECK(Expanded(live));
    CHECK(Retains(frames, update));
}

TEST_CASE("Chat transcript disclosures that cannot expand ignore clicks", "[core][sakura][chat]") {
    Scene scene;
    Twin twin;
    Frames frames(twin.livePlace, twin.referencePlace, kScreen, scene.context());
    Chat::Snapshot snapshot{.generation = 1};
    snapshot.messages.push_back(AssistantMessage({DisclosurePart(3, 1, "Status", "", "Hidden", false, false)}));
    const auto update = [&] {
        twin.update({.id = "chat", .snapshot = snapshot});
    };
    REQUIRE(frames.settle(update));
    twin.click(TogglePoint(twin.live));
    REQUIRE(frames.settle(update));
    CHECK(Visible<Sakura::TextMarkdown>(twin.live).empty());
    CHECK_FALSE(Expanded(twin.live));
}

TEST_CASE("Chat transcript toggles repaint once per hover transition and show a hand cursor",
          "[core][sakura][chat][retention]") {
    Scene scene;
    Transcript transcript;
    Place place(transcript, kRect);
    Frames frames(place, kScreen, scene.context());
    Chat::Snapshot snapshot{.generation = 1};
    snapshot.messages.push_back(AssistantMessage({DisclosurePart(1, 1, "Search"), MarkdownPart(2, 1, "Result")}));
    const auto update = [&] {
        transcript.update({.id = "chat", .snapshot = snapshot});
    };
    REQUIRE(frames.settle(update));
    const auto inside = TogglePoint(transcript);
    const Extent2D<F32> outside = {300.0f, 250.0f};
    CheckHoverRepaintsOnce(frames, place, update, inside, outside);
    CheckLeaveRepaintsOnce(frames, place, update, inside);

    place.send(Mouse(MouseEventType::Move, inside.x, inside.y));
    CHECK(RepaintOnly(frames.run(update)));
    const auto& highlights = Highlights(transcript).config();
    const auto label = Toggles(transcript).config().instances.at(0).rect;
    REQUIRE(highlights.instances.size() == 1);
    CHECK(highlights.instances[0].backgroundColor == frames.ctx.palette.get().at("button_hovered"));
    CHECK(highlights.instances[0].rect.x == kRect.x);
    CHECK(highlights.instances[0].rect.y == label.y);
    CHECK(highlights.instances[0].rect.height == Approx(kToggleHeight));
    CHECK(highlights.cornerRadius == 6.0f);
    CHECK(highlights.capacity == 1);

    CHECK(Sakura::Access::cursor(place, inside) == SurfaceCursor::Hand);
    CHECK_FALSE(Sakura::Access::cursor(place, outside).has_value());
}

TEST_CASE("Chat transcript toggles ignore releases outside and cancelled presses", "[core][sakura][chat]") {
    Scene scene;
    Twin twin;
    Frames frames(twin.livePlace, twin.referencePlace, kScreen, scene.context());
    Chat::Snapshot snapshot{.generation = 1};
    snapshot.messages.push_back(AssistantMessage({DisclosurePart(1, 1, "Search", "", "Body")}));
    const auto update = [&] {
        twin.update({.id = "chat", .snapshot = snapshot});
    };
    REQUIRE(frames.settle(update));
    const auto toggle = TogglePoint(twin.live);

    twin.send(Mouse(MouseEventType::Click, toggle.x, toggle.y));
    CHECK(RepaintOnly(frames.run(update)));
    REQUIRE(Highlights(twin.live).config().instances.size() == 1);
    CHECK(Highlights(twin.live).config().instances[0].backgroundColor ==
          frames.ctx.palette.get().at("button_active"));
    twin.send(Mouse(MouseEventType::Cancel, toggle.x, toggle.y));
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Highlights(twin.live).config().instances.empty());
    CHECK_FALSE(twin.send(Mouse(MouseEventType::Release, toggle.x, toggle.y)));
    CHECK(frames.run(update).clean());
    CHECK_FALSE(Expanded(twin.live));

    twin.send(Mouse(MouseEventType::Click, toggle.x, toggle.y));
    frames.run(update);
    CHECK(twin.send(Mouse(MouseEventType::Release, 300.0f, 250.0f)));
    REQUIRE(frames.settle(update));
    CHECK_FALSE(Expanded(twin.live));

    twin.send(Mouse(MouseEventType::Click, toggle.x, toggle.y));
    twin.send(Mouse(MouseEventType::Leave, toggle.x, toggle.y));
    twin.send(Mouse(MouseEventType::Release, toggle.x, toggle.y));
    REQUIRE(frames.settle(update));
    CHECK_FALSE(Expanded(twin.live));
    CHECK(Visible<Sakura::TextMarkdown>(twin.live).empty());
}

TEST_CASE("Chat transcripts settle under a persistent snapshot and repaint as the conversation changes",
          "[core][sakura][chat][retention]") {
    Scene scene;
    Transcript transcript;
    Place place(transcript, kRect);
    Frames frames(place, kScreen, scene.context());
    Chat::Snapshot snapshot = Conversation(4);
    const auto update = [&] {
        transcript.update({.id = "chat", .snapshot = snapshot});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CHECK(QuietMoves(frames, update, {{60.0f, 40.0f}, {200.0f, 120.0f}, {380.0f, 20.0f}, {500.0f, 500.0f}},
                     [&](const MouseEvent& event) {
                         return place.send(event);
                     }));
    const U64 restructures = frames.restructures;
    const U64 rebuilds = frames.rebuilds;

    snapshot = Conversation(2);
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Visible<Sakura::TextView>(transcript).size() == 1);
    CHECK(Visible<Sakura::TextMarkdown>(transcript).size() == 1);
    CHECK(Retains(frames, update));

    snapshot.messages.push_back(UserMessage("Question 2"));
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));

    snapshot.messages.push_back(AssistantMessage({MarkdownPart(0, 1, "Answer 3")}));
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));

    const F32 before = Visible<Sakura::TextMarkdown>(transcript).back().height;
    std::string streamed = "Answer 3";
    for (int i = 0; i < 30; ++i) {
        streamed += " continued";
    }
    snapshot.messages.tail = std::make_shared<const Chat::Message>(AssistantMessage({MarkdownPart(0, 2, streamed)}));
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Visible<Sakura::TextMarkdown>(transcript).back().height > before);
    CHECK(Retains(frames, update));

    CHECK(frames.restructures == restructures);
    CHECK(frames.rebuilds == rebuilds);
}

TEST_CASE("Chat transcripts restructure once when a slot pool is first used", "[core][sakura][chat][retention]") {
    Scene scene;
    Transcript transcript;
    Place place(transcript, kRect);
    Frames frames(place, kScreen, scene.context());
    Chat::Snapshot snapshot{.generation = 1};
    snapshot.messages.push_back(UserMessage("Hello"));
    const auto update = [&] {
        transcript.update({.id = "chat", .snapshot = snapshot});
    };
    REQUIRE(frames.settle(update));
    CHECK(Visible<Sakura::TextView>(transcript).size() == 1);
    CHECK(Visible<Sakura::TextMarkdown>(transcript).empty());
    CHECK(Pooled<Sakura::TextView>(transcript) == Transcript::UserSlots + Transcript::DisclosureSlots);
    CHECK(Pooled<Sakura::TextMarkdown>(transcript) == Transcript::MarkdownSlots);
    const U64 restructures = frames.restructures;
    const U64 rebuilds = frames.rebuilds;

    snapshot.messages.push_back(AssistantMessage({MarkdownPart(0, 1, "Hi")}));
    REQUIRE(frames.settle(update));
    CHECK(Visible<Sakura::TextMarkdown>(transcript).size() == 1);
    CHECK(frames.restructures == restructures + 1);
    CHECK(frames.rebuilds == rebuilds);
    CHECK(Retains(frames, update));
    CHECK(Pooled<Sakura::TextMarkdown>(transcript) == Transcript::MarkdownSlots);
}

TEST_CASE("Chat transcripts reuse slots while scrolling through history", "[core][sakura][chat][retention]") {
    Scene scene;
    Transcript transcript;
    Place place(transcript, kRect);
    Frames frames(place, kScreen, scene.context());
    const Chat::Snapshot snapshot = Conversation(40);
    const auto update = [&] {
        transcript.update({.id = "chat", .snapshot = snapshot});
    };
    const auto wheel = [&](F32 dy) {
        place.send(Wheel(200.0f, 150.0f, dy));
    };
    REQUIRE(frames.settle(update));
    for (int step = 0; step < 40; ++step) {
        wheel(1.0f);
        frames.run(update);
    }
    for (int step = 0; step < 40; ++step) {
        wheel(-1.0f);
        frames.run(update);
    }
    REQUIRE(frames.settle(update));
    const U64 restructures = frames.restructures;
    const U64 rebuilds = frames.rebuilds;

    for (int step = 0; step < 40; ++step) {
        wheel(1.0f);
        CHECK(Kept(frames.run(update)));
        CHECK(Visible<Sakura::TextView>(transcript).size() <= Transcript::UserSlots);
        CHECK(Visible<Sakura::TextMarkdown>(transcript).size() <= Transcript::MarkdownSlots);
    }
    CHECK(Visible<Sakura::TextView>(transcript).at(0).y == kRect.y);
    for (int step = 0; step < 40; ++step) {
        wheel(-1.0f);
        CHECK(Kept(frames.run(update)));
    }
    CHECK(frames.restructures == restructures);
    CHECK(frames.rebuilds == rebuilds);
    CHECK(Visible<Sakura::TextMarkdown>(transcript).back().bottom() == Approx(kRect.bottom()).margin(1.0f));
    CHECK(Retains(frames, update));
}

TEST_CASE("Chat transcripts drop keyboard focus from slots they park or rebind", "[core][sakura][chat]") {
    Scene scene;
    Transcript transcript;
    Place place(transcript, kRect);
    Frames frames(place, kScreen, scene.context());
    const Chat::Snapshot snapshot = Conversation(40);
    const auto update = [&] {
        transcript.update({.id = "chat", .snapshot = snapshot});
    };
    REQUIRE(frames.settle(update));

    Sakura::TextGrid* focused = nullptr;
    for (auto* markdown : Find<Sakura::TextMarkdown>(transcript)) {
        if (markdown->visible()) {
            focused = Find<Sakura::TextGrid>(*markdown).at(0);
            break;
        }
    }
    REQUIRE(focused != nullptr);
    REQUIRE(Sakura::Access::focus(*focused));
    REQUIRE(FocusOwner(place) == focused);

    for (int step = 0; step < 40; ++step) {
        place.send(Wheel(200.0f, 150.0f, 1.0f));
        frames.run(update);
    }
    CHECK(FocusOwner(place) == nullptr);
}

TEST_CASE("Busy chat disclosures shimmer only while visible and settle when busy ends",
          "[core][sakura][chat][retention]") {
    Scene scene;
    Transcript transcript;
    Place place(transcript, kRect);
    Frames frames(place, kScreen, scene.context());
    const auto busySnapshot = [](bool busy) {
        Chat::Snapshot snapshot{.generation = 1};
        snapshot.messages.push_back(AssistantMessage({DisclosurePart(4, busy ? 1 : 2, "Running tool", "", "", busy)}));
        return snapshot;
    };
    Chat::Snapshot snapshot = busySnapshot(true);
    const auto update = [&] {
        transcript.update({.id = "chat", .snapshot = snapshot, .surfaceColorKey = "agent_panel"});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));

    const auto strips = Shimmer(transcript).config().instances;
    const auto label = Toggles(transcript).config().instances.at(0).rect;
    const auto panel = frames.ctx.palette.get().at("agent_panel");
    REQUIRE(strips.size() == Transcript::ShimmerSteps);
    CHECK(strips.front().rect.x == Approx(kRect.x + 15.0f));
    CHECK(strips.back().rect.right() == Approx(label.right()));
    CHECK(strips.front().rect.y == label.y);
    CHECK(strips.front().rect.height == label.height);
    CHECK(strips.front().backgroundColor.r == panel.r);
    CHECK(strips.front().backgroundColor.a <= panel.a * 0.55f);
    CHECK(Shimmer(transcript).config().capacity == Transcript::DisclosureSlots * Transcript::ShimmerSteps);

    for (int frame = 0; frame < 3; ++frame) {
        frames.ctx.time += 0.1;
        CHECK(RepaintOnly(frames.run(update)));
    }
    CHECK(Shimmer(transcript).config().instances != strips);

    const auto toggle = TogglePoint(transcript);
    place.send(Mouse(MouseEventType::Move, toggle.x, toggle.y));
    frames.run(update);
    CHECK(Highlights(transcript).config().instances.empty());
    CHECK(Sakura::Access::cursor(place, toggle) == SurfaceCursor::Hand);

    snapshot = busySnapshot(false);
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Shimmer(transcript).config().instances.empty());
    for (int frame = 0; frame < 3; ++frame) {
        frames.ctx.time += 0.1;
        CHECK(frames.run(update).clean());
    }

    Chat::Snapshot offscreen{.generation = 3};
    offscreen.messages.push_back(AssistantMessage({DisclosurePart(4, 1, "Running tool", "", "", true)}));
    for (U64 i = 0; i < 30; ++i) {
        offscreen.messages.push_back(UserMessage("Message " + std::to_string(i)));
    }
    snapshot = offscreen;
    REQUIRE(frames.settle(update));
    CHECK(Shimmer(transcript).config().instances.empty());
    for (int frame = 0; frame < 3; ++frame) {
        frames.ctx.time += 0.1;
        CHECK(frames.run(update).clean());
    }
}

TEST_CASE("Chat transcripts keep fixed drawable capacities and pool sizes", "[core][sakura][chat][retention]") {
    Scene scene;
    Transcript transcript;
    Place place(transcript, kRect);
    Frames frames(place, kScreen, scene.context());
    Chat::Snapshot snapshot{.generation = 1};
    const auto update = [&] {
        transcript.update({.id = "chat", .snapshot = snapshot});
    };
    const auto capacities = [&] {
        return std::vector<U64>{
            Bubbles(transcript).config().capacity,
            Highlights(transcript).config().capacity,
            Shimmer(transcript).config().capacity,
            Toggles(transcript).config().capacity,
            Toggles(transcript).config().maxCharacters,
            Carets(transcript).config().capacity,
        };
    };
    const auto pools = [&] {
        return std::vector<U64>{Pooled<Sakura::TextView>(transcript), Pooled<Sakura::TextMarkdown>(transcript)};
    };
    REQUIRE(frames.settle(update));
    const auto empty = capacities();
    const auto mounted = pools();
    CHECK(empty[0] == Transcript::UserSlots);
    CHECK(empty[1] == 1);
    CHECK(empty[2] == Transcript::DisclosureSlots * Transcript::ShimmerSteps);
    CHECK(empty[3] == Transcript::DisclosureSlots);
    CHECK(Toggles(transcript).config().fontName == "default_body_bold");

    snapshot = Chat::Snapshot{.generation = 2};
    for (U64 i = 0; i < 40; ++i) {
        if (i % 2 == 0) {
            snapshot.messages.push_back(UserMessage("Question " + std::to_string(i)));
        } else {
            snapshot.messages.push_back(AssistantMessage({
                DisclosurePart(0, 1, "Tool " + std::to_string(i), "Summary", "", true),
                MarkdownPart(1, 1, "Answer " + std::to_string(i)),
            }));
        }
    }
    REQUIRE(frames.settle(update, 16));
    CHECK(capacities() == empty);
    CHECK(pools() == mounted);
    CHECK(Bubbles(transcript).config().instances.size() <= Transcript::UserSlots);
    CHECK(Toggles(transcript).config().instances.size() <= Transcript::DisclosureSlots);
    CHECK_FALSE(Shimmer(transcript).config().instances.empty());
    CHECK(Shimmer(transcript).config().instances.size() <= Transcript::DisclosureSlots * Transcript::ShimmerSteps);
    CHECK(Visible<Sakura::TextMarkdown>(transcript).size() <= Transcript::MarkdownSlots);
    CHECK(Visible<Sakura::TextView>(transcript).size() <= Transcript::UserSlots + Transcript::DisclosureSlots);
}
