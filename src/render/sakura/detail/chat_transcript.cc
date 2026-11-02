#include "chat_transcript.hh"

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/components/polyline.hh>
#include <jetstream/render/sakura/components/scroller.hh>
#include <jetstream/render/sakura/components/text_grid.hh>
#include <jetstream/render/sakura/components/text_markdown.hh>
#include <jetstream/render/sakura/components/text_view.hh>
#include <jetstream/logger.hh>

#include "../context.hh"
#include "../metrics.hh"
#include "helpers.hh"
#include "text_metrics.hh"

#include <algorithm>
#include <cmath>
#include <compare>
#include <functional>
#include <limits>
#include <map>
#include <numbers>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

namespace {

using Role = Chat::Role;
using PartKind = Chat::PartKind;
using Part = Chat::Part;
using Message = Chat::Message;

constexpr F32 kBubbleGap = 10.0f;
constexpr F32 kTurnGap = 16.0f;
constexpr F32 kPartGap = 6.0f;
constexpr F32 kBubbleMaxWidthRatio = 0.72f;
constexpr F32 kBubbleBorder = 1.0f;
constexpr F32 kTextPad = 9.0f;
constexpr F32 kScrollbarReserve = 14.0f;
constexpr F32 kThinScrollbar = 5.0f;
constexpr F32 kWheelLines = 3.0f;
constexpr F32 kTailSlack = 1.0f;
constexpr F32 kOverscanLines = 2.0f;
constexpr F32 kSummaryInset = 17.0f;
constexpr F32 kSummaryGap = 2.0f;
constexpr F32 kCaretInset = 8.0f;
constexpr F32 kCaretLength = 7.0f;
constexpr F32 kCaretDepth = 4.0f;
constexpr F32 kLabelInset = 15.0f;
constexpr F32 kTogglePadX = 8.0f;
constexpr F32 kTogglePadY = 4.0f;
constexpr F32 kToggleRounding = 6.0f;
constexpr U64 kToggleCharacters = 128;
constexpr F64 kShimmerPeriod = 1.8;
constexpr F32 kShimmerBandRatio = 0.35f;
constexpr F32 kShimmerDim = 0.55f;
constexpr const char* kToggleFont = Typography::BodyBoldFont;
constexpr const char* kSummaryFont = Typography::BodyItalicFont;
constexpr F32 kUnbounded = std::numeric_limits<F32>::infinity();

TextView::Config UserText(std::string id, std::string value, F32 font, std::optional<Padding> padding) {
    return {
        .id = std::move(id),
        .value = std::move(value),
        .fontSize = font,
        .fontName = Typography::BodyFont,
        .monospace = false,
        .scrollbar = false,
        .wrap = TextGrid::Wrap::Word,
        .padding = padding,
        .textColorKey = "action_btn_text",
    };
}

TextView::Config SummaryText(std::string id, std::string value, F32 font) {
    return {
        .id = std::move(id),
        .value = std::move(value),
        .fontSize = font,
        .fontName = kSummaryFont,
        .monospace = false,
        .scrollbar = false,
        .wrap = TextGrid::Wrap::Word,
        .textColorKey = "text_secondary",
    };
}

TextMarkdown::Config MarkdownText(std::string id, std::string value, F32 font, bool disclosure,
                                  std::vector<bool> details) {
    return {
        .id = std::move(id),
        .value = std::move(value),
        .fontSize = font,
        .textColorKey = disclosure ? "editor_text" : "text_primary",
        .detailsOpen = std::move(details),
    };
}

U64 DetailsRevision(U64 revision, const std::vector<bool>& details) {
    return revision ^ (std::hash<std::vector<bool>>{}(details) * 0x9E3779B97F4A7C15ull);
}

bool HasExpandablePart(const std::vector<Part>& parts, U64 key) {
    return std::any_of(parts.begin(), parts.end(), [key](const Part& part) {
        return part.kind == PartKind::Disclosure && part.expandable && part.key == key;
    });
}

Polyline::Instance Caret(const Rect& zone, bool down, F32 length, F32 depth, ColorRGBA<F32> color, bool visible) {
    const auto center = zone.center();
    std::vector<Extent2D<F32>> points;
    if (down) {
        points = {{center.x - length * 0.5f, center.y - depth * 0.5f},
                  {center.x + length * 0.5f, center.y - depth * 0.5f},
                  {center.x, center.y + depth * 0.5f}};
    } else {
        points = {{center.x - depth * 0.5f, center.y - length * 0.5f},
                  {center.x + depth * 0.5f, center.y},
                  {center.x - depth * 0.5f, center.y + length * 0.5f}};
    }
    return {.points = std::move(points), .color = color, .closed = true, .filled = true, .visible = visible};
}

void AppendShimmer(std::vector<Box::Instance>& strips, const Rect& zone, F64 time, ColorRGBA<F32> color,
                   bool visible) {
    const U64 steps = ChatTranscript::ShimmerSteps;
    const F32 step = zone.width / static_cast<F32>(steps);
    const F32 band = std::max(1.0f, zone.width * kShimmerBandRatio);
    const F32 phase = static_cast<F32>(std::fmod(std::max(0.0, time), kShimmerPeriod) / kShimmerPeriod);
    const F32 center = zone.x - band + phase * (zone.width + 2.0f * band);
    for (U64 k = 0; k < steps; ++k) {
        const F32 x = zone.x + (static_cast<F32>(k) + 0.5f) * step;
        const F32 distance = std::abs(x - center) / band;
        const F32 highlight = distance < 1.0f ? 0.5f * (1.0f + std::cos(std::numbers::pi_v<F32> * distance)) : 0.0f;
        strips.push_back({
            .rect = {zone.x + static_cast<F32>(k) * step, zone.y, step, zone.height},
            .visible = visible && !zone.empty(),
            .backgroundColor = WithAlpha(color, kShimmerDim * (1.0f - highlight)),
        });
    }
}

}  // namespace

struct ChatTranscript::Impl {
    enum class Kind : U8 {
        User,
        Markdown,
        Disclosure,
    };

    struct Key {
        U64 generation = 0;
        U64 message = 0;
        U64 part = 0;
        Kind kind = Kind::Markdown;

        auto operator<=>(const Key&) const = default;
    };

    struct Metrics {
        Rect bounds;
        bool visible = false;
        F32 unit = 1.0f;
        F32 font = 0.0f;
        F32 lineHeight = 0.0f;
        F32 textPad = 0.0f;
        F32 textInset = 0.0f;
        F32 reserve = 0.0f;
        F32 contentWidth = 0.0f;
        F32 maxBubbleWidth = 0.0f;
        F32 maxUserTextWidth = 0.0f;
        F32 partGap = 0.0f;
        F32 summaryInset = 0.0f;
        F32 summaryGap = 0.0f;
        F32 caretInset = 0.0f;
        F32 caretLength = 0.0f;
        F32 caretDepth = 0.0f;
        F32 labelInset = 0.0f;
        Extent2D<F32> togglePadding = {0.0f, 0.0f};
        F32 visibleTop = 0.0f;
        F32 visibleBottom = 0.0f;
        const Render::Window* window = nullptr;
        U64 fontRevision = 0;

        bool inViewport(const Rect& rect) const {
            return visible && !rect.empty() && rect.bottom() > visibleTop && rect.y < visibleBottom;
        }

        bool inBounds(const Rect& rect) const {
            return visible && !rect.empty() && rect.bottom() > bounds.y && rect.y < bounds.bottom();
        }
    };

    struct Block {
        Key key;
        U64 revision = 0;
        const std::string* text = nullptr;
        const std::string* summary = nullptr;
        std::string label;
        bool expanded = false;
        bool busy = false;
        F32 width = 0.0f;
        F32 height = 0.0f;
        F32 bodyHeight = 0.0f;
        F32 summaryHeight = 0.0f;
        Extent2D<F32> toggleSize = {0.0f, 0.0f};
    };

    struct Measurement {
        U64 revision = std::numeric_limits<U64>::max();
        F32 width = -1.0f;
        F32 font = -1.0f;
        const Render::Window* window = nullptr;
        U64 fontRevision = 0;
        bool expanded = false;
        bool busy = false;
        F32 measuredWidth = 0.0f;
        F32 height = 0.0f;
        F32 bodyHeight = 0.0f;
        F32 summaryHeight = 0.0f;
        Extent2D<F32> toggleSize = {0.0f, 0.0f};
        std::string label;

        bool matches(const Block& block, F32 at, const Metrics& m) const {
            return revision == block.revision && width == at && font == m.font && window == m.window &&
                   fontRevision == m.fontRevision && expanded == block.expanded && busy == block.busy;
        }

        void store(const Block& block, F32 at, const Metrics& m) {
            revision = block.revision;
            width = at;
            font = m.font;
            window = m.window;
            fontRevision = m.fontRevision;
            expanded = block.expanded;
            busy = block.busy;
        }
    };

    struct RowContext {
        U64 generation = 0;
        F32 contentWidth = 0.0f;
        F32 font = 0.0f;
        const Render::Window* window = nullptr;
        U64 fontRevision = 0;
        U64 disclosureRevision = 0;

        bool operator==(const RowContext&) const = default;
    };

    template<typename View>
    struct Slot {
        View view;
        std::optional<Key> key;
        bool identityChanged = false;
        U64 revision = 0;
        const std::string* text = nullptr;
        U64 size = 0;
        F32 font = -1.0f;
    };

    struct UserPlacement {
        const Block* block = nullptr;
        Rect bubble;
    };

    struct MarkdownPlacement {
        const Block* block = nullptr;
        Rect rect;
    };

    struct DisclosurePlacement {
        const Block* block = nullptr;
        Rect toggle;
        Rect summary;
    };

    struct Placements {
        std::vector<UserPlacement> users;
        std::vector<MarkdownPlacement> markdown;
        std::vector<DisclosurePlacement> disclosures;
    };

    struct Toggle {
        Key key;
        Rect rect;
    };

    Config config;
    Box bubbles;
    Scroller scroll;
    std::vector<std::unique_ptr<Slot<TextView>>> users;
    std::vector<std::unique_ptr<Slot<TextMarkdown>>> markdowns;
    Box highlights;
    Label labels;
    Box shimmer;
    Polyline carets;
    std::vector<std::unique_ptr<Slot<TextView>>> summaries;

    TextView userMeasurer;
    TextMarkdown markdownMeasurer;
    TextView summaryMeasurer;
    TextMetrics textMetrics;

    std::map<Key, Measurement> measurements;
    std::vector<std::vector<Block>> rows;
    std::vector<F32> rowWidth;
    std::vector<F32> rowHeight;
    std::vector<std::shared_ptr<const Message>> rowMessages;
    RowContext rowContext;
    std::map<std::pair<U64, U64>, bool> expansions;
    std::map<Key, std::vector<bool>> markdownDetails;
    std::vector<F32> markdownTops;
    F32 revealMargin = 0.0f;
    U64 disclosureRevision = 0;

    std::vector<Toggle> toggles;
    std::optional<Key> hovered;
    std::optional<Key> pressed;

    F32 scrollY = 0.0f;
    F32 overscrollY = 0.0f;
    bool followTail = true;
    F32 lastMaxScroll = 0.0f;
    F32 tailSlack = 0.0f;

    template<typename View>
    static void Allocate(std::vector<std::unique_ptr<Slot<View>>>& slots, U64 count) {
        while (slots.size() < count) {
            auto& slot = *slots.emplace_back(std::make_unique<Slot<View>>());
            slot.view.setVisible(false);
        }
    }

    template<typename SlotType, typename Placement>
    static bool HasSlot(const std::vector<std::unique_ptr<SlotType>>& slots, const Placement& placement) {
        return std::any_of(slots.begin(), slots.end(), [&](const auto& slot) {
            return slot->key == placement.block->key;
        });
    }

    template<typename Placement, typename SlotType>
    static void Limit(std::vector<Placement>& placements,
                      const std::vector<std::unique_ptr<SlotType>>& slots,
                      bool preferTail) {
        const U64 capacity = slots.size();
        if (placements.size() <= capacity) {
            return;
        }
        std::vector<bool> selected(placements.size(), false);
        U64 count = 0;
        for (U64 i = 0; i < placements.size() && count < capacity; ++i) {
            if (HasSlot(slots, placements[i])) {
                selected[i] = true;
                ++count;
            }
        }
        for (U64 n = 0; n < placements.size() && count < capacity; ++n) {
            const U64 i = preferTail ? placements.size() - 1 - n : n;
            if (!selected[i]) {
                selected[i] = true;
                ++count;
            }
        }
        std::vector<Placement> kept;
        kept.reserve(count);
        for (U64 i = 0; i < placements.size(); ++i) {
            if (selected[i]) {
                kept.push_back(placements[i]);
            }
        }
        placements = std::move(kept);
    }

    template<typename SlotType, typename Placement>
    static std::vector<const Placement*> Assign(std::vector<std::unique_ptr<SlotType>>& slots,
                                                const std::vector<Placement>& placements) {
        std::vector<const Placement*> bySlot(slots.size(), nullptr);
        std::vector<bool> placed(placements.size(), false);
        for (U64 i = 0; i < placements.size(); ++i) {
            for (U64 s = 0; s < slots.size(); ++s) {
                if (!bySlot[s] && slots[s]->key == placements[i].block->key) {
                    bySlot[s] = &placements[i];
                    placed[i] = true;
                    break;
                }
            }
        }
        for (U64 i = 0; i < placements.size(); ++i) {
            if (placed[i]) {
                continue;
            }
            for (U64 s = 0; s < slots.size(); ++s) {
                if (bySlot[s]) {
                    continue;
                }
                bySlot[s] = &placements[i];
                slots[s]->identityChanged = true;
                slots[s]->key = placements[i].block->key;
                break;
            }
        }
        return bySlot;
    }

    template<typename View>
    static void Park(ChatTranscript& self, Slot<View>& slot) {
        if (!slot.view.visible()) {
            return;
        }
        self.cancelMouse(slot.view);
        self.releaseFocusWithin(slot.view);
        slot.view.setVisible(false);
    }

    template<typename View, typename Make>
    static void Bind(ChatTranscript& self, Slot<View>& slot, U64 revision, const std::string& text, F32 font,
                     Make make) {
        if (slot.identityChanged) {
            self.cancelMouse(slot.view);
            self.releaseFocusWithin(slot.view);
            slot.view.update(make(std::string()));
        }
        if (slot.identityChanged || slot.revision != revision || slot.text != &text || slot.size != text.size() ||
            slot.font != font) {
            slot.view.update(make(text));
            slot.revision = revision;
            slot.text = &text;
            slot.size = text.size();
            slot.font = font;
        }
        slot.identityChanged = false;
        if (!slot.view.visible()) {
            slot.view.setVisible(true);
        }
    }

    void allocate() {
        Allocate(users, UserSlots);
        Allocate(markdowns, MarkdownSlots);
        Allocate(summaries, DisclosureSlots);
    }

    void reset() {
        scrollY = 0.0f;
        overscrollY = 0.0f;
        scroll.reset();
        followTail = true;
        measurements.clear();
        rowMessages.clear();
        expansions.clear();
        markdownDetails.clear();
        ++disclosureRevision;
        hovered.reset();
        pressed.reset();
        for (auto& slot : users) {
            slot->key.reset();
        }
        for (auto& slot : markdowns) {
            slot->key.reset();
        }
        for (auto& slot : summaries) {
            slot->key.reset();
        }
    }

    std::string measureId(const char* element, const Key& key) const {
        return jst::fmt::format("{}:measure-{}:{}:{}:{}", config.id, element, key.generation, key.message, key.part);
    }

    std::string slotId(const char* element, U64 index) const {
        return jst::fmt::format("{}:{}{}", config.id, element, index);
    }

    bool expanded(U64 message, U64 part) const {
        const auto found = expansions.find({message, part});
        return found != expansions.end() && found->second;
    }

    void toggle(ChatTranscript& self, const Key& key) {
        const auto& messages = config.snapshot.messages;
        if (key.generation != config.snapshot.generation || key.message >= messages.size()) {
            return;
        }
        const auto& message = messages.at(key.message);
        if (!message || !HasExpandablePart(message->parts, key.part)) {
            return;
        }
        auto& state = expansions[{key.message, key.part}];
        state = !state;
        ++disclosureRevision;
        self.invalidate(Dirty::Layout);
    }

    Metrics metrics(const ChatTranscript& self, const Context& ctx) const {
        const Rect bounds = self.frame();
        const F32 unit = Px(ctx, config.fontSize / Typography::FontSize);
        const F32 font = Px(ctx, config.fontSize);
        const F32 lineHeight = font * Typography::CodeLineHeight;
        const F32 textPad = kTextPad * unit;
        const F32 reserve = kScrollbarReserve * unit;
        const F32 contentWidth = std::max(1.0f, bounds.width - reserve);
        const F32 maxBubbleWidth = std::max(1.0f, contentWidth * kBubbleMaxWidthRatio);
        const F32 overscan = kOverscanLines * lineHeight;
        return {
            .bounds = bounds,
            .visible = !bounds.empty(),
            .unit = unit,
            .font = font,
            .lineHeight = lineHeight,
            .textPad = textPad,
            .textInset = font * Typography::TextPaddingRatio,
            .reserve = reserve,
            .contentWidth = contentWidth,
            .maxBubbleWidth = maxBubbleWidth,
            .maxUserTextWidth = std::max(1.0f, maxBubbleWidth - 2.0f * textPad),
            .partGap = kPartGap * unit,
            .summaryInset = kSummaryInset * unit,
            .summaryGap = kSummaryGap * unit,
            .caretInset = kCaretInset * unit,
            .caretLength = kCaretLength * unit,
            .caretDepth = kCaretDepth * unit,
            .labelInset = kLabelInset * unit,
            .togglePadding = {Px(ctx, kTogglePadX), Px(ctx, kTogglePadY)},
            .visibleTop = bounds.y - overscan,
            .visibleBottom = bounds.bottom() + overscan,
            .window = ctx.render,
            .fontRevision = ctx.render ? ctx.render->fontRevision() : 0,
        };
    }

    Block makeBlock(U64 message, U64 part, Kind kind) const {
        Block block;
        block.key = {
            .generation = config.snapshot.generation,
            .message = message,
            .part = part,
            .kind = kind,
        };
        return block;
    }

    void sync(const Metrics& m) {
        const RowContext next = {
            .generation = config.snapshot.generation,
            .contentWidth = m.contentWidth,
            .font = m.font,
            .window = m.window,
            .fontRevision = m.fontRevision,
            .disclosureRevision = disclosureRevision,
        };
        if (rowContext != next) {
            rowMessages.clear();
            rowContext = next;
        }
        const U64 count = config.snapshot.messages.size();
        rows.resize(count);
        rowWidth.resize(count, 0.0f);
        rowHeight.resize(count, 0.0f);
        rowMessages.resize(count);
    }

    void buildRows(ChatTranscript& self, const Context& ctx, const Metrics& m) {
        const auto& messages = config.snapshot.messages;
        for (U64 i = 0; i < messages.size(); ++i) {
            const auto& message = messages.at(i);
            if (message && rowMessages[i] == message) {
                continue;
            }
            rowMessages[i] = message;
            rows[i].clear();
            rowWidth[i] = 0.0f;
            rowHeight[i] = 0.0f;
            if (!message) {
                continue;
            }
            if (message->role == Role::User) {
                buildUser(self, ctx, m, i, *message);
            } else {
                buildAssistant(self, ctx, m, i, *message);
            }
        }
    }

    void buildUser(ChatTranscript& self, const Context& ctx, const Metrics& m, U64 index, const Message& message) {
        Block block = makeBlock(index, 0, Kind::User);
        block.text = &message.text;
        block.revision = std::hash<std::string>{}(message.text);
        auto& cached = measurements[block.key];
        if (!cached.matches(block, m.maxUserTextWidth, m)) {
            userMeasurer.update(UserText(config.id + ":measure-user", message.text, m.font, std::nullopt));
            const auto measured = self.measureChild(userMeasurer, ctx, {m.maxUserTextWidth, kUnbounded});
            cached.store(block, m.maxUserTextWidth, m);
            cached.measuredWidth = std::min(m.maxBubbleWidth, measured.x + 2.0f * m.textPad);
            cached.height = std::max(m.lineHeight, measured.y) + 2.0f * m.textPad;
        }
        block.width = cached.measuredWidth;
        block.height = cached.height;
        rowWidth[index] = block.width;
        rowHeight[index] = block.height;
        rows[index].push_back(std::move(block));
    }

    void buildAssistant(ChatTranscript& self, const Context& ctx, const Metrics& m, U64 index,
                        const Message& message) {
        auto& blocks = rows[index];
        for (const auto& part : message.parts) {
            if (part.kind == PartKind::Disclosure) {
                addDisclosure(self, ctx, m, index, part);
            } else {
                addMarkdown(self, ctx, m, index, part);
            }
        }
        F32 height = 0.0f;
        for (U64 b = 0; b < blocks.size(); ++b) {
            height += blocks[b].height + (b > 0 ? m.partGap : 0.0f);
        }
        rowWidth[index] = m.contentWidth;
        rowHeight[index] = height;
    }

    void addMarkdown(ChatTranscript& self, const Context& ctx, const Metrics& m, U64 index, const Part& part) {
        Block block = makeBlock(index, part.key, Kind::Markdown);
        const auto& details = detailsFor(block.key);
        block.revision = DetailsRevision(part.revision, details);
        block.text = &part.text;
        auto& cached = measurements[block.key];
        if (!cached.matches(block, m.contentWidth, m)) {
            markdownMeasurer.update(MarkdownText(measureId("markdown", block.key), part.text, m.font, false,
                                                 details));
            cached.store(block, m.contentWidth, m);
            cached.height =
                std::max(m.lineHeight, self.measureChild(markdownMeasurer, ctx, {m.contentWidth, kUnbounded}).y);
        }
        block.height = cached.height;
        rows[index].push_back(std::move(block));
    }

    void addDisclosure(ChatTranscript& self, const Context& ctx, const Metrics& m, U64 index, const Part& part) {
        Block block = makeBlock(index, part.key, Kind::Disclosure);
        const auto& details = detailsFor(block.key);
        block.revision = DetailsRevision(part.revision, details);
        block.expanded = part.expandable && expanded(index, part.key);
        block.busy = part.busy;
        block.summary = &part.summary;
        if (block.expanded) {
            block.text = &part.text;
        }
        auto& cached = measurements[block.key];
        if (!cached.matches(block, m.contentWidth, m)) {
            const F32 room = std::max(0.0f, m.contentWidth - m.labelInset - 2.0f * m.togglePadding.x);
            cached.label = textMetrics.ellipsize(kToggleFont, part.label, m.font, room);
            const F32 textWidth = textMetrics.measure(kToggleFont, cached.label, m.font);
            cached.toggleSize = {
                std::min(m.contentWidth, textWidth + 2.0f * m.togglePadding.x + m.labelInset),
                m.font + 2.0f * m.togglePadding.y,
            };
            cached.summaryHeight = 0.0f;
            cached.bodyHeight = 0.0f;
            cached.height = cached.toggleSize.y;
            if (!part.summary.empty()) {
                summaryMeasurer.update(SummaryText(config.id + ":measure-summary", part.summary, m.font));
                const F32 summaryWidth = std::max(1.0f, m.contentWidth - m.summaryInset);
                cached.summaryHeight = self.measureChild(summaryMeasurer, ctx, {summaryWidth, kUnbounded}).y;
                cached.height += m.summaryGap + cached.summaryHeight;
            }
            if (block.expanded) {
                markdownMeasurer.update(MarkdownText(measureId("body", block.key), part.text, m.font, true,
                                                     details));
                cached.bodyHeight = self.measureChild(markdownMeasurer, ctx, {m.contentWidth, kUnbounded}).y;
                cached.height += m.partGap + cached.bodyHeight;
            }
            cached.store(block, m.contentWidth, m);
        }
        block.label = cached.label;
        block.toggleSize = cached.toggleSize;
        block.summaryHeight = cached.summaryHeight;
        block.bodyHeight = cached.bodyHeight;
        block.height = cached.height;
        rows[index].push_back(std::move(block));
    }

    bool userAt(U64 index) const {
        return index < rowMessages.size() && rowMessages[index] && rowMessages[index]->role == Role::User;
    }

    F32 gapAfter(const Metrics& m, U64 index) const {
        const bool turn = index + 1 < rowMessages.size() && userAt(index + 1) != userAt(index);
        return (turn ? kTurnGap : kBubbleGap) * m.unit;
    }

    F32 contentHeight(const Metrics& m) const {
        F32 height = 0.0f;
        for (U64 i = 0; i < rowHeight.size(); ++i) {
            height += rowHeight[i] + (i + 1 < rowHeight.size() ? gapAfter(m, i) : 0.0f);
        }
        return height;
    }

    void clampScroll(const Metrics& m, F32 height) {
        const F32 maxScroll = std::max(0.0f, height - m.bounds.height);
        lastMaxScroll = maxScroll;
        if (followTail) {
            scrollY = maxScroll;
        }
        scrollY = std::clamp(scrollY, 0.0f, maxScroll);
    }

    Placements place(const Metrics& m) const {
        Placements placements;
        F32 y = std::round(m.bounds.y - scrollY - overscrollY);
        for (U64 i = 0; i < rows.size(); ++i) {
            if (userAt(i)) {
                placeUser(m, i, y, placements);
            } else {
                placeAssistant(m, i, y, placements);
            }
            y += rowHeight[i] + (i + 1 < rows.size() ? gapAfter(m, i) : 0.0f);
        }
        return placements;
    }

    void placeUser(const Metrics& m, U64 index, F32 y, Placements& placements) const {
        if (rows[index].empty()) {
            return;
        }
        const Rect bubble = {m.bounds.right() - m.reserve - rowWidth[index], y, rowWidth[index], rowHeight[index]};
        if (m.inViewport(bubble)) {
            placements.users.push_back({.block = &rows[index].front(), .bubble = bubble});
        }
    }

    void placeAssistant(const Metrics& m, U64 index, F32 y, Placements& placements) const {
        const auto& blocks = rows[index];
        const F32 width = rowWidth[index];
        for (U64 b = 0; b < blocks.size(); ++b) {
            const auto& block = blocks[b];
            if (b > 0) {
                y += m.partGap;
            }
            if (block.key.kind == Kind::Disclosure) {
                y = placeDisclosure(m, block, width, y, placements);
                continue;
            }
            const Rect rect = {m.bounds.x, y, width, block.height};
            if (m.inViewport(rect)) {
                placements.markdown.push_back({.block = &block, .rect = rect});
            }
            y += block.height;
        }
    }

    F32 placeDisclosure(const Metrics& m, const Block& block, F32 width, F32 y, Placements& placements) const {
        const Rect toggleRect = {m.bounds.x, y, block.toggleSize.x, block.toggleSize.y};
        y += block.toggleSize.y;
        Rect summaryRect;
        if (block.summaryHeight > 0.0f) {
            y += m.summaryGap;
            summaryRect = {m.bounds.x + m.summaryInset, y, std::max(0.0f, width - m.summaryInset), block.summaryHeight};
            y += block.summaryHeight;
        }
        const Rect chrome = {
            toggleRect.x,
            toggleRect.y,
            std::max(toggleRect.width, summaryRect.right() - toggleRect.x),
            std::max(toggleRect.height, summaryRect.bottom() - toggleRect.y),
        };
        if (m.inBounds(chrome)) {
            placements.disclosures.push_back({.block = &block, .toggle = toggleRect, .summary = summaryRect});
        }
        if (!block.expanded) {
            return y;
        }
        y += m.partGap;
        const Rect body = {m.bounds.x, y, width, block.bodyHeight};
        if (m.inViewport(body)) {
            placements.markdown.push_back({.block = &block, .rect = body});
        }
        return y + block.bodyHeight;
    }

    void layoutScroll(ChatTranscript& self, const Context& ctx, const Metrics& m, F32 height) {
        tailSlack = Px(ctx, kTailSlack);
        scroll.update({
            .id = config.id + ":scroll",
            .contentHeight = height,
            .scrollY = scrollY,
            .scrollbar = m.visible,
            .wheelStep = m.lineHeight * kWheelLines,
            .thickness = kThinScrollbar * m.unit,
            .margin = kScrollbarMargin * m.unit,
            .onScrollY = [this, &self](F32 next) {
                scrollY = next;
                followTail = next >= lastMaxScroll - tailSlack;
                self.invalidate(Dirty::Layout);
            },
            .onOverscroll = [&self] {
                self.requestFrame();
            },
        });
        self.layoutChild(ctx, scroll, m.bounds);
    }

    void layoutUsers(ChatTranscript& self, const Context& ctx, const Metrics& m,
                     const std::vector<const UserPlacement*>& bySlot) {
        std::vector<Box::Instance> fills;
        const auto fill = ctx.color("chat_user_bubble");
        const Padding padding = {m.textPad + m.textInset, m.textPad, m.textInset, m.textPad};
        for (U64 i = 0; i < users.size(); ++i) {
            auto& slot = *users[i];
            const auto* placement = bySlot[i];
            if (!placement) {
                Park(self, slot);
                continue;
            }
            Bind(self, slot, placement->block->revision, *placement->block->text, m.font, [&](std::string value) {
                return UserText(slotId("user", i), std::move(value), m.font, padding);
            });
            self.layoutChild(ctx, slot.view, placement->bubble);
            fills.push_back({.rect = placement->bubble, .visible = m.visible, .backgroundColor = fill});
        }
        bubbles.update({
            .instances = std::move(fills),
            .clip = m.bounds,
            .cornerRadius = m.lineHeight * 0.5f + m.textPad,
            .borderWidth = kBubbleBorder * m.unit,
            .borderColor = ctx.color("chat_user_bubble_outline"),
            .strokeFringe = true,
            .capacity = UserSlots,
        });
        self.layoutChild(ctx, bubbles, m.bounds);
    }

    const std::vector<bool>& detailsFor(const Key& key) const {
        static const std::vector<bool> kNone;
        const auto found = markdownDetails.find(key);
        return found != markdownDetails.end() ? found->second : kNone;
    }

    void toggleDetails(ChatTranscript& self, U64 slot, const std::vector<bool>& open) {
        if (slot >= markdowns.size() || !markdowns[slot]->key) {
            return;
        }
        markdownDetails[*markdowns[slot]->key] = open;
        ++disclosureRevision;
        self.invalidate(Dirty::Layout);
    }

    bool reveal(ChatTranscript& self, U64 slot, F32 offset) {
        if (slot >= markdownTops.size()) {
            return false;
        }
        scrollY = std::max(0.0f, markdownTops[slot] + offset - revealMargin);
        followTail = false;
        self.invalidate(Dirty::Layout);
        return true;
    }

    void layoutMarkdown(ChatTranscript& self, const Context& ctx, const Metrics& m,
                        const std::vector<const MarkdownPlacement*>& bySlot) {
        markdownTops.assign(markdowns.size(), 0.0f);
        revealMargin = m.lineHeight;
        for (U64 i = 0; i < markdowns.size(); ++i) {
            auto& slot = *markdowns[i];
            const auto* placement = bySlot[i];
            if (!placement) {
                Park(self, slot);
                continue;
            }
            const bool disclosure = placement->block->key.kind == Kind::Disclosure;
            Bind(self, slot, placement->block->revision, *placement->block->text, m.font, [&](std::string value) {
                auto text = MarkdownText(slotId("markdown", i), std::move(value), m.font, disclosure,
                                         detailsFor(placement->block->key));
                text.onDetailsToggle = [this, &self, i](const std::vector<bool>& open) {
                    toggleDetails(self, i, open);
                };
                text.onReveal = [this, &self, i](F32 offset) {
                    return reveal(self, i, offset);
                };
                return text;
            });
            markdownTops[i] = placement->rect.y - m.bounds.y + scrollY + overscrollY;
            self.layoutChild(ctx, slot.view, placement->rect);
        }
    }

    void layoutDisclosures(ChatTranscript& self, const Context& ctx, const Metrics& m,
                           const std::vector<const DisclosurePlacement*>& bySlot) {
        std::vector<Box::Instance> fills;
        std::vector<Label::Instance> texts;
        std::vector<Box::Instance> strips;
        std::vector<Polyline::Instance> marks;
        const auto textColor = ctx.color("text_primary");
        const auto caretColor = ctx.color("text_secondary");
        const auto surface = ctx.color(config.surfaceColorKey);
        bool animating = false;
        toggles.clear();
        for (U64 i = 0; i < summaries.size(); ++i) {
            auto& slot = *summaries[i];
            const auto* placement = bySlot[i];
            if (!placement) {
                Park(self, slot);
                continue;
            }
            const Block& block = *placement->block;
            const Rect& toggle = placement->toggle;
            const Rect hit = Intersect(toggle, m.bounds);
            toggles.push_back({.key = block.key, .rect = hit});
            if (!block.busy && hovered == block.key) {
                fills.push_back({
                    .rect = hit,
                    .visible = m.visible && !hit.empty(),
                    .backgroundColor = ctx.color(pressed == block.key ? "button_active" : "button_hovered"),
                });
            }
            const F32 labelX = toggle.x + m.labelInset + m.togglePadding.x;
            const Rect labelRect = {labelX, toggle.y, std::max(0.0f, toggle.right() - labelX), toggle.height};
            texts.push_back({
                .rect = labelRect,
                .str = block.label,
                .visible = m.visible && !labelRect.empty(),
                .color = textColor,
                .fontSize = m.font,
                .alignment = {0, 1},
            });
            const Rect caretZone = {toggle.x + m.caretInset, toggle.y, m.caretLength, toggle.height};
            marks.push_back(Caret(caretZone, block.expanded, m.caretLength, m.caretDepth, caretColor, m.visible));
            if (block.busy) {
                const Rect zone = {toggle.x + m.labelInset, toggle.y, std::max(0.0f, toggle.width - m.labelInset),
                                   toggle.height};
                AppendShimmer(strips, zone, ctx.time, surface, m.visible);
                animating = true;
            }
            if (block.summary && !block.summary->empty() && !placement->summary.empty()) {
                Bind(self, slot, block.revision, *block.summary, m.font, [&](std::string value) {
                    return SummaryText(slotId("summary", i), std::move(value), m.font);
                });
                self.layoutChild(ctx, slot.view, placement->summary);
            } else {
                Park(self, slot);
            }
        }
        highlights.update({
            .instances = std::move(fills),
            .clip = m.bounds,
            .cornerRadius = kToggleRounding * m.unit,
            .capacity = 1,
        });
        self.layoutChild(ctx, highlights, m.bounds);
        labels.update({
            .instances = std::move(texts),
            .clip = m.bounds,
            .fontName = kToggleFont,
            .maxCharacters = kToggleCharacters,
            .capacity = DisclosureSlots,
        });
        self.layoutChild(ctx, labels, m.bounds);
        shimmer.update({
            .instances = std::move(strips),
            .clip = m.bounds,
            .capacity = DisclosureSlots * ShimmerSteps,
        });
        self.layoutChild(ctx, shimmer, m.bounds);
        carets.update({
            .instances = std::move(marks),
            .clip = m.bounds,
            .fringe = Fringe(ctx),
        });
        self.layoutChild(ctx, carets, m.bounds);
        if (animating) {
            self.requestFrame();
        }
    }

    const Toggle* toggleAt(const ChatTranscript& self, Extent2D<F32> point) const {
        if (!self.clip().contains(point.x, point.y)) {
            return nullptr;
        }
        for (const auto& toggle : toggles) {
            if (toggle.rect.contains(point.x, point.y)) {
                return &toggle;
            }
        }
        return nullptr;
    }

    bool pointer(ChatTranscript& self, const MouseEvent& event) {
        const Toggle* hit = toggleAt(self, event.position);
        switch (event.type) {
            case MouseEventType::Move: {
                const std::optional<Key> next = hit ? std::optional<Key>(hit->key) : std::nullopt;
                if (next != hovered) {
                    hovered = next;
                    self.invalidate(Dirty::Layout);
                }
                return false;
            }
            case MouseEventType::Leave:
            case MouseEventType::Cancel:
                if (hovered || pressed) {
                    hovered.reset();
                    pressed.reset();
                    self.invalidate(Dirty::Layout);
                }
                return false;
            case MouseEventType::Click:
                if (!hit || event.button != MouseButton::Left) {
                    return false;
                }
                pressed = hit->key;
                hovered = hit->key;
                self.invalidate(Dirty::Layout);
                return true;
            case MouseEventType::Release: {
                if (event.button != MouseButton::Left || !pressed) {
                    return false;
                }
                const Key key = *pressed;
                pressed.reset();
                self.invalidate(Dirty::Layout);
                if (hit && hit->key == key) {
                    toggle(self, key);
                }
                return true;
            }
            default:
                return false;
        }
    }
};

ChatTranscript::ChatTranscript() {
    impl = std::make_unique<Impl>();
    setClipsChildren(true);
    impl->allocate();
    addChild(impl->bubbles);
    addChild(impl->scroll);
    for (auto& slot : impl->users) {
        addChild(slot->view);
    }
    for (auto& slot : impl->markdowns) {
        addChild(slot->view);
    }
    addChild(impl->highlights);
    addChild(impl->labels);
    addChild(impl->shimmer);
    addChild(impl->carets);
    for (auto& slot : impl->summaries) {
        addChild(slot->view);
    }
}

ChatTranscript::~ChatTranscript() = default;

void ChatTranscript::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    if (config.id != impl->config.id || config.snapshot.generation != impl->config.snapshot.generation) {
        impl->reset();
    }
    if (config.tailRequest != impl->config.tailRequest) {
        impl->followTail = true;
    }
    impl->config = std::move(config);
}

void ChatTranscript::layout(const Context& ctx) {
    auto& state = *impl;
    state.textMetrics.setWindow(ctx.render);
    const auto m = state.metrics(*this, ctx);
    state.sync(m);
    state.buildRows(*this, ctx, m);
    const F32 height = state.contentHeight(m);
    state.clampScroll(m, height);
    state.overscrollY = state.scroll.overscroll(ctx).y;

    auto placements = state.place(m);
    Impl::Limit(placements.users, state.users, state.followTail);
    Impl::Limit(placements.markdown, state.markdowns, state.followTail);
    Impl::Limit(placements.disclosures, state.summaries, state.followTail);
    const auto users = Impl::Assign(state.users, placements.users);
    const auto markdown = Impl::Assign(state.markdowns, placements.markdown);
    const auto disclosures = Impl::Assign(state.summaries, placements.disclosures);

    state.layoutScroll(*this, ctx, m, height);
    state.layoutUsers(*this, ctx, m, users);
    state.layoutMarkdown(*this, ctx, m, markdown);
    state.layoutDisclosures(*this, ctx, m, disclosures);
}

bool ChatTranscript::mouseEvent(const MouseEvent& event) {
    if (impl->pointer(*this, event)) {
        return true;
    }
    return mouseEventChildren(event);
}

std::optional<SurfaceCursor> ChatTranscript::cursorAt(const Extent2D<F32>& point) const {
    if (!frame().contains(point.x, point.y)) {
        return std::nullopt;
    }
    for (const auto& toggle : impl->toggles) {
        if (toggle.rect.contains(point.x, point.y)) {
            return SurfaceCursor::Hand;
        }
    }
    return std::nullopt;
}

}  // namespace Jetstream::Sakura
