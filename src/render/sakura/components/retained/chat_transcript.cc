#include "chat_transcript.hh"

#include <jetstream/logger.hh>
#include <jetstream/render/sakura/typography.hh>

#include "../../context.hh"
#include "../../retained/helpers.hh"
#include "../../retained/shapes.hh"

#include <algorithm>
#include <cmath>
#include <functional>

namespace Jetstream::Sakura::Retained {

namespace {

using Role = Chat::Role;
using PartKind = Chat::PartKind;
using Part = Chat::Part;

constexpr F32 kBubbleGap = 10.0f;
constexpr F32 kTurnGap = 16.0f;
constexpr F32 kPartGap = 6.0f;
constexpr F32 kBubbleMaxWidthRatio = 0.72f;
constexpr F32 kTranscriptOverscanLines = 2.0f;
constexpr U64 kUserSlotPoolSize = 8;
constexpr U64 kMarkdownSlotPoolSize = 12;
constexpr U64 kDisclosureSlotPoolSize = 12;
constexpr U64 kShimmerSteps = 48;
constexpr F32 kShimmerPeriod = 1.8f;
constexpr F32 kShimmerBandRatio = 0.35f;
constexpr F32 kShimmerDim = 0.55f;
constexpr F32 kUnbounded = std::numeric_limits<F32>::infinity();

std::vector<Box::Instance> ShimmerStrips(const Rect& zone,
                                         const F32 time,
                                         const ColorRGBA<F32>& color,
                                         const bool visible) {
    std::vector<Box::Instance> strips(kShimmerSteps);
    const F32 step = zone.width / static_cast<F32>(kShimmerSteps);
    const F32 band = std::max(1.0f, zone.width * kShimmerBandRatio);
    const F32 phase = std::fmod(time, kShimmerPeriod) / kShimmerPeriod;
    const F32 center = zone.x - band + phase * (zone.width + 2.0f * band);
    for (U64 k = 0; k < kShimmerSteps; ++k) {
        const F32 x = zone.x + (static_cast<F32>(k) + 0.5f) * step;
        const F32 distance = std::abs(x - center) / band;
        const F32 highlight = distance < 1.0f ? 0.5f * (1.0f + std::cos(3.14159265f * distance)) : 0.0f;
        auto tint = color;
        tint.a *= kShimmerDim * (1.0f - highlight);
        strips[k] = {
            .rect = {zone.x + static_cast<F32>(k) * step, zone.y, step, zone.height},
            .visible = visible,
            .backgroundColor = tint,
        };
    }
    return strips;
}

bool HasExpandablePart(const std::vector<Part>& parts, U64 key) {
    return std::any_of(parts.begin(), parts.end(), [key](const Part& part) {
        return part.kind == PartKind::Disclosure && part.expandable && part.key == key;
    });
}

template<typename Slot, typename Key>
bool HasSlot(const std::vector<std::unique_ptr<Slot>>& slots, const Key& key) {
    for (const auto& slot : slots) {
        if (slot->key == key) {
            return true;
        }
    }
    return false;
}

template<typename Placement, typename Key>
bool HasPlacement(const std::vector<Placement>& placements, const Key& key) {
    for (const auto& placement : placements) {
        if (placement.block->key == key) {
            return true;
        }
    }
    return false;
}

template<typename Placement, typename Slot>
void LimitPlacements(std::vector<Placement>& placements,
                     const std::vector<std::unique_ptr<Slot>>& slots,
                     U64 capacity,
                     bool preferTail) {
    if (placements.size() <= capacity) {
        return;
    }

    std::vector<bool> selected(placements.size(), false);
    U64 selectedCount = 0;
    for (U64 i = 0; i < placements.size() && selectedCount < capacity; ++i) {
        if (HasSlot(slots, placements[i].block->key)) {
            selected[i] = true;
            selectedCount++;
        }
    }

    for (U64 n = 0; n < placements.size() && selectedCount < capacity; ++n) {
        const U64 i = preferTail ? placements.size() - 1 - n : n;
        if (!selected[i]) {
            selected[i] = true;
            selectedCount++;
        }
    }

    std::vector<Placement> retained;
    retained.reserve(selectedCount);
    for (U64 i = 0; i < placements.size(); ++i) {
        if (selected[i]) {
            retained.push_back(std::move(placements[i]));
        }
    }
    placements = std::move(retained);
}

template<typename Slot, typename Placement>
std::vector<const Placement*> AssignSlots(std::vector<std::unique_ptr<Slot>>& slots,
                                          const std::vector<Placement>& placements,
                                          bool quarantineActive) {
    std::vector<const Placement*> bySlot(slots.size(), nullptr);
    std::vector<bool> placed(placements.size(), false);
    for (auto& slot : slots) {
        slot->identityChanged = false;
    }

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
            if (bySlot[s] || (quarantineActive && slots[s]->activeLastFrame)) {
                continue;
            }
            bySlot[s] = &placements[i];
            slots[s]->identityChanged = slots[s]->key != placements[i].block->key;
            slots[s]->key = placements[i].block->key;
            break;
        }
    }
    return bySlot;
}

}  // namespace

bool ChatMessageList::Metrics::inViewport(const Rect& rect) const {
    return visible && !rect.empty() && rect.bottom() > visibleTop && rect.y < visibleBottom;
}

bool ChatMessageList::Metrics::inBounds(const Rect& rect) const {
    return visible && !rect.empty() && rect.bottom() > bounds.y && rect.y < bounds.bottom();
}

bool ChatMessageList::Measurement::matches(const Block& block, F32 measuredAt,
                                          const Metrics& m) const {
    return revision == block.revision &&
           width == measuredAt &&
           fontSize == m.font &&
           unit == m.unit &&
           window == m.window &&
           expanded == block.expanded &&
           busy == block.busy;
}

void ChatMessageList::Measurement::store(const Block& block, F32 measuredAt, const Metrics& m) {
    revision = block.revision;
    width = measuredAt;
    fontSize = m.font;
    unit = m.unit;
    window = m.window;
    expanded = block.expanded;
    busy = block.busy;
}

ChatMessageList::ChatMessageList(const Chat::Config& config) : config(config) {
    setClipsChildren(true);
    add(bubbleBoxes);
    add(scroll);
    ensureSlots(userSlots, kUserSlotPoolSize);
    ensureSlots(markdownSlots, kMarkdownSlotPoolSize);
    ensureSlots(disclosureSlots, kDisclosureSlotPoolSize);
}

void ChatMessageList::reset() {
    resetScroll();
    measurements.clear();
    rowMessages.clear();
    expansions.clear();
    ++disclosureRevision;
}

void ChatMessageList::resetScroll() {
    scrollY = 0.0f;
    followTail = true;
}

void ChatMessageList::stickToTail() {
    followTail = true;
}

void ChatMessageList::toggleDisclosure(U64 expectedGeneration, U64 messageIndex, U64 partKey) {
    if (config.snapshot.generation != expectedGeneration ||
        messageIndex >= config.snapshot.messages.size() ||
        !HasExpandablePart(config.snapshot.messages[messageIndex].parts, partKey)) {
        return;
    }
    expansions[{messageIndex, partKey}] = !expanded(messageIndex, partKey);
    ++disclosureRevision;
    invalidate(Dirty::Paint);
}

bool ChatMessageList::expanded(U64 message, U64 part) const {
    const auto found = expansions.find({message, part});
    return found != expansions.end() && found->second;
}

void ChatMessageList::layout(const Context& ctx) {
    const Metrics m = metricsFor(ctx);
    resetCaches(m);
    buildRows(ctx, m);
    const F32 height = contentHeight(m);
    clampScroll(m, height);

    auto placements = place(m);
    LimitPlacements(placements.users, userSlots, userSlots.size(), followTail);
    LimitPlacements(placements.markdown, markdownSlots, markdownSlots.size(), followTail);
    LimitPlacements(placements.disclosures, disclosureSlots,
                    assignableDisclosures(placements.disclosures), followTail);

    const auto users = AssignSlots(userSlots, placements.users, false);
    const auto markdown = AssignSlots(markdownSlots, placements.markdown, false);
    const auto disclosure = AssignSlots(disclosureSlots, placements.disclosures, true);

    auto bubbles = bindUserSlots(ctx, m, users);
    bindMarkdownSlots(ctx, m, markdown);
    bindDisclosureSlots(ctx, m, disclosure);
    bindBubbles(ctx, m, std::move(bubbles));
    bindScroll(ctx, m, height);
}

ChatMessageList::Metrics ChatMessageList::metricsFor(const Context& ctx) const {
    const Rect bounds = frame();
    const F32 font = config.fontSize;
    const F32 unit = font / Typography::FontSize;
    const F32 lineHeight = font * Typography::CodeLineHeight;
    const F32 textPad = 9.0f * unit;
    const F32 scrollbarReserve = 14.0f * unit;
    const F32 contentWidth = std::max(1.0f, bounds.width - scrollbarReserve);
    const F32 maxBubbleWidth = std::max(1.0f, contentWidth * kBubbleMaxWidthRatio);
    const F32 overscan = kTranscriptOverscanLines * lineHeight;
    return {
        .bounds = bounds,
        .clip = Intersect(bounds, clip()),
        .visible = !bounds.empty(),
        .unit = unit,
        .font = font,
        .lineHeight = lineHeight,
        .textPad = textPad,
        .scrollbarReserve = scrollbarReserve,
        .contentWidth = contentWidth,
        .maxBubbleWidth = maxBubbleWidth,
        .maxUserTextWidth = std::max(1.0f, maxBubbleWidth - 2.0f * textPad),
        .partGap = kPartGap * unit,
        .summaryInset = 17.0f * unit,
        .caretInset = 8.0f * unit,
        .caret = {7.0f * unit, 4.0f * unit},
        .labelInset = 15.0f * unit,
        .visibleTop = bounds.y - overscan,
        .visibleBottom = bounds.bottom() + overscan,
        .window = ctx.render,
    };
}

std::string ChatMessageList::scopedId(std::string_view suffix) const {
    return jst::fmt::format("{}:{}", config.id, suffix);
}

std::string ChatMessageList::slotId(std::string_view element, U64 index) const {
    return jst::fmt::format("{}:{}-slot{}", config.id, element, index);
}

std::string ChatMessageList::blockId(const BlockKey& key, std::string_view element) const {
    return jst::fmt::format("{}:{}:{}:{}:{}{}", config.id,
                            key.generation, key.message, key.part,
                            static_cast<U64>(key.kind), element);
}

ChatMessageList::Block ChatMessageList::makeBlock(U64 message, U64 part, BlockKind kind) const {
    Block block;
    block.key = {
        .generation = config.snapshot.generation,
        .message = message,
        .part = part,
        .kind = kind,
    };
    return block;
}

void ChatMessageList::resetCaches(const Metrics& m) {
    const auto& snapshot = config.snapshot;
    if (!measurements.empty() && measurements.begin()->first.generation != snapshot.generation) {
        measurements.clear();
    }

    const RowContext next = {
        .generation = snapshot.generation,
        .contentWidth = m.contentWidth,
        .unit = m.unit,
        .window = m.window,
        .disclosureRevision = disclosureRevision,
    };
    if (rowContext != next) {
        rowMessages.clear();
        rowContext = next;
    }

    const U64 count = snapshot.messages.size();
    rowWidth.resize(count);
    rowHeight.resize(count);
    messageBlocks.resize(count);
    rowMessages.resize(count);
}

void ChatMessageList::buildRows(const Context& ctx, const Metrics& m) {
    const auto& messages = config.snapshot.messages;
    const U64 count = messages.size();
    for (U64 i = 0; i < count; ++i) {
        if (rowMessages[i] == messages.at(i)) {
            continue;
        }
        rowMessages[i] = messages.at(i);
        messageBlocks[i].clear();
        if (messages[i].role == Role::User) {
            buildUserRow(ctx, m, i);
        } else {
            buildAssistantRow(ctx, m, i);
        }
    }
}

void ChatMessageList::buildUserRow(const Context& ctx, const Metrics& m, U64 index) {
    Block block = makeBlock(index, 0, BlockKind::User);
    block.value = config.snapshot.messages[index].text;
    block.revision = std::hash<std::string>{}(block.value);

    auto& cached = measurements[block.key];
    if (!cached.matches(block, m.maxUserTextWidth, m)) {
        userMeasurer.update(userTextConfig(scopedId("measure-user"), block.value, m.font));
        const auto measured = measureChild(userMeasurer, ctx, {m.maxUserTextWidth, kUnbounded});
        cached.store(block, m.maxUserTextWidth, m);
        cached.measuredWidth = std::min(m.maxBubbleWidth, measured.x + 2.0f * m.textPad);
        cached.height = std::max(m.lineHeight, measured.y) + 2.0f * m.textPad;
    }

    block.width = cached.measuredWidth;
    block.height = cached.height;
    rowWidth[index] = block.width;
    rowHeight[index] = block.height;
    messageBlocks[index].push_back(std::move(block));
}

void ChatMessageList::buildAssistantRow(const Context& ctx, const Metrics& m, U64 index) {
    auto& blocks = messageBlocks[index];
    for (const auto& part : config.snapshot.messages[index].parts) {
        if (part.kind == PartKind::Disclosure) {
            addDisclosureBlock(ctx, m, index, part);
        } else {
            addMarkdownBlock(ctx, m, index, part);
        }
    }

    F32 height = 0.0f;
    for (U64 b = 0; b < blocks.size(); ++b) {
        height += blocks[b].height + (b > 0 ? m.partGap : 0.0f);
    }
    rowWidth[index] = m.contentWidth;
    rowHeight[index] = height;
}

void ChatMessageList::addMarkdownBlock(const Context& ctx, const Metrics& m,
                                     U64 index, const Part& part) {
    Block block = makeBlock(index, part.key, BlockKind::Markdown);
    block.revision = part.revision;
    block.value = part.text;

    auto& cached = measurements[block.key];
    if (!cached.matches(block, m.contentWidth, m)) {
        markdownMeasurer.update(markdownConfig(
            scopedId("measure-markdown"), block.value, m.font, false));
        cached.store(block, m.contentWidth, m);
        cached.height = std::max(m.lineHeight,
                                 measureChild(markdownMeasurer, ctx, {m.contentWidth, kUnbounded}).y);
    }

    block.height = cached.height;
    messageBlocks[index].push_back(std::move(block));
}

void ChatMessageList::addDisclosureBlock(const Context& ctx, const Metrics& m,
                                       U64 index, const Part& part) {
    Block block = makeBlock(index, part.key, BlockKind::Disclosure);
    block.revision = part.revision;
    block.expanded = part.expandable && expanded(index, part.key);
    block.busy = part.busy;
    block.toggleLabel = part.label;
    block.summary = part.summary;
    if (block.expanded) {
        block.value = part.text;
    }

    measureDisclosure(ctx, m, block);
    messageBlocks[index].push_back(std::move(block));
}

void ChatMessageList::measureDisclosure(const Context& ctx, const Metrics& m, Block& block) {
    auto& cached = measurements[block.key];
    if (!cached.matches(block, m.contentWidth, m)) {
        toggleMeasurer.update(toggleConfig(scopedId("measure-disclosure-toggle"), block.toggleLabel, m, false));
        cached.toggleSize = measureChild(toggleMeasurer, ctx, {m.contentWidth, kUnbounded});
        cached.summaryHeight = 0.0f;
        cached.bodyHeight = 0.0f;
        cached.height = cached.toggleSize.y;

        if (!block.summary.empty()) {
            summaryMeasurer.update(summaryTextConfig(
                scopedId("measure-disclosure-summary"), block.summary, m.font));
            cached.summaryHeight = measureChild(summaryMeasurer, ctx,
                                                {m.contentWidth - m.summaryInset, kUnbounded}).y;
            cached.height += 2.0f * m.unit + cached.summaryHeight;
        }
        if (block.expanded) {
            markdownMeasurer.update(markdownConfig(
                scopedId("measure-disclosure-body"), block.value, m.font, true));
            cached.bodyHeight = measureChild(markdownMeasurer, ctx, {m.contentWidth, kUnbounded}).y;
            cached.height += m.partGap + cached.bodyHeight;
        }
        cached.store(block, m.contentWidth, m);
    }

    block.toggleSize = cached.toggleSize;
    block.summaryHeight = cached.summaryHeight;
    block.bodyHeight = cached.bodyHeight;
    block.height = cached.height;
}

F32 ChatMessageList::gapAfter(const Metrics& m, U64 index) const {
    const auto& messages = config.snapshot.messages;
    const bool turnChange = index + 1 < messages.size() &&
                            messages[index + 1].role != messages[index].role;
    return (turnChange ? kTurnGap : kBubbleGap) * m.unit;
}

F32 ChatMessageList::contentHeight(const Metrics& m) const {
    const U64 count = config.snapshot.messages.size();
    F32 height = 0.0f;
    for (U64 i = 0; i < count; ++i) {
        height += rowHeight[i];
        if (i + 1 < count) {
            height += gapAfter(m, i);
        }
    }
    return height;
}

void ChatMessageList::clampScroll(const Metrics& m, F32 height) {
    const F32 maxScroll = std::max(0.0f, height - m.bounds.height);
    lastMaxScroll = maxScroll;
    if (followTail) {
        scrollY = maxScroll;
    }
    scrollY = std::clamp(scrollY, 0.0f, maxScroll);
}

ChatMessageList::Placements ChatMessageList::place(const Metrics& m) {
    Placements placements;
    const auto& messages = config.snapshot.messages;
    const U64 count = messages.size();
    F32 y = m.bounds.y - scrollY;
    for (U64 i = 0; i < count; ++i) {
        if (messages[i].role == Role::User) {
            placeUser(m, i, y, placements);
        } else {
            placeAssistant(m, i, y, placements);
        }
        y += rowHeight[i];
        if (i + 1 < count) {
            y += gapAfter(m, i);
        }
    }
    return placements;
}

void ChatMessageList::placeUser(const Metrics& m, U64 index, F32 y, Placements& placements) {
    const Rect bubble = {
        m.bounds.right() - m.scrollbarReserve - rowWidth[index],
        y,
        rowWidth[index],
        rowHeight[index],
    };
    if (!m.inViewport(bubble)) {
        return;
    }
    placements.users.push_back({
        .block = &messageBlocks[index].front(),
        .bubble = bubble,
        .text = {
            bubble.x + m.textPad,
            bubble.y + m.textPad,
            std::max(0.0f, bubble.width - m.textPad),
            std::max(0.0f, bubble.height - 2.0f * m.textPad),
        },
    });
}

void ChatMessageList::placeAssistant(const Metrics& m, U64 index, F32 y, Placements& placements) {
    auto& blocks = messageBlocks[index];
    const F32 width = rowWidth[index];
    for (U64 b = 0; b < blocks.size(); ++b) {
        auto& block = blocks[b];
        if (b > 0) {
            y += m.partGap;
        }
        if (block.key.kind == BlockKind::Disclosure) {
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

F32 ChatMessageList::placeDisclosure(const Metrics& m, Block& block, F32 width,
                                    F32 y, Placements& placements) {
    const Rect toggleRect = {m.bounds.x, y, block.toggleSize.x, block.toggleSize.y};
    y += block.toggleSize.y;

    Rect summaryRect;
    if (block.summaryHeight > 0.0f) {
        y += 2.0f * m.unit;
        summaryRect = {
            m.bounds.x + m.summaryInset,
            y,
            std::max(0.0f, width - m.summaryInset),
            block.summaryHeight,
        };
        y += block.summaryHeight;
    }

    const Rect chromeRect = {
        toggleRect.x,
        toggleRect.y,
        std::max(toggleRect.width, summaryRect.right() - toggleRect.x),
        std::max(toggleRect.height, summaryRect.bottom() - toggleRect.y),
    };
    if (m.inBounds(chromeRect)) {
        placements.disclosures.push_back({
            .block = &block,
            .toggle = Intersect(toggleRect, m.bounds),
            .caret = toggleRect,
            .summary = summaryRect,
        });
    }

    if (!block.expanded) {
        return y;
    }
    y += m.partGap;
    const Rect bodyRect = {m.bounds.x, y, width, block.bodyHeight};
    if (m.inViewport(bodyRect)) {
        placements.markdown.push_back({.block = &block, .rect = bodyRect});
    }
    return y + block.bodyHeight;
}

U64 ChatMessageList::assignableDisclosures(const std::vector<DisclosurePlacement>& placements) const {
    U64 matched = 0;
    for (const auto& placement : placements) {
        matched += HasSlot(disclosureSlots, placement.block->key) ? 1 : 0;
    }
    U64 idle = 0;
    for (const auto& slot : disclosureSlots) {
        const bool placed = slot->key.has_value() && HasPlacement(placements, slot->key.value());
        if (!slot->activeLastFrame && !placed) {
            idle++;
        }
    }
    return std::min<U64>(placements.size(), matched + idle);
}

std::vector<Box::Instance> ChatMessageList::bindUserSlots(const Context& ctx, const Metrics& m,
                                                          const std::vector<const UserPlacement*>& bySlot) {
    std::vector<Box::Instance> bubbles;
    bubbles.reserve(bySlot.size());
    for (U64 i = 0; i < userSlots.size(); ++i) {
        auto& slot = *userSlots[i];
        const auto* placement = bySlot[i];
        if (!placement) {
            slot.view.update(userTextConfig(slotId("idle-user", i), "", m.font));
            layoutChild(ctx, slot.view, {});
            continue;
        }
        if (slot.identityChanged) {
            slot.view.update(userTextConfig(slotId("reset-user", i), "", m.font));
        }
        slot.view.update(userTextConfig(blockId(placement->block->key, ":user"),
                                        placement->block->value, m.font));
        layoutChild(ctx, slot.view, placement->text);
        bubbles.push_back({
            .rect = placement->bubble,
            .visible = m.visible,
            .backgroundColor = ctx.color("chat_user_bubble"),
        });
    }
    return bubbles;
}

void ChatMessageList::bindMarkdownSlots(const Context& ctx, const Metrics& m,
                                        const std::vector<const MarkdownPlacement*>& bySlot) {
    for (U64 i = 0; i < markdownSlots.size(); ++i) {
        auto& slot = *markdownSlots[i];
        const auto* placement = bySlot[i];
        if (!placement) {
            slot.view.update(markdownConfig(slotId("idle-markdown", i), "", m.font, false));
            layoutChild(ctx, slot.view, {});
            continue;
        }
        const bool disclosure = placement->block->key.kind == BlockKind::Disclosure;
        if (slot.identityChanged) {
            slot.view.update(markdownConfig(slotId("reset-markdown", i), "", m.font, disclosure));
        }
        slot.view.update(markdownConfig(
            blockId(placement->block->key, ":markdown"), placement->block->value, m.font, disclosure));
        layoutChild(ctx, slot.view, placement->rect);
    }
}

void ChatMessageList::bindDisclosureSlots(const Context& ctx, const Metrics& m,
                                          const std::vector<const DisclosurePlacement*>& bySlot) {
    for (U64 i = 0; i < disclosureSlots.size(); ++i) {
        if (bySlot[i]) {
            bindDisclosureSlot(ctx, m, *disclosureSlots[i], i, *bySlot[i]);
        } else {
            bindIdleDisclosureSlot(ctx, m, *disclosureSlots[i], i);
        }
    }
    for (U64 i = 0; i < disclosureSlots.size(); ++i) {
        disclosureSlots[i]->activeLastFrame = bySlot[i] != nullptr;
    }
}

void ChatMessageList::bindIdleDisclosureSlot(const Context& ctx, const Metrics& m,
                                            DisclosureSlot& slot, U64 index) {
    slot.toggle.update({
        .id = slotId("idle-disclosure-toggle", index),
        .str = "",
        .disabled = true,
        .fontSize = m.font,
        .fontName = "default_body_bold",
    });
    slot.caret.update({
        .id = slotId("idle-disclosure-caret", index),
        .instances = {},
        .clip = m.clip,
        .shape = Box::Shape::Triangle,
        .capacity = 1,
    });
    slot.summary.update(summaryTextConfig(slotId("idle-disclosure-summary", index), "", m.font));
    slot.shimmer.update({
        .id = slotId("idle-disclosure-shimmer", index),
        .instances = {},
        .clip = m.clip,
        .capacity = kShimmerSteps,
    });
    layoutChild(ctx, slot.toggle, {});
    layoutChild(ctx, slot.caret, {});
    layoutChild(ctx, slot.summary, {});
    layoutChild(ctx, slot.shimmer, {});
}

void ChatMessageList::bindDisclosureSlot(const Context& ctx, const Metrics& m,
                                        DisclosureSlot& slot, U64 index,
                                         const DisclosurePlacement& placement) {
    const Block& block = *placement.block;
    const BlockKey key = block.key;
    if (slot.identityChanged) {
        slot.summary.update(summaryTextConfig(
            slotId("reset-disclosure-summary", index), "", m.font));
    }

    const Rect caretRect = {
        m.bounds.x + m.caretInset,
        placement.caret.y,
        m.caret.length,
        placement.caret.height,
    };
    slot.caret.update({
        .id = blockId(key, ":disclosure-caret"),
        .instances = {Caret(caretRect, block.expanded, m.caret, ctx.color("text_secondary"), m.visible)},
        .clip = m.clip,
        .shape = Box::Shape::Triangle,
        .capacity = 1,
    });
    layoutChild(ctx, slot.caret, caretRect);

    auto toggle = toggleConfig(blockId(key, ":disclosure-toggle"), block.toggleLabel, m, block.busy);
    toggle.onClick = [this, key]() {
        toggleDisclosure(key.generation, key.message, key.part);
    };
    slot.toggle.update(std::move(toggle));
    layoutChild(ctx, slot.toggle, placement.toggle);

    const Rect shimmerRect = {
        placement.caret.x + m.labelInset,
        placement.caret.y,
        std::max(0.0f, placement.caret.width - m.labelInset),
        placement.caret.height,
    };
    const F32 time = std::chrono::duration<F32>(std::chrono::steady_clock::now() - epoch).count();
    slot.shimmer.update({
        .id = blockId(key, ":disclosure-shimmer"),
        .instances = block.busy ? ShimmerStrips(
            shimmerRect, time, ctx.color(config.surfaceColorKey), m.visible)
                                    : std::vector<Box::Instance>{},
        .clip = Intersect(placement.toggle, m.clip),
        .capacity = kShimmerSteps,
    });
    layoutChild(ctx, slot.shimmer, shimmerRect);

    slot.summary.update(summaryTextConfig(
        blockId(key, ":disclosure-summary"), block.summary, m.font));
    layoutChild(ctx, slot.summary, placement.summary);
}

void ChatMessageList::bindBubbles(const Context& ctx, const Metrics& m,
                                 std::vector<Box::Instance> bubbles) {
    bubbleBoxes.update({
        .id = scopedId("bubbles"),
        .instances = std::move(bubbles),
        .clip = m.clip,
        .cornerRadius = m.lineHeight * 0.5f + m.textPad,
        .borderWidth = 1.0f * m.unit,
        .borderColor = ctx.color("chat_user_bubble_outline"),
        .capacity = std::max<U64>(1, userSlots.size()),
    });
    layoutChild(ctx, bubbleBoxes, m.bounds);
}

void ChatMessageList::bindScroll(const Context& ctx, const Metrics& m, F32 height) {
    scroll.update({
        .id = scopedId("scroll"),
        .contentHeight = height,
        .scrollY = scrollY,
        .scrollbar = m.visible,
        .wheelStep = m.lineHeight * 3.0f,
        .thickness = 5.0f * m.unit,
        .margin = 3.0f * m.unit,
        .trackColorKey = "editor_scrollbar_track",
        .thumbColorKey = "editor_scrollbar_thumb",
        .onScrollY = [this](F32 next) {
            scrollY = next;
            followTail = next >= lastMaxScroll - 1.0f;
        },
    });
    layoutChild(ctx, scroll, m.bounds);
}

TextView::Config ChatMessageList::userTextConfig(std::string id, std::string value, F32 font) {
    return {
        .id = std::move(id),
        .value = std::move(value),
        .fontSize = font,
        .fontName = "default_body",
        .monospace = false,
        .scrollbar = false,
        .wrap = TextGrid::Wrap::Word,
        .textColorKey = "action_btn_text",
    };
}

TextView::Config ChatMessageList::summaryTextConfig(std::string id, std::string value, F32 font) {
    return {
        .id = std::move(id),
        .value = std::move(value),
        .fontSize = font,
        .fontName = "default_body_italic",
        .monospace = false,
        .scrollbar = false,
        .wrap = TextGrid::Wrap::Word,
        .textColorKey = "text_secondary",
    };
}

TextMarkdown::Config ChatMessageList::markdownConfig(
    std::string id, std::string value, F32 font, bool disclosure) {
    return {
        .id = std::move(id),
        .value = std::move(value),
        .fontSize = font,
        .textColorKey = disclosure ? "editor_text" : "text_primary",
    };
}

Button::Config ChatMessageList::toggleConfig(
    std::string id, std::string label, const Metrics& m, bool busy) {
    return {
        .id = std::move(id),
        .str = std::move(label),
        .colorKey = "transparent",
        .hoveredColorKey = busy ? "transparent" : "button_hovered",
        .activeColorKey = busy ? "transparent" : "button_active",
        .borderColorKey = "transparent",
        .textColorKey = "text_primary",
        .fontSize = m.font,
        .fontName = "default_body_bold",
        .cornerRadius = 6.0f * m.unit,
        .labelInsetLeft = m.labelInset,
    };
}

}  // namespace Jetstream::Sakura::Retained
