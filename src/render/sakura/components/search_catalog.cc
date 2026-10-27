#include <jetstream/render/sakura/components/search_catalog.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/button.hh>
#include <jetstream/render/sakura/components/card.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/components/markdown.hh>
#include <jetstream/render/sakura/components/scroll_area.hh>
#include <jetstream/render/sakura/components/stack.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/sakura/typography.hh>
#include <jetstream/render/tools/icons_lucide.hh>

#include "../context.hh"
#include "../detail/text_metrics.hh"
#include "../metrics.hh"
#include "tools/text.hh"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

namespace {

constexpr const char* kTitleFont = Typography::DisplaySemiboldFont;
constexpr const char* kBodyFont = Typography::DisplayRegularFont;
constexpr const char* kSummaryFont = Typography::BodyFont;
constexpr const char* kAllLabel = "All";
constexpr const char* kActionLabel = "Add on";
constexpr F32 kCardRadius = kPanelRounding;
constexpr F32 kCardGap = 8.0f;
constexpr F32 kCardPadding = 6.0f;
constexpr F32 kCardShadowSoftness = 8.0f;
constexpr F32 kCardShadowDrop = 3.0f;
constexpr F32 kHeaderHeight = 24.0f;
constexpr F32 kRowHeight = 48.0f;
constexpr F32 kRowPadX = 12.0f;
constexpr F32 kRowRadius = 8.0f;
constexpr F32 kRowTop = 8.0f;
constexpr F32 kTitleSize = 16.0f;
constexpr F32 kSummarySize = 14.0f;
constexpr F32 kLineGap = 3.0f;
constexpr F32 kHeaderSize = 12.0f;
constexpr F32 kRowIcon = 16.0f;
constexpr F32 kRowIconGap = 10.0f;
constexpr F32 kDocsMinHeight = 360.0f;
constexpr F32 kDocsPadding = 10.0f;
constexpr F32 kDocsFooterPadding = 10.0f;
constexpr F32 kDocsFooterGap = 10.0f;
constexpr U64 kRowCapacity = 64;
constexpr F32 kChipHeight = 24.0f;
constexpr F32 kChipPadX = 10.0f;
constexpr F32 kChipGap = 6.0f;
constexpr F32 kChipSize = 12.0f;

std::string Upper(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return value;
}

U64 Capacity(U64 needed) {
    U64 capacity = kRowCapacity;
    while (capacity < needed) {
        capacity *= 2;
    }
    return capacity;
}

struct ResultList : public Component {
    struct Row {
        bool header = false;
        std::string label;
        U64 result = 0;

        bool operator==(const Row&) const = default;
    };

    struct Config {
        std::vector<Row> rows;
        const std::vector<SearchCatalog::Item>* items = nullptr;
        U64 revision = 0;
        std::vector<U64> order;
        U64 selected = 0;
        std::string emptyLabel;
        Callback<void(U64)> onSelect;
        Callback<void(const std::string&, std::optional<U64>)> onActivate;

        bool operator==(const Config&) const = default;
    };

    struct Placed {
        Rect rect;
        U64 result = 0;
    };

    Config config;
    F32 scroll = 0.0f;
    F32 maxScroll = 0.0f;
    bool reveal = true;
    std::optional<U64> hovered;
    std::vector<Placed> placed;
    Rect bounds;
    F32 pixelRatio = 1.0f;
    TextMetrics metrics;
    Box rowBoxes;
    Label headers;
    Label icons;
    Label titles;
    Label summaries;
    Label empty;

    ResultList() {
        setClipsChildren(true);
        addChild(rowBoxes);
        addChild(headers);
        addChild(icons);
        addChild(titles);
        addChild(summaries);
        addChild(empty);
    }

    void update(Config next) {
        if (next != config) {
            invalidate(Dirty::Layout);
        }
        if (next.selected != config.selected || next.order != config.order) {
            reveal = true;
        }
        if (next.order != config.order) {
            hovered.reset();
        }
        config = std::move(next);
        markConfigured();
    }

    F32 rowHeight(const Row& row) const {
        return (row.header ? kHeaderHeight : kRowHeight) * pixelRatio;
    }

    F32 contentHeight() const {
        F32 height = 0.0f;
        for (const auto& row : config.rows) {
            height += rowHeight(row);
        }
        return height;
    }

    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override {
        pixelRatio = ctx.pixelRatio;
        const F32 height = config.rows.empty() ? Px(ctx, kRowHeight) : contentHeight();
        return {FillLength(available.x), FitLength(height, available.y)};
    }

    void layout(const Context& ctx) override {
        pixelRatio = ctx.pixelRatio;
        metrics.setWindow(ctx.render);
        bounds = frame();
        const F32 total = contentHeight();
        maxScroll = std::max(0.0f, total - bounds.height);
        if (reveal) {
            reveal = false;
            F32 y = 0.0f;
            for (const auto& row : config.rows) {
                const F32 height = rowHeight(row);
                if (!row.header && row.result == config.selected) {
                    if (y < scroll) {
                        scroll = y;
                    } else if (y + height > scroll + bounds.height) {
                        scroll = y + height - bounds.height;
                    }
                    break;
                }
                y += height;
            }
        }
        scroll = std::clamp(scroll, 0.0f, maxScroll);

        const F32 padX = Px(ctx, kRowPadX);
        const F32 titleSize = Px(ctx, kTitleSize);
        const F32 summarySize = Px(ctx, kSummarySize);
        const F32 iconSize = Px(ctx, kRowIcon);
        const F32 inset = Px(ctx, 4.0f);
        std::vector<Box::Instance> rowInstances;
        std::vector<Label::Instance> headerInstances;
        std::vector<Label::Instance> iconInstances;
        std::vector<Label::Instance> titleInstances;
        std::vector<Label::Instance> summaryInstances;
        placed.clear();

        F32 y = bounds.y - scroll;
        for (const auto& row : config.rows) {
            const F32 height = rowHeight(row);
            const Rect rect = {bounds.x, y, bounds.width, height};
            y += height;
            if (rect.bottom() < bounds.y || rect.y > bounds.bottom()) {
                continue;
            }
            if (row.header) {
                headerInstances.push_back({
                    .rect = {rect.x + padX, rect.y + Px(ctx, 8.0f), std::max(0.0f, rect.width - 2.0f * padX),
                             Px(ctx, kHeaderSize)},
                    .str = Upper(row.label),
                    .color = ctx.color("text_secondary"),
                    .fontSize = Px(ctx, kHeaderSize),
                    .alignment = {0, 1},
                });
                continue;
            }
            if (!config.items || row.result >= config.order.size() || config.order[row.result] >= config.items->size()) {
                continue;
            }
            const auto& item = (*config.items)[config.order[row.result]];
            const Rect tile = {rect.x + inset, rect.y + Px(ctx, 1.0f), std::max(0.0f, rect.width - 2.0f * inset),
                               rect.height - Px(ctx, 2.0f)};
            const bool selected = row.result == config.selected;
            if (selected || hovered == row.result) {
                rowInstances.push_back({
                    .rect = tile,
                    .backgroundColor = ctx.color(selected ? "pill_active_bg" : "pill_hover_bg"),
                });
            }

            const Placed place{.rect = tile, .result = row.result};
            const F32 titleY = rect.y + Px(ctx, kRowTop);
            F32 textX = tile.x + padX - inset;
            if (!item.icon.empty()) {
                iconInstances.push_back({
                    .rect = {textX, tile.y, iconSize, tile.height},
                    .str = item.icon,
                    .color = ctx.color(selected ? "text_primary" : "text_secondary"),
                    .fontSize = iconSize,
                    .alignment = {1, 1},
                });
                textX += iconSize + Px(ctx, kRowIconGap);
            }
            const F32 textWidth = std::max(0.0f, tile.right() - padX + inset - textX);
            titleInstances.push_back({
                .rect = {textX, titleY, textWidth, titleSize},
                .str = metrics.ellipsize(kTitleFont, item.title, titleSize, textWidth),
                .color = ctx.color("text_primary"),
                .fontSize = titleSize,
                .alignment = {0, 1},
            });
            summaryInstances.push_back({
                .rect = {textX, titleY + titleSize + Px(ctx, kLineGap), textWidth, summarySize},
                .str = metrics.ellipsize(kSummaryFont, item.summary, summarySize, textWidth),
                .color = ctx.color("text_secondary"),
                .fontSize = summarySize,
                .alignment = {0, 1},
            });
            placed.push_back(place);
        }

        const F32 viewport = static_cast<F32>(ctx.framebufferSize.y);
        const U64 capacity = Capacity(static_cast<U64>(std::ceil(viewport / Px(ctx, kHeaderHeight))) + 2);
        rowBoxes.update({.instances = std::move(rowInstances), .clip = bounds, .cornerRadius = Px(ctx, kRowRadius),
                         .capacity = 4});
        headers.update({.instances = std::move(headerInstances), .clip = bounds, .fontName = kTitleFont,
                        .capacity = capacity});
        icons.update({.instances = std::move(iconInstances), .clip = bounds, .fontName = Typography::IconFont,
                      .maxCharacters = 4, .capacity = capacity});
        titles.update({.instances = std::move(titleInstances), .clip = bounds, .fontName = kTitleFont,
                       .capacity = capacity});
        summaries.update({.instances = std::move(summaryInstances), .clip = bounds, .fontName = kSummaryFont,
                          .capacity = capacity});
        empty.update({
            .instances = {{
                .rect = bounds,
                .str = config.emptyLabel,
                .visible = config.rows.empty(),
                .color = ctx.color("text_secondary"),
                .fontSize = Px(ctx, kSummarySize),
                .alignment = {1, 1},
            }},
            .clip = bounds,
            .fontName = kBodyFont,
        });
        for (auto* part : std::initializer_list<Component*>{&rowBoxes, &headers, &icons, &titles, &summaries,
                                                            &empty}) {
            layoutChild(ctx, *part, bounds);
        }
    }

    const Placed* at(const Extent2D<F32>& point) const {
        if (!bounds.contains(point.x, point.y)) {
            return nullptr;
        }
        for (const auto& place : placed) {
            if (place.rect.contains(point.x, point.y)) {
                return &place;
            }
        }
        return nullptr;
    }

    bool mouseEvent(const MouseEvent& event) override {
        const Extent2D<F32> point = event.position;
        switch (event.type) {
            case MouseEventType::Move: {
                const auto* place = at(point);
                const std::optional<U64> next = place ? std::optional<U64>(place->result) : std::nullopt;
                if (next != hovered) {
                    hovered = next;
                    invalidate(Dirty::Layout);
                }
                return place != nullptr;
            }
            case MouseEventType::Click: {
                const auto* place = at(point);
                if (!place || event.button != MouseButton::Left) {
                    return bounds.contains(point.x, point.y);
                }
                const std::string id = config.items && place->result < config.order.size()
                                           ? (*config.items)[config.order[place->result]].id
                                           : std::string();
                if (event.clicks >= 2) {
                    if (config.onActivate) {
                        config.onActivate(id, std::nullopt);
                    }
                    return true;
                }
                if (config.onSelect) {
                    config.onSelect(place->result);
                }
                return true;
            }
            case MouseEventType::Scroll: {
                if (!bounds.contains(point.x, point.y) || maxScroll <= 0.0f) {
                    return false;
                }
                const F32 step = WheelStep(event, kWheelStep * pixelRatio, pixelRatio);
                const F32 next = std::clamp(scroll - event.scroll.y * step, 0.0f, maxScroll);
                if (next != scroll) {
                    scroll = next;
                    invalidate(Dirty::Layout);
                }
                return true;
            }
            case MouseEventType::Leave:
            case MouseEventType::Cancel:
                if (hovered.has_value()) {
                    hovered.reset();
                    invalidate(Dirty::Layout);
                }
                return false;
            default:
                return bounds.contains(point.x, point.y);
        }
    }

    std::optional<SurfaceCursor> cursorAt(const Extent2D<F32>& point) const override {
        if (at(point)) {
            return SurfaceCursor::Hand;
        }
        return std::nullopt;
    }
};

struct ChipStrip : public Component {
    struct Config {
        std::vector<std::string> labels;
        U64 active = 0;
        Callback<void(U64)> onPick;

        bool operator==(const Config&) const = default;
    };

    Config config;
    TextMetrics metrics;
    std::vector<Rect> rects;
    std::optional<U64> hovered;
    Box fills;
    Box selection;
    Label texts;

    ChipStrip() {
        addChild(fills);
        addChild(selection);
        addChild(texts);
    }

    void update(Config next) {
        if (next != config) {
            invalidate(Dirty::Layout);
        }
        if (next.labels != config.labels) {
            hovered.reset();
        }
        config = std::move(next);
        markConfigured();
    }

    std::vector<Rect> arrange(const Context& ctx, F32 x, F32 y, F32 width) {
        metrics.setWindow(ctx.render);
        const F32 height = Px(ctx, kChipHeight);
        const F32 gap = Px(ctx, kChipGap);
        std::vector<Rect> placed;
        F32 cursor = x;
        F32 row = y;
        for (const auto& label : config.labels) {
            const F32 chip = std::ceil(metrics.measure(kBodyFont, label, Px(ctx, kChipSize)) + 2.0f * Px(ctx, kChipPadX));
            if (cursor > x && cursor + chip > x + width) {
                cursor = x;
                row += height + gap;
            }
            placed.push_back({cursor, row, chip, height});
            cursor += chip + gap;
        }
        return placed;
    }

    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override {
        const F32 width = FillLength(available.x);
        if (config.labels.empty()) {
            return {width, 0.0f};
        }
        const auto placed = arrange(ctx, 0.0f, 0.0f, width);
        return {width, FitLength(placed.back().bottom(), available.y)};
    }

    void layout(const Context& ctx) override {
        const Rect bounds = frame();
        rects = arrange(ctx, bounds.x, bounds.y, bounds.width);
        std::vector<Box::Instance> fillInstances;
        std::vector<Box::Instance> selectionInstances;
        std::vector<Label::Instance> textInstances;
        for (U64 index = 0; index < rects.size(); ++index) {
            const bool active = index == config.active;
            const char* fill = active ? "pill_active_bg" : hovered == index ? "pill_hover_bg" : "pill_bg";
            fillInstances.push_back({.rect = rects[index], .backgroundColor = ctx.color(fill)});
            if (active) {
                selectionInstances.push_back({.rect = rects[index], .backgroundColor = {0.0f, 0.0f, 0.0f, 0.0f}});
            }
            textInstances.push_back({
                .rect = rects[index],
                .str = config.labels[index],
                .color = ctx.color(active ? "text_primary" : "text_secondary"),
                .fontSize = Px(ctx, kChipSize),
                .alignment = {1, 1},
            });
        }
        const F32 radius = Px(ctx, kChipHeight) * 0.5f;
        fills.update({.instances = std::move(fillInstances), .cornerRadius = radius, .capacity = 16});
        selection.update({.instances = std::move(selectionInstances), .cornerRadius = radius,
                          .borderWidth = Px(ctx, 1.0f), .borderColor = ctx.color("tile_border"), .strokeFringe = true,
                          .capacity = 1});
        texts.update({.instances = std::move(textInstances), .fontName = kBodyFont,
                      .maxCharacters = 32, .capacity = 16});
        for (auto* part : std::initializer_list<Component*>{&fills, &selection, &texts}) {
            layoutChild(ctx, *part, bounds);
        }
    }

    std::optional<U64> at(const Extent2D<F32>& point) const {
        for (U64 index = 0; index < rects.size(); ++index) {
            if (rects[index].contains(point.x, point.y)) {
                return index;
            }
        }
        return std::nullopt;
    }

    bool mouseEvent(const MouseEvent& event) override {
        switch (event.type) {
            case MouseEventType::Move: {
                const auto next = at(event.position);
                if (next != hovered) {
                    hovered = next;
                    invalidate(Dirty::Layout);
                }
                return next.has_value();
            }
            case MouseEventType::Click: {
                const auto chip = at(event.position);
                if (chip.has_value() && event.button == MouseButton::Left && config.onPick) {
                    config.onPick(*chip);
                }
                return chip.has_value();
            }
            case MouseEventType::Leave:
            case MouseEventType::Cancel:
                if (hovered.has_value()) {
                    hovered.reset();
                    invalidate(Dirty::Layout);
                }
                return false;
            default:
                return at(event.position).has_value();
        }
    }

    std::optional<SurfaceCursor> cursorAt(const Extent2D<F32>& point) const override {
        if (at(point).has_value()) {
            return SurfaceCursor::Hand;
        }
        return std::nullopt;
    }
};

}  // namespace

struct SearchCatalog::Impl {
    Config config;
    std::vector<U64> order;
    std::vector<std::string> ids;
    Card cards;
    ChipStrip chips;
    std::vector<std::string> groups;
    bool chipsShown = false;
    ResultList list;
    std::vector<std::unique_ptr<Button>> docsButtons;
    Stack docsButtonRow;
    Markdown docsMarkdown;
    ScrollArea docsScroll;
    Box docsDivider;
    Text docsFooterLabel;
    bool docs = false;
    bool docsFit = true;
    U64 itemsRevision = 0;
    std::vector<Component*> mounted;
    Rect listCard;
    Rect docsCard;

    F32 chipsHeight(SearchCatalog& self, const Context& ctx, F32 width) {
        if (!chipsShown) {
            return 0.0f;
        }
        const F32 inset = kChipGap * ctx.pixelRatio;
        const F32 inner = std::max(0.0f, width - 2.0f * inset);
        return self.measureChild(chips, ctx, {inner, std::numeric_limits<F32>::infinity()}).y + 2.0f * inset;
    }

    F32 listHeight(SearchCatalog& self, const Context& ctx, F32 width) {
        const F32 padding = kCardPadding * ctx.pixelRatio;
        const F32 inner = std::max(0.0f, width - 2.0f * padding);
        return chipsHeight(self, ctx, inner) +
               self.measureChild(list, ctx, {inner, std::numeric_limits<F32>::infinity()}).y;
    }
};

SearchCatalog::SearchCatalog() {
    impl = std::make_unique<Impl>();
}

SearchCatalog::~SearchCatalog() = default;

std::vector<std::string> SearchCatalog::Groups(const std::vector<Item>& items, const std::vector<std::string>& groups) {
    std::vector<std::string> sequence;
    for (const auto& group : groups) {
        const bool used = std::any_of(items.begin(), items.end(), [&](const Item& item) {
            return item.group == group;
        });
        if (used && std::find(sequence.begin(), sequence.end(), group) == sequence.end()) {
            sequence.push_back(group);
        }
    }
    for (const auto& item : items) {
        if (!item.group.empty() && std::find(sequence.begin(), sequence.end(), item.group) == sequence.end()) {
            sequence.push_back(item.group);
        }
    }
    const bool unnamed = std::any_of(items.begin(), items.end(), [](const Item& item) {
        return item.group.empty();
    });
    if (unnamed && std::find(sequence.begin(), sequence.end(), std::string()) == sequence.end()) {
        sequence.push_back({});
    }
    return sequence;
}

std::vector<U64> SearchCatalog::Rank(const std::vector<Item>& items,
                                     const std::vector<std::string>& groups,
                                     const std::string& query,
                                     const std::string& activeGroup) {
    std::vector<U64> order;
    const auto included = [&](const Item& item) {
        return activeGroup.empty() || item.group == activeGroup;
    };
    if (query.empty()) {
        for (const auto& group : Groups(items, groups)) {
            for (U64 index = 0; index < items.size(); ++index) {
                if (items[index].group == group && included(items[index])) {
                    order.push_back(index);
                }
            }
        }
        return order;
    }
    const std::string needle = Jetstream::Text::ToLower(query);
    std::vector<std::pair<U64, U64>> ranked;
    for (U64 index = 0; index < items.size(); ++index) {
        if (!included(items[index])) {
            continue;
        }
        const std::string title = Jetstream::Text::ToLower(items[index].title);
        U64 tier = 0;
        if (title.rfind(needle, 0) == 0) {
            tier = 0;
        } else if (title.find(needle) != std::string::npos) {
            tier = 1;
        } else if (Jetstream::Text::ToLower(items[index].summary).find(needle) != std::string::npos) {
            tier = 2;
        } else if (Jetstream::Text::ToLower(items[index].group).find(needle) != std::string::npos) {
            tier = 3;
        } else {
            continue;
        }
        ranked.emplace_back(tier, index);
    }
    std::stable_sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });
    for (const auto& [_, index] : ranked) {
        order.push_back(index);
    }
    return order;
}

void SearchCatalog::update(Config config) {
    if (config.items != impl->config.items) {
        ++impl->itemsRevision;
    }
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    auto& s = *impl;
    s.config = std::move(config);
    s.groups = Groups(s.config.items, s.config.groups);
    s.order = Rank(s.config.items, s.config.groups, s.config.query, s.config.activeGroup);
    std::vector<std::string> labels;
    U64 activeChip = 0;
    std::vector<std::string> named;
    for (const auto& name : s.groups) {
        if (!name.empty()) {
            named.push_back(name);
        }
    }
    if (!named.empty()) {
        labels.push_back(kAllLabel);
        for (U64 index = 0; index < named.size(); ++index) {
            labels.push_back(named[index]);
            if (named[index] == s.config.activeGroup) {
                activeChip = index + 1;
            }
        }
    }
    s.chipsShown = !named.empty();
    s.chips.update({
        .labels = std::move(labels),
        .active = activeChip,
        .onPick = [onGroup = s.config.onGroup, named](U64 index) {
            if (onGroup) {
                onGroup(index == 0 || index > named.size() ? std::string() : named[index - 1]);
            }
        },
    });
    s.ids.clear();
    for (const U64 index : s.order) {
        s.ids.push_back(s.config.items[index].id);
    }
    const U64 selected = s.order.empty() ? 0 : std::min<U64>(s.config.selected, s.order.size() - 1);

    std::vector<ResultList::Row> rows;
    std::string group;
    for (U64 result = 0; result < s.order.size(); ++result) {
        const auto& item = s.config.items[s.order[result]];
        if (s.config.query.empty() && s.config.activeGroup.empty() && (result == 0 || item.group != group)) {
            group = item.group;
            rows.push_back({.header = true, .label = group.empty() ? "Other" : group});
        }
        rows.push_back({.result = result});
    }
    s.list.update({
        .rows = std::move(rows),
        .items = &s.config.items,
        .revision = s.itemsRevision,
        .order = s.order,
        .selected = selected,
        .emptyLabel = s.config.emptyLabel,
        .onSelect = s.config.onSelect,
        .onActivate = s.config.onActivate,
    });

    s.docs = !s.order.empty();
    if (s.docs) {
        const auto& item = s.config.items[s.order[selected]];
        while (s.docsButtons.size() < item.badges.size()) {
            s.docsButtons.push_back(std::make_unique<Button>());
        }
        std::vector<Stack::Item> buttons;
        for (U64 index = 0; index < item.badges.size(); ++index) {
            s.docsButtons[index]->update({
                .str = jst::fmt::format("{} {}", item.badges[index], ICON_LC_ARROW_RIGHT),
                .colorKey = "action_btn",
                .hoveredColorKey = "action_btn",
                .activeColorKey = "action_btn",
                .borderColorKey = "action_btn_outline",
                .textColorKey = "action_btn_text",
                .textScale = 1.05f,
                .onClick = [onActivate = s.config.onActivate, id = item.id, index]() {
                    if (onActivate) {
                        onActivate(id, index);
                    }
                },
            });
            buttons.push_back({.child = s.docsButtons[index].get()});
        }
        s.docsButtonRow.update({.axis = Stack::Axis::Horizontal, .spacing = 10.0f, .items = std::move(buttons)});
        s.docsMarkdown.update({.value = item.documentation});
        s.docsScroll.update({.id = item.id, .content = &s.docsMarkdown, .alwaysScrollbar = false, .border = false});
        s.docsFooterLabel.update({.str = kActionLabel, .tone = Text::Tone::Secondary});
    }

    mountParts();
}

void SearchCatalog::mountParts() {
    auto& s = *impl;
    std::vector<Component*> children = {&s.cards};
    if (s.config.field) {
        children.push_back(s.config.field);
    }
    if (s.chipsShown) {
        children.push_back(&s.chips);
    }
    children.push_back(&s.list);
    if (s.docs) {
        children.push_back(&s.docsScroll);
        children.push_back(&s.docsDivider);
        children.push_back(&s.docsFooterLabel);
        children.push_back(&s.docsButtonRow);
    }
    const bool owned = std::all_of(children.begin(), children.end(), [this](const Component* child) {
        return hasChild(*child);
    });
    if (children != s.mounted || !owned) {
        mountChildren(children);
        s.mounted = std::move(children);
    }
}

const std::vector<std::string>& SearchCatalog::results() const {
    return impl->ids;
}

Extent2D<F32> SearchCatalog::measure(const Context& ctx, Extent2D<F32> available) {
    auto& s = *impl;
    const F32 infinite = std::numeric_limits<F32>::infinity();
    const F32 padding = Px(ctx, kCardPadding);
    const F32 gap = Px(ctx, kCardGap);
    const F32 listWidth = FitLength(Px(ctx, s.config.listWidth), available.x);
    const F32 full = Px(ctx, s.config.listWidth) + gap + Px(ctx, s.config.docsWidth);
    const bool docsShell = (s.docs || s.config.height != 0.0f) && full <= available.x;
    const F32 width = FitLength(std::ceil(docsShell ? full : listWidth), available.x);
    const F32 field = s.config.field ? measureChild(*s.config.field, ctx, {width, infinite}).y + gap : 0.0f;
    F32 natural = s.listHeight(*this, ctx, listWidth) + 2.0f * padding;
    if (s.docs) {
        natural = std::max(natural, Px(ctx, kDocsMinHeight));
    }
    natural += field;
    if (s.config.maxHeight > 0.0f) {
        natural = std::min(natural, Px(ctx, s.config.maxHeight));
    }
    return {width, ResolveLength(ctx, s.config.height, std::ceil(natural), available.y)};
}

void SearchCatalog::layout(const Context& ctx) {
    auto& s = *impl;
    const Rect bounds = frame();
    const F32 infinite = std::numeric_limits<F32>::infinity();
    const F32 gap = Px(ctx, kCardGap);
    const F32 listWidth = std::min(Px(ctx, s.config.listWidth), bounds.width);
    const bool fits = bounds.width >= Px(ctx, s.config.listWidth) + gap + Px(ctx, s.config.docsWidth);
    s.docsFit = fits;
    const bool docs = s.docs && s.docsFit;
    const bool steady = s.config.height != 0.0f;
    const bool docsShell = (s.docs || steady) && s.docsFit;
    for (auto* part : std::initializer_list<Component*>{&s.docsScroll, &s.docsDivider, &s.docsFooterLabel, &s.docsButtonRow}) {
        part->setVisible(docs);
    }
    const F32 docsWidth = docsShell ? Px(ctx, s.config.docsWidth) : 0.0f;
    const bool docsLeft = s.config.docsSide == DocsSide::Left;
    const F32 listX = docsShell && docsLeft ? bounds.right() - listWidth : bounds.x;
    const F32 docsX = docsLeft ? bounds.x : listX + listWidth + gap;

    F32 top = bounds.y;
    if (s.config.field) {
        const F32 height = measureChild(*s.config.field, ctx, {bounds.width, infinite}).y;
        layoutChild(ctx, *s.config.field, {bounds.x, top, bounds.width, height});
        top += height + gap;
    }
    const F32 cardHeight = std::max(0.0f, bounds.bottom() - top);
    const F32 padding = Px(ctx, kCardPadding);
    const F32 listContent = s.listHeight(*this, ctx, listWidth) + 2.0f * padding;
    s.listCard = {listX, top, listWidth, docs || steady ? cardHeight : std::min(cardHeight, listContent)};
    s.docsCard = docsShell ? Rect{docsX, top, docsWidth, cardHeight} : Rect{};

    std::vector<Rect> rects = {s.listCard};
    if (docsShell) {
        rects.push_back(s.docsCard);
    }
    s.cards.update({
        .rects = std::move(rects),
        .clip = clip(),
        .cornerRadius = Px(ctx, kCardRadius),
        .softness = Px(ctx, kCardShadowSoftness),
        .drop = Px(ctx, kCardShadowDrop),
        .borderWidth = Px(ctx, 1.0f),
        .lineWidth = Px(ctx, 1.0f),
        .fillKey = kPillFill,
    });
    layoutChild(ctx, s.cards, bounds);
    Rect listArea = {s.listCard.x + padding, s.listCard.y + padding, std::max(0.0f, s.listCard.width - 2.0f * padding),
                     std::max(0.0f, s.listCard.height - 2.0f * padding)};
    if (s.chipsShown) {
        const F32 chipInset = Px(ctx, kChipGap);
        const F32 chipWidth = std::max(0.0f, listArea.width - 2.0f * chipInset);
        const F32 chipHeight = s.chipsHeight(*this, ctx, listArea.width);
        layoutChild(ctx, s.chips, {listArea.x + chipInset, listArea.y + chipInset, chipWidth, chipHeight - 2.0f * chipInset});
        listArea.y += chipHeight;
        listArea.height = std::max(0.0f, listArea.height - chipHeight);
    }
    layoutChild(ctx, s.list, listArea, listArea);
    if (docs) {
        const F32 inner = Px(ctx, kDocsPadding);
        const F32 width = std::max(0.0f, s.docsCard.width - 2.0f * inner);
        const Extent2D<F32> buttons = measureChild(s.docsButtonRow, ctx, {infinite, infinite});
        const F32 footer = std::ceil(buttons.y + 2.0f * Px(ctx, kDocsFooterPadding));
        const F32 footerY = s.docsCard.bottom() - footer;
        const F32 centerY = footerY + footer * 0.5f;
        F32 x = s.docsCard.x + inner;
        const Extent2D<F32> label = measureChild(s.docsFooterLabel, ctx, {infinite, infinite});
        layoutChild(ctx, s.docsFooterLabel, {x, std::round(centerY - label.y * 0.5f), label.x, label.y});
        x += label.x + Px(ctx, kDocsFooterGap);
        layoutChild(ctx, s.docsButtonRow, {x, std::round(centerY - buttons.y * 0.5f),
                                           std::max(0.0f, s.docsCard.right() - inner - x), buttons.y});
        s.docsDivider.update({
            .instances = {{.rect = {s.docsCard.x + inner, footerY, width, Px(ctx, 1.0f)},
                           .backgroundColor = ctx.color("tile_border")}},
            .capacity = 1,
        });
        layoutChild(ctx, s.docsDivider, bounds);
        const Rect docsArea = {s.docsCard.x + inner, s.docsCard.y + inner, width,
                               std::max(0.0f, footerY - s.docsCard.y - 2.0f * inner)};
        layoutChild(ctx, s.docsScroll, docsArea, docsArea);
    }
}

}  // namespace Jetstream::Sakura
