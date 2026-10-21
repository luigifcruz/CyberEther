#include <jetstream/render/sakura/components/setting_card.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/sakura/typography.hh>

#include "../context.hh"
#include "../detail/helpers.hh"
#include "../metrics.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

namespace {

constexpr F32 kCardPadY = 14.0f;
constexpr F32 kCardControlGap = 24.0f;
constexpr F32 kCardControlMax = 220.0f;
constexpr F32 kCardWideControl = 0.6f;
constexpr F32 kCardLineGap = 3.0f;
constexpr F32 kCardLabel = 15.5f;
constexpr F32 kCardDescription = 15.0f;
constexpr F32 kCardDescriptionLineHeight = 1.05f;
constexpr F32 kCardEmptyPadY = 22.0f;
constexpr U64 kCardSeparatorBlock = 128;

}  // namespace

struct SettingCard::Impl {
    struct Line {
        Text label;
        Text description;
    };

    struct Metrics {
        Extent2D<F32> control = {0.0f, 0.0f};
        F32 label = 0.0f;
        F32 description = 0.0f;
        F32 height = 0.0f;
    };

    Config config;
    Box card;
    Box highlight;
    Box separators;
    Text emptyText;
    std::vector<std::unique_ptr<Line>> lines;
    std::vector<Rect> rows;
    std::vector<Component*> mounted;
    std::optional<U64> hovered;
    std::optional<U64> pressed;

    std::optional<U64> rowAt(const Rect& bounds, const Extent2D<F32>& point) const {
        if (!bounds.contains(point.x, point.y)) {
            return std::nullopt;
        }
        for (U64 index = 0; index < rows.size() && index < config.rows.size(); ++index) {
            if (config.rows[index].onClick && rows[index].contains(point.x, point.y)) {
                return index;
            }
        }
        return std::nullopt;
    }

    Metrics metrics(SettingCard& self, const Context& ctx, U64 index, F32 width) {
        const auto& row = config.rows[index];
        auto& line = *lines[index];
        const F32 infinite = std::numeric_limits<F32>::infinity();
        const F32 inner = std::max(0.0f, width - 2.0f * Px(ctx, kCardPadX));
        Metrics result;
        if (row.control) {
            const F32 cap = row.wide ? inner * kCardWideControl : std::min(Px(ctx, kCardControlMax), inner * 0.5f);
            result.control = self.measureChild(*row.control, ctx, {cap, infinite});
            result.control.x = std::min(result.control.x, cap);
        }
        const F32 gap = row.control ? Px(ctx, kCardControlGap) : 0.0f;
        const F32 text = std::max(0.0f, inner - result.control.x - gap);
        result.label = row.label.empty() ? 0.0f : self.measureChild(line.label, ctx, {text, infinite}).y;
        result.description = row.description.empty() ? 0.0f
                                                     : self.measureChild(line.description, ctx, {text, infinite}).y;
        const F32 spacing = result.label > 0.0f && result.description > 0.0f ? Px(ctx, kCardLineGap) : 0.0f;
        result.height = std::max(result.label + spacing + result.description, result.control.y) +
                        2.0f * Px(ctx, kCardPadY);
        return result;
    }
};

SettingCard::SettingCard() {
    impl = std::make_unique<Impl>();
    setClipsChildren(true);
}

SettingCard::~SettingCard() = default;

void SettingCard::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    const auto& rows = impl->config.rows;
    while (impl->lines.size() < rows.size()) {
        impl->lines.push_back(std::make_unique<Impl::Line>());
    }
    if (impl->hovered && *impl->hovered >= rows.size()) {
        impl->hovered.reset();
    }
    if (impl->pressed && *impl->pressed >= rows.size()) {
        impl->pressed.reset();
    }

    impl->emptyText.update({
        .str = impl->config.empty,
        .font = Text::Font::Sans,
        .tone = Text::Tone::Secondary,
        .align = Text::Align::Center,
        .wrapped = true,
        .scale = kCardDescription / Typography::FontSize,
        .lineHeight = kCardDescriptionLineHeight,
    });
    std::vector<Component*> children = {&impl->card, &impl->highlight, &impl->separators};
    if (rows.empty() && !impl->config.empty.empty()) {
        children.push_back(&impl->emptyText);
    }
    for (U64 index = 0; index < rows.size(); ++index) {
        auto& line = *impl->lines[index];
        line.label.update({
            .str = rows[index].label,
            .font = Text::Font::Sans,
            .wrapped = true,
            .scale = kCardLabel / Typography::FontSize,
            .weight = kLabelWeight,
        });
        line.description.update({
            .str = rows[index].description,
            .font = Text::Font::Sans,
            .tone = Text::Tone::Secondary,
            .wrapped = true,
            .scale = kCardDescription / Typography::FontSize,
            .lineHeight = kCardDescriptionLineHeight,
        });
        if (!rows[index].label.empty()) {
            children.push_back(&line.label);
        }
        if (!rows[index].description.empty()) {
            children.push_back(&line.description);
        }
        if (rows[index].control) {
            children.push_back(rows[index].control);
        }
    }
    const bool owned = std::all_of(children.begin(), children.end(), [this](const Component* child) {
        return hasChild(*child);
    });
    if (children != impl->mounted || !owned) {
        mountChildren(children);
        impl->mounted = std::move(children);
    }
}

Extent2D<F32> SettingCard::measure(const Context& ctx, Extent2D<F32> available) {
    const F32 width = FillLength(available.x);
    if (impl->config.rows.empty() && !impl->config.empty.empty()) {
        const F32 inner = std::max(0.0f, width - 2.0f * Px(ctx, kCardPadX));
        const F32 text = measureChild(impl->emptyText, ctx, {inner, std::numeric_limits<F32>::infinity()}).y;
        return {width, FitLength(text + 2.0f * Px(ctx, kCardEmptyPadY), available.y)};
    }
    F32 height = 0.0f;
    for (U64 index = 0; index < impl->config.rows.size(); ++index) {
        height += impl->metrics(*this, ctx, index, width).height;
    }
    if (impl->config.rows.size() > 1) {
        height += Hairline(ctx) * static_cast<F32>(impl->config.rows.size() - 1);
    }
    return {width, FitLength(height, available.y)};
}

void SettingCard::layout(const Context& ctx) {
    auto& s = *impl;
    const Rect bounds = frame();
    const F32 padX = Px(ctx, kCardPadX);
    const F32 line = Hairline(ctx);

    s.card.update({
        .instances = {{.rect = bounds, .visible = !bounds.empty(), .backgroundColor = ctx.color("setting_card_bg")}},
        .clip = bounds,
        .cornerRadius = Px(ctx, kCardRounding),
        .borderWidth = Px(ctx, 1.0f),
        .borderColor = ctx.color("setting_card_border"),
        .strokeFringe = true,
        .capacity = 1,
    });
    layoutChild(ctx, s.card, bounds);

    if (s.config.rows.empty() && !s.config.empty.empty()) {
        const F32 inner = std::max(0.0f, bounds.width - 2.0f * padX);
        const F32 height = measureChild(s.emptyText, ctx, {inner, std::numeric_limits<F32>::infinity()}).y;
        layoutChild(ctx, s.emptyText, {bounds.x + padX, std::round(bounds.y + (bounds.height - height) * 0.5f),
                                       inner, height});
    }

    std::vector<Box::Instance> separators;
    s.rows.clear();
    F32 y = bounds.y;
    for (U64 index = 0; index < s.config.rows.size(); ++index) {
        const auto& row = s.config.rows[index];
        auto& text = *s.lines[index];
        const auto metrics = s.metrics(*this, ctx, index, bounds.width);
        s.rows.push_back({bounds.x, y, bounds.width, metrics.height});
        const F32 right = bounds.right() - padX;
        if (row.control) {
            layoutChild(ctx, *row.control, {right - metrics.control.x,
                                            std::round(y + (metrics.height - metrics.control.y) * 0.5f),
                                            metrics.control.x, metrics.control.y});
        }
        const F32 gap = row.control ? Px(ctx, kCardControlGap) : 0.0f;
        const F32 width = std::max(0.0f, right - metrics.control.x - gap - (bounds.x + padX));
        const F32 spacing = metrics.label > 0.0f && metrics.description > 0.0f ? Px(ctx, kCardLineGap) : 0.0f;
        const F32 textHeight = metrics.label + spacing + metrics.description;
        F32 textY = std::round(y + (metrics.height - textHeight) * 0.5f);
        if (!row.label.empty()) {
            layoutChild(ctx, text.label, {bounds.x + padX, textY, width, metrics.label});
        }
        textY += metrics.label + spacing;
        if (!row.description.empty()) {
            layoutChild(ctx, text.description, {bounds.x + padX, textY, width, metrics.description});
        }
        y += metrics.height;
        if (index + 1 < s.config.rows.size()) {
            separators.push_back({.rect = {bounds.x + padX, y, std::max(0.0f, bounds.width - 2.0f * padX), line},
                                  .backgroundColor = ctx.color("setting_card_border")});
            y += line;
        }
    }
    const U64 capacity = kCardSeparatorBlock * (s.config.rows.size() / kCardSeparatorBlock + 1);
    s.separators.update({.instances = std::move(separators), .clip = bounds, .capacity = capacity});
    layoutChild(ctx, s.separators, bounds);

    const bool lit = s.hovered.has_value() && *s.hovered < s.rows.size();
    U8 corners = 0;
    if (lit && *s.hovered == 0) {
        corners |= static_cast<U8>(Box::TopLeft | Box::TopRight);
    }
    if (lit && *s.hovered + 1 == s.rows.size()) {
        corners |= static_cast<U8>(Box::BottomLeft | Box::BottomRight);
    }
    s.highlight.update({
        .instances = {{.rect = lit ? Inset(s.rows[*s.hovered], line) : Rect{},
                       .visible = lit,
                       .backgroundColor = ctx.color("setting_card_hovered"),
                       .corners = corners}},
        .clip = bounds,
        .cornerRadius = std::max(0.0f, Px(ctx, kCardRounding) - line),
        .capacity = 1,
    });
    layoutChild(ctx, s.highlight, bounds);
}

bool SettingCard::mouseEvent(const MouseEvent& event) {
    auto& s = *impl;
    if (mouseEventChildren(event)) {
        return true;
    }
    switch (event.type) {
        case MouseEventType::Move: {
            const auto row = s.rowAt(frame(), event.position);
            if (row != s.hovered) {
                s.hovered = row;
                invalidate(Dirty::Layout);
            }
            return false;
        }
        case MouseEventType::Click: {
            const auto row = s.rowAt(frame(), event.position);
            if (!row || event.button != MouseButton::Left) {
                return false;
            }
            s.pressed = row;
            return true;
        }
        case MouseEventType::Release: {
            if (event.button != MouseButton::Left || !s.pressed) {
                return false;
            }
            const auto pressed = *s.pressed;
            s.pressed.reset();
            if (s.rowAt(frame(), event.position) == pressed) {
                s.config.rows[pressed].onClick();
            }
            return true;
        }
        case MouseEventType::Leave:
        case MouseEventType::Cancel:
            if (s.hovered || s.pressed) {
                s.hovered.reset();
                s.pressed.reset();
                invalidate(Dirty::Layout);
            }
            return false;
        default:
            return false;
    }
}

std::optional<SurfaceCursor> SettingCard::cursorAt(const Extent2D<F32>& point) const {
    if (impl->rowAt(frame(), point)) {
        return SurfaceCursor::Hand;
    }
    return std::nullopt;
}

}  // namespace Jetstream::Sakura
