#include <jetstream/render/sakura/components/feature_card.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/sakura/typography.hh>

#include "../context.hh"
#include "../metrics.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

namespace {

constexpr F32 kFeaturePadY = 18.0f;
constexpr F32 kFeatureLineGap = 4.0f;
constexpr F32 kFeatureTitle = 16.5f;
constexpr F32 kFeatureDescription = 14.0f;
constexpr F32 kFeatureActionGap = 24.0f;

}  // namespace

struct FeatureCard::Impl {
    Config config;
    Box card;
    Text title;
    Text description;
    Box separator;
    std::vector<Component*> mounted;

    struct Metrics {
        Extent2D<F32> action = {0.0f, 0.0f};
        F32 text = 0.0f;
        F32 title = 0.0f;
        F32 description = 0.0f;
        F32 header = 0.0f;
        F32 content = 0.0f;
    };

    Metrics metrics(FeatureCard& self, const Context& ctx, F32 width) {
        const F32 infinite = std::numeric_limits<F32>::infinity();
        const F32 inner = std::max(0.0f, width - 2.0f * Px(ctx, kCardPadX));
        Metrics result;
        if (config.action) {
            result.action = self.measureChild(*config.action, ctx, {inner * 0.5f, infinite});
        }
        const F32 actionGap = config.action ? Px(ctx, kFeatureActionGap) : 0.0f;
        result.text = std::max(0.0f, inner - result.action.x - actionGap);
        result.title = self.measureChild(title, ctx, {result.text, infinite}).y;
        result.description = config.description.empty() ? 0.0f
                                                       : self.measureChild(description, ctx, {result.text, infinite}).y;
        const F32 gap = result.description > 0.0f ? Px(ctx, kFeatureLineGap) : 0.0f;
        const F32 block = std::max(result.title + gap + result.description, result.action.y);
        result.header = block + 2.0f * Px(ctx, kFeaturePadY);
        if (config.content) {
            result.content = self.measureChild(*config.content, ctx, {inner, infinite}).y + 2.0f * Px(ctx, kFeaturePadY);
        }
        return result;
    }
};

FeatureCard::FeatureCard() {
    impl = std::make_unique<Impl>();
    setClipsChildren(true);
}

FeatureCard::~FeatureCard() = default;

void FeatureCard::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    impl->title.update({
        .str = impl->config.title,
        .font = Text::Font::Sans,
        .wrapped = true,
        .scale = kFeatureTitle / Typography::FontSize,
        .weight = kLabelWeight,
    });
    impl->description.update({
        .str = impl->config.description,
        .font = Text::Font::Sans,
        .tone = Text::Tone::Secondary,
        .wrapped = true,
        .scale = kFeatureDescription / Typography::FontSize,
        .lineHeight = 1.05f,
    });
    std::vector<Component*> children = {&impl->card, &impl->title};
    if (!impl->config.description.empty()) {
        children.push_back(&impl->description);
    }
    if (impl->config.action) {
        children.push_back(impl->config.action);
    }
    if (impl->config.content) {
        children.push_back(&impl->separator);
        children.push_back(impl->config.content);
    }
    const bool owned = std::all_of(children.begin(), children.end(), [this](const Component* child) {
        return hasChild(*child);
    });
    if (children != impl->mounted || !owned) {
        mountChildren(children);
        impl->mounted = std::move(children);
    }
}

Extent2D<F32> FeatureCard::measure(const Context& ctx, Extent2D<F32> available) {
    const F32 width = FillLength(available.x);
    const auto metrics = impl->metrics(*this, CardSurface(ctx), width);
    const F32 separator = impl->config.content ? Hairline(ctx) : 0.0f;
    return {width, FitLength(metrics.header + separator + metrics.content, available.y)};
}

void FeatureCard::layout(const Context& ctx) {
    auto& s = *impl;
    const Rect bounds = frame();
    const Context inner = CardSurface(ctx);
    const auto metrics = s.metrics(*this, inner, bounds.width);
    const F32 padX = Px(ctx, kCardPadX);
    const F32 padY = Px(ctx, kFeaturePadY);
    const F32 width = std::max(0.0f, bounds.width - 2.0f * padX);
    const F32 line = Hairline(ctx);
    s.card.update({
        .instances = {{.rect = bounds, .visible = !bounds.empty(), .backgroundColor = ctx.color(kCardSurface)}},
        .clip = bounds,
        .cornerRadius = Px(ctx, kPanelRounding),
        .borderWidth = Px(ctx, 1.0f),
        .borderColor = ctx.color("setting_card_border"),
        .strokeFringe = true,
        .capacity = 1,
    });
    layoutChild(ctx, s.card, bounds);

    const F32 gap = metrics.description > 0.0f ? Px(ctx, kFeatureLineGap) : 0.0f;
    const F32 textHeight = metrics.title + gap + metrics.description;
    F32 y = std::round(bounds.y + (metrics.header - textHeight) * 0.5f);
    layoutChild(inner, s.title, {bounds.x + padX, y, metrics.text, metrics.title});
    y += metrics.title + gap;
    if (!s.config.description.empty()) {
        layoutChild(inner, s.description, {bounds.x + padX, y, metrics.text, metrics.description});
    }
    if (s.config.action) {
        layoutChild(inner, *s.config.action, {bounds.right() - padX - metrics.action.x,
                                              std::round(bounds.y + (metrics.header - metrics.action.y) * 0.5f),
                                              metrics.action.x, metrics.action.y});
    }

    if (s.config.content) {
        const F32 top = bounds.y + metrics.header;
        s.separator.update({
            .instances = {{.rect = {bounds.x + padX, top, width, line},
                           .backgroundColor = ctx.color("setting_card_border")}},
            .clip = bounds,
            .capacity = 1,
        });
        layoutChild(ctx, s.separator, bounds);
        layoutChild(inner, *s.config.content, {bounds.x + padX, top + line + padY, width,
                                               std::max(0.0f, metrics.content - 2.0f * padY)});
    }
}

}  // namespace Jetstream::Sakura
