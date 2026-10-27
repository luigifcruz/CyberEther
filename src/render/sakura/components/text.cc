#include <jetstream/render/sakura/components/text.hh>

#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/typography.hh>

#include <jetstream/render/components/text.hh>
#include <jetstream/render/tools/icons_lucide.hh>

#include "../context.hh"
#include "../metrics.hh"
#include "../detail/helpers.hh"
#include "../detail/text_metrics.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

namespace {

constexpr F32 kSharpness = 0.45f;
constexpr F32 kDisplaySharpness = 0.3f;

using Unicode = Jetstream::Render::Components::Text::Unicode;

constexpr const char* kIconFont = Typography::IconFont;

const char* FontName(Text::Font font) {
    switch (font) {
        case Text::Font::H1:
        case Text::Font::H2:
        case Text::Font::Bold:
            return Typography::MonoBoldFont;
        case Text::Font::Display:
            return Typography::DisplayFont;
        case Text::Font::DisplaySemiBold:
            return Typography::DisplaySemiboldFont;
        case Text::Font::Sans:
            return Typography::BodyFont;
        case Text::Font::SansBold:
            return Typography::BodyBoldFont;
        case Text::Font::Body:
            break;
    }
    return Typography::MonoFont;
}

const char* ToneKey(Text::Tone tone) {
    switch (tone) {
        case Text::Tone::Secondary:
            return "text_secondary";
        case Text::Tone::Disabled:
            return "text_disabled";
        case Text::Tone::Accent:
            return "cyber_blue";
        case Text::Tone::Success:
            return "success_green";
        case Text::Tone::Warning:
            return "warning_yellow";
        case Text::Tone::Primary:
            break;
    }
    return "text_primary";
}

}  // namespace

struct Text::Impl {
    struct Run {
        std::string str;
        bool icon = false;
        F32 width = 0.0f;
    };

    struct Line {
        std::vector<Run> runs;
        F32 width = 0.0f;
    };

    Config config;
    Label label;
    Label icons;
    TextMetrics metrics;
    std::vector<Line> lines;
    F32 linesWidth = -1.0f;
    F32 linesFontSize = -1.0f;
    U64 linesFonts = 0;
    F32 fit = 1.0f;

    static constexpr U64 kFitPasses = 3;

    F32 baseSize(const Context& ctx) const {
        return Typography::FontSize * config.scale * ctx.pixelRatio;
    }

    F32 fontSize(const Context& ctx) const {
        return baseSize(ctx) * fit;
    }

    F32 iconSize(const Context& ctx) const {
        return fontSize(ctx);
    }

    F32 widest() const {
        F32 width = 0.0f;
        for (const auto& line : lines) {
            width = std::max(width, line.width);
        }
        return width;
    }

    Line shape(const std::string& str, const Context& ctx) {
        Line line;
        const F32 size = fontSize(ctx);
        const F32 glyphSize = iconSize(ctx);
        U64 pos = 0;
        while (pos < str.size()) {
            U32 codepoint = 0;
            const U64 next = pos + std::max<U64>(1, Unicode::Decode(str, pos, codepoint));
            const bool icon = codepoint >= ICON_MIN_LC && codepoint <= ICON_MAX_LC;
            if (line.runs.empty() || line.runs.back().icon != icon || icon) {
                line.runs.push_back({.icon = icon});
            }
            line.runs.back().str += str.substr(pos, next - pos);
            pos = next;
        }
        for (auto& run : line.runs) {
            run.width = run.icon ? std::max(glyphSize, metrics.measure(kIconFont, run.str, glyphSize))
                                 : metrics.measure(FontName(config.font), run.str, size);
            line.width += run.width;
        }
        return line;
    }

    void wrap(const Context& ctx, F32 width) {
        const F32 size = baseSize(ctx);
        const U64 fonts = ctx.render ? ctx.render->fontRevision() : 0;
        if (width == linesWidth && size == linesFontSize && fonts == linesFonts) {
            return;
        }
        linesWidth = width;
        linesFontSize = size;
        linesFonts = fonts;
        metrics.setWindow(ctx.render);
        fit = 1.0f;
        build(ctx, width);
        if (!config.shrinkToFit || config.wrapped || !std::isfinite(width) || width <= 0.0f) {
            return;
        }
        for (U64 pass = 0; pass < kFitPasses && widest() > width; ++pass) {
            fit *= width / widest();
            build(ctx, width);
        }
    }

    void build(const Context& ctx, F32 width) {
        lines.clear();
        const auto trimmed = [](std::string value) {
            while (!value.empty() && value.back() == ' ') {
                value.pop_back();
            }
            return value;
        };
        const std::string& str = config.str;
        U64 start = 0;
        while (true) {
            const U64 end = std::min<U64>(str.find('\n', start), str.size());
            const std::string paragraph = str.substr(start, end - start);
            if (!config.wrapped || !std::isfinite(width)) {
                lines.push_back(shape(paragraph, ctx));
            } else {
                const auto fits = [&](const std::string& value) {
                    return shape(trimmed(value), ctx).width <= width;
                };
                std::string current;
                U64 cursor = 0;
                while (cursor < paragraph.size()) {
                    const U64 space = paragraph.find(' ', cursor);
                    const U64 wordEnd = space == std::string::npos ? paragraph.size() : space + 1;
                    const std::string word = paragraph.substr(cursor, wordEnd - cursor);
                    cursor = wordEnd;
                    if (!current.empty() && !fits(current + word)) {
                        lines.push_back(shape(trimmed(current), ctx));
                        current.clear();
                    }
                    if (!current.empty() || fits(word)) {
                        current += word;
                        continue;
                    }
                    U64 index = 0;
                    while (index < word.size()) {
                        const U8 lead = static_cast<U8>(word[index]);
                        const U64 length = lead < 0x80 ? 1 : (lead >> 5) == 0x6 ? 2 : (lead >> 4) == 0xE ? 3 : 4;
                        const std::string glyph = word.substr(index, length);
                        if (!current.empty() && !fits(current + glyph)) {
                            lines.push_back(shape(trimmed(current), ctx));
                            current.clear();
                        }
                        current += glyph;
                        index += length;
                    }
                }
                lines.push_back(shape(trimmed(current), ctx));
            }
            if (end >= str.size()) {
                break;
            }
            start = end + 1;
        }
    }
};

Text::Text() {
    impl = std::make_unique<Impl>();
    addChild(impl->label);
    addChild(impl->icons);
}

Text::~Text() = default;

const Text::Config& Text::config() const {
    return impl->config;
}

void Text::update(Config config) {
    if (config == impl->config) {
        impl->config = std::move(config);
        return;
    }
    invalidate(Dirty::Layout);
    if (config.str != impl->config.str || config.wrapped != impl->config.wrapped ||
        config.font != impl->config.font || config.scale != impl->config.scale ||
        config.shrinkToFit != impl->config.shrinkToFit) {
        impl->linesWidth = -1.0f;
    }
    impl->config = std::move(config);
}

Extent2D<F32> Text::measure(const Context& ctx, Extent2D<F32> available) {
    impl->wrap(ctx, available.x);
    F32 width = impl->widest();
    if (std::isfinite(available.x) && (impl->config.wrapped || impl->config.align != Align::Left)) {
        width = std::max(width, available.x);
    }
    return FitExtent({width, std::round(impl->fontSize(ctx) * impl->config.lineHeight) * static_cast<F32>(impl->lines.size()) +
                                 impl->config.verticalOffset * ctx.pixelRatio},
                     available);
}

void Text::layout(const Context& ctx) {
    const Rect bounds = frame();
    impl->wrap(ctx, bounds.width);
    const F32 size = impl->fontSize(ctx);
    const F32 glyphSize = impl->iconSize(ctx);
    auto color = ctx.color(impl->config.colorKey.empty() ? ToneKey(impl->config.tone) : impl->config.colorKey);
    color = WithAlpha(color, impl->config.opacity);
    const bool visible = !bounds.empty();

    std::vector<Label::Instance> texts;
    std::vector<Label::Instance> glyphs;
    for (U64 i = 0; i < impl->lines.size(); ++i) {
        const auto& line = impl->lines[i];
        const F32 pitch = std::round(size * impl->config.lineHeight);
        const F32 y = bounds.y + impl->config.verticalOffset * ctx.pixelRatio + pitch * static_cast<F32>(i) +
                      std::round((pitch - size) * 0.5f);
        F32 x = bounds.x;
        if (impl->config.align == Align::Center) {
            x += std::floor((bounds.width - line.width) * 0.5f);
        } else if (impl->config.align == Align::Right) {
            x += bounds.width - line.width;
        }
        for (const auto& run : line.runs) {
            if (run.icon) {
                glyphs.push_back({
                    .rect = {x, y, run.width, size},
                    .str = run.str,
                    .visible = visible,
                    .color = color,
                    .fontSize = glyphSize,
                    .alignment = {1, 1},
                });
            } else {
                texts.push_back({
                    .rect = {x, y, run.width, size},
                    .str = run.str,
                    .visible = visible,
                    .color = color,
                    .fontSize = size,
                    .alignment = {0, 1},
                });
            }
            x += run.width;
        }
    }
    impl->label.update({
        .instances = std::move(texts),
        .clip = bounds,
        .fontName = FontName(impl->config.font),
        .sharpness = impl->config.font == Font::Display ? kDisplaySharpness : kSharpness,
        .weight = impl->config.weight * Fringe(ctx),
        .maxCharacters = impl->config.maxCharacters,
    });
    impl->icons.update({
        .instances = std::move(glyphs),
        .clip = bounds,
        .fontName = kIconFont,
        .maxCharacters = 4,
    });
    layoutChild(ctx, impl->label, bounds);
    layoutChild(ctx, impl->icons, bounds);
}

}  // namespace Jetstream::Sakura
