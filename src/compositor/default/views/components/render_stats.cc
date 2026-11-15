#include "render_stats.hh"

namespace Jetstream {

namespace {

std::string Bytes(F64 bytes) {
    if (bytes >= 1024.0 * 1024.0) {
        return jst::fmt::format("{:.1f} MB", bytes / (1024.0 * 1024.0));
    }
    return jst::fmt::format("{:.0f} KB", bytes / 1024.0);
}

}  // namespace

void RenderStatsView::update(Config config) {
    using Text = Sakura::Text;

    this->config = std::move(config);
    const auto& stats = this->config.stats;
    const F32 period = stats.refreshRate > 0.0f ? 1000.0f / stats.refreshRate : 0.0f;

    const F32 shown = stats.active ? stats.presentRate : 0.0f;
    rate.update({
        .str = stats.refreshRate > 0.0f ? jst::fmt::format("{:03.0f} / {:03.0f} FPS", shown, stats.refreshRate)
                                        : jst::fmt::format("{:03.0f} FPS", shown),
        .font = Text::Font::Bold,
        .scale = Sakura::Typography::Scale::Headline,
    });

    strip.update({
        .values = stats.intervals,
        .limit = period,
        .alert = period * MissFactor,
        .size = {Sakura::Fill, StripHeight},
        .colorKey = "success_green",
    });

    const F32 budget = period > 0.0f ? stats.workPeak / period : 0.0f;
    fill(timing, timingMetrics, {{
        {"Smooth", jst::fmt::format("{:.1f}%", stats.smoothness * 100.0f)},
        {"Idle", jst::fmt::format("{:.0f}%", stats.idle * 100.0f)},
        {"Work", jst::fmt::format("{:.2f} ms", stats.workAverage)},
        {"Layouts", jst::fmt::format("{:.0f}/s", stats.layoutRate)},
        {"Leaves", jst::fmt::format("{}", stats.leaves)},
        {"Buffers", jst::fmt::format("{}", stats.buffers)},
        {"Uploads", jst::fmt::format("{}/s", Bytes(stats.uploadRate))},
    }});
    fill(drawing, drawingMetrics, {{
        {"Missed", jst::fmt::format("{}", stats.missedFrames)},
        {"Budget", jst::fmt::format("{:.0f}%", budget * 100.0f)},
        {"Peak", jst::fmt::format("{:.2f} ms", stats.workPeak)},
        {"Redraws", jst::fmt::format("{:.0f}/s", stats.redrawRate)},
        {"Draws", jst::fmt::format("{}", stats.draws)},
        {"Memory", Bytes(static_cast<F64>(stats.bufferBytes))},
        {"Rebuilds", jst::fmt::format("{}", stats.rebuilds)},
    }});
    tables.update({
        .axis = Sakura::Stack::Axis::Horizontal,
        .spacing = TableGap,
        .centerCross = false,
        .items = {
            {.child = &timing, .size = TableWidth},
            {.child = &drawing, .size = TableWidth},
        },
    });

    stack.update({
        .spacing = Sakura::Spacing::Small,
        .centerCross = false,
        .items = {
            {.child = &rate},
            {.child = &strip},
            {.child = &tables},
        },
    });
}

void RenderStatsView::fill(Sakura::Table& table, std::array<Metric, Rows>& metrics,
                           const std::array<std::pair<const char*, std::string>, Rows>& entries) {
    std::vector<std::vector<Sakura::Component*>> rows;
    rows.reserve(Rows);
    for (U64 i = 0; i < Rows; ++i) {
        metrics[i].label.update({.str = entries[i].first});
        metrics[i].value.update({.str = entries[i].second, .align = Sakura::Text::Align::Right});
        rows.push_back({&metrics[i].label, &metrics[i].value});
    }
    table.update({
        .columns = {"", ""},
        .cellRows = std::move(rows),
        .fixedColumnWidths = {LabelWidth},
        .showHeaders = false,
    });
}

Sakura::Component& RenderStatsView::component() {
    return stack;
}

Sakura::HudLayer::Hud RenderStatsView::hud() {
    return {
        .content = &stack,
        .anchor = Sakura::Anchor::BottomLeft,
        .margin = Sakura::Spacing::Medium,
        .padding = {Padding, Padding},
        .borderSize = 1.0f,
        .rounding = Sakura::Rounding::Card,
        .backgroundColorKey = "popup_bg",
        .borderColorKey = "setting_card_border",
        .size = {Width, 0.0f},
    };
}

}  // namespace Jetstream
