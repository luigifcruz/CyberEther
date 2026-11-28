#include "render_stats.hh"

#include <algorithm>
#include <cctype>

namespace Jetstream {

namespace {

std::string Bytes(F64 bytes) {
    if (bytes >= 1024.0 * 1024.0) {
        return jst::fmt::format("{:.1f} MB", bytes / (1024.0 * 1024.0));
    }
    return jst::fmt::format("{:.0f} KB", bytes / 1024.0);
}

std::string Input(const Instance::Remote::Stats& remote) {
    if (remote.format == "videotoolbox_vld") {
        return "VT BGRA";
    }
    if (remote.format == "cuda") {
        return "CUDA BGRA";
    }
    std::string format = remote.format;
    std::transform(format.begin(), format.end(), format.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return format;
}

std::string Bits(F64 bits) {
    if (bits >= 1000.0 * 1000.0) {
        return jst::fmt::format("{:.1f} Mbps", bits / (1000.0 * 1000.0));
    }
    return jst::fmt::format("{:.0f} kbps", bits / 1000.0);
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

    std::vector<Sakura::Stack::Item> items = {
        {.child = &rate},
        {.child = &strip},
        {.child = &tables},
    };
    if (this->config.remote) {
        updateRemote(*this->config.remote);
        items.push_back({.child = &remoteEncoder});
        items.push_back({.child = &remoteTable});
    }

    stack.update({
        .spacing = Sakura::Spacing::Small,
        .centerCross = false,
        .items = std::move(items),
    });
}

void RenderStatsView::updateRemote(const Instance::Remote::Stats& remote) {
    const bool hardware = remote.encoder != Instance::Remote::EncoderType::Software;

    remoteEncoder.update({
        .str = jst::fmt::format("Remote {} {} ({})",
                                GetRemoteEncoderPrettyName(remote.encoder),
                                GetRemoteCodecPrettyName(remote.codec),
                                hardware ? "Hardware" : "Software"),
        .tone = Sakura::Text::Tone::Secondary,
        .ellipsis = true,
    });

    fill(remoteTable, remoteMetrics, {{
        {"Rate", jst::fmt::format("{:.0f} / {:.0f} FPS", remote.frameRate, remote.targetRate)},
        {"Size", jst::fmt::format("{}x{}", remote.size.x, remote.size.y)},
        {"Input", Input(remote)},
        {"Encode", jst::fmt::format("{:.1f} ms", remote.encodeAverage)},
        {"Bitrate", Bits(remote.bitrate)},
        {"Keyframe", Bytes(static_cast<F64>(remote.keyframeBytes))},
        {"RTT", remote.roundTrip ? jst::fmt::format("{:.0f} ms", *remote.roundTrip) : std::string("--")},
    }}, RemoteLabelWidth);
}

template<std::size_t Count>
void RenderStatsView::fill(Sakura::Table& table, std::array<Metric, Count>& metrics,
                           const std::array<std::pair<const char*, std::string>, Count>& entries,
                           F32 labelWidth) {
    std::vector<std::vector<Sakura::Component*>> rows;
    rows.reserve(Count);
    for (std::size_t i = 0; i < Count; ++i) {
        metrics[i].label.update({.str = entries[i].first});
        metrics[i].value.update({.str = entries[i].second, .align = Sakura::Text::Align::Right});
        rows.push_back({&metrics[i].label, &metrics[i].value});
    }
    table.update({
        .columns = {"", ""},
        .cellRows = std::move(rows),
        .fixedColumnWidths = {labelWidth},
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
