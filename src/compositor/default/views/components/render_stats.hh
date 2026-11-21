#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_RENDER_STATS_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_RENDER_STATS_HH

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <array>
#include <string>
#include <utility>

namespace Jetstream {

struct RenderStatsView {
    static constexpr F32 Width = 312.0f;
    static constexpr F32 StripHeight = 28.0f;
    static constexpr F32 Padding = Sakura::Spacing::Medium;
    static constexpr F32 LabelWidth = 70.0f;
    static constexpr F32 TableGap = Sakura::Spacing::Small;
    static constexpr F32 TableWidth = (Width - 2.0f * Padding - TableGap) * 0.5f;
    static constexpr F32 MissFactor = 1.5f;

    struct Config {
        Sakura::Canvas::Stats stats;
    };

    void update(Config config);
    Sakura::Component& component();
    Sakura::HudLayer::Hud hud();

 private:
    static constexpr U64 Rows = 7;

    struct Metric {
        Sakura::Text label;
        Sakura::Text value;
    };

    void fill(Sakura::Table& table, std::array<Metric, Rows>& metrics,
              const std::array<std::pair<const char*, std::string>, Rows>& entries);

    Config config;
    std::array<Metric, Rows> timingMetrics;
    std::array<Metric, Rows> drawingMetrics;
    Sakura::Text rate;
    Sakura::Sparkline strip;
    Sakura::Table timing;
    Sakura::Table drawing;
    Sakura::Stack tables;
    Sakura::Stack stack;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_RENDER_STATS_HH
