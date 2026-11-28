#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_RENDER_STATS_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_RENDER_STATS_HH

#include "jetstream/instance_remote.hh"
#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <array>
#include <optional>
#include <string>
#include <utility>

namespace Jetstream {

struct RenderStatsView {
    static constexpr F32 Width = 312.0f;
    static constexpr F32 StripHeight = 28.0f;
    static constexpr F32 Padding = Sakura::Spacing::Medium;
    static constexpr F32 LabelWidth = 70.0f;
    static constexpr F32 RemoteLabelWidth = 96.0f;
    static constexpr F32 TableGap = Sakura::Spacing::Small;
    static constexpr F32 TableWidth = (Width - 2.0f * Padding - TableGap) * 0.5f;
    static constexpr F32 MissFactor = 1.5f;

    struct Config {
        Sakura::Canvas::Stats stats;
        std::optional<Instance::Remote::Stats> remote;
    };

    void update(Config config);
    Sakura::Component& component();
    Sakura::HudLayer::Hud hud();

 private:
    static constexpr U64 Rows = 7;
    static constexpr U64 RemoteRows = 7;

    struct Metric {
        Sakura::Text label;
        Sakura::Text value;
    };

    template<std::size_t Count>
    void fill(Sakura::Table& table, std::array<Metric, Count>& metrics,
              const std::array<std::pair<const char*, std::string>, Count>& entries,
              F32 labelWidth = LabelWidth);
    void updateRemote(const Instance::Remote::Stats& remote);

    Config config;
    std::array<Metric, Rows> timingMetrics;
    std::array<Metric, Rows> drawingMetrics;
    Sakura::Text rate;
    Sakura::Sparkline strip;
    Sakura::Table timing;
    Sakura::Table drawing;
    Sakura::Stack tables;
    std::array<Metric, RemoteRows> remoteMetrics;
    Sakura::Text remoteEncoder;
    Sakura::Table remoteTable;
    Sakura::Stack stack;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_RENDER_STATS_HH
