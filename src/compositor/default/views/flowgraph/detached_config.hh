#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_DETACHED_CONFIG_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_DETACHED_CONFIG_HH

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include "editor/config/stack.hh"
#include "editor/metrics/base.hh"

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct FlowgraphDetachedConfig {
    struct Config {
        std::string id;
        std::string title;
        std::vector<FlowgraphMetricConfig> metrics;
        std::vector<FlowgraphConfigFieldConfig> configFields;
        bool metricsOpen = true;
        Sakura::Callback<void()> onClose;
        Sakura::Callback<void(bool)> onToggleMetricsOpen;
    };

    void update(Config config);
    std::vector<Sakura::Dock::Action> actions();
    Sakura::Component& component();
    const std::string& title() const;
    Sakura::Callback<void()> closeCallback() const;

 private:
    Config config;
    bool metricsVisible = true;
    bool initialized = false;
    Sakura::NodeContentStack content;
    Sakura::NodeFieldPanel panel;
    std::vector<std::unique_ptr<FlowgraphMetricInstance>> metrics;
    FlowgraphConfigFieldStack fields;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_DETACHED_CONFIG_HH
