#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_DETACHED_CONFIG_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_DETACHED_CONFIG_HH

#include "jetstream/render/sakura/base.hh"

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
        Sakura::Callback<void()> onClose;
    };

    void update(Config config) {
        using Stack = Sakura::NodeContentStack;

        this->config = std::move(config);

        while (metrics.size() > this->config.metrics.size()) {
            metrics.pop_back();
        }
        while (metrics.size() < this->config.metrics.size()) {
            metrics.push_back(std::make_unique<FlowgraphMetricInstance>());
        }
        for (U64 i = 0; i < metrics.size(); ++i) {
            metrics[i]->update(std::move(this->config.metrics[i]));
        }

        fields.update(std::move(this->config.configFields));
        std::vector<Stack::Item> items;
        for (const auto& metric : metrics) {
            items.push_back({.child = &metric->component()});
        }
        if (!metrics.empty()) {
            items.back().trailingGap = Stack::Spacing;
        }
        fields.appendItems(items);

        content.update({
            .id = this->config.id + ":content",
            .items = std::move(items),
        });
        panel.update({
            .id = this->config.id + ":panel",
            .content = &content,
            .height = Sakura::Fill,
        });
    }

    Sakura::Component& component() {
        return panel;
    }

    const std::string& title() const {
        return config.title;
    }

    Sakura::Callback<void()> closeCallback() const {
        return config.onClose;
    }

 private:
    Config config;
    Sakura::NodeContentStack content;
    Sakura::NodeFieldPanel panel;
    std::vector<std::unique_ptr<FlowgraphMetricInstance>> metrics;
    FlowgraphConfigFieldStack fields;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_DETACHED_CONFIG_HH
