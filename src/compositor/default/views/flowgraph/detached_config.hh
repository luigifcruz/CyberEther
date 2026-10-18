#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_DETACHED_CONFIG_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_DETACHED_CONFIG_HH

#include "jetstream/render/sakura/base.hh"

#include "editor/config/stack.hh"

#include <algorithm>
#include <functional>
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
        std::vector<FlowgraphConfigFieldConfig> configFields;
        std::function<void()> onClose;
    };

    void update(Config config) {
        using Stack = Sakura::NodeContentStack;

        this->config = std::move(config);
        fields.update(std::move(this->config.configFields));
        std::vector<Stack::Item> items;
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

    std::function<void()> closeCallback() const {
        return config.onClose;
    }

 private:
    Config config;
    Sakura::NodeContentStack content;
    Sakura::NodeFieldPanel panel;
    FlowgraphConfigFieldStack fields;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_DETACHED_CONFIG_HH
