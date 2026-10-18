#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_DOCUMENTATION_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_DOCUMENTATION_HH

#include "jetstream/render/sakura/base.hh"

#include <functional>
#include <string>
#include <utility>

namespace Jetstream {

struct FlowgraphDocumentation {
    struct Config {
        std::string id;
        std::string title;
        std::string markdown;
        std::function<void()> onClose;
    };

    void update(Config config) {
        this->config = std::move(config);
        markdown.update({
            .id = this->config.id + ":markdown",
            .value = this->config.markdown,
            .scrollbar = true,
        });
    }

    Sakura::Component& component() {
        return markdown;
    }

    const std::string& id() const {
        return config.id;
    }

    const std::string& title() const {
        return config.title;
    }

    std::function<void()> closeCallback() const {
        return config.onClose;
    }

 private:
    Config config;
    Sakura::Markdown markdown;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_DOCUMENTATION_HH
