#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_DOCUMENTATION_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_DOCUMENTATION_HH

#include "jetstream/render/sakura/base.hh"

#include <string>
#include <utility>

namespace Jetstream {

struct FlowgraphDocumentation {
    static constexpr F32 MaxWidth = 720.0f;

    struct Config {
        std::string id;
        std::string title;
        std::string markdown;
        Sakura::Callback<void()> onClose;
        Sakura::Callback<void(const std::string&)> onOpenUrl;
    };

    void update(Config config) {
        this->config = std::move(config);
        markdown.update({
            .id = this->config.id + ":markdown",
            .value = this->config.markdown,
            .scrollbar = true,
            .onLink = [this](const std::string& url) {
                if (!this->config.onOpenUrl) {
                    return false;
                }
                this->config.onOpenUrl(url);
                return true;
            },
            .maxWidth = MaxWidth,
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

    Sakura::Callback<void()> closeCallback() const {
        return config.onClose;
    }

 private:
    Config config;
    Sakura::Markdown markdown;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_DOCUMENTATION_HH
