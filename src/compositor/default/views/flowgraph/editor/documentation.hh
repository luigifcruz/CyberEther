#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_DOCUMENTATION_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_DOCUMENTATION_HH

#include "jetstream/render/sakura/sakura.hh"

#include <functional>
#include <string>
#include <utility>

namespace Jetstream {

struct FlowgraphNodeDocumentation : public Sakura::Component {
    struct Config {
        std::string id;
        std::string title;
        std::string name;
        std::string value;
        std::function<void()> onClose;
    };

    void update(Config config) {
        this->config = std::move(config);
        window.update({
            .id = this->config.id,
            .title = this->config.title + " Documentation (" + this->config.name + ")",
            .onClose = this->config.onClose,
        });
        markdown.update({
            .id = this->config.id + ":markdown",
            .value = this->config.value,
        });
    }

    void render(const Sakura::Context& ctx) {
        window.render(ctx, [this](const Sakura::Context& ctx) {
            markdown.render(ctx);
        });
    }

 private:
    Config config;
    Sakura::Window window;
    Sakura::Markdown markdown;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_DOCUMENTATION_HH
