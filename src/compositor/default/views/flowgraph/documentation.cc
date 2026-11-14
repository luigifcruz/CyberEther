#include "documentation.hh"

namespace Jetstream {

void FlowgraphDocumentation::update(Config config) {
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

Sakura::Component& FlowgraphDocumentation::component() {
    return markdown;
}

const std::string& FlowgraphDocumentation::id() const {
    return config.id;
}

const std::string& FlowgraphDocumentation::title() const {
    return config.title;
}

Sakura::Callback<void()> FlowgraphDocumentation::closeCallback() const {
    return config.onClose;
}

}  // namespace Jetstream
