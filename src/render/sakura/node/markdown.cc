#include <jetstream/render/sakura/node/markdown.hh>

#include <jetstream/render/sakura/markdown.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct NodeMarkdown::Impl {
    Config config;
    Markdown markdown;
};

NodeMarkdown::NodeMarkdown() {
    this->impl = std::make_unique<Impl>();
}

NodeMarkdown::~NodeMarkdown() = default;
NodeMarkdown::NodeMarkdown(NodeMarkdown&&) noexcept = default;
NodeMarkdown& NodeMarkdown::operator=(NodeMarkdown&&) noexcept = default;

bool NodeMarkdown::update(Config config) {
    auto& impl = *this->impl;
    impl.config = std::move(config);
    impl.markdown.update({
        .id = impl.config.id,
        .value = impl.config.value,
    });
    return true;
}

void NodeMarkdown::render(const Context& ctx) const {
    ImGui::PushStyleColor(ImGuiCol_Text, Private::ImColor(ctx, "text_primary"));
    this->impl->markdown.render(ctx);
    ImGui::PopStyleColor();
}

}  // namespace Jetstream::Sakura
