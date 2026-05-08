#include <jetstream/render/sakura/node/subtitle.hh>

#include <jetstream/render/sakura/text.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct NodeSubtitle::Impl {
    Config config;
    Text text;
};

NodeSubtitle::NodeSubtitle() {
    this->impl = std::make_unique<Impl>();
}

NodeSubtitle::~NodeSubtitle() = default;
NodeSubtitle::NodeSubtitle(NodeSubtitle&&) noexcept = default;
NodeSubtitle& NodeSubtitle::operator=(NodeSubtitle&&) noexcept = default;

bool NodeSubtitle::update(Config config) {
    this->impl->config = std::move(config);
    this->impl->text.update({
        .id = "NodeSubtitle" + this->impl->config.text,
        .str = this->impl->config.text,
        .tone = Text::Tone::Secondary,
        .align = Text::Align::Center,
        .scale = this->impl->config.fontScale,
    });
    return true;
}

void NodeSubtitle::render(const Context& ctx) const {
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() - Scale(ctx, 14.0f));
    this->impl->text.render(ctx);
}

}  // namespace Jetstream::Sakura
