#include <jetstream/render/sakura/node/label.hh>

#include <jetstream/render/sakura/text.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct NodeLabel::Impl {
    Config config;
    Text label;
};

NodeLabel::NodeLabel() {
    this->impl = std::make_unique<Impl>();
}

NodeLabel::~NodeLabel() = default;
NodeLabel::NodeLabel(NodeLabel&&) noexcept = default;
NodeLabel& NodeLabel::operator=(NodeLabel&&) noexcept = default;

bool NodeLabel::update(Config config) {
    auto& impl = *this->impl;
    impl.config = std::move(config);
    impl.label.update({
        .id = impl.config.id,
        .str = impl.config.str,
        .font = impl.config.font,
        .tone = impl.config.tone,
        .align = impl.config.align,
        .wrapped = impl.config.wrapped,
        .scale = impl.config.scale,
    });
    return true;
}

void NodeLabel::render(const Context& ctx) const {
    this->impl->label.render(ctx);
}

}  // namespace Jetstream::Sakura
