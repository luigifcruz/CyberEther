#include <jetstream/render/sakura/components/node/console.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct NodeConsole::Impl {
    Config config;
    Retained::Console console;
};

NodeConsole::NodeConsole() {
    this->impl = std::make_unique<Impl>();
}

NodeConsole::~NodeConsole() = default;
NodeConsole::NodeConsole(NodeConsole&&) noexcept = default;
NodeConsole& NodeConsole::operator=(NodeConsole&&) noexcept = default;

bool NodeConsole::update(Config config) {
    impl->config = std::move(config);

    impl->console.update({
        .id = impl->config.id,
        .output = impl->config.output,
        .status = impl->config.status,
        .emptyText = impl->config.emptyText,
        .statusTone = impl->config.statusTone,
        .size = {0.0f, impl->config.height.value_or(160.0f)},
        .fontSize = impl->config.fontSize,
    });
    return true;
}

void NodeConsole::render(const Context& ctx) const {
    impl->console.render(ctx);
}

}  // namespace Jetstream::Sakura
