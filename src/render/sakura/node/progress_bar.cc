#include <jetstream/render/sakura/node/progress_bar.hh>

#include <jetstream/render/sakura/progress_bar.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct NodeProgressBar::Impl {
    Config config;
    ProgressBar progress;
};

NodeProgressBar::NodeProgressBar() {
    this->impl = std::make_unique<Impl>();
}

NodeProgressBar::~NodeProgressBar() = default;
NodeProgressBar::NodeProgressBar(NodeProgressBar&&) noexcept = default;
NodeProgressBar& NodeProgressBar::operator=(NodeProgressBar&&) noexcept = default;

bool NodeProgressBar::update(Config config) {
    auto& impl = *this->impl;
    impl.config = std::move(config);
    impl.progress.update({
        .id = impl.config.id,
        .value = impl.config.value,
        .overlay = impl.config.overlay,
        .size = impl.config.size,
        .colorKey = "action_btn",
    });
    return true;
}

void NodeProgressBar::render(const Context& ctx) const {
    this->impl->progress.render(ctx);
}

}  // namespace Jetstream::Sakura
