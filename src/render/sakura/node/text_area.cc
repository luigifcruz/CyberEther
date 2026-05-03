#include <jetstream/render/sakura/node/text_area.hh>

#include <jetstream/render/sakura/text_area.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct NodeTextArea::Impl {
    Config config;
    TextArea input;
};

NodeTextArea::NodeTextArea() {
    this->impl = std::make_unique<Impl>();
}

NodeTextArea::~NodeTextArea() = default;
NodeTextArea::NodeTextArea(NodeTextArea&&) noexcept = default;
NodeTextArea& NodeTextArea::operator=(NodeTextArea&&) noexcept = default;

bool NodeTextArea::update(Config config) {
    auto& impl = *this->impl;
    impl.config = std::move(config);
    impl.input.update({
        .id = impl.config.id,
        .value = impl.config.value,
        .size = {0.0f, 80.0f},
        .submit = TextArea::Submit::OnEdit,
        .onChange = impl.config.onChange,
    });
    return true;
}

void NodeTextArea::render(const Context& ctx) const {
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(Scale(ctx, 6.0f), Scale(ctx, 3.0f)));
    this->impl->input.render(ctx);
    ImGui::PopStyleVar();
}

}  // namespace Jetstream::Sakura
