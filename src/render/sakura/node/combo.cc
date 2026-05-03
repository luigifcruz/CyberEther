#include <jetstream/render/sakura/node/combo.hh>

#include <jetstream/render/sakura/combo.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct NodeCombo::Impl {
    Config config;
    Combo combo;
};

NodeCombo::NodeCombo() {
    this->impl = std::make_unique<Impl>();
}

NodeCombo::~NodeCombo() = default;
NodeCombo::NodeCombo(NodeCombo&&) noexcept = default;
NodeCombo& NodeCombo::operator=(NodeCombo&&) noexcept = default;

bool NodeCombo::update(Config config) {
    auto& impl = *this->impl;
    impl.config = std::move(config);
    impl.combo.update({
        .id = impl.config.id,
        .options = impl.config.options,
        .value = impl.config.value,
        .onChange = impl.config.onChange,
    });
    return true;
}

void NodeCombo::render(const Context& ctx) const {
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(Scale(ctx, 6.0f), Scale(ctx, 3.0f)));
    ImGui::PushStyleColor(ImGuiCol_Button, Private::ImColor(ctx, "card"));
    this->impl->combo.render(ctx);
    ImGui::PopStyleColor(1);
    ImGui::PopStyleVar();
}

}  // namespace Jetstream::Sakura
