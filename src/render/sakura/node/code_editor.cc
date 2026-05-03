#include <jetstream/render/sakura/node/code_editor.hh>

#include <jetstream/render/sakura/code_editor.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct NodeCodeEditor::Impl {
    Config config;
    CodeEditor editor;
};

NodeCodeEditor::NodeCodeEditor() {
    this->impl = std::make_unique<Impl>();
}

NodeCodeEditor::~NodeCodeEditor() = default;
NodeCodeEditor::NodeCodeEditor(NodeCodeEditor&&) noexcept = default;
NodeCodeEditor& NodeCodeEditor::operator=(NodeCodeEditor&&) noexcept = default;

bool NodeCodeEditor::update(Config config) {
    auto& impl = *this->impl;
    impl.config = std::move(config);
    impl.editor.update({
        .id = impl.config.id,
        .value = impl.config.value,
        .collapsible = impl.config.collapsible,
        .size = {0.0f, 200.0f},
        .onChange = impl.config.onChange,
    });
    return true;
}

void NodeCodeEditor::render(const Context& ctx) const {
    const ImVec4 background = Private::ImColor(ctx, "card");
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(Scale(ctx, 6.0f), Scale(ctx, 3.0f)));
    ImGui::PushStyleColor(ImGuiCol_Button, background);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, background);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, background);
    this->impl->editor.render(ctx);
    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar();
}

}  // namespace Jetstream::Sakura
