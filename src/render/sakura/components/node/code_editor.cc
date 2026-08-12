#include <jetstream/render/sakura/components/node/code_editor.hh>

#include <jetstream/render/sakura/components/retained/code_editor.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct NodeCodeEditor::Impl {
    Config config;
    Retained::CodeEditor editor;
};

NodeCodeEditor::NodeCodeEditor() {
    this->impl = std::make_unique<Impl>();
}

NodeCodeEditor::~NodeCodeEditor() = default;
NodeCodeEditor::NodeCodeEditor(NodeCodeEditor&&) noexcept = default;
NodeCodeEditor& NodeCodeEditor::operator=(NodeCodeEditor&&) noexcept = default;

bool NodeCodeEditor::update(Config config) {
    impl->config = std::move(config);

    impl->editor.update({
        .id = impl->config.id,
        .value = impl->config.value,
        .consoleOutput = impl->config.consoleOutput,
        .status = impl->config.status,
        .size = {0.0f, 200.0f},
        .statusTone = impl->config.statusTone,
        .consoleVisible = impl->config.consoleVisible,
        .collapsible = impl->config.collapsible,
        .autoHeight = impl->config.autoHeight,
        .maxAutoHeightWindowRatio = impl->config.maxAutoHeightWindowRatio,
        .language = impl->config.language,
        .lineNumbers = impl->config.lineNumbers,
        .lineWrapping = impl->config.lineWrapping,
        .editorFontSize = impl->config.editorFontSize,
        .backgroundColorKey = impl->config.backgroundColorKey,
        .onChange = impl->config.onChange,
        .onSubmit = impl->config.onSubmit,
    });
    return true;
}

void NodeCodeEditor::render(const Context& ctx) const {
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(Scale(ctx, 6.0f), Scale(ctx, 3.0f)));
    impl->editor.render(ctx);
    ImGui::PopStyleVar();
}

}  // namespace Jetstream::Sakura
