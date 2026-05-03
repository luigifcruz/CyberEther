#include <jetstream/render/sakura/code_editor.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct CodeEditor::Impl {
    Config config;
};

CodeEditor::CodeEditor() {
    this->impl = std::make_unique<Impl>();
}

CodeEditor::~CodeEditor() = default;
CodeEditor::CodeEditor(CodeEditor&&) noexcept = default;
CodeEditor& CodeEditor::operator=(CodeEditor&&) noexcept = default;

bool CodeEditor::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void CodeEditor::render(const Context& ctx) const {
    const auto& config = this->impl->config;

    ImGui::PushID(config.id.c_str());
    std::string value = config.value;
    bool changed = false;
    if (!config.collapsible) {
        changed = ImGui::InputTextCodeEditor("##multiline", &value, Private::ToImVec2(Scale(ctx, config.size)));
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            changed = true;
        }
        if (changed && config.onChange) {
            config.onChange(value);
        }
        ImGui::PopID();
        return;
    }

    const ImGuiID editingId = ImGui::GetID("##editing");
    ImGuiStorage* storage = ImGui::GetStateStorage();
    const bool editing = storage->GetBool(editingId, false);

    static std::unordered_map<ImGuiID, std::string> buffers;
    if (editing) {
        auto it = buffers.find(editingId);
        if (it == buffers.end()) {
            it = buffers.emplace(editingId, config.value).first;
        }
        ImGui::InputTextCodeEditor("##multiline", &it->second, Private::ToImVec2(Scale(ctx, config.size)));
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() - ImGui::GetStyle().ItemSpacing.y * 0.5f);
        if (ImGui::Button("Done", ImVec2(-1.0f, 0.0f))) {
            changed = true;
            if (config.onChange) {
                config.onChange(it->second);
            }
            buffers.erase(it);
            storage->SetBool(editingId, false);
        }
        ImGui::PopID();
        return;
    }

    buffers.erase(editingId);
    if (ImGui::Button("Edit", ImVec2(-1.0f, 0.0f))) {
        storage->SetBool(editingId, true);
    }
    ImGui::PopID();
}

}  // namespace Jetstream::Sakura
