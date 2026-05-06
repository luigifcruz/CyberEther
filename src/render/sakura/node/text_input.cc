#include <jetstream/render/sakura/node/text_input.hh>

#include <jetstream/render/sakura/text_input.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct NodeTextInput::Impl {
    Config config;
    TextInput input;
};

NodeTextInput::NodeTextInput() {
    this->impl = std::make_unique<Impl>();
}

NodeTextInput::~NodeTextInput() = default;
NodeTextInput::NodeTextInput(NodeTextInput&&) noexcept = default;
NodeTextInput& NodeTextInput::operator=(NodeTextInput&&) noexcept = default;

bool NodeTextInput::update(Config config) {
    auto& impl = *this->impl;
    impl.config = std::move(config);
    impl.input.update({
        .id = impl.config.id,
        .value = impl.config.value,
        .submit = impl.config.submit,
        .onChange = impl.config.onChange,
    });
    return true;
}

void NodeTextInput::render(const Context& ctx) const {
    const auto& config = this->impl->config;

    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(Scale(ctx, 6.0f), Scale(ctx, 3.0f)));
    this->impl->input.render(ctx);
    if (!config.unit.empty()) {
        const ImVec2 inputPos = ImGui::GetItemRectMin();
        const ImVec2 inputSize = ImGui::GetItemRectSize();
        const ImVec2 textSize = ImGui::CalcTextSize(config.unit.c_str());
        const ImVec2 textPos(inputPos.x + inputSize.x - textSize.x - Scale(ctx, 3.0f),
                             inputPos.y + (inputSize.y - textSize.y) * 0.5f);
        ImGui::GetWindowDrawList()->AddText(textPos,
                                            ImGui::ColorConvertFloat4ToU32(Private::ImColor(ctx, "text_secondary")),
                                            config.unit.c_str());
    }
    ImGui::PopStyleVar();
}

}  // namespace Jetstream::Sakura
