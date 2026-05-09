#include <jetstream/render/sakura/node/int_input.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct NodeIntInput::Impl {
    Config config;
};

NodeIntInput::NodeIntInput() {
    this->impl = std::make_unique<Impl>();
}

NodeIntInput::~NodeIntInput() = default;
NodeIntInput::NodeIntInput(NodeIntInput&&) noexcept = default;
NodeIntInput& NodeIntInput::operator=(NodeIntInput&&) noexcept = default;

bool NodeIntInput::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void NodeIntInput::render(const Context& ctx) const {
    const auto& config = this->impl->config;

    U64 value = config.value;
    ImGui::PushID(config.id.c_str());
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(Scale(ctx, 6.0f), Scale(ctx, 3.0f)));
    ImGui::SetNextItemWidth(-FLT_MIN);
    const bool changed = ImGui::InputScalar("##int",
                                            ImGuiDataType_U64,
                                            &value,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            ImGuiInputTextFlags_EnterReturnsTrue);
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
    if (changed && config.onChange) {
        config.onChange(value);
    }
    ImGui::PopStyleVar();
    ImGui::PopID();
}

}  // namespace Jetstream::Sakura
