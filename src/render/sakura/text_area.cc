#include <jetstream/render/sakura/text_area.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct TextArea::Impl {
    Config config;
};

TextArea::TextArea() {
    this->impl = std::make_unique<Impl>();
}

TextArea::~TextArea() = default;
TextArea::TextArea(TextArea&&) noexcept = default;
TextArea& TextArea::operator=(TextArea&&) noexcept = default;

bool TextArea::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void TextArea::render(const Context& ctx) const {
    const auto& config = this->impl->config;

    std::string value = config.value;
    ImGui::PushID(config.id.c_str());
    ImGui::SetNextItemWidth(-FLT_MIN);

    const ImGuiInputTextFlags flags = config.submit == Submit::OnEdit ? ImGuiInputTextFlags_None
                                                                       : ImGuiInputTextFlags_EnterReturnsTrue;
    bool changed = ImGui::InputTextMultiline("##textarea",
                                             &value,
                                             Private::ToImVec2(Scale(ctx, config.size)),
                                             flags);
    if (config.submit == Submit::OnCommit && ImGui::IsItemDeactivatedAfterEdit()) {
        changed = true;
    }
    if (changed && config.onChange) {
        config.onChange(value);
    }
    ImGui::PopID();
}

}  // namespace Jetstream::Sakura
