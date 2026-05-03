#include <jetstream/render/sakura/modal.hh>

#include <jetstream/render/sakura/button.hh>
#include <jetstream/render/sakura/divider.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct Modal::Impl {
    Config config;
    bool visible = false;
    std::string modalId;
    Divider footerDivider;
    Button closeButton;
};

Modal::Modal() {
    this->impl = std::make_unique<Impl>();
}

Modal::~Modal() = default;
Modal::Modal(Modal&&) noexcept = default;
Modal& Modal::operator=(Modal&&) noexcept = default;

bool Modal::update(Config config) {
    auto& impl = *this->impl;
    impl.modalId = config.id.empty() ? "##sakura_test_modal" : config.id;
    impl.config = std::move(config);
    impl.footerDivider.update({
        .id = impl.modalId + "FooterDivider",
    });
    impl.closeButton.update({
        .id = impl.modalId + "Close",
        .str = "Close",
        .size = {-1.0f, 40.0f},
        .onClick = []() {
            ImGui::CloseCurrentPopup();
        },
    });
    return true;
}

void Modal::render(const Context& ctx, Child child) {
    auto& impl = *this->impl;
    const auto& config = impl.config;

    ImGui::OpenPopup(impl.modalId.c_str());

    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(),
                            ImGuiCond_Always,
                            ImVec2(0.5f, 0.5f));

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar |
                             ImGuiWindowFlags_NoResize |
                             ImGuiWindowFlags_NoMove |
                             ImGuiWindowFlags_NoScrollbar;
    if (config.size.has_value()) {
        ImGui::SetNextWindowSize(Private::ToImVec2(Scale(ctx, *config.size)), ImGuiCond_Always);
    } else {
        flags |= ImGuiWindowFlags_AlwaysAutoResize;
    }

    if (!ImGui::BeginPopupModal(impl.modalId.c_str(), nullptr, flags)) {
        if (impl.visible) {
            if (config.onClose) {
                config.onClose();
            }
            impl.visible = false;
        }
        return;
    }

    if (!impl.visible) {
        if (config.onOpen) {
            config.onOpen();
        }
        impl.visible = true;
    }

    if (child) {
        child(ctx);
    }
    impl.footerDivider.render(ctx);
    impl.closeButton.render(ctx);

    ImGui::SetItemDefaultFocus();
    if (config.minWidth > 0.0f) {
        ImGui::Dummy(Private::ToImVec2({Scale(ctx, config.minWidth), 0.0f}));
    }
    ImGui::EndPopup();

    const bool modalOpen = ImGui::IsPopupOpen(impl.modalId.c_str());
    if (impl.visible && !modalOpen) {
        if (config.onClose) {
            config.onClose();
        }
        impl.visible = false;
    }
}

}  // namespace Jetstream::Sakura
