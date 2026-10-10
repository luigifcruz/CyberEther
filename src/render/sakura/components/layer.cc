#include <jetstream/render/sakura/components/layer.hh>

#include "../helpers.hh"

namespace Jetstream::Sakura {

struct Layer::Impl {
    Config config;
    std::string windowId;
    ImGuiWindow* window = nullptr;
    U64 focusRequest = 0;
    U64 blurRequest = 0;
};

Layer::Layer() {
    this->impl = std::make_unique<Impl>();
}

Layer::~Layer() = default;
Layer::Layer(Layer&&) noexcept = default;
Layer& Layer::operator=(Layer&&) noexcept = default;

bool Layer::update(Config config) {
    this->impl->windowId = "##" + config.id;
    this->impl->config = std::move(config);
    return true;
}

void Layer::render(const Context& ctx, Child child) {
    auto& impl = *this->impl;
    const auto& config = impl.config;
    const ImGuiViewport* viewport = ImGui::GetMainViewport();

    const F32 top = viewport->Pos.y + Scale(ctx, config.topOffset);
    const F32 bottom = viewport->Pos.y + viewport->Size.y;
    const F32 left = viewport->Pos.x;
    const F32 right = viewport->Pos.x + viewport->Size.x;
    const F32 availableHeight = std::max(0.0f, bottom - top);

    ImVec2 size = Private::ToImVec2(Scale(ctx, config.size));
    if (size.x <= 0.0f || size.x > right - left) {
        size.x = right - left;
    }
    if (size.y <= 0.0f || size.y > availableHeight) {
        size.y = availableHeight;
    }
    if (size.x <= 0.0f || size.y <= 0.0f) {
        return;
    }

    const ImVec2 point = Private::AnchorPoint(config.anchor, {left, top},
                                             {right - left, availableHeight}, {});
    const ImVec2 pivot = Private::AnchorPivot(config.anchor);
    const ImVec2 position(point.x - pivot.x * size.x, point.y - pivot.y * size.y);

    ImGui::SetNextWindowPos(position, ImGuiCond_Always);
    ImGui::SetNextWindowSize(size, ImGuiCond_Always);
    ImGui::SetNextWindowViewport(viewport->ID);
    if (Private::ConsumeRequest(impl.focusRequest, config.focusRequest)) {
        ImGui::SetNextWindowFocus();
    }

    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration |
                             ImGuiWindowFlags_NoMove |
                             ImGuiWindowFlags_NoDocking |
                             ImGuiWindowFlags_NoSavedSettings |
                             ImGuiWindowFlags_NoFocusOnAppearing |
                             ImGuiWindowFlags_NoBackground |
                             ImGuiWindowFlags_NoScrollWithMouse;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowMinSize, ImVec2(1.0f, 1.0f));
    const bool visible = ImGui::Begin(impl.windowId.c_str(), nullptr, flags);
    ImGui::PopStyleVar(3);

    impl.window = ImGui::GetCurrentWindow();
    if (visible && child) {
        child(ctx);
    }

    if (Private::ConsumeRequest(impl.blurRequest, config.blurRequest) &&
        ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)) {
        ImGui::FocusWindow(nullptr);
    }
    if (!ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel)) {
        ImGui::BringWindowToDisplayFront(impl.window);
    }
    ImGui::End();
}

}  // namespace Jetstream::Sakura
