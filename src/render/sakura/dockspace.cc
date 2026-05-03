#include <jetstream/render/sakura/dockspace.hh>

#include "base.hh"

namespace Jetstream::Sakura {

namespace {

ImGuiWindowFlags DefaultWindowFlags() {
    return ImGuiWindowFlags_NoDocking |
           ImGuiWindowFlags_NoTitleBar |
           ImGuiWindowFlags_NoCollapse |
           ImGuiWindowFlags_NoResize |
           ImGuiWindowFlags_NoMove |
           ImGuiWindowFlags_NoBringToFrontOnFocus |
           ImGuiWindowFlags_NoNavFocus |
           ImGuiWindowFlags_NoBackground;
}

ImGuiDockNodeFlags DefaultDockFlags() {
    return ImGuiDockNodeFlags_PassthruCentralNode;
}

}  // namespace

U64 DockspaceId(const DockspaceConfig& config) {
    return static_cast<U64>(ImHashStr(config.id));
}

U64 DockspaceId() {
    return DockspaceId(DockspaceConfig{});
}

void Dockspace(const DockspaceConfig& config) {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const ImVec2 pos = ImVec2(viewport->Pos.x, viewport->Pos.y + config.topOffset);
    const ImVec2 size = ImVec2(viewport->Size.x, viewport->Size.y - config.topOffset);
    const ImGuiID id = static_cast<ImGuiID>(DockspaceId(config));
    const ImGuiWindowFlags windowFlags = static_cast<ImGuiWindowFlags>(config.windowFlags.value_or(DefaultWindowFlags()));
    const ImGuiDockNodeFlags dockFlags = static_cast<ImGuiDockNodeFlags>(config.dockFlags.value_or(DefaultDockFlags()));

    ImGui::SetNextWindowPos(pos);
    ImGui::SetNextWindowSize(size);
    ImGui::SetNextWindowViewport(viewport->ID);

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin(config.windowTitle, nullptr, windowFlags);
    ImGui::DockSpace(id, ImVec2(0.0f, 0.0f), dockFlags);
    ImGui::End();
    ImGui::PopStyleVar(3);
}

}  // namespace Jetstream::Sakura
