#include "jetstream/render/base/window.hh"

#include "jetstream/render/tools/compressed_jbmm.hh"
#include "jetstream/render/tools/compressed_fa.hh"

namespace Jetstream::Render {

Result Window::create() {
    _scalingFactor = 0.0f;
    _previousScalingFactor = 0.0f;

    graphicalLoopThreadStarted = false;

    return Result::SUCCESS;
}

Result Window::begin() {
    // Process surface bind and unbind queue.
    JST_CHECK(processSurfaceUnbindQueue());
    JST_CHECK(processSurfaceBindQueue());

    // Record graphical thread ID.
    graphicalLoopThreadStarted = true;
    graphicalLoopThreadId = std::this_thread::get_id();

    return Result::SUCCESS;
}

Result Window::bind(const std::shared_ptr<Surface>& surface) {
    // Push new surface to the bind queue.
    surfaceBindQueue.push(surface);

    // This is overcomplicated because of Emscripten.
    // The browser won't allow calling WebGPU function from other thread. 
    // So we need to find a way to make it work for everyone.

    // If graphical loop didn't start yet. Call the function directly.
    if (!graphicalLoopThreadStarted) {
        JST_CHECK(processSurfaceBindQueue());
    } 
    // Wait for graphical loop to process queue if current thread is different.
    else if (graphicalLoopThreadId != std::this_thread::get_id()) {
        while (!surfaceBindQueue.empty()) {
            std::this_thread::yield();
        }
    }
    // Call the function directly as fallback.
    else {
        JST_CHECK(processSurfaceBindQueue());
    }

    return Result::SUCCESS;
}

Result Window::unbind(const std::shared_ptr<Surface>& surface) {
    // Push new surface to the unbind queue.
    surfaceUnbindQueue.push(surface);

    // Return immediately.

    return Result::SUCCESS;
}

Result Window::processSurfaceBindQueue() {
    while (!surfaceBindQueue.empty()) {
        JST_CHECK(bindSurface(surfaceBindQueue.front()));
        surfaceBindQueue.pop();
    }

    return Result::SUCCESS;
}

Result Window::processSurfaceUnbindQueue() {
    while (!surfaceUnbindQueue.empty()) {
        JST_CHECK(unbindSurface(surfaceUnbindQueue.front()));
        surfaceUnbindQueue.pop();
    }

    return Result::SUCCESS;
}

void Window::ScaleStyle(const Viewport::Generic& viewport) {
    if (_previousScalingFactor == 0.0f) {
        _scalingFactor = viewport.scale(config.scale);

        ImGuiStyleSetup();
        ImNodesStyleSetup();
    }

    if (_scalingFactor != _previousScalingFactor) {
        ImGuiStyleScale();
        ImNodesStyleScale();
    }

    _previousScalingFactor = _scalingFactor;
}

void Window::ImGuiStyleSetup() {
    // Setup Theme
    // Inspired from: https://github.com/ocornut/imgui/issues/707#issuecomment-917151020

    auto &colors = ImGui::GetStyle().Colors;
    colors[ImGuiCol_Text]                   = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
    colors[ImGuiCol_TextDisabled]           = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
    colors[ImGuiCol_WindowBg]               = ImVec4(0.10f, 0.10f, 0.10f, 1.00f);
    colors[ImGuiCol_ChildBg]                = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_PopupBg]                = ImVec4(0.19f, 0.19f, 0.19f, 0.92f);
    colors[ImGuiCol_Border]                 = ImVec4(0.19f, 0.19f, 0.19f, 0.29f);
    colors[ImGuiCol_BorderShadow]           = ImVec4(0.00f, 0.00f, 0.00f, 0.24f);
    colors[ImGuiCol_FrameBg]                = ImVec4(0.05f, 0.05f, 0.05f, 0.54f);
    colors[ImGuiCol_FrameBgHovered]         = ImVec4(0.19f, 0.19f, 0.19f, 0.54f);
    colors[ImGuiCol_FrameBgActive]          = ImVec4(0.20f, 0.22f, 0.23f, 1.00f);
    colors[ImGuiCol_TitleBg]                = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_TitleBgActive]          = ImVec4(0.06f, 0.06f, 0.06f, 1.00f);
    colors[ImGuiCol_TitleBgCollapsed]       = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_MenuBarBg]              = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
    colors[ImGuiCol_ScrollbarBg]            = ImVec4(0.05f, 0.05f, 0.05f, 0.54f);
    colors[ImGuiCol_ScrollbarGrab]          = ImVec4(0.34f, 0.34f, 0.34f, 0.54f);
    colors[ImGuiCol_ScrollbarGrabHovered]   = ImVec4(0.40f, 0.40f, 0.40f, 0.54f);
    colors[ImGuiCol_ScrollbarGrabActive]    = ImVec4(0.56f, 0.56f, 0.56f, 0.54f);
    colors[ImGuiCol_CheckMark]              = ImVec4(0.33f, 0.67f, 0.86f, 1.00f);
    colors[ImGuiCol_SliderGrab]             = ImVec4(0.34f, 0.34f, 0.34f, 0.54f);
    colors[ImGuiCol_SliderGrabActive]       = ImVec4(0.56f, 0.56f, 0.56f, 0.54f);
    colors[ImGuiCol_Button]                 = ImVec4(0.05f, 0.05f, 0.05f, 0.54f);
    colors[ImGuiCol_ButtonHovered]          = ImVec4(0.19f, 0.19f, 0.19f, 0.54f);
    colors[ImGuiCol_ButtonActive]           = ImVec4(0.20f, 0.22f, 0.23f, 1.00f);
    colors[ImGuiCol_Header]                 = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
    colors[ImGuiCol_HeaderHovered]          = ImVec4(0.00f, 0.00f, 0.00f, 0.36f);
    colors[ImGuiCol_HeaderActive]           = ImVec4(0.20f, 0.22f, 0.23f, 0.33f);
    colors[ImGuiCol_Separator]              = ImVec4(0.28f, 0.28f, 0.28f, 0.29f);
    colors[ImGuiCol_SeparatorHovered]       = ImVec4(0.44f, 0.44f, 0.44f, 0.29f);
    colors[ImGuiCol_SeparatorActive]        = ImVec4(0.40f, 0.44f, 0.47f, 1.00f);
    colors[ImGuiCol_ResizeGrip]             = ImVec4(0.28f, 0.28f, 0.28f, 0.29f);
    colors[ImGuiCol_ResizeGripHovered]      = ImVec4(0.44f, 0.44f, 0.44f, 0.29f);
    colors[ImGuiCol_ResizeGripActive]       = ImVec4(0.40f, 0.44f, 0.47f, 1.00f);
    colors[ImGuiCol_Tab]                    = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
    colors[ImGuiCol_TabHovered]             = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
    colors[ImGuiCol_TabActive]              = ImVec4(0.20f, 0.20f, 0.20f, 0.36f);
    colors[ImGuiCol_TabUnfocused]           = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
    colors[ImGuiCol_TabUnfocusedActive]     = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
    colors[ImGuiCol_DockingPreview]         = ImVec4(0.33f, 0.67f, 0.86f, 1.00f);
    colors[ImGuiCol_DockingEmptyBg]         = ImVec4(0.05f, 0.05f, 0.05f, 0.54f);
    colors[ImGuiCol_PlotLines]              = ImVec4(0.86f, 0.33f, 0.33f, 1.00f);
    colors[ImGuiCol_PlotLinesHovered]       = ImVec4(0.92f, 0.40f, 0.40f, 1.00f);
    colors[ImGuiCol_PlotHistogram]          = ImVec4(0.23f, 0.65f, 0.58f, 1.00f);
    colors[ImGuiCol_PlotHistogramHovered]   = ImVec4(0.92f, 0.40f, 0.40f, 1.00f);
    colors[ImGuiCol_TableHeaderBg]          = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
    colors[ImGuiCol_TableBorderStrong]      = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
    colors[ImGuiCol_TableBorderLight]       = ImVec4(0.28f, 0.28f, 0.28f, 0.29f);
    colors[ImGuiCol_TableRowBg]             = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_TableRowBgAlt]          = ImVec4(1.00f, 1.00f, 1.00f, 0.06f);
    colors[ImGuiCol_TextSelectedBg]         = ImVec4(0.20f, 0.22f, 0.23f, 1.00f);
    colors[ImGuiCol_DragDropTarget]         = ImVec4(0.33f, 0.67f, 0.86f, 1.00f);
    colors[ImGuiCol_NavHighlight]           = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
    colors[ImGuiCol_NavWindowingHighlight]  = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg]      = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
    colors[ImGuiCol_ModalWindowDimBg]       = ImVec4(0.10f, 0.10f, 0.10f, 0.65f);

    // Setup Fonts

    auto& io = ImGui::GetIO();

    ImFontConfig font_config;
    font_config.OversampleH = 5;
    font_config.OversampleV = 5;
    font_config.FontBuilderFlags = 1;
    io.Fonts->Clear();

    io.Fonts->AddFontFromMemoryCompressedTTF(
        jbmm_compressed_data,
        jbmm_compressed_size,
        15.0f * _scalingFactor,
        &font_config,
        nullptr);

    ImFontConfig icon_font_config;
    icon_font_config.OversampleH = 5;
    icon_font_config.OversampleV = 5;
    icon_font_config.FontBuilderFlags = 1;
    icon_font_config.MergeMode = true;
    icon_font_config.GlyphMinAdvanceX = 15.0f * _scalingFactor;
    icon_font_config.GlyphOffset = { 0.0f, 2.0f };

    static const ImWchar icon_ranges[] = { ICON_MIN_FA, ICON_MAX_FA, 0 };

    io.Fonts->AddFontFromMemoryCompressedTTF(far_compressed_data,
                                             far_compressed_size,
                                             15.0f * _scalingFactor,
                                             &icon_font_config,
                                             icon_ranges);

    io.Fonts->AddFontFromMemoryCompressedTTF(fas_compressed_data,
                                             fas_compressed_size,
                                             15.0f * _scalingFactor,
                                             &icon_font_config,
                                             icon_ranges);
}

void Window::ImGuiStyleScale() {
    auto& style = ImGui::GetStyle();

    // Rewrite Style Values.
    style.WindowPadding                     = ImVec2(8.00f, 8.00f);
    style.FramePadding                      = ImVec2(5.00f, 2.00f);
    style.ItemSpacing                       = ImVec2(6.00f, 6.00f);
    style.ItemInnerSpacing                  = ImVec2(6.00f, 6.00f);
    style.TouchExtraPadding                 = ImVec2(0.00f, 0.00f);
    style.CellPadding                       = ImVec2(3.00f, 2.50f);
    style.IndentSpacing                     = 25;
    style.ScrollbarSize                     = 15;
    style.GrabMinSize                       = 10;
    style.WindowBorderSize                  = 1;
    style.ChildBorderSize                   = 1;
    style.PopupBorderSize                   = 1;
    style.FrameBorderSize                   = 1;
    style.TabBorderSize                     = 1;
    style.WindowRounding                    = 7;
    style.ChildRounding                     = 4;
    style.FrameRounding                     = 3;
    style.PopupRounding                     = 4;
    style.ScrollbarRounding                 = 9;
    style.GrabRounding                      = 3;
    style.LogSliderDeadzone                 = 4;
    style.TabRounding                       = 4;

    style.ScaleAllSizes(_scalingFactor);
}

void Window::ImNodesStyleSetup() {
    // Setup Theme

    auto &colors = ImNodes::GetStyle().Colors;
    colors[ImNodesCol_NodeBackground]         = IM_COL32(30, 30, 30, 255);
    colors[ImNodesCol_NodeBackgroundHovered]  = IM_COL32(30, 30, 30, 255);
    colors[ImNodesCol_NodeBackgroundSelected] = IM_COL32(35, 35, 35, 255);
    colors[ImNodesCol_NodeOutline]            = IM_COL32(20, 20, 20, 255);
    colors[ImNodesCol_Link]                   = IM_COL32(75, 75, 75, 255);
    colors[ImNodesCol_LinkHovered]            = IM_COL32(75, 75, 75, 255);
    colors[ImNodesCol_LinkSelected]           = IM_COL32(75, 75, 75, 255);
}

void Window::ImNodesStyleScale() {
    auto& style = ImNodes::GetStyle();

    style.NodePadding               = ImVec2(4.0f * _scalingFactor, 4.0f * _scalingFactor);
    style.PinCircleRadius           = 2.0f  * _scalingFactor;
    style.GridSpacing               = 20.0f * _scalingFactor;
    style.NodeBorderThickness       = 0.5f  * _scalingFactor;
    style.NodeCornerRounding        = 2.0f  * _scalingFactor;
    style.LinkThickness             = 1.5f  * _scalingFactor;
    style.PinLineThickness          = 0.5f  * _scalingFactor;
    style.LinkLineSegmentsPerLength = 0.2f  / _scalingFactor;
}

}  // namespace Jetstream::Render
