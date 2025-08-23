#include "jetstream/compositor.hh"
#include "jetstream/instance.hh"

#include "resources/fonts/compressed_jbmm.hh"
#include "resources/fonts/compressed_jbmb.hh"
#include "resources/fonts/compressed_fa.hh"

namespace Jetstream {

void Compositor::ImGuiLoadFonts() {
    const auto& scalingFactor = instance.window().scalingFactor();
    auto& io = ImGui::GetIO();

    ImFontConfig font_config;
    font_config.OversampleH = 5;
    font_config.OversampleV = 5;
    font_config.FontBuilderFlags = 1;
    io.Fonts->Clear();

    _bodyFont = io.Fonts->AddFontFromMemoryCompressedTTF(jbmm_compressed_data,
                                                         jbmm_compressed_size,
                                                         15.0f * scalingFactor,
                                                         &font_config,
                                                         nullptr);

    ImFontConfig icon_font_config;
    icon_font_config.OversampleH = 5;
    icon_font_config.OversampleV = 5;
    icon_font_config.FontBuilderFlags = 1;
    icon_font_config.MergeMode = true;
    icon_font_config.GlyphMinAdvanceX = 15.0f * scalingFactor;
    icon_font_config.GlyphOffset = { 0.0f, 2.0f };

    static const ImWchar icon_ranges[] = { ICON_MIN_FA, ICON_MAX_FA, 0 };

    io.Fonts->AddFontFromMemoryCompressedTTF(far_compressed_data,
                                             far_compressed_size,
                                             15.0f * scalingFactor,
                                             &icon_font_config,
                                             icon_ranges);

    io.Fonts->AddFontFromMemoryCompressedTTF(fas_compressed_data,
                                             fas_compressed_size,
                                             15.0f * scalingFactor,
                                             &icon_font_config,
                                             icon_ranges);

    _h1Font = io.Fonts->AddFontFromMemoryCompressedTTF(jbmb_compressed_data,
                                                       jbmb_compressed_size,
                                                       15.0f * scalingFactor * 1.15,
                                                       &font_config,
                                                       nullptr);

    _h2Font = io.Fonts->AddFontFromMemoryCompressedTTF(jbmb_compressed_data,
                                                       jbmb_compressed_size,
                                                       15.0f * scalingFactor * 1.10,
                                                       &font_config,
                                                       nullptr);

    _boldFont = io.Fonts->AddFontFromMemoryCompressedTTF(jbmb_compressed_data,
                                                         jbmb_compressed_size,
                                                         15.0f * scalingFactor * 1.04,
                                                         &font_config,
                                                         nullptr);
}

void Compositor::ImGuiStyleSetup() {
    // Setup Options

    auto& io = ImGui::GetIO();
    io.ConfigWindowsMoveFromTitleBarOnly = true;

    // Setup Theme

    // Inspired from: https://github.com/ocornut/imgui/issues/707#issuecomment-917151020
    // Original work is licensed under CC BY 4.0 DEED by Jan Bielak (janekb04).
    // Tweaked by Luigi F. Cruz (luigifcruz) to match CyberEther style.

    auto &colors = ImGui::GetStyle().Colors;
    colors[ImGuiCol_Text]                   = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
    colors[ImGuiCol_TextDisabled]           = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
    colors[ImGuiCol_WindowBg]               = ImVec4(0.10f, 0.10f, 0.10f, 1.00f);
    colors[ImGuiCol_ChildBg]                = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_PopupBg]                = ImVec4(0.18f, 0.18f, 0.18f, 1.00f);
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
}

void Compositor::ImGuiStyleScale() {
    auto& style = ImGui::GetStyle();
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
}

}  // namespace Jetstream
