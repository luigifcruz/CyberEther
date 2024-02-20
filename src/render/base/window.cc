#include "jetstream/render/base/window.hh"
#include "jetstream/platform.hh"

#include "compressed_jbmm.hh"
#include "compressed_jbmb.hh"
#include "compressed_fa.hh"

namespace Jetstream::Render {

Result Window::create() {
    // Set variables.

    _scalingFactor = 0.0f;
    _previousScalingFactor = 0.0f;
    graphicalLoopThreadStarted = false;

    // Lock the frame queue.
    newFrameQueueMutex.lock();

    // Call underlying create.
    const auto& res = underlyingCreate();

    // Unlock the frame queue.
    newFrameQueueMutex.unlock();

    return res;
}

Result Window::destroy() {
    graphicalLoopThreadStarted = false;

    // Lock the frame queue.
    newFrameQueueMutex.lock();

    // Call underlying destroy.
    const auto& res = underlyingDestroy();

    // Unlock the frame queue.
    newFrameQueueMutex.unlock();

    return res;
}

Result Window::begin() {
    // Process surface bind and unbind queue.
    JST_CHECK(processSurfaceUnbindQueue());
    JST_CHECK(processSurfaceBindQueue());

    // Record graphical thread ID.
    graphicalLoopThreadStarted = true;
    graphicalLoopThreadId = std::this_thread::get_id();

    // Lock the frame queue.
    newFrameQueueMutex.lock();

    // Call frame begin.
    const auto& res = underlyingBegin();

    // Unlock the frame queue if failed.
    if (res != Result::SUCCESS) {
        newFrameQueueMutex.unlock();
    }
    
    return res;
}

Result Window::end() {
    // Call frame end.
    const auto& res = underlyingEnd();

    // Unlock the frame queue.
    newFrameQueueMutex.unlock();

    return res;
}

Result Window::synchronize() {
    // Lock the frame queue.
    newFrameQueueMutex.lock();

    // Call frame synchronize.
    const auto& res = underlyingSynchronize();

    // Unlock the frame queue.
    newFrameQueueMutex.unlock();

    return res;
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

    // Wait completion.
    if (graphicalLoopThreadId != std::this_thread::get_id()) {
        while (!surfaceUnbindQueue.empty()) {
            std::this_thread::yield();
        }
    }
    // Call the function directly as fallback.
    else {
        JST_CHECK(processSurfaceUnbindQueue());
    }

    return Result::SUCCESS;
}

Result Window::processSurfaceBindQueue() {
    std::lock_guard<std::mutex> lock(newFrameQueueMutex);

    while (!surfaceBindQueue.empty()) {
        JST_CHECK(bindSurface(surfaceBindQueue.front()));
        surfaceBindQueue.pop();
    }

    return Result::SUCCESS;
}

Result Window::processSurfaceUnbindQueue() {
    std::lock_guard<std::mutex> lock(newFrameQueueMutex);

    while (!surfaceUnbindQueue.empty()) {
        JST_CHECK(unbindSurface(surfaceUnbindQueue.front()));
        surfaceUnbindQueue.pop();
    }

    return Result::SUCCESS;
}

void Window::ScaleStyle(const Viewport::Generic& viewport) {
    if (_previousScalingFactor == 0.0f) {
        _scalingFactor = viewport.scale(config.scale);

        ImGuiLoadFonts();
        ImGuiMarkdownSetup();
        ImGuiStyleSetup();
        ImNodesStyleSetup();
    }

    if (_scalingFactor != _previousScalingFactor) {
        ImGuiStyleScale();
        ImNodesStyleScale();
    }

    _previousScalingFactor = _scalingFactor;
}

void Window::ImGuiMarkdownLinkCallback(ImGui::MarkdownLinkCallbackData data) {
    std::string url(data.link, data.linkLength);
    if(!data.isImage) {
        Platform::OpenUrl(url);
    }
}

void Window::ImGuiMarkdownFormatCallback(const ImGui::MarkdownFormatInfo& md_info, bool start) {
    switch (md_info.type) {
        case ImGui::MarkdownFormatType::NORMAL_TEXT:
            break;
        case ImGui::MarkdownFormatType::EMPHASIS: {
            ImGui::MarkdownHeadingFormat fmt;
            fmt = md_info.config->headingFormats[ImGui::MarkdownConfig::NUMHEADINGS - 1];
            if (start) {
                ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
                if (fmt.font) {
                    ImGui::PushFont( fmt.font );
                }
            } else {
                if (fmt.font) {
                    ImGui::PopFont();
                }
                ImGui::PopStyleColor();
            }
            break;
        }
        case ImGui::MarkdownFormatType::HEADING: {
            ImGui::MarkdownHeadingFormat fmt;
            if (md_info.level > ImGui::MarkdownConfig::NUMHEADINGS) {
                fmt = md_info.config->headingFormats[ImGui::MarkdownConfig::NUMHEADINGS - 1];
            } else {
                fmt = md_info.config->headingFormats[md_info.level - 1];
            }
            if (start) {
                if (fmt.font) {
                    ImGui::PushFont(fmt.font);
                }
            } else {
                if (fmt.separator) {
                    ImGui::Separator();
                }
                if (fmt.font) {
                    ImGui::PopFont();
                }
            }
            break;
        }
        case ImGui::MarkdownFormatType::UNORDERED_LIST:
            break;
        case ImGui::MarkdownFormatType::LINK: {
            if (start) {
                ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(71, 127, 239, 255));
            } else {
                ImGui::PopStyleColor();
                if (md_info.itemHovered) {
                    ImGui::UnderLine(IM_COL32(71, 127, 239, 255));
                }
            }
            break;
        }
    }
}

void Window::ImGuiMarkdownSetup() {
    _markdownConfig.linkCallback        = &Window::ImGuiMarkdownLinkCallback;
    _markdownConfig.tooltipCallback     = nullptr;
    _markdownConfig.imageCallback       = nullptr;
    _markdownConfig.linkIcon            = ICON_FA_LINK;
    _markdownConfig.headingFormats[0]   = { _h1Font, true };
    _markdownConfig.headingFormats[1]   = { _h2Font, true };
    _markdownConfig.headingFormats[2]   = { _boldFont, false };
    _markdownConfig.userData            = this;
    _markdownConfig.formatCallback      = &Window::ImGuiMarkdownFormatCallback;
}

void Window::ImGuiLoadFonts() {
    auto& io = ImGui::GetIO();

    ImFontConfig font_config;
    font_config.OversampleH = 5;
    font_config.OversampleV = 5;
    font_config.FontBuilderFlags = 1;
    io.Fonts->Clear();

    _bodyFont = io.Fonts->AddFontFromMemoryCompressedTTF(jbmm_compressed_data,
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

    _h1Font = io.Fonts->AddFontFromMemoryCompressedTTF(jbmb_compressed_data,
                                                       jbmb_compressed_size,
                                                       15.0f * _scalingFactor * 1.15,
                                                       &font_config,
                                                       nullptr);

    _h2Font = io.Fonts->AddFontFromMemoryCompressedTTF(jbmb_compressed_data,
                                                       jbmb_compressed_size,
                                                       15.0f * _scalingFactor * 1.10,
                                                       &font_config,
                                                       nullptr);

    _boldFont = io.Fonts->AddFontFromMemoryCompressedTTF(jbmb_compressed_data,
                                                         jbmb_compressed_size,
                                                         15.0f * _scalingFactor * 1.04,
                                                         &font_config,
                                                         nullptr);
}

void Window::ImGuiStyleSetup() {
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
