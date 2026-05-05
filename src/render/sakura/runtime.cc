#include <jetstream/render/sakura/runtime.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct Runtime::Impl {
    ~Impl() {
        destroyNodeContexts();
    }

    void create(FontConfig fontConfig) {
        this->fontConfig = fontConfig;
        loadFonts();
        setupMarkdown();
    }

    void update(Config config) {
        this->config = std::move(config);
        applyImGuiStyle();
        applyImNodesStyle();
        setupMarkdown();
    }

    Context context() {
        return Context{
            .palette = activePalette(),
            .render = config.render,
            .fonts = fonts,
            .markdownConfig = Private::ToMarkdownConfigHandle(&markdownConfig),
            .nodeContextResolver = [this](const std::string& id) {
                return Private::ToNodeContextHandle(nodeContext(id));
            },
        };
    }

    void syncNodeContexts(const std::vector<std::string>& ids) {
        for (const auto& id : ids) {
            if (!nodeContexts.contains(id)) {
                nodeContexts[id] = ImNodes::CreateContext();
                applyImNodesStyle(nodeContexts[id]);
            }
        }

        if (nodeContexts.size() != ids.size()) {
            std::vector<std::string> staleIds;
            for (const auto& [id, _] : nodeContexts) {
                if (std::find(ids.begin(), ids.end(), id) == ids.end()) {
                    staleIds.push_back(id);
                }
            }
            for (const auto& id : staleIds) {
                ImNodes::DestroyContext(nodeContexts[id]);
                nodeContexts.erase(id);
            }
        }
    }

    ImNodesContext* nodeContext(const std::string& id) const {
        const auto it = nodeContexts.find(id);
        if (it == nodeContexts.end()) {
            return nullptr;
        }
        return it->second;
    }

    const Palette& activePalette() const {
        return config.palette ? *config.palette : EmptyPalette();
    }

    ImVec4 color(const std::string& key) const {
        return Private::ToImVec4(activePalette().at(key));
    }

    F32 scalingFactor() const {
        if (!config.render) {
            JST_FATAL("Sakura::Runtime is missing render window.");
            std::abort();
        }
        return config.render->scalingFactor();
    }

    void setupMarkdown() {
        markdownConfig.linkCallback        = &Runtime::Impl::markdownLinkCallback;
        markdownConfig.tooltipCallback     = nullptr;
        markdownConfig.imageCallback       = nullptr;
        markdownConfig.linkIcon            = ICON_FA_LINK;
        markdownConfig.headingFormats[0]   = {Private::NativeFont(fonts.h1), true};
        markdownConfig.headingFormats[1]   = {Private::NativeFont(fonts.h2), true};
        markdownConfig.headingFormats[2]   = {Private::NativeFont(fonts.bold), false};
        markdownConfig.userData            = this;
        markdownConfig.formatCallback      = &Runtime::Impl::markdownFormatCallback;
    }

    static void markdownLinkCallback(ImGui::MarkdownLinkCallbackData data) {
        if (!data.isImage) {
            const std::string url(data.link, data.linkLength);
            Platform::OpenUrl(url);
        }
    }

    static void markdownFormatCallback(const ImGui::MarkdownFormatInfo& mdInfo, bool start) {
        switch (mdInfo.type) {
            case ImGui::MarkdownFormatType::NORMAL_TEXT:
                break;
            case ImGui::MarkdownFormatType::EMPHASIS: {
                ImGui::MarkdownHeadingFormat fmt;
                fmt = mdInfo.config->headingFormats[ImGui::MarkdownConfig::NUMHEADINGS - 1];
                if (start) {
                    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
                    if (fmt.font) {
                        ImGui::PushFont(fmt.font, 0.0f);
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
                if (mdInfo.level > ImGui::MarkdownConfig::NUMHEADINGS) {
                    fmt = mdInfo.config->headingFormats[ImGui::MarkdownConfig::NUMHEADINGS - 1];
                } else {
                    fmt = mdInfo.config->headingFormats[mdInfo.level - 1];
                }
                if (start) {
                    if (fmt.font) {
                        ImGui::PushFont(fmt.font, 0.0f);
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
                const ImVec4 linkColor(0.278f, 0.498f, 0.937f, 1.0f);
                if (start) {
                    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::ColorConvertFloat4ToU32(linkColor));
                } else {
                    ImGui::PopStyleColor();
                    if (mdInfo.itemHovered) {
                        ImGui::UnderLine(ImGui::ColorConvertFloat4ToU32(linkColor));
                    }
                }
                break;
            }
        }
    }

    void loadFonts() {
        if (!fontConfig.body.valid() || !fontConfig.bold.valid()) {
            return;
        }

        auto& io = ImGui::GetIO();

        ImFontConfig fontConfig;
        fontConfig.OversampleH = 5;
        fontConfig.OversampleV = 5;
        fontConfig.FontLoaderFlags = 1;
        io.Fonts->Clear();

        fonts.body = Private::ToFontHandle(io.Fonts->AddFontFromMemoryCompressedTTF(
            this->fontConfig.body.data,
            static_cast<int>(this->fontConfig.body.size),
            15.0f * scalingFactor(),
            &fontConfig,
            nullptr));

        if (this->fontConfig.iconRegular.valid() || this->fontConfig.iconSolid.valid()) {
            ImFontConfig iconFontConfig;
            iconFontConfig.OversampleH = 5;
            iconFontConfig.OversampleV = 5;
            iconFontConfig.FontLoaderFlags = 1;
            iconFontConfig.MergeMode = true;
            iconFontConfig.GlyphMinAdvanceX = 15.0f * scalingFactor();
            iconFontConfig.GlyphOffset = {0.0f, 2.0f};

            static const ImWchar iconRanges[] = {ICON_MIN_FA, ICON_MAX_FA, 0};

            if (this->fontConfig.iconRegular.valid()) {
                io.Fonts->AddFontFromMemoryCompressedTTF(this->fontConfig.iconRegular.data,
                                                         static_cast<int>(this->fontConfig.iconRegular.size),
                                                         15.0f * scalingFactor(),
                                                         &iconFontConfig,
                                                         iconRanges);
            }
            if (this->fontConfig.iconSolid.valid()) {
                io.Fonts->AddFontFromMemoryCompressedTTF(this->fontConfig.iconSolid.data,
                                                         static_cast<int>(this->fontConfig.iconSolid.size),
                                                         15.0f * scalingFactor(),
                                                         &iconFontConfig,
                                                         iconRanges);
            }
        }

        fonts.h1 = Private::ToFontHandle(io.Fonts->AddFontFromMemoryCompressedTTF(
            this->fontConfig.bold.data,
            static_cast<int>(this->fontConfig.bold.size),
            15.0f * scalingFactor() * 1.15f,
            &fontConfig,
            nullptr));
        fonts.h2 = Private::ToFontHandle(io.Fonts->AddFontFromMemoryCompressedTTF(
            this->fontConfig.bold.data,
            static_cast<int>(this->fontConfig.bold.size),
            15.0f * scalingFactor() * 1.10f,
            &fontConfig,
            nullptr));
        fonts.bold = Private::ToFontHandle(io.Fonts->AddFontFromMemoryCompressedTTF(
            this->fontConfig.bold.data,
            static_cast<int>(this->fontConfig.bold.size),
            15.0f * scalingFactor() * 1.04f,
            &fontConfig,
            nullptr));
    }

    void applyImGuiStyle() const {
        if (!config.palette) {
            return;
        }

        auto& io = ImGui::GetIO();
        io.ConfigWindowsMoveFromTitleBarOnly = true;

        auto& style = ImGui::GetStyle();
        auto& colors = style.Colors;
        colors[ImGuiCol_Text]                      = color("text_primary");
        colors[ImGuiCol_TextDisabled]              = color("text_secondary");
        colors[ImGuiCol_TextSelectedBg]            = color("text_selected_bg");
        colors[ImGuiCol_WindowBg]                  = color("background");
        colors[ImGuiCol_PopupBg]                   = color("popup_bg");
        colors[ImGuiCol_ModalWindowDimBg]          = color("modal_dim");
        colors[ImGuiCol_Border]                    = color("border");
        colors[ImGuiCol_BorderShadow]              = color("border_shadow");
        colors[ImGuiCol_FrameBg]                   = color("card");
        colors[ImGuiCol_FrameBgHovered]            = color("frame_bg_hovered");
        colors[ImGuiCol_FrameBgActive]             = color("frame_bg_active");
        colors[ImGuiCol_TitleBg]                   = color("panel");
        colors[ImGuiCol_TitleBgActive]             = color("title_bg_active");
        colors[ImGuiCol_TitleBgCollapsed]          = color("title_bg_collapsed");
        colors[ImGuiCol_MenuBarBg]                 = color("background");
        colors[ImGuiCol_ScrollbarBg]               = color("scrollbar_bg");
        colors[ImGuiCol_ScrollbarGrab]             = color("scrollbar_grab");
        colors[ImGuiCol_ScrollbarGrabHovered]      = color("scrollbar_grab_hovered");
        colors[ImGuiCol_ScrollbarGrabActive]       = color("scrollbar_grab_active");
        colors[ImGuiCol_CheckMark]                 = color("accent_color");
        colors[ImGuiCol_SliderGrab]                = color("accent_color");
        colors[ImGuiCol_SliderGrabActive]          = color("accent_active");
        colors[ImGuiCol_Button]                    = color("button");
        colors[ImGuiCol_ButtonHovered]             = color("button_hovered");
        colors[ImGuiCol_ButtonActive]              = color("button_active");
        colors[ImGuiCol_Header]                    = color("header");
        colors[ImGuiCol_HeaderHovered]             = color("header_hovered");
        colors[ImGuiCol_HeaderActive]              = color("header_active");
        colors[ImGuiCol_Separator]                 = color("separator");
        colors[ImGuiCol_SeparatorHovered]          = color("separator_hovered");
        colors[ImGuiCol_SeparatorActive]           = color("separator_active");
        colors[ImGuiCol_ResizeGrip]                = color("resize_grip");
        colors[ImGuiCol_ResizeGripHovered]         = color("resize_grip_hovered");
        colors[ImGuiCol_ResizeGripActive]          = color("resize_grip_active");
        colors[ImGuiCol_Tab]                       = color("tab");
        colors[ImGuiCol_TabHovered]                = color("tab_hovered");
        colors[ImGuiCol_TabSelected]               = color("tab_selected");
        colors[ImGuiCol_TabDimmed]                 = color("tab_dimmed");
        colors[ImGuiCol_TabDimmedSelected]         = color("tab_dimmed_selected");
        colors[ImGuiCol_TabDimmedSelectedOverline] = color("tab_dimmed_selected");
        colors[ImGuiCol_TabSelectedOverline]       = color("tab_selected");
        colors[ImGuiCol_DockingPreview]            = color("docking_preview");
        colors[ImGuiCol_DockingEmptyBg]            = color("docking_empty_bg");
        colors[ImGuiCol_PlotLines]                 = color("plot_lines");
        colors[ImGuiCol_PlotLinesHovered]          = color("plot_lines_hovered");
        colors[ImGuiCol_PlotHistogram]             = color("plot_histogram");
        colors[ImGuiCol_PlotHistogramHovered]      = color("plot_histogram_hovered");
        colors[ImGuiCol_TableHeaderBg]             = color("table_header_bg");
        colors[ImGuiCol_TableBorderStrong]         = color("table_border_strong");
        colors[ImGuiCol_TableBorderLight]          = color("table_border_light");
        colors[ImGuiCol_TableRowBg]                = color("table_row_bg");
        colors[ImGuiCol_TableRowBgAlt]             = color("table_row_bg_alt");
        colors[ImGuiCol_DragDropTarget]            = color("drag_drop_target");
        colors[ImGuiCol_NavCursor]                 = color("accent_color");
        colors[ImGuiCol_NavWindowingHighlight]     = color("nav_windowing_highlight");
        colors[ImGuiCol_NavWindowingDimBg]         = color("nav_windowing_dim_bg");

        style._MainScale                        = 1.0f;
        style.WindowPadding                     = ImVec2(12.00f, 12.00f);
        style.FramePadding                      = ImVec2(12.00f, 4.00f);
        style.ItemSpacing                       = ImVec2(8.00f, 8.00f);
        style.ItemInnerSpacing                  = ImVec2(8.00f, 6.00f);
        style.TouchExtraPadding                 = ImVec2(0.00f, 0.00f);
        style.CellPadding                       = ImVec2(6.00f, 4.00f);
        style.IndentSpacing                     = 20.0f;
        style.ColumnsMinSpacing                 = 6.0f;
        style.ScrollbarSize                     = 12.0f;
        style.GrabMinSize                       = 12.0f;
        style.WindowMinSize                     = ImVec2(32.0f, 32.0f);
        style.WindowBorderSize                  = 0.5f;
        style.WindowBorderHoverPadding          = 4.0f;
        style.ChildBorderSize                   = 0.5f;
        style.PopupBorderSize                   = 0.5f;
        style.FrameBorderSize                   = 0.0f;
        style.ImageBorderSize                   = 0.0f;
        style.TabBorderSize                     = 0.0f;
        style.TabBarBorderSize                  = 1.0f;
        style.TabBarOverlineSize                = 1.0f;
        style.TabMinWidthBase                   = 1.0f;
        style.TabMinWidthShrink                 = 80.0f;
        style.TabCloseButtonMinWidthSelected    = -1.0f;
        style.TabCloseButtonMinWidthUnselected  = 0.0f;
        style.WindowRounding                    = 12.0f;
        style.ChildRounding                     = 0.0f;
        style.FrameRounding                     = 8.0f;
        style.PopupRounding                     = 18.0f;
        style.ScrollbarRounding                 = 18.0f;
        style.GrabRounding                      = 8.0f;
        style.LogSliderDeadzone                 = 4.0f;
        style.TabRounding                       = 10.0f;
        style.TreeLinesRounding                 = 0.0f;
        style.WindowTitleAlign                  = ImVec2(0.5f, 0.5f);
        style.SeparatorTextPadding              = ImVec2(20.0f, 3.0f);
        style.DockingSeparatorSize              = 2.0f;
        style.DisplayWindowPadding              = ImVec2(19.0f, 19.0f);
        style.DisplaySafeAreaPadding            = ImVec2(3.0f, 3.0f);
        style.MouseCursorScale                  = 1.0f;
        style.CircleTessellationMaxError        = 0.1f;
        style.ScaleAllSizes(scalingFactor());
    }

    void applyImNodesStyle() const {
        for (const auto& [_, context] : nodeContexts) {
            applyImNodesStyle(context);
        }
    }

    void applyImNodesStyle(ImNodesContext* context) const {
        if (!config.palette || !context) {
            return;
        }

        const auto previousContext = ImNodes::GetCurrentContext();
        ImNodes::SetCurrentContext(context);

        auto& colors = ImNodes::GetStyle().Colors;
        colors[ImNodesCol_NodeBackground]         = ImGui::ColorConvertFloat4ToU32(color("node_background"));
        colors[ImNodesCol_NodeBackgroundHovered]  = ImGui::ColorConvertFloat4ToU32(color("node_background"));
        colors[ImNodesCol_NodeBackgroundSelected] = ImGui::ColorConvertFloat4ToU32(color("node_background"));
        colors[ImNodesCol_NodeOutline]            = ImGui::ColorConvertFloat4ToU32(color("node_outline"));
        colors[ImNodesCol_TitleBar]               = ImGui::ColorConvertFloat4ToU32(color("node_title_bar"));
        colors[ImNodesCol_TitleBarHovered]        = ImGui::ColorConvertFloat4ToU32(color("node_title_bar"));
        colors[ImNodesCol_TitleBarSelected]       = ImGui::ColorConvertFloat4ToU32(color("node_title_bar"));
        colors[ImNodesCol_Pin]                    = ImGui::ColorConvertFloat4ToU32(color("node_pin"));
        colors[ImNodesCol_PinHovered]             = ImGui::ColorConvertFloat4ToU32(color("node_pin"));
        colors[ImNodesCol_Link]                   = ImGui::ColorConvertFloat4ToU32(color("node_link"));
        colors[ImNodesCol_LinkHovered]            = ImGui::ColorConvertFloat4ToU32(color("node_link"));
        colors[ImNodesCol_LinkSelected]           = ImGui::ColorConvertFloat4ToU32(color("node_link"));
        colors[ImNodesCol_GridLine]               = ImGui::ColorConvertFloat4ToU32(color("grid_line"));
        colors[ImNodesCol_GridBackground]         = ImGui::ColorConvertFloat4ToU32(color("grid_background"));

        const F32 scale = scalingFactor();
        auto& style = ImNodes::GetStyle();
        style.NodePadding               = ImVec2(8.0f * scale,  8.0f * scale);
        style.PinCircleRadius           = 4.0f  * scale;
        style.GridSpacing               = 23.0f * scale;
        style.NodeBorderThickness       = 2.0f  * scale;
        style.NodeCornerRounding        = 12.0f * scale;
        style.LinkThickness             = 1.5f  * scale;
        style.PinLineThickness          = 1.0f  * scale;
        style.LinkLineSegmentsPerLength = 0.2f  / scale;
        style.MiniMapOffset             = ImVec2(8.0f * scale, 8.0f * scale);

        ImNodes::SetCurrentContext(previousContext);
    }

    void destroyNodeContexts() {
        for (const auto& [_, context] : nodeContexts) {
            ImNodes::DestroyContext(context);
        }
        nodeContexts.clear();
    }

    FontConfig fontConfig;
    Config config;
    Fonts fonts;
    ImGui::MarkdownConfig markdownConfig;
    std::unordered_map<std::string, ImNodesContext*> nodeContexts;
};

Runtime::Runtime() {
    this->impl = std::make_unique<Impl>();
}

Runtime::~Runtime() = default;
Runtime::Runtime(Runtime&&) noexcept = default;
Runtime& Runtime::operator=(Runtime&&) noexcept = default;

bool Runtime::FontData::valid() const {
    return data && size > 0;
}

void Runtime::create(FontConfig fontConfig) {
    this->impl->create(std::move(fontConfig));
}

void Runtime::update(Config config) {
    this->impl->update(std::move(config));
}

Context Runtime::context() {
    return this->impl->context();
}

void Runtime::syncNodeContexts(const std::vector<std::string>& ids) {
    this->impl->syncNodeContexts(ids);
}

}  // namespace Jetstream::Sakura
