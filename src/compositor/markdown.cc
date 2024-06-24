#include "jetstream/compositor.hh"
#include "jetstream/platform.hh"

namespace Jetstream {

void Compositor::ImGuiMarkdownStyleSetup(const F32&) {
    _markdownConfig.linkCallback        = &Compositor::ImGuiMarkdownLinkCallback;
    _markdownConfig.tooltipCallback     = nullptr;
    _markdownConfig.imageCallback       = nullptr;
    _markdownConfig.linkIcon            = ICON_FA_LINK;
    _markdownConfig.headingFormats[0]   = { _h1Font, true };
    _markdownConfig.headingFormats[1]   = { _h2Font, true };
    _markdownConfig.headingFormats[2]   = { _boldFont, false };
    _markdownConfig.userData            = this;
    _markdownConfig.formatCallback      = &Compositor::ImGuiMarkdownFormatCallback;
}

void Compositor::ImGuiMarkdownLinkCallback(ImGui::MarkdownLinkCallbackData data) {
    std::string url(data.link, data.linkLength);
    if(!data.isImage) {
        Platform::OpenUrl(url);
    }
}

void Compositor::ImGuiMarkdownFormatCallback(const ImGui::MarkdownFormatInfo& md_info, bool start) {
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

}  // namespace Jetstream