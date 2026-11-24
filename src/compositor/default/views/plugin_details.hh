#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_PLUGIN_DETAILS_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_PLUGIN_DETAILS_HH

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"
#include "jetstream/types.hh"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct PluginDetailsView {
    static constexpr F32 Width = 600.0f;
    static constexpr F32 IconSize = 64.0f;
    static constexpr F32 IconRounding = 16.0f;
    static constexpr F32 IconScale = 2.0f;
    static constexpr F32 IconInset = (IconSize - IconScale * Sakura::Typography::FontSize) * 0.5f;
    static constexpr F32 TitleScale = Sakura::Typography::Scale::Headline;
    static constexpr F32 SummaryScale = Sakura::Typography::Scale::Body;
    static constexpr F32 InfoTitleScale = Sakura::Typography::Scale::Heading;
    static constexpr F32 InfoTitleGap = Sakura::Spacing::Medium;
    static constexpr F32 InstallHeight = 38.0f;
    static constexpr F32 InstallText = 15.0f;
    static constexpr F32 IconButtonPad = 12.0f;
    static constexpr F32 DescriptionPadding = Sakura::Spacing::Small;
    static constexpr F32 AreaInset = 6.0f;
    static constexpr F32 AreaBorder = 1.5f;
    static constexpr F32 AreaShadowSoftness = 4.0f;
    static constexpr F32 AreaShadowDrop = 1.0f;
    static constexpr const char* AreaFill = "pill_field_bg";
    static constexpr F32 PanePadding = Sakura::Spacing::Medium;
    static constexpr F32 SummaryHeight = IconSize + Sakura::Spacing::Medium + InstallHeight;

    struct Detail {
        std::string label;
        std::string value;
    };

    struct Config {
        std::string id;
        std::string title;
        std::string summary;
        std::string description;
        std::vector<Detail> details;
        std::string installLabel;
        bool installDisabled = false;
        Sakura::Callback<void()> onInstall;
        Sakura::Callback<void()> onHowToInstall;
        Sakura::Callback<void()> onCopyLink;
        Sakura::Callback<void(const std::string&)> onOpenUrl;
        Sakura::Callback<void()> onClose;
    };

    void update(Config config);
    Sakura::Component& component();

 private:
    struct Row {
        Sakura::Text label;
        Sakura::Text value;
        Sakura::Stack stack;
        Sakura::Divider rule;
    };

    void updateHeader();
    void updateActions();
    void updateInfo();

    Config config;

    Sakura::Text appIcon;
    Sakura::Div appTile;
    Sakura::Text title;
    Sakura::Text summary;
    Sakura::Stack titles;
    Sakura::Stack header;

    Sakura::PillButton install;
    Sakura::PillButton guide;
    Sakura::PillButton copyLink;
    Sakura::Stack actions;
    Sakura::Stack summaryStack;
    Sakura::Spacer summaryGap;

    std::vector<std::unique_ptr<Row>> rows;
    Sakura::Text infoTitle;
    Sakura::Stack infoRows;
    Sakura::Stack info;
    Sakura::Markdown description;
    Sakura::Div descriptionCard;
    Sakura::Stack body;
    Sakura::ScrollArea scroll;

    Sakura::Overlay layers;
    Sakura::Dock dock;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_PLUGIN_DETAILS_HH
