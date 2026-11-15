#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_MODALS_INFO_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_MODALS_INFO_HH

#include "../../components/compact_button.hh"
#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <string>
#include <string_view>
#include <utility>

namespace Jetstream {

struct FlowgraphInfoView {
    static constexpr F32 Width = 940.0f;
    static constexpr F32 DetailsWidth = 350.0f;
    static constexpr F32 DescriptionHeight = 432.0f;
    static constexpr F32 SummaryHeight = 66.0f;
    static constexpr F32 LabelText = 15.5f;
    static constexpr F32 InputText = 14.0f;
    static constexpr F32 InputIcon = 13.0f;
    static constexpr F32 AreaInset = 6.0f;
    static constexpr F32 AreaBorder = 1.5f;
    static constexpr F32 AreaShadowSoftness = 4.0f;
    static constexpr F32 AreaShadowDrop = 1.0f;
    static constexpr const char* AreaFill = "pill_field_bg";
    static constexpr F32 ButtonHeight = 34.0f;

    struct Config {
        std::string flowgraphId;
        std::string title;
        std::string summary;
        std::string author;
        std::string license;
        std::string description;
        std::string path;
        Sakura::Callback<void(const std::string&)> onTitleChange;
        Sakura::Callback<void(const std::string&)> onSummaryChange;
        Sakura::Callback<void(const std::string&)> onAuthorChange;
        Sakura::Callback<void(const std::string&)> onLicenseChange;
        Sakura::Callback<void(const std::string&)> onDescriptionChange;
        Sakura::Callback<void(const std::string&, Sakura::Callback<void(std::string)>)> onBrowse;
        Sakura::Callback<void(const std::string&)> onSave;
        Sakura::Callback<void()> onClose;
    };

    void update(Config config);
    Sakura::Component& component();

 private:
    bool canSave() const;
    void label(Sakura::Text& text, const std::string& name);
    void field(Sakura::Stack& stack, Sakura::Text& text, const std::string& name, Sakura::Component& control);

    Config config;
    std::string filename;
    bool previewing = false;
    Sakura::Text icon;
    Sakura::Text title;
    Sakura::Stack titleRow;
    Sakura::Text description;
    Sakura::Stack headerText;
    Sakura::Stack header;
    Sakura::Text titleLabel;
    Sakura::PillInput titleInput;
    Sakura::Stack titleField;
    Sakura::Text summaryLabel;
    Sakura::CodeEditor summaryInput;
    Sakura::Text summaryIcon;
    Sakura::Spacer summaryIconInset;
    Sakura::Spacer summaryIconDrop;
    Sakura::Stack summaryIconColumn;
    Sakura::Stack summaryRow;
    Sakura::Div summaryCard;
    Sakura::Stack summaryField;
    Sakura::Text authorLabel;
    Sakura::PillInput authorInput;
    Sakura::Stack authorField;
    Sakura::Text licenseLabel;
    Sakura::PillInput licenseInput;
    Sakura::Stack licenseField;
    Sakura::Text pathLabel;
    Sakura::PillInput pathInput;
    CompactButton browseButton;
    Sakura::Stack pathRow;
    Sakura::Text pathHint;
    Sakura::Stack pathField;
    Sakura::Stack detailsColumn;
    Sakura::PillTabs descriptionTabs;
    Sakura::CodeEditor descriptionEditor;
    Sakura::Markdown descriptionPreview;
    Sakura::Div descriptionCard;
    Sakura::Text descriptionOffset;
    Sakura::Stack descriptionColumn;
    Sakura::Stack body;
    Sakura::Spacer footerSpacer;
    Sakura::PillButton cancel;
    Sakura::PillButton save;
    Sakura::Stack footer;
    Sakura::Stack stack;
    Sakura::Panel panel;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_MODALS_INFO_HH
