#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_MODALS_INFO_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_MODALS_INFO_HH

#include "../../components/modal_header.hh"
#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <optional>
#include <string>

namespace Jetstream {

struct FlowgraphInfoView {
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
    };

    void update(Config config);
    Sakura::Component& component();

 private:
    Config config;
    std::string filename;
    ModalHeader header;
    Sakura::TextInput titleInput;
    Sakura::TextInput summaryInput;
    Sakura::TextInput authorInput;
    Sakura::TextInput licenseInput;
    Sakura::TextInput descriptionInput;
    Sakura::TextInput pathInput;
    Sakura::Button browseButton;
    Sakura::Stack pathControls;
    Sakura::SettingField titleField;
    Sakura::SettingField summaryField;
    Sakura::SettingField authorField;
    Sakura::SettingField licenseField;
    Sakura::SettingField descriptionField;
    Sakura::SettingField pathField;
    Sakura::Divider divider;
    Sakura::Button saveButton;
    Sakura::Stack stack;
    Sakura::Panel panel;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_MODALS_INFO_HH
