#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_MODALS_INFO_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_MODALS_INFO_HH

#include "../../components/modal_header.hh"
#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <functional>
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
        std::function<void(const std::string&)> onTitleChange;
        std::function<void(const std::string&)> onSummaryChange;
        std::function<void(const std::string&)> onAuthorChange;
        std::function<void(const std::string&)> onLicenseChange;
        std::function<void(const std::string&)> onDescriptionChange;
        std::function<void(const std::string&, std::function<void(std::string)>)> onBrowse;
        std::function<void(const std::string&)> onSave;
    };

    void update(Config config) {
        using TextInput = Sakura::TextInput;

        if (this->config.flowgraphId != config.flowgraphId) {
            filename = config.path;
        }
        this->config = std::move(config);
        header.update({
            .title = ICON_LC_INFO " Flowgraph Information",
            .description = "Edit the focused flowgraph metadata and save location.",
        });
        titleInput.update({
            .value = this->config.title,
            .submit = TextInput::Submit::OnEdit,
            .onChange = this->config.onTitleChange,
        });
        summaryInput.update({
            .value = this->config.summary,
            .submit = TextInput::Submit::OnEdit,
            .onChange = this->config.onSummaryChange,
        });
        authorInput.update({
            .value = this->config.author,
            .submit = TextInput::Submit::OnEdit,
            .onChange = this->config.onAuthorChange,
        });
        licenseInput.update({
            .value = this->config.license,
            .submit = TextInput::Submit::OnEdit,
            .onChange = this->config.onLicenseChange,
        });
        descriptionInput.update({
            .value = this->config.description,
            .submit = TextInput::Submit::OnEdit,
            .multiline = true,
            .height = 80.0f,
            .onChange = this->config.onDescriptionChange,
        });
        pathInput.update({
            .value = filename,
            .submit = TextInput::Submit::OnEdit,
            .onChange = [this](const std::string& value) {
                filename = value;
            },
        });
        browseButton.update({
            .str = "Browse File",
            .size = {Sakura::Fill, 0.0f},
            .onClick = [this]() {
                if (this->config.onBrowse) {
                    this->config.onBrowse(filename, [this](std::string path) {
                        filename = std::move(path);
                    });
                }
            },
        });
        saveButton.update({
            .str = ICON_LC_SAVE " Save Flowgraph",
            .size = {Sakura::Fill, 40.0f},
            .variant = Sakura::Button::Variant::Action,
            .onClick = [this]() {
                if (this->config.onSave) {
                    this->config.onSave(filename);
                }
            },
        });
        divider.update({});
        pathControls.update({
            .items = {
                {.child = &pathInput},
                {.child = &browseButton},
            },
        });
        titleField.update({
            .label = "Title",
            .description = "Display title for this flowgraph.",
            .control = &titleInput,
        });
        summaryField.update({
            .label = "Summary",
            .description = "Short one-line summary.",
            .control = &summaryInput,
        });
        authorField.update({
            .label = "Author",
            .description = "Flowgraph author or organization.",
            .control = &authorInput,
        });
        licenseField.update({
            .label = "License",
            .description = "License identifier for this flowgraph.",
            .control = &licenseInput,
        });
        descriptionField.update({
            .label = "Description",
            .description = "Longer flowgraph description.",
            .control = &descriptionInput,
        });
        pathField.update({
            .label = "File Path",
            .description = "Save path for this flowgraph file.",
            .control = &pathControls,
            .divider = false,
        });
        stack.update({
            .items = {
                {.child = &header},
                {.child = &titleField},
                {.child = &summaryField},
                {.child = &authorField},
                {.child = &licenseField},
                {.child = &descriptionField},
                {.child = &pathField},
                {.child = &divider},
                {.child = &saveButton},
            },
        });
        panel.update({
            .id = "FlowgraphInfo",
            .content = &stack,
            .backgroundColorKey = "popup_bg",
        });
    }

    Sakura::Component& component() {
        return panel;
    }

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
