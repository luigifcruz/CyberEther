#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_MODALS_INFO_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_MODALS_INFO_HH

#include "../../components/modal_header.hh"
#include "jetstream/render/sakura/sakura.hh"
#include "jetstream/render/tools/imgui_icons_ext.hh"

#include <functional>
#include <optional>
#include <string>

namespace Jetstream {

struct FlowgraphInfoView : public Sakura::Component {
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
        if (this->config.flowgraphId != config.flowgraphId) {
            filename = config.path;
        }
        this->config = std::move(config);

        header.update({
            .id = "FlowgraphInfoHeader",
            .title = ICON_FA_CIRCLE_INFO " Flowgraph Information",
            .description = "Edit the focused flowgraph metadata and save location.",
        });
        titleField.update({
            .id = "FlowgraphInfoTitleField",
            .label = "Title",
            .description = "Display title for this flowgraph.",
        });
        summaryField.update({
            .id = "FlowgraphInfoSummaryField",
            .label = "Summary",
            .description = "Short one-line summary.",
        });
        authorField.update({
            .id = "FlowgraphInfoAuthorField",
            .label = "Author",
            .description = "Flowgraph author or organization.",
        });
        licenseField.update({
            .id = "FlowgraphInfoLicenseField",
            .label = "License",
            .description = "License identifier for this flowgraph.",
        });
        descriptionField.update({
            .id = "FlowgraphInfoDescriptionField",
            .label = "Description",
            .description = "Longer flowgraph description.",
        });
        pathField.update({
            .id = "FlowgraphInfoPathField",
            .label = "File Path",
            .description = "Save path for this flowgraph file.",
            .divider = false,
        });

        titleInput.update({
            .id = "##flowgraph-info-title",
            .value = this->config.title,
            .submit = Sakura::TextInput::Submit::OnEdit,
            .onChange = this->config.onTitleChange,
        });
        summaryInput.update({
            .id = "##flowgraph-info-summary",
            .value = this->config.summary,
            .submit = Sakura::TextInput::Submit::OnEdit,
            .onChange = this->config.onSummaryChange,
        });
        authorInput.update({
            .id = "##flowgraph-info-author",
            .value = this->config.author,
            .submit = Sakura::TextInput::Submit::OnEdit,
            .onChange = this->config.onAuthorChange,
        });
        licenseInput.update({
            .id = "##flowgraph-info-license",
            .value = this->config.license,
            .submit = Sakura::TextInput::Submit::OnEdit,
            .onChange = this->config.onLicenseChange,
        });
        descriptionInput.update({
            .id = "##flowgraph-info-description",
            .value = this->config.description,
            .onChange = this->config.onDescriptionChange,
        });
        pathInput.update({
            .id = "##flowgraph-info-filename",
            .value = filename,
            .submit = Sakura::TextInput::Submit::OnEdit,
            .onChange = [this](const std::string& value) {
                filename = value;
            },
        });
        browseButton.update({
            .id = "FlowgraphInfoBrowse",
            .str = "Browse File",
            .size = {-1.0f, 0.0f},
            .onClick = [this]() {
                if (this->config.onBrowse) {
                    this->config.onBrowse(filename, [this](std::string path) {
                        filename = std::move(path);
                    });
                }
            },
        });
        saveButton.update({
            .id = "FlowgraphInfoSave",
            .str = ICON_FA_FLOPPY_DISK " Save Flowgraph",
            .size = {-1.0f, 40.0f},
            .variant = Sakura::Button::Variant::Action,
            .onClick = [this]() {
                if (this->config.onSave) {
                    this->config.onSave(filename);
                }
            },
        });
    }

    void render(const Sakura::Context& ctx) const {
        header.render(ctx);
        titleField.render(ctx, [this](const Sakura::Context& ctx) { titleInput.render(ctx); });
        summaryField.render(ctx, [this](const Sakura::Context& ctx) { summaryInput.render(ctx); });
        authorField.render(ctx, [this](const Sakura::Context& ctx) { authorInput.render(ctx); });
        licenseField.render(ctx, [this](const Sakura::Context& ctx) { licenseInput.render(ctx); });
        descriptionField.render(ctx, [this](const Sakura::Context& ctx) { descriptionInput.render(ctx); });
        pathField.render(ctx, [this](const Sakura::Context& ctx) {
            pathInput.render(ctx);
            browseButton.render(ctx);
        });
        divider.render(ctx);
        saveButton.render(ctx);
    }

 private:
    Config config;
    std::string filename;
    ModalHeader header;
    Sakura::SettingField titleField;
    Sakura::SettingField summaryField;
    Sakura::SettingField authorField;
    Sakura::SettingField licenseField;
    Sakura::SettingField descriptionField;
    Sakura::SettingField pathField;
    Sakura::TextInput titleInput;
    Sakura::TextInput summaryInput;
    Sakura::TextInput authorInput;
    Sakura::TextInput licenseInput;
    Sakura::TextArea descriptionInput;
    Sakura::TextInput pathInput;
    Sakura::Button browseButton;
    Sakura::Divider divider;
    Sakura::Button saveButton;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_MODALS_INFO_HH
