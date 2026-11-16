#include "info.hh"

namespace Jetstream {

void FlowgraphInfoView::update(Config config) {
    using Text = Sakura::Text;
    using TextInput = Sakura::TextInput;

    if (this->config.flowgraphId != config.flowgraphId || this->config.path != config.path) {
        filename = config.path;
    }
    if (this->config.flowgraphId != config.flowgraphId) {
        previewing = false;
    }
    this->config = std::move(config);

    icon.update({
        .str = ICON_LC_INFO,
        .colorKey = "text_primary",
        .scale = Sakura::Typography::Scale::Heading,
    });
    title.update({
        .str = "Flowgraph Information",
        .font = Text::Font::Display,
        .scale = Sakura::Typography::Scale::Headline,
    });
    titleRow.update({
        .axis = Sakura::Stack::Axis::Horizontal,
        .spacing = Sakura::Spacing::Small,
        .items = {
            {.child = &icon},
            {.child = &title},
        },
    });
    description.update({
        .str = "Describe this flowgraph and choose where it is saved.",
        .font = Text::Font::Sans,
        .tone = Text::Tone::Secondary,
        .wrapped = true,
        .scale = Sakura::Typography::Scale::Lead,
    });
    headerText.update({
        .spacing = Sakura::Spacing::XSmall,
        .items = {
            {.child = &titleRow},
            {.child = &description},
        },
    });
    descriptionTabs.update({
        .items = {
            {.icon = ICON_LC_PENCIL, .label = "Write"},
            {.icon = ICON_LC_EYE, .label = "Preview"},
        },
        .selected = previewing ? 1u : 0u,
        .compact = true,
        .track = true,
        .onSelect = [this](U64 index) {
            previewing = index == 1;
        },
    });
    header.update({
        .axis = Sakura::Stack::Axis::Horizontal,
        .spacing = Sakura::Spacing::Medium,
        .items = {
            {.child = &headerText, .grow = 1.0f, .shrink = 1.0f},
            {.child = &descriptionTabs},
        },
    });

    titleInput.update({
        .icon = ICON_LC_TYPE,
        .value = this->config.title,
        .hint = "Untitled Flowgraph",
        .submit = TextInput::Submit::OnEdit,
        .onChange = this->config.onTitleChange,
    });
    summaryInput.update({
        .id = "FlowgraphInfoSummary",
        .value = this->config.summary,
        .height = SummaryHeight - 2.0f * AreaInset,
        .language = Sakura::CodeEditor::Language::Plain,
        .lineNumbers = false,
        .lineWrapping = true,
        .scrollPastEnd = false,
        .showActiveLine = false,
        .padding = Sakura::Padding{AreaPadding, SummaryPadding, AreaPadding, SummaryPadding},
        .fontSize = InputText,
        .fontName = Sakura::Typography::DisplayFont,
        .backgroundColorKey = AreaFill,
        .onChange = [this](std::string value) {
            if (this->config.onSummaryChange) {
                this->config.onSummaryChange(value);
            }
        },
    });
    summaryIcon.update({
        .str = ICON_LC_TEXT,
        .colorKey = "text_secondary",
        .scale = InputIcon / Sakura::Typography::FontSize,
    });
    summaryIconInset.update({});
    summaryIconDrop.update({});
    summaryIconColumn.update({
        .spacing = 0.0f,
        .items = {
            {.child = &summaryIconDrop, .size = IconDrop},
            {.child = &summaryIcon},
        },
    });
    summaryRow.update({
        .axis = Sakura::Stack::Axis::Horizontal,
        .spacing = 0.0f,
        .centerCross = false,
        .items = {
            {.child = &summaryIconInset, .size = IconInset},
            {.child = &summaryIconColumn},
            {.child = &summaryInput, .grow = 1.0f, .shrink = 1.0f},
        },
    });
    summaryCard.update({
        .content = &summaryRow,
        .size = {Sakura::Fill, SummaryHeight},
        .padding = AreaInset,
        .rounding = Sakura::Rounding::Card,
        .border = true,
        .borderWidth = AreaBorder,
        .chrome = true,
        .shadow = true,
        .shadowSoftness = AreaShadowSoftness,
        .shadowDrop = AreaShadowDrop,
        .colorKey = AreaFill,
        .hoveredColorKey = AreaFill,
    });
    authorInput.update({
        .icon = ICON_LC_USER,
        .value = this->config.author,
        .hint = "Name or organization",
        .submit = TextInput::Submit::OnEdit,
        .onChange = this->config.onAuthorChange,
    });
    licenseInput.update({
        .icon = ICON_LC_SCALE,
        .value = this->config.license,
        .hint = "MIT",
        .submit = TextInput::Submit::OnEdit,
        .onChange = this->config.onLicenseChange,
    });
    descriptionEditor.update({
        .id = "FlowgraphInfoDescription",
        .value = this->config.description,
        .height = DescriptionHeight - 2.0f * AreaInset,
        .language = Sakura::CodeEditor::Language::Markdown,
        .lineNumbers = false,
        .lineWrapping = true,
        .scrollPastEnd = false,
        .showActiveLine = false,
        .padding = Sakura::Padding{AreaPadding, DescriptionPadding, AreaPadding, DescriptionPadding},
        .fontName = Sakura::Typography::BodyFont,
        .backgroundColorKey = AreaFill,
        .onChange = [this](std::string value) {
            if (this->config.onDescriptionChange) {
                this->config.onDescriptionChange(value);
            }
        },
    });
    field(titleField, titleLabel, "Title", titleInput);
    field(summaryField, summaryLabel, "Summary", summaryCard);
    field(authorField, authorLabel, "Author", authorInput);
    field(licenseField, licenseLabel, "License", licenseInput);

    pathInput.update({
        .icon = ICON_LC_FILE_BRACES,
        .value = filename,
        .hint = "/path/to/flowgraph.yml",
        .submit = TextInput::Submit::OnEdit,
        .onChange = [this](const std::string& value) {
            filename = value;
        },
    });
    browseButton.update({
        .icon = ICON_LC_FOLDER_OPEN,
        .onClick = [this]() {
            if (this->config.onBrowse) {
                this->config.onBrowse(filename, [this](std::string path) {
                    filename = std::move(path);
                });
            }
        },
    });
    pathRow.update({
        .axis = Sakura::Stack::Axis::Horizontal,
        .spacing = Sakura::Spacing::Small,
        .items = {
            {.child = &pathInput, .grow = 1.0f, .shrink = 1.0f},
            {.child = &browseButton},
        },
    });
    label(pathLabel, "File");
    pathHint.update({
        .str = filename.empty()  ? "Choose where this flowgraph is saved."
               : canSave()       ? "Saving writes the flowgraph to this file."
                                 : "The file name must end in .yml or .yaml.",
        .font = Text::Font::Sans,
        .tone = filename.empty() || canSave() ? Text::Tone::Disabled : Text::Tone::Warning,
        .wrapped = true,
        .scale = Sakura::Typography::Scale::Small,
    });
    pathField.update({
        .spacing = Sakura::Spacing::XSmall,
        .items = {
            {.child = &pathLabel},
            {.child = &pathRow},
            {.child = &pathHint},
        },
    });
    detailsColumn.update({
        .spacing = Sakura::Spacing::Medium,
        .items = {
            {.child = &titleField},
            {.child = &summaryField},
            {.child = &authorField},
            {.child = &licenseField},
            {.child = &pathField},
        },
    });

    descriptionHint.update({
        .str = "Write a description in Markdown. It shows in the Description panel.",
        .font = Text::Font::Body,
        .tone = Text::Tone::Secondary,
    });
    std::vector<Sakura::Overlay::Item> writeItems = {{.child = &descriptionEditor}};
    if (this->config.description.empty()) {
        writeItems.push_back({
            .child = &descriptionHint,
            .anchor = Sakura::Anchor::TopLeft,
            .offset = {AreaPadding, DescriptionPadding},
        });
    }
    descriptionWrite.update({.items = std::move(writeItems)});
    descriptionPreview.update({
        .id = "FlowgraphInfoPreview",
        .value = this->config.description.empty() ? "*Nothing to preview yet.*" : this->config.description,
        .padding = Sakura::Padding{AreaPadding, DescriptionPadding, AreaPadding, DescriptionPadding},
        .scrollbar = true,
    });
    descriptionCard.update({
        .content = previewing ? static_cast<Sakura::Component*>(&descriptionPreview)
                              : static_cast<Sakura::Component*>(&descriptionWrite),
        .size = {Sakura::Fill, DescriptionHeight},
        .padding = AreaInset,
        .rounding = Sakura::Rounding::Card,
        .border = true,
        .borderWidth = AreaBorder,
        .chrome = true,
        .shadow = true,
        .shadowSoftness = AreaShadowSoftness,
        .shadowDrop = AreaShadowDrop,
        .colorKey = AreaFill,
        .hoveredColorKey = AreaFill,
    });
    field(descriptionColumn, descriptionOffset, " ", descriptionCard);
    body.update({
        .axis = Sakura::Stack::Axis::Horizontal,
        .spacing = Sakura::Spacing::XLarge,
        .centerCross = false,
        .items = {
            {.child = &detailsColumn, .size = DetailsWidth},
            {.child = &descriptionColumn, .grow = 1.0f, .shrink = 1.0f},
        },
    });

    cancel.update({
        .label = "Cancel",
        .hoverColorKey = "chrome_button_hover",
        .height = ButtonHeight,
        .onClick = [this]() {
            if (this->config.onClose) {
                this->config.onClose();
            }
        },
    });
    save.update({
        .icon = ICON_LC_SAVE,
        .label = "Save Flowgraph",
        .colorKey = "action_btn_text",
        .backgroundColorKey = "action_btn",
        .hoverColorKey = "action_btn_hovered",
        .raised = true,
        .disabled = !canSave(),
        .height = ButtonHeight,
        .onClick = [this]() {
            if (canSave() && this->config.onSave) {
                this->config.onSave(filename);
            }
        },
    });
    footerSpacer.update({});
    footer.update({
        .axis = Sakura::Stack::Axis::Horizontal,
        .spacing = Sakura::Spacing::Small,
        .items = {
            {.child = &footerSpacer, .grow = 1.0f},
            {.child = &cancel},
            {.child = &save},
        },
    });

    stack.update({
        .spacing = Sakura::Spacing::Large,
        .items = {
            {.child = &header},
            {.child = &body},
            {.child = &footer},
        },
    });
    panel.update({
        .id = "FlowgraphInfo",
        .content = &stack,
        .backgroundColorKey = "popup_bg",
    });
}

Sakura::Component& FlowgraphInfoView::component() {
    return panel;
}

void FlowgraphInfoView::label(Sakura::Text& text, const std::string& name) {
    text.update({
        .str = name,
        .font = Sakura::Text::Font::DisplaySemiBold,
        .scale = LabelText / Sakura::Typography::FontSize,
    });
}

void FlowgraphInfoView::field(Sakura::Stack& stack,
                              Sakura::Text& text,
                              const std::string& name,
                              Sakura::Component& control) {
    label(text, name);
    stack.update({
        .spacing = Sakura::Spacing::XSmall,
        .items = {
            {.child = &text},
            {.child = &control},
        },
    });
}

bool FlowgraphInfoView::canSave() const {
    for (const std::string_view extension : {".yml", ".yaml"}) {
        if (filename.size() > extension.size() && filename.ends_with(extension)) {
            return true;
        }
    }
    return false;
}

}  // namespace Jetstream
