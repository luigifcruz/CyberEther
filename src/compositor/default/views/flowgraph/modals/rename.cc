#include "rename.hh"

namespace Jetstream {

void RenameBlockView::update(Config config) {
    if (this->config.oldName != config.oldName) {
        newName = config.oldName;
    }
    this->config = std::move(config);

    header.update({
        .title = ICON_LC_TAG " Rename Block",
        .description = "Enter a new block identifier.",
    });
    input.update({
        .value = newName,
        .hint = "Block name",
        .submit = Sakura::TextInput::Submit::OnEdit,
        .focus = true,
        .selectAllOnFocus = true,
        .onChange = [this](const std::string& value) {
            newName = value;
        },
        .onSubmit = [this](const std::string& value) {
            newName = value;
            submit();
        },
    });
    renameButton.update({
        .str = "Rename Block",
        .size = {Sakura::Fill, 40.0f},
        .variant = Sakura::Button::Variant::Action,
        .disabled = !canSubmit(),
        .onClick = [this]() {
            submit();
        },
    });
    divider.update({});
    stack.update({
        .items = {
            {.child = &header},
            {.child = &input},
            {.child = &divider},
            {.child = &renameButton},
        },
    });
    panel.update({
        .id = "RenameBlock",
        .content = &stack,
        .backgroundColorKey = "popup_bg",
    });
}

Sakura::Component& RenameBlockView::component() {
    return panel;
}

bool RenameBlockView::canSubmit() const {
    return newName != config.oldName;
}

void RenameBlockView::submit() const {
    if (canSubmit() && config.onRename) {
        config.onRename(newName);
    }
}

}  // namespace Jetstream
