#include "registry_add_plugin.hh"

namespace Jetstream {

void RegistryAddPluginView::reset() {
    pendingPath.clear();
    trusted = false;
}

void RegistryAddPluginView::update(Config config) {
    this->config = std::move(config);
    breadcrumb.update({
        .root = "Registry",
        .current = "Add Plugin",
        .onRoot = this->config.onBack,
    });
    const bool ready = trusted && !pendingPath.empty();
    registerButton.update({
        .str = "Register Plugin",
        .size = {Sakura::Fill, 40.0f},
        .variant = Sakura::Button::Variant::Action,
        .disabled = !ready,
        .rounding = Sakura::Rounding::Card,
        .onClick = [this]() {
            if (trusted && !pendingPath.empty() && this->config.onRegister) {
                this->config.onRegister(pendingPath);
            }
        },
        .font = Sakura::Text::Font::SansBold,
    });
    header.update({
        .title = "Add Plugin",
        .description = "Load a plugin bundle that registers additional blocks.",
    });
    pathField.update({
        .icon = ICON_LC_PACKAGE,
        .value = pendingPath,
        .hint = "/path/to/plugin.cep",
        .submit = Sakura::TextInput::Submit::OnEdit,
        .onChange = [this](const std::string& value) {
            setPendingPath(value);
        },
        .onBrowse = [this]() {
            if (this->config.onBrowse) {
                this->config.onBrowse(pendingPath, [this](std::string path) {
                    setPendingPath(std::move(path));
                });
            }
        },
    });
    trustToggle.update({
        .value = trusted,
        .onChange = [this](bool value) {
            trusted = value;
        },
    });
    sourceGroup.update({
        .warning = "Only register plugins you built yourself or received from a source you trust.",
        .rows = {
            {
                .label = "Plugin Path",
                .description = "Path to a .cep plugin bundle that registers additional blocks.",
                .control = &pathField,
            },
            {
                .label = "Trust Source",
                .description = "Confirm that you trust the source of this plugin.",
                .control = &trustToggle,
            },
        },
        .footer = &registerButton,
    });
    stack.update({
        .spacing = Sakura::Spacing::XSmall,
        .items = {
            {.child = &breadcrumb},
            {.child = &header},
            {.child = &sourceGroup},
        },
    });
}

Sakura::Component& RegistryAddPluginView::content() {
    return stack;
}

void RegistryAddPluginView::setPendingPath(std::string path) {
    if (path != pendingPath) {
        trusted = false;
    }
    pendingPath = std::move(path);
}

}  // namespace Jetstream
