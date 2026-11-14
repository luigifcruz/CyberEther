#include "dependencies.hh"

namespace Jetstream {

void DependencyReviewView::update(Config config) {
    using Text = Sakura::Text;
    using Button = Sakura::Button;

    this->config = std::move(config);
    header.update({
        .title = ICON_LC_PACKAGE " Install Python Dependencies",
        .description = "Review and install the Python packages requested by blocks.",
    });
    overviewText.update({
        .str = jst::fmt::format("Blocks declared {} that are not installed yet. "
                                 "Approving installs them with pip into a shared environment managed by "
                                 "CyberEther, separate from your system Python. Blocks that depend on them "
                                 "stay paused until installation completes, then reload automatically.",
                                 packageNoun()),
        .font = Text::Font::Sans,
        .tone = Text::Tone::Secondary,
        .wrapped = true,
    });
    thirdPartyCallout.update({
        .str = "Python packages are third-party code. Only approve dependencies you trust.",
        .icon = ICON_LC_SHIELD_HALF,
        .tone = Callout::Tone::Warning,
    });
    while (dependencyCells.size() < this->config.dependencies.size()) {
        dependencyCells.push_back(std::make_unique<DependencyCells>());
    }
    std::vector<std::vector<Sakura::Component*>> cellRows;
    for (U64 i = 0; i < this->config.dependencies.size(); ++i) {
        const auto& entry = this->config.dependencies[i];
        auto& cells = *dependencyCells[i];
        cells.requirement.update({
            .str = entry.requirement,
            .wrapped = true,
        });
        const bool attributed = !entry.requestedBy.empty();
        cells.requestedBy.update({
            .str = attributed ? entry.requestedBy : "Unknown",
            .font = Text::Font::Sans,
            .tone = attributed ? Text::Tone::Secondary : Text::Tone::Disabled,
        });
        cellRows.push_back({&cells.requirement, &cells.requestedBy});
    }
    dependencyTable.update({
        .columns = {"Package", "Requested By"},
        .cellRows = std::move(cellRows),
        .fixedColumnWidths = {0.0f, 190.0f},
        .maxHeight = 320.0f,
    });
    emptySpacingTop.update({.lines = 2});
    emptySpacingBottom.update({.lines = 2});
    emptyText.update({
        .str = "No dependencies are pending approval.",
        .font = Text::Font::Sans,
        .tone = Text::Tone::Disabled,
        .align = Text::Align::Center,
    });
    staleCallout.update({
        .str = "The dependency list changed. Close this window and review it again.",
        .tone = Callout::Tone::Warning,
    });
    statusCallout.update({
        .str = statusLine(),
        .tone = statusTone(),
    });
    console.update({
        .value = consoleText(),
        .height = 300.0f,
        .stickToBottom = installing() || installed(),
    });
    installDivider.update({});
    installButton.update({
        .str = installLabel(),
        .size = {Sakura::Fill, 40.0f},
        .variant = Button::Variant::Action,
        .disabled = !this->config.installEnabled,
        .onClick = [this]() {
            if (this->config.onInstall) {
                this->config.onInstall();
            }
        },
    });

    std::vector<Sakura::Stack::Item> items = {{.child = &header}};
    if (reviewLayout() || denied()) {
        if (reviewLayout() && !this->config.dependencies.empty()) {
            items.push_back({.child = &overviewText});
        }
        if (this->config.dependencies.empty()) {
            items.push_back({.child = &emptySpacingTop});
            items.push_back({.child = &emptyText});
            items.push_back({.child = &emptySpacingBottom});
        } else {
            items.push_back({.child = &dependencyTable});
        }
    }
    if (this->config.stale) {
        items.push_back({.child = &staleCallout});
    }
    if (hasStatus()) {
        items.push_back({.child = &statusCallout});
    }
    if (showConsole()) {
        items.push_back({.child = &console});
    }
    if (showInstallButton()) {
        if (reviewLayout()) {
            items.push_back({.child = &thirdPartyCallout});
        }
        items.push_back({.child = &installDivider});
        items.push_back({.child = &installButton});
    }
    stack.update({.items = std::move(items)});
    panel.update({
        .id = "DependencyReview",
        .content = &stack,
        .backgroundColorKey = "popup_bg",
    });
}

Sakura::Component& DependencyReviewView::component() {
    return panel;
}

bool DependencyReviewView::denied() const { return config.state == State::Denied; }

bool DependencyReviewView::installing() const { return config.state == State::Installing; }

bool DependencyReviewView::installed() const { return config.state == State::Installed; }

bool DependencyReviewView::failed() const { return config.state == State::Failed; }

bool DependencyReviewView::reviewLayout() const {
    return config.state == State::Review || (failed() && !config.approved);
}

bool DependencyReviewView::hasStatus() const {
    return config.state != State::Review || !config.message.empty();
}

bool DependencyReviewView::showConsole() const {
    return installing() || installed() || failed();
}

bool DependencyReviewView::showInstallButton() const {
    return config.state == State::Review || failed();
}

U64 DependencyReviewView::packageCount() const {
    std::unordered_set<std::string_view> requirements;
    for (const auto& entry : config.dependencies) {
        requirements.insert(entry.requirement);
    }
    return requirements.size();
}

std::string DependencyReviewView::packageNoun() const {
    const auto count = packageCount();
    if (count == 0) {
        return "Python packages";
    }
    return jst::fmt::format("{} Python {}", count, count == 1 ? "package" : "packages");
}

std::string DependencyReviewView::installLabel() const {
    return failed() && config.approved ? "Retry" : "Install Once";
}

std::string DependencyReviewView::consoleText() const {
    if (!config.output.empty()) {
        return config.output;
    }
    if (failed()) {
        return config.message.empty() ? "Installation failed." : config.message;
    }
    return "Waiting for pip output...";
}

std::string DependencyReviewView::statusLine() const {
    switch (config.state) {
        case State::Denied:
            return config.message.empty()
                       ? "Dependency installation is disabled by policy."
                       : config.message;
        case State::Installing:
            return config.message.empty()
                       ? jst::fmt::format("Installing {} with pip. This may take a few minutes.", packageNoun())
                       : config.message;
        case State::Installed:
            return config.message.empty() ? jst::fmt::format("Installed {} successfully.", packageNoun())
                                          : config.message;
        case State::Failed:
            return config.approved
                       ? "Installation failed. Check the output below, then retry."
                       : "Preparing the environment failed. Check the output below, then "
                         "review the packages and install again.";
        case State::Review:
            return config.message;
    }
    return "";
}

Callout::Tone DependencyReviewView::statusTone() const {
    switch (config.state) {
        case State::Denied:
            return Callout::Tone::Warning;
        case State::Installed:
            return Callout::Tone::Success;
        case State::Failed:
            return Callout::Tone::Error;
        default:
            return Callout::Tone::Info;
    }
}

}  // namespace Jetstream
