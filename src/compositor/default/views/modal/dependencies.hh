#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_DEPENDENCIES_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_DEPENDENCIES_HH

#include "../components/callout.hh"
#include "../components/modal_header.hh"
#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/imgui_icons_ext.hh"

#include <functional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Jetstream {

struct DependencyReviewView {
    enum class State {
        Review,      // Waiting for the user's decision.
        Denied,      // Installation is disabled by policy.
        Installing,  // Installation is in progress.
        Installed,   // The shared environment is ready.
        Failed,      // Installation failed.
    };

    struct DependencyEntry {
        std::string requirement;
        std::string requestedBy;
    };

    struct Config {
        std::vector<DependencyEntry> dependencies;
        State state = State::Review;
        bool stale = false;
        bool installEnabled = false;
        std::string output;
        std::string message;
        std::function<void()> onInstall;
    };

    DependencyReviewView() {
        consoleCanvas.mount(consoleTextView);
    }

    void update(Config config) {
        this->config = std::move(config);

        header.update({
            .id = "DependencyReviewHeader",
            .title = ICON_FA_BOX " Install Python Dependencies",
            .description = "Review and install the Python packages requested by blocks.",
        });

        overviewText.update({
            .id = "DependencyReviewOverview",
            .str = "Blocks declared " + packageNoun() + " that are not installed yet. "
                   "Approving installs them with pip into a shared environment managed by "
                   "CyberEther, separate from your system Python. Blocks that depend on them "
                   "stay paused until installation completes, then reload automatically.",
            .tone = Sakura::Text::Tone::Secondary,
            .wrapped = true,
        });

        thirdPartyCallout.update({
            .id = "DependencyReviewThirdParty",
            .str = "Python packages are third-party code. Only approve dependencies you trust.",
            .icon = ICON_FA_SHIELD_HALVED,
            .tone = Callout::Tone::Warning,
        });

        dependencyTable.update({
            .id = "DependencyReviewTable",
            .columns = {"Package", "Requested By"},
            .fixedColumnWidths = {0.0f, 190.0f},
            .maxHeight = 320.0f,
        });
        requirementTexts.resize(this->config.dependencies.size());
        requestedByTexts.resize(this->config.dependencies.size());
        for (U64 i = 0; i < this->config.dependencies.size(); ++i) {
            const auto& entry = this->config.dependencies[i];
            requirementTexts[i].update({
                .id = "DependencyReviewRequirement" + std::to_string(i),
                .str = entry.requirement,
                .wrapped = true,
            });
            const bool attributed = !entry.requestedBy.empty();
            requestedByTexts[i].update({
                .id = "DependencyReviewRequestedBy" + std::to_string(i),
                .str = attributed ? entry.requestedBy : "Unknown",
                .tone = attributed ? Sakura::Text::Tone::Secondary : Sakura::Text::Tone::Disabled,
            });
        }
        emptySpacing.update({
            .id = "DependencyReviewEmptySpacing",
            .lines = 2,
        });
        emptyText.update({
            .id = "DependencyReviewEmpty",
            .str = "No dependencies are pending approval.",
            .tone = Sakura::Text::Tone::Disabled,
            .align = Sakura::Text::Align::Center,
        });

        staleCallout.update({
            .id = "DependencyReviewStale",
            .str = "The dependency list changed. Close this window and review it again.",
            .tone = Callout::Tone::Warning,
        });
        statusCallout.update({
            .id = "DependencyReviewStatus",
            .str = statusLine(),
            .tone = statusTone(),
        });
        consoleCanvas.update({
            .id = "DependencyReviewConsoleCanvas",
            .size = {0.0f, 300.0f},
            .clearColor = {0.0f, 0.0f, 0.0f, 0.0f},
            .onLayout = [this](const Sakura::Retained::Canvas::Layout& layout) {
                consoleFontSizePixels = 15.0f * layout.pixelRatio;
                consolePadding = {6.0f * layout.pixelRatio, 6.0f * layout.pixelRatio,
                                  6.0f * layout.pixelRatio, 6.0f * layout.pixelRatio};
            },
        });
        consoleTextView.update({
            .id = "DependencyReviewConsoleText",
            .value = consoleText(),
            .fontSize = consoleFontSizePixels,
            .fontName = "default_mono",
            .monospace = true,
            .lineNumbers = false,
            .stickToBottom = installing() || installed(),
            .scrollbar = true,
            .wrap = Sakura::Retained::TextGrid::Wrap::Word,
            .padding = consolePadding,
            .backgroundColorKey = "editor_console_background",
            .textColorKey = "editor_text",
        });

        installDivider.update({
            .id = "DependencyReviewInstallDivider",
        });
        installButton.update({
            .id = "DependencyReviewInstall",
            .str = installLabel(),
            .size = {-1.0f, 40.0f},
            .variant = Sakura::Button::Variant::Action,
            .disabled = !this->config.installEnabled,
            .onClick = [this]() {
                if (this->config.onInstall) {
                    this->config.onInstall();
                }
            },
        });
    }

    void render(const Sakura::Context& ctx) {
        header.render(ctx);

        if (!installing() && !installed()) {
            if (config.state == State::Review && !config.dependencies.empty()) {
                overviewText.render(ctx);
            }
            if (config.dependencies.empty()) {
                emptySpacing.render(ctx);
                emptyText.render(ctx);
                emptySpacing.render(ctx);
            } else {
                Sakura::Table::Rows rows;
                rows.reserve(config.dependencies.size());
                for (U64 i = 0; i < config.dependencies.size(); ++i) {
                    Sakura::Table::Row row;
                    row.push_back([this, i](const Sakura::Context& ctx) {
                        requirementTexts[i].render(ctx);
                    });
                    row.push_back([this, i](const Sakura::Context& ctx) {
                        requestedByTexts[i].render(ctx);
                    });
                    rows.push_back(std::move(row));
                }
                dependencyTable.render(ctx, std::move(rows));
            }
        }

        if (config.stale) {
            staleCallout.render(ctx);
        }

        if (hasStatus()) {
            statusCallout.render(ctx);
        }

        if (showConsole()) {
            consoleCanvas.render(ctx);
        }

        if (showInstallButton()) {
            thirdPartyCallout.render(ctx);
            installDivider.render(ctx);
            installButton.render(ctx);
        }
    }

 private:
    bool denied() const { return config.state == State::Denied; }
    bool installing() const { return config.state == State::Installing; }
    bool installed() const { return config.state == State::Installed; }
    bool failed() const { return config.state == State::Failed; }

    bool hasStatus() const {
        return config.state != State::Review || !config.message.empty();
    }

    bool showConsole() const {
        return installing() || installed() || (failed() && !config.output.empty());
    }

    bool showInstallButton() const {
        return config.state == State::Review || failed();
    }

    U64 packageCount() const {
        std::unordered_set<std::string_view> requirements;
        for (const auto& entry : config.dependencies) {
            requirements.insert(entry.requirement);
        }
        return requirements.size();
    }

    std::string packageNoun() const {
        const auto count = packageCount();
        if (count == 0) {
            return "Python packages";
        }
        return std::to_string(count) + (count == 1 ? " Python package" : " Python packages");
    }

    std::string installLabel() const {
        return failed() ? "Retry" : "Install Once";
    }

    std::string consoleText() const {
        return config.output.empty() ? "Waiting for pip output..." : config.output;
    }

    std::string statusLine() const {
        switch (config.state) {
            case State::Denied:
                return config.message.empty()
                           ? "Dependency installation is disabled by policy."
                           : config.message;
            case State::Installing:
                return config.message.empty()
                           ? "Installing " + packageNoun() +
                                 " with pip. This may take a few minutes."
                           : config.message;
            case State::Installed:
                return config.message.empty() ? "Installed " + packageNoun() + " successfully."
                                              : config.message;
            case State::Failed:
                return config.message.empty() ? "Installation failed." : config.message;
            case State::Review:
                return config.message;
        }
        return "";
    }

    Callout::Tone statusTone() const {
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

    Config config;
    F32 consoleFontSizePixels = 15.0f;
    Sakura::Padding consolePadding = {6.0f, 6.0f, 6.0f, 6.0f};

    ModalHeader header;

    Sakura::Text overviewText;
    Callout thirdPartyCallout;

    Sakura::Table dependencyTable;
    std::vector<Sakura::Text> requirementTexts;
    std::vector<Sakura::Text> requestedByTexts;
    Sakura::Spacing emptySpacing;
    Sakura::Text emptyText;

    Callout staleCallout;

    Callout statusCallout;
    Sakura::Retained::Canvas consoleCanvas;
    Sakura::Retained::TextView consoleTextView;

    Sakura::Divider installDivider;
    Sakura::Button installButton;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_DEPENDENCIES_HH
