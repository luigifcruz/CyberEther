#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_DEPENDENCIES_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_DEPENDENCIES_HH

#include "../components/modal_header.hh"
#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/imgui_icons_ext.hh"

#include <functional>
#include <string>
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
        errorCanvas.mount(errorTextView);
    }

    void update(Config config) {
        this->config = std::move(config);

        const bool denied = this->config.state == State::Denied;
        const bool installing = this->config.state == State::Installing;
        const bool failed = this->config.state == State::Failed;
        const bool installed = this->config.state == State::Installed;

        header.update({
            .id = "DependencyReviewHeader",
            .title = ICON_FA_BOX " Install Python Dependencies",
            .description = "Review the Python packages before approving their installation.",
            .dividerSpacing = 0.0f,
        });

        thirdPartyNotice.update({
            .id = "DependencyReviewThirdPartyNotice",
            .str = std::string(ICON_FA_SHIELD_HALVED) +
                   " Python packages are third-party code. Only approve dependencies you trust.",
            .tone = Sakura::Text::Tone::Warning,
            .wrapped = true,
        });
        warningDivider.update({
            .id = "DependencyReviewWarningDivider",
            .spacing = 0.0f,
        });

        dependencyTable.update({
            .id = "DependencyReviewTable",
            .columns = {"Package", "Requested By"},
            .fixedColumnWidths = {0.0f, 190.0f},
            .size = {0.0f, this->config.dependencies.size() > 4 ? 300.0f : 0.0f},
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

        staleText.update({
            .id = "DependencyReviewStale",
            .str = "The dependency list changed. Close this window and review it again.",
            .tone = Sakura::Text::Tone::Warning,
            .wrapped = true,
        });

        reloadNotice.update({
            .id = "DependencyReviewReloadNotice",
            .str = "Dependencies are installed into a shared environment. Running blocks "
                   "will be reloaded with the new environment once installation completes.",
            .tone = Sakura::Text::Tone::Secondary,
            .wrapped = true,
        });
        statusText.update({
            .id = "DependencyReviewStatus",
            .str = statusLine(),
            .colorKey = statusColorKey(),
            .wrapped = true,
        });
        errorCanvas.update({
            .id = "DependencyReviewErrorCanvas",
            .size = {0.0f, 300.0f},
            .clearColor = {0.0f, 0.0f, 0.0f, 0.0f},
            .onLayout = [this](const Sakura::Retained::Canvas::Layout& layout) {
                errorFontSizePixels = 15.0f * layout.pixelRatio;
                errorPadding = {6.0f * layout.pixelRatio, 6.0f * layout.pixelRatio,
                                6.0f * layout.pixelRatio, 6.0f * layout.pixelRatio};
            },
        });
        errorTextView.update({
            .id = "DependencyReviewErrorText",
            .value = failed
                         ? (this->config.message.empty() ? "Installation failed."
                                                         : this->config.message)
                         : (this->config.output.empty() ? "Waiting for pip output..."
                                                        : this->config.output),
            .fontSize = errorFontSizePixels,
            .fontName = "default_mono",
            .monospace = true,
            .lineNumbers = false,
            .stickToBottom = installing || installed,
            .scrollbar = true,
            .wrap = Sakura::Retained::TextGrid::Wrap::Word,
            .padding = errorPadding,
            .backgroundColorKey = "editor_console_background",
            .textColorKey = failed ? "error_red" : "editor_text",
        });

        installDivider.update({
            .id = "DependencyReviewInstallDivider",
        });
        installButton.update({
            .id = "DependencyReviewInstall",
            .str = denied ? "Installation Disabled"
                          : (installing ? "Installing..."
                                        : (failed ? "Retry"
                                                  : (installed ? "Installed" : "Install Once"))),
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
        const bool installing = config.state == State::Installing;
        const bool installed = config.state == State::Installed;
        const bool failed = config.state == State::Failed;
        const bool showDependencies = config.state == State::Review ||
                                      config.state == State::Denied ||
                                      failed;

        header.render(ctx);

        if (showDependencies) {
            thirdPartyNotice.render(ctx);
            warningDivider.render(ctx);
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
            staleText.render(ctx);
        }

        if (installing) {
            statusText.render(ctx);
        }

        if (installing || installed || failed) {
            errorCanvas.render(ctx);
        } else if (hasStatus()) {
            statusText.render(ctx);
        }

        if (config.state != State::Denied) {
            reloadNotice.render(ctx);
        }

        installDivider.render(ctx);
        installButton.render(ctx);
    }

 private:
    bool hasStatus() const {
        return config.state != State::Review || !config.message.empty();
    }

    std::string statusLine() const {
        switch (config.state) {
            case State::Denied:
                return config.message.empty()
                           ? "Dependency installation is disabled by policy."
                           : config.message;
            case State::Installing:
                return config.message.empty() ? "Installing dependencies…"
                                              : config.message;
            case State::Installed:
                return "";
            case State::Failed:
                return config.message.empty() ? "Installation failed." : config.message;
            case State::Review:
                return config.message;
        }
        return "";
    }

    std::string statusColorKey() const {
        switch (config.state) {
            case State::Denied:
                return "warning_yellow";
            case State::Installed:
                return "success_green";
            case State::Failed:
                return "error_red";
            default:
                return "text_secondary";
        }
    }

    Config config;
    F32 errorFontSizePixels = 15.0f;
    Sakura::Padding errorPadding = {6.0f, 6.0f, 6.0f, 6.0f};

    ModalHeader header;

    Sakura::Divider warningDivider;

    Sakura::Table dependencyTable;
    std::vector<Sakura::Text> requirementTexts;
    std::vector<Sakura::Text> requestedByTexts;
    Sakura::Spacing emptySpacing;
    Sakura::Text emptyText;

    Sakura::Text staleText;
    Sakura::Text reloadNotice;
    Sakura::Text thirdPartyNotice;

    Sakura::Text statusText;
    Sakura::Retained::Canvas errorCanvas;
    Sakura::Retained::TextView errorTextView;

    Sakura::Divider installDivider;
    Sakura::Button installButton;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_DEPENDENCIES_HH
