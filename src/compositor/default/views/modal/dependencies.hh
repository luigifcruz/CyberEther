#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_DEPENDENCIES_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_DEPENDENCIES_HH

#include "../components/callout.hh"
#include "../components/modal_header.hh"
#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <memory>
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
        bool approved = false;
        bool installEnabled = false;
        std::string output;
        std::string message;
        Sakura::Callback<void()> onInstall;
    };

    void update(Config config);
    Sakura::Component& component();

 private:
    bool denied() const;
    bool installing() const;
    bool installed() const;
    bool failed() const;
    bool reviewLayout() const;
    bool hasStatus() const;
    bool showConsole() const;
    bool showInstallButton() const;
    U64 packageCount() const;
    std::string packageNoun() const;
    std::string installLabel() const;
    std::string consoleText() const;
    std::string statusLine() const;
    Callout::Tone statusTone() const;

    struct DependencyCells {
        Sakura::Text requirement;
        Sakura::Text requestedBy;
    };

    Config config;
    ModalHeader header;
    Sakura::Text overviewText;
    Callout thirdPartyCallout;
    std::vector<std::unique_ptr<DependencyCells>> dependencyCells;
    Sakura::Table dependencyTable;
    Sakura::Spacer emptySpacingTop;
    Sakura::Spacer emptySpacingBottom;
    Sakura::Text emptyText;
    Callout staleCallout;
    Callout statusCallout;
    Sakura::LogView console;
    Sakura::Divider installDivider;
    Sakura::Button installButton;
    Sakura::Stack stack;
    Sakura::Panel panel;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_DEPENDENCIES_HH
