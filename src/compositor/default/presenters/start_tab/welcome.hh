#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_START_TAB_WELCOME_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_START_TAB_WELCOME_HH

#include "remote.hh"
#include "../agent/chat.hh"
#include "../context.hh"

#include "../../model/messages.hh"
#include "../../support/terminal.hh"
#include "../../views/start_tab/welcome.hh"

#include "jetstream/config.hh"
#include "jetstream/platform.hh"
#include "jetstream/registry.hh"
#include "tools/text.hh"

#include <algorithm>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream {

struct WelcomePresenter {
    static constexpr U64 PreviewLimit = 6;
    static constexpr U64 PageSize = 12;

    const PresenterContext& context;
    RemoteSectionPresenter remote;

    explicit WelcomePresenter(const PresenterContext& context);
    WelcomeView::Config build() const;
    static std::string Progress(const DefaultCompositorState::RegistryState::GalleryFetch& gallery);
    static std::string ProgressResult(const DefaultCompositorState::RegistryState::GalleryFetch& gallery);

 private:
    struct CategoryInfo {
        const char* title;
        const char* description;
    };

    static std::string CategoryDescription(const std::string& category);
    static WelcomeView::Example Example(const Registry::FlowgraphRegistration& entry);
    std::vector<WelcomeView::Recent> entries(const std::vector<Settings::Interface::FlowgraphEntry>& items) const;
    static bool Matches(const WelcomeView::Example& example, const std::string& query);
    static void Catalog(WelcomeView::Config& config,
                        const std::vector<Registry::FlowgraphRegistration>& available,
                        const std::string& query,
                        const std::string& category);
    static void Paginate(WelcomeView::Config& config, U64 page);
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_START_TAB_WELCOME_HH
