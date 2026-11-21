#include "welcome.hh"

namespace Jetstream {

WelcomePresenter::WelcomePresenter(const PresenterContext& context) : context(context), remote(context) {}

WelcomeView::Config WelcomePresenter::build() const {
    const auto enqueue = context.callbacks.enqueueMail;
    WelcomeView::Config config{
        .version = "v" JETSTREAM_VERSION_STR,
        .assistantEnabled = context.state.agent.settings.enabled,
        .pageCard = !context.state.tabsHidden(),
        .assistantClearRequest = context.state.agent.chat.welcomeAccepted,
        .assistantSelector = AgentChatPresenter::ModelSelector(context),
        .onAssistantSubmit = [enqueue](const std::string& text) {
            enqueue(MailSubmitAgentChat{.composer = AgentComposer::Welcome, .text = text, .reveal = true});
        },
        .onAssistantVoice = [enqueue]() {
            enqueue(MailSetAgentVoiceVisible{.value = true});
        },

        .onNewFlowgraph = [enqueue]() {
            enqueue(MailNewFlowgraph{});
        },
        .onOpenFlowgraph = [enqueue]() {
            enqueue(MailOpenFlowgraph{});
        },
        .onOpenTerminal = TerminalSession::Supported() ? Sakura::Callback<void()>([enqueue]() {
            enqueue(MailOpenTerminal{});
        })
                                                       : Sakura::Callback<void()>(),
        .onOpenSettings = [enqueue]() {
            enqueue(MailOpenSettings{.section = SettingsSection::General});
        },
        .darkMode = context.state.sakura.dark(),
        .onToggleTheme = [enqueue, dark = context.state.sakura.dark()]() {
            enqueue(MailApplyTheme{dark ? "Light" : "Dark"});
        },
        .sidebarOpen = context.state.interface.welcomeSidebarOpen,

        .favorites = entries(context.state.interface.favoriteFlowgraphs),
        .recent = entries(context.state.interface.recentFlowgraphs),
        .query = context.state.interface.welcomeQuery,
        .category = context.state.interface.welcomeCategory,
        .onQueryChange = [enqueue](const std::string& query) {
            enqueue(MailSetWelcomeQuery{.query = query});
        },
        .onSelectCategory = [enqueue](const std::string& category) {
            enqueue(MailSetWelcomeCategory{.category = category});
        },
        .onSelectPage = [enqueue](U64 page) {
            enqueue(MailSetWelcomePage{.page = page});
        },
        .section = context.state.interface.welcomeSection,
        .progress = Progress(context.state.registry.gallery),
        .progressResult = ProgressResult(context.state.registry.gallery),
        .progressFailed = context.state.registry.gallery.failed,
        .onSelectSection = [enqueue](WelcomeSection section) {
            enqueue(MailSetWelcomeSection{.section = section});
        },
        .onOpenRecent = [enqueue](const std::string& path) {
            enqueue(MailOpenFlowgraphPath{.path = path});
        },
        .onToggleFavorite = [enqueue](const std::string& path) {
            enqueue(MailToggleFavoriteFlowgraph{.path = path});
        },
        .onForgetRecent = [enqueue](const std::string& path) {
            enqueue(MailForgetRecentFlowgraph{.path = path});
        },
        .onRevealRecent = [enqueue](const std::string& path) {
            enqueue(MailRevealPath{.path = path});
        },
        .onCopyRecentPath = [enqueue](const std::string& path) {
            enqueue(MailCopyText{.label = "Path", .value = path});
        },
        .onOpenExample = [enqueue](const std::string& key) {
            enqueue(MailOpenFlowgraphExample{.key = key});
        },
    };
    config.remote = remote.build();
    const auto query = Text::ToLower(Text::TrimBlank(config.query));
    Catalog(config, context.state.registry.flowgraphs, query, config.category);
    Paginate(config, context.state.interface.welcomePage);
    return config;
}

std::string WelcomePresenter::Progress(const DefaultCompositorState::RegistryState::GalleryFetch& gallery) {
    if (!gallery.loading || gallery.attempted) {
        return "";
    }
    return gallery.cached ? "Updating flowgraphs..." : "Loading flowgraphs...";
}

std::string WelcomePresenter::ProgressResult(const DefaultCompositorState::RegistryState::GalleryFetch& gallery) {
    if (gallery.loading || !gallery.attempted) {
        return "";
    }
    if (gallery.failed) {
        return "Using bundled flowgraphs";
    }
    return gallery.cached ? "Gallery updated successfully!" : "Gallery loaded.";
}

std::string WelcomePresenter::CategoryDescription(const std::string& category) {
    static const CategoryInfo infos[] = {
        {"Basics", "Start here to learn the editor, sources, and plots."},
        {"DSP", "Signal processing techniques, taken apart block by block."},
        {"Receivers", "Tune, demodulate, and decode signals off the air."},
        {"Visualization", "Turn numbers into color, shape, and motion."},
        {"Plugins", "Examples bundled with your installed plugins."},
    };
    for (const auto& info : infos) {
        if (category == info.title) {
            return info.description;
        }
    }
    return "Complete pipelines, ready to run in Web or Native.";
}

WelcomeView::Example WelcomePresenter::Example(const Registry::FlowgraphRegistration& entry) {
    WelcomeView::Example example{
        .title = entry.title,
        .summary = entry.summary,
        .key = entry.key,
        .category = entry.category.empty() ? "Plugins" : entry.category,
        .diagram = entry.diagram,
    };
    return example;
}

std::vector<WelcomeView::Recent> WelcomePresenter::entries(const std::vector<Settings::Interface::FlowgraphEntry>& items) const {
    const auto& favorites = context.state.interface.favoriteFlowgraphs;
    std::vector<WelcomeView::Recent> result;
    for (const auto& item : items) {
        const auto file = Platform::PathFromUtf8(item.path);
        result.push_back({
            .title = item.title.empty() ? Platform::PathToUtf8(file.stem()) : item.title,
            .path = item.path,
            .favorite = std::any_of(favorites.begin(), favorites.end(), [&](const auto& favorite) {
                return favorite.path == item.path;
            }),
        });
    }
    return result;
}

bool WelcomePresenter::Matches(const WelcomeView::Example& example, const std::string& query) {
    if (query.empty()) {
        return true;
    }
    return Text::ToLower(example.title).find(query) != std::string::npos ||
           Text::ToLower(example.summary).find(query) != std::string::npos ||
           Text::ToLower(example.category).find(query) != std::string::npos;
}

void WelcomePresenter::Catalog(WelcomeView::Config& config,
                               const std::vector<Registry::FlowgraphRegistration>& available,
                               const std::string& query,
                               const std::string& category) {
    if (query.empty() && category.empty()) {
        for (const auto& entry : available) {
            if (entry.featured) {
                config.featured.push_back(Example(entry));
            }
        }
    }
    for (const auto& entry : available) {
        auto example = Example(entry);
        if (std::find(config.categoryNames.begin(), config.categoryNames.end(), example.category) ==
            config.categoryNames.end()) {
            config.categoryNames.push_back(example.category);
        }
        if ((!category.empty() && example.category != category) || !Matches(example, query)) {
            continue;
        }
        auto bucket = std::find_if(config.categories.begin(), config.categories.end(), [&](const auto& item) {
            return item.title == example.category;
        });
        if (bucket == config.categories.end()) {
            config.categories.push_back({
                .title = example.category,
                .description = CategoryDescription(example.category),
            });
            bucket = std::prev(config.categories.end());
        }
        bucket->examples.push_back(std::move(example));
    }
}

void WelcomePresenter::Paginate(WelcomeView::Config& config, U64 page) {
    for (auto& category : config.categories) {
        category.total = category.examples.size();
    }
    if (config.category.empty()) {
        for (auto& category : config.categories) {
            if (category.examples.size() > PreviewLimit) {
                category.examples.resize(PreviewLimit);
            }
        }
        return;
    }
    if (config.categories.empty()) {
        return;
    }
    auto& examples = config.categories.front().examples;
    config.pages = (examples.size() + PageSize - 1) / PageSize;
    config.page = std::min(page, config.pages > 0 ? config.pages - 1 : 0);
    const U64 first = config.page * PageSize;
    const U64 last = std::min<U64>(first + PageSize, examples.size());
    examples = std::vector<WelcomeView::Example>(examples.begin() + static_cast<I64>(first),
                                                 examples.begin() + static_cast<I64>(last));
}

}  // namespace Jetstream
