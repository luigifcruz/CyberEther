#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_START_TAB_WELCOME_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_START_TAB_WELCOME_HH

#include "remote.hh"
#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/start_tab/welcome.hh"

#include "jetstream/config.hh"
#include "jetstream/platform.hh"
#include "jetstream/registry.hh"
#include "render/sakura/themes.hh"
#include "tools/text.hh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iterator>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace Jetstream {

struct WelcomePresenter {
    static constexpr U64 PreviewLimit = 6;
    static constexpr U64 PageSize = 12;

    const PresenterContext& context;
    RemoteSectionPresenter remote;

    explicit WelcomePresenter(const PresenterContext& context) : context(context), remote(context) {}

    WelcomeView::Config build() const {
        const auto enqueue = context.callbacks.enqueueMail;
        WelcomeView::Config config{
            .version = "v" JETSTREAM_VERSION_STR,
            .assistantEnabled = context.state.agent.settings.enabled,
            .pageCard = !context.state.tabsHidden(),
            .assistantClearRequest = context.state.agent.chat.welcomeAccepted,
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
            .onOpenExample = [enqueue](const std::string& key) {
                enqueue(MailOpenFlowgraphExample{.key = key});
            },
        };
        config.remote = remote.build();
        const auto query = Text::ToLower(Text::TrimBlank(config.query));
        Catalog(config, context.state.registry.flowgraphs, query, config.category);
        Paginate(config, context.state.interface.welcomePage);
        if (config.section != WelcomeSection::All) {
            config.featured.clear();
        }
        return config;
    }

    static std::string Progress(const DefaultCompositorState::RegistryState::GalleryFetch& gallery) {
        if (!gallery.loading || gallery.attempted) {
            return "";
        }
        return gallery.cached ? "Updating gallery..." : "Loading flowgraphs from the gallery...";
    }

    static std::string ProgressResult(const DefaultCompositorState::RegistryState::GalleryFetch& gallery) {
        if (gallery.loading || !gallery.attempted) {
            return "";
        }
        if (gallery.failed) {
            return "Could not reach the gallery. Showing bundled flowgraphs.";
        }
        return gallery.cached ? "Gallery updated successfully!" : "Gallery loaded.";
    }

    static std::string Accessed(U64 accessed, U64 now) {
        if (accessed == 0 || accessed > now) {
            return "Opened recently";
        }
        const U64 seconds = now - accessed;
        const auto ago = [](U64 count, const char* unit) {
            return jst::fmt::format("Opened {} {}{} ago", count, unit, count == 1 ? "" : "s");
        };
        if (seconds < 60) {
            return "Opened just now";
        }
        if (seconds < 3600) {
            return ago(seconds / 60, "minute");
        }
        if (seconds < 86400) {
            return ago(seconds / 3600, "hour");
        }
        if (seconds < 172800) {
            return "Opened yesterday";
        }
        if (seconds < 2592000) {
            return ago(seconds / 86400, "day");
        }
        if (seconds < 31536000) {
            return ago(seconds / 2592000, "month");
        }
        return ago(seconds / 31536000, "year");
    }

 private:
    struct CategoryInfo {
        const char* title;
        const char* description;
    };

    static std::string CategoryDescription(const std::string& category) {
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

    static U32 Channel(F32 value) {
        return static_cast<U32>(std::lround(std::clamp(value, 0.0f, 1.0f) * 255.0f));
    }

    static std::optional<std::string> AccentKey(const std::string& accent) {
        static const auto accents = [] {
            std::unordered_map<std::string, std::string> accents;
            for (const auto& [key, color] : Sakura::Themes.at("Dark")) {
                if (key.starts_with("gallery_")) {
                    accents.emplace(jst::fmt::format("#{:02x}{:02x}{:02x}", Channel(color.r), Channel(color.g),
                                                     Channel(color.b)),
                                    key);
                }
            }
            return accents;
        }();
        const auto match = accents.find(Text::ToLower(accent));
        if (match == accents.end()) {
            return std::nullopt;
        }
        return match->second;
    }

    static WelcomeView::Example Example(const Registry::FlowgraphRegistration& entry) {
        WelcomeView::Example example{
            .title = entry.title,
            .summary = entry.summary,
            .key = entry.key,
            .category = entry.category.empty() ? "Plugins" : entry.category,
            .diagram = entry.diagram,
        };
        if (const auto key = AccentKey(entry.accent)) {
            example.accentColorKey = *key;
        }
        return example;
    }

    std::vector<WelcomeView::Recent> entries(const std::vector<Settings::Interface::FlowgraphEntry>& items) const {
        const auto now = static_cast<U64>(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
        const auto& favorites = context.state.interface.favoriteFlowgraphs;
        std::vector<WelcomeView::Recent> result;
        for (const auto& item : items) {
            const auto file = Platform::PathFromUtf8(item.path);
            result.push_back({
                .title = item.title.empty() ? Platform::PathToUtf8(file.stem()) : item.title,
                .accessed = Accessed(item.accessed, now),
                .path = item.path,
                .favorite = std::any_of(favorites.begin(), favorites.end(), [&](const auto& favorite) {
                    return favorite.path == item.path;
                }),
            });
        }
        return result;
    }

    static bool Matches(const WelcomeView::Example& example, const std::string& query) {
        if (query.empty()) {
            return true;
        }
        return Text::ToLower(example.title).find(query) != std::string::npos ||
               Text::ToLower(example.summary).find(query) != std::string::npos ||
               Text::ToLower(example.category).find(query) != std::string::npos;
    }

    static void Catalog(WelcomeView::Config& config,
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

    static void Paginate(WelcomeView::Config& config, U64 page) {
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
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_START_TAB_WELCOME_HH
