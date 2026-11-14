#include "terminal.hh"

namespace Jetstream {

bool TerminalActions::Openable(const std::string& uri) {
    static constexpr std::array<std::string_view, 4> kSchemes = {"https://", "http://", "mailto:", "file://"};
    return std::any_of(kSchemes.begin(), kSchemes.end(), [&](std::string_view scheme) {
        return uri.size() > scheme.size() && std::equal(scheme.begin(), scheme.end(), uri.begin(), [](char a, char b) {
            return a == std::tolower(static_cast<unsigned char>(b));
        });
    });
}

TerminalActions::TerminalActions(DefaultCompositorState& state, DefaultCompositorCallbacks& callbacks) :
    state(state),
    callbacks(callbacks) {}

Result TerminalActions::handle(const MailOpenTerminal& msg) {
    auto& terminals = state.terminal;
    const U64 id = terminals.next++;
    auto& item = terminals.items[id];
    item.directory = directory(msg.flowgraph);
    const std::string key = MakeTerminalPaneKey(id);
    if (msg.split.has_value()) {
        const auto source = terminals.items.find(*msg.split);
        if (source != terminals.items.end()) {
            item.directory = source->second.directory;
            const Terminal::Placement beside{.anchor = MakeTerminalPaneKey(*msg.split), .stacked = msg.stacked};
            if (source->second.flowgraph.has_value()) {
                state.flowgraph.restoredLayouts.insert(*source->second.flowgraph);
                item.flowgraph = source->second.flowgraph;
                item.placement = beside;
                return Result::SUCCESS;
            }
            if (source->second.tab.has_value() && terminals.tabs.contains(*source->second.tab)) {
                item.tab = source->second.tab;
                item.placement = beside;
                return Result::SUCCESS;
            }
        }
    }
    if (msg.tile && state.flowgraph.items.contains(msg.flowgraph)) {
        state.flowgraph.restoredLayouts.insert(msg.flowgraph);
        item.flowgraph = msg.flowgraph;
        item.placement = Terminal::Placement{};
        return Result::SUCCESS;
    }
    item.tab = open(key);
    return Result::SUCCESS;
}

Result TerminalActions::handle(const MailCloseTerminal& msg) {
    state.terminal.items.erase(msg.terminal);
    settle();
    return Result::SUCCESS;
}

Result TerminalActions::handle(const MailCloseTerminalTab& msg) {
    settle();
    std::erase_if(state.terminal.items, [&](const auto& entry) {
        return entry.second.tab == msg.tab;
    });
    state.terminal.tabs.erase(msg.tab);
    settle();
    return Result::SUCCESS;
}

Result TerminalActions::handle(const MailFocusTerminalTab& msg) {
    auto& terminals = state.terminal;
    if (!terminals.tabs.contains(msg.tab)) {
        return Result::SUCCESS;
    }
    WorkbenchActions::Focus(state, std::nullopt);
    terminals.focused = msg.tab;
    Terminal* target = nullptr;
    for (auto& [_, terminal] : terminals.items) {
        if (terminal.tab == msg.tab && (!target || terminal.focusedAt > target->focusedAt)) {
            target = &terminal;
        }
    }
    if (target) {
        ++target->focusRequests;
    }
    return Result::SUCCESS;
}

Result TerminalActions::handle(const MailSetTerminalTabLayout& msg) {
    auto& tabs = state.terminal.tabs;
    if (!tabs.contains(msg.tab)) {
        return Result::SUCCESS;
    }
    if (msg.tiles.has_value()) {
        tabs[msg.tab] = *msg.tiles;
    } else {
        tabs.erase(msg.tab);
    }
    return Result::SUCCESS;
}

Result TerminalActions::handle(const MailTerminalInput& msg) {
    const auto session = sessions.find(msg.terminal);
    const auto terminal = state.terminal.items.find(msg.terminal);
    if (session == sessions.end() || terminal == state.terminal.items.end()) {
        return Result::SUCCESS;
    }
    auto& target = *session->second.session;
    std::visit([&](const auto& input) {
        using Input = std::decay_t<decltype(input)>;
        if constexpr (std::is_same_v<Input, Sakura::Terminal::Grid>) {
            target.resize(input);
        } else if constexpr (std::is_same_v<Input, KeyEvent>) {
            target.key(input);
        } else if constexpr (std::is_same_v<Input, MailTerminalInput::Text>) {
            target.text(input.value, input.modifiers);
        } else if constexpr (std::is_same_v<Input, MailTerminalInput::Paste>) {
            target.paste(input.value);
        } else if constexpr (std::is_same_v<Input, MailTerminalInput::DropFiles>) {
            target.paste(TerminalSession::EscapePaths(input.paths));
            ++terminal->second.focusRequests;
        } else if constexpr (std::is_same_v<Input, MailTerminalInput::Scroll>) {
            target.scroll(input.rows, input.position, input.modifiers);
        } else if constexpr (std::is_same_v<Input, MailTerminalInput::ScrollTo>) {
            target.scrollTo(input.row);
        } else if constexpr (std::is_same_v<Input, Sakura::Terminal::Selection>) {
            target.select(input);
        } else if constexpr (std::is_same_v<Input, MouseEvent>) {
            target.pointer(input);
        } else if constexpr (std::is_same_v<Input, MailTerminalInput::Focus>) {
            target.focus(input.value);
        } else if constexpr (std::is_same_v<Input, MailTerminalInput::Search>) {
            terminal->second.search = input.query;
            target.search(input.query.value_or(""));
        } else if constexpr (std::is_same_v<Input, MailTerminalInput::SearchStep>) {
            target.searchStep(input.older);
        } else if constexpr (std::is_same_v<Input, MailTerminalInput::OpenLink>) {
            if (Openable(input.uri)) {
                callbacks.enqueueMail(MailOpenUrl{.url = input.uri, .notifyResult = true});
            }
        } else if constexpr (std::is_same_v<Input, MailTerminalInput::Zoom>) {
            auto& zoom = terminal->second.zoom;
            zoom = input.step == 0 ? 0 : std::clamp(zoom + input.step, kMinZoom, kMaxZoom);
        }
    }, msg.input);
    if (const auto* focus = std::get_if<MailTerminalInput::Focus>(&msg.input); focus && focus->value) {
        terminal->second.focusedAt = ++state.terminal.focusSerial;
    }
    publish(terminal->second, session->second);
    return Result::SUCCESS;
}

void TerminalActions::reconcile() {
    settle();
    auto& terminals = state.terminal.items;
    std::erase_if(sessions, [&](const auto& entry) {
        return !terminals.contains(entry.first);
    });
    if (terminals.empty()) {
        return;
    }

    const TerminalSession::Theme theme = palette();
    for (auto& [id, terminal] : terminals) {
        auto session = sessions.find(id);
        if (session == sessions.end()) {
            auto started = std::make_unique<TerminalSession>();
            if (started->start({.directory = terminal.directory, .theme = theme}) != Result::SUCCESS) {
                callbacks.notify(Sakura::ToastType::Error, 5000,
                                 TerminalSession::Supported() ? "Failed to start the terminal."
                                                              : "The terminal is not available in this build.");
                callbacks.enqueueMail(MailCloseTerminal{.terminal = id});
                continue;
            }
            session = sessions.emplace(id, Session{.session = std::move(started)}).first;
        }

        auto& running = *session->second.session;
        running.theme(theme);
        running.poll();
        if (!running.running()) {
            callbacks.enqueueMail(MailCloseTerminal{.terminal = id});
            continue;
        }
        publish(terminal, session->second);
    }
}

void TerminalActions::shutdown() {
    sessions.clear();
}

void TerminalActions::publish(Terminal& terminal, Session& entry) {
    const auto& session = *entry.session;
    if (terminal.revision == 0 || entry.revision != session.revision()) {
        entry.revision = session.revision();
        terminal.screen = session.screen();
        terminal.revision = ++published;
    }
    if (terminal.title != session.title()) {
        terminal.title = session.title();
    }
}

TerminalSession::Theme TerminalActions::palette() const {
    const auto color = [&](const std::string& key, ColorRGBA<F32> fallback) {
        const auto entry = state.sakura.colorMap.find(key);
        return entry == state.sakura.colorMap.end() ? fallback : entry->second;
    };
    const TerminalSession::Theme fallback;
    return {
        .foreground = color("text_primary", fallback.foreground),
        .background = color("tile_bg", fallback.background),
    };
}

U64 TerminalActions::open(const std::string& key) {
    return open(TileMeta{.pane = key});
}

U64 TerminalActions::open(TileMeta layout) {
    auto& terminals = state.terminal;
    const U64 tab = terminals.nextTab++;
    terminals.tabs[tab] = std::move(layout);
    WorkbenchActions::Focus(state, std::nullopt);
    terminals.focused = tab;
    terminals.requested = tab;
    ++terminals.requests;
    return tab;
}

void TerminalActions::splice(TileMeta layout, const std::optional<U64>& tab,
                             const std::function<void(const std::string&, const TileMeta&)>& replace) {
    auto& terminals = state.terminal;
    bool spliced = true;
    while (spliced) {
        spliced = false;
        std::vector<std::string> panes;
        CollectTilePanes(layout, panes);
        for (const auto& pane : panes) {
            const auto other = ParseTerminalTabKey(pane);
            if (!other.has_value() || *other == tab) {
                continue;
            }
            const auto source = terminals.tabs.find(*other);
            if (source == terminals.tabs.end()) {
                continue;
            }
            const TileMeta content = source->second;
            terminals.tabs.erase(source);
            layout = ReplaceTilePane(layout, pane, content);
            replace(pane, content);
            spliced = true;
            break;
        }
    }
}

std::function<bool(const std::string&)> TerminalActions::claimer(const std::optional<std::string>& flowgraph,
                                                                 const std::optional<U64>& tab) {
    return [this, flowgraph, tab](const std::string& pane) {
        if (ParseTerminalTabKey(pane).has_value()) {
            return false;
        }
        const auto id = ParseTerminalPaneKey(pane);
        if (!id.has_value()) {
            return true;
        }
        auto& items = state.terminal.items;
        const auto terminal = items.find(*id);
        if (terminal == items.end()) {
            return false;
        }
        auto& item = terminal->second;
        if (item.placement.has_value()) {
            if (item.flowgraph != flowgraph || item.tab != tab) {
                return false;
            }
            item.placement.reset();
            return true;
        }
        if (item.flowgraph.has_value() || item.tab.has_value()) {
            return false;
        }
        item.flowgraph = flowgraph;
        item.tab = tab;
        return true;
    };
}

void TerminalActions::settle() {
    auto& terminals = state.terminal;
    for (auto& [_, terminal] : terminals.items) {
        const bool owned = (terminal.flowgraph.has_value() && state.flowgraph.items.contains(*terminal.flowgraph)) ||
                           (terminal.tab.has_value() && terminals.tabs.contains(*terminal.tab));
        if (terminal.placement.has_value() && owned) {
            continue;
        }
        terminal.placement.reset();
        terminal.flowgraph.reset();
        terminal.tab.reset();
    }
    std::vector<std::string> flowgraphs;
    for (const auto& [flowgraphId, _] : state.flowgraph.layouts) {
        flowgraphs.push_back(flowgraphId);
    }
    for (const auto& flowgraphId : flowgraphs) {
        if (state.flowgraph.items.contains(flowgraphId)) {
            splice(state.flowgraph.layouts.at(flowgraphId), std::nullopt,
                   [&](const std::string& pane, const TileMeta& content) {
                       LayoutActions::ReplacePane(state.flowgraph, flowgraphId, pane, content);
                   });
            continue;
        }
        if (auto orphans = LayoutActions::TakePanes(state.flowgraph, flowgraphId, [](const std::string& pane) {
                return ParseTerminalPaneKey(pane).has_value() || ParseTerminalTabKey(pane).has_value();
            })) {
            (void)open(std::move(*orphans));
        }
    }
    std::vector<U64> tabs;
    for (const auto& [tab, _] : terminals.tabs) {
        tabs.push_back(tab);
    }
    for (const U64 tab : tabs) {
        if (const auto layout = terminals.tabs.find(tab); layout != terminals.tabs.end()) {
            splice(layout->second, tab, [&](const std::string& pane, const TileMeta& content) {
                auto& target = terminals.tabs.at(tab);
                target = ReplaceTilePane(target, pane, content);
            });
        }
    }
    for (const auto& flowgraphId : flowgraphs) {
        if (state.flowgraph.items.contains(flowgraphId)) {
            LayoutActions::KeepPanes(state.flowgraph, flowgraphId, claimer(flowgraphId, std::nullopt));
        }
    }
    std::erase_if(terminals.tabs, [&](auto& entry) {
        auto kept = FilterTiles(entry.second, claimer(std::nullopt, entry.first));
        if (!kept.has_value()) {
            return true;
        }
        entry.second = std::move(*kept);
        return false;
    });
    for (auto& [id, terminal] : terminals.items) {
        if (terminal.tab.has_value() && !terminals.tabs.contains(*terminal.tab)) {
            terminal.placement.reset();
            terminal.tab.reset();
        }
        if (!terminal.flowgraph.has_value() && !terminal.tab.has_value()) {
            terminal.tab = open(MakeTerminalPaneKey(id));
        }
    }
    if (terminals.focused.has_value() && !terminals.tabs.contains(*terminals.focused)) {
        terminals.focused.reset();
    }
}

std::string TerminalActions::directory(const std::string& flowgraphId) const {
    const auto flowgraph = state.flowgraph.items.find(flowgraphId);
    if (flowgraph == state.flowgraph.items.end() || !flowgraph->second || flowgraph->second->path().empty()) {
        return {};
    }
    return Platform::PathToUtf8(Platform::PathFromUtf8(flowgraph->second->path()).parent_path());
}

}  // namespace Jetstream
