#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_TERMINAL_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_TERMINAL_HH

#include "context.hh"
#include "tiles.hh"

#include "../model/messages.hh"
#include "../model/tiles.hh"
#include "../support/terminal.hh"
#include "../views/terminal.hh"

#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct TerminalPresenter {
    const PresenterContext& context;

    explicit TerminalPresenter(const PresenterContext& context) : context(context) {}

    static std::string Title(const std::string& title) {
        return title.empty() ? std::string("Terminal") : jst::fmt::format("Terminal ({})", title);
    }

    std::vector<TerminalView::Config> build() const {
        std::vector<TerminalView::Config> configs;
        const auto enqueue = context.callbacks.enqueueMail;
        for (const auto& [id, terminal] : context.state.terminal.items) {
            const auto send = [enqueue, id](MailTerminalInput::Input input) {
                enqueue(MailTerminalInput{.terminal = id, .input = std::move(input)});
            };
            configs.push_back({
                .id = MakeTerminalPaneKey(id),
                .title = Title(terminal.title),
                .flowgraph = terminal.flowgraph,
                .tab = terminal.tab.has_value() ? std::optional<std::string>(MakeTerminalTabKey(*terminal.tab))
                                                : std::nullopt,
                .terminal = {
                    .id = MakeTerminalPaneKey(id),
                    .screen = &terminal.screen,
                    .revision = terminal.revision,
                    .focusRequest = terminal.focusRequests,
                    .fontSize = Sakura::Typography::FontSize + static_cast<F32>(terminal.zoom),
                    .altSendsEscape = TerminalSession::AltSendsEscape(),
                    .search = terminal.search,
                    .onResize = [send](Sakura::Terminal::Grid grid) {
                        send(grid);
                    },
                    .onKey = [send](const KeyEvent& event) {
                        send(event);
                    },
                    .onText = [send](std::string text, KeyModifiers modifiers) {
                        send(MailTerminalInput::Text{.value = std::move(text), .modifiers = modifiers});
                    },
                    .onPaste = [send](std::string text) {
                        send(MailTerminalInput::Paste{std::move(text)});
                    },
                    .onScroll = [send](I32 rows, Extent2D<F32> position, KeyModifiers modifiers) {
                        send(MailTerminalInput::Scroll{.rows = rows, .position = position, .modifiers = modifiers});
                    },
                    .onScrollTo = [send](U64 row) {
                        send(MailTerminalInput::ScrollTo{.row = row});
                    },
                    .onSelect = [send](Sakura::Terminal::Selection selection) {
                        send(selection);
                    },
                    .onPointer = [send](const MouseEvent& event) {
                        send(event);
                    },
                    .onFocusChange = [send](bool focused) {
                        send(MailTerminalInput::Focus{.value = focused});
                    },
                    .onZoom = [send](I32 step) {
                        send(MailTerminalInput::Zoom{.step = step});
                    },
                    .onOpenLink = [send](std::string uri) {
                        send(MailTerminalInput::OpenLink{.uri = std::move(uri)});
                    },
                    .onSearch = [send](std::optional<std::string> query) {
                        send(MailTerminalInput::Search{.query = std::move(query)});
                    },
                    .onSearchStep = [send](bool older) {
                        send(MailTerminalInput::SearchStep{.older = older});
                    },
                },
                .onClose = [enqueue, id]() {
                    enqueue(MailCloseTerminal{.terminal = id});
                },
                .onSplit = [enqueue, id](bool stacked) {
                    enqueue(MailOpenTerminal{.split = id, .stacked = stacked});
                },
            });
        }
        return configs;
    }

    std::vector<TerminalTabView::Config> tabs() const {
        std::vector<TerminalTabView::Config> configs;
        const auto enqueue = context.callbacks.enqueueMail;
        const auto& terminals = context.state.terminal;
        for (const auto& [tab, layout] : terminals.tabs) {
            std::vector<std::string> panes;
            CollectTilePanes(layout, panes);
            std::string title;
            for (const auto& pane : panes) {
                if (const auto id = ParseTerminalPaneKey(pane); id && terminals.items.contains(*id)) {
                    title = terminals.items.at(*id).title;
                    break;
                }
            }
            configs.push_back({
                .id = MakeTerminalTabKey(tab),
                .title = Title(title),
                .tiles = ToDockTiles(layout),
                .onTiles = [enqueue, tab](std::optional<Sakura::DockTiles> tiles) {
                    enqueue(MailSetTerminalTabLayout{
                        .tab = tab,
                        .tiles = tiles.has_value() ? ToTileMeta(*tiles) : std::nullopt,
                    });
                },
                .onClose = [enqueue, tab]() {
                    enqueue(MailCloseTerminalTab{.tab = tab});
                },
                .onFocus = [enqueue, tab]() {
                    enqueue(MailFocusTerminalTab{.tab = tab});
                },
            });
        }
        return configs;
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_TERMINAL_HH
