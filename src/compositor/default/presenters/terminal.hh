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

    explicit TerminalPresenter(const PresenterContext& context);
    static std::string Title(const std::string& title);
    static std::optional<Sakura::Dock::Placement> Placement(const DefaultCompositorState::TerminalState::Item& item);
    std::vector<TerminalView::Config> build() const;
    std::vector<TerminalTabView::Config> tabs() const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_TERMINAL_HH
