#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_HUD_WELCOME_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_HUD_WELCOME_HH

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/hud/welcome.hh"

#include "jetstream/config.hh"

#include <optional>

namespace Jetstream {

struct WelcomeHudPresenter {
    const PresenterContext& context;

    explicit WelcomeHudPresenter(const PresenterContext& context) : context(context) {}

    std::optional<WelcomeHudView::Config> build() const {
        if (!context.state.flowgraph.items.empty() || context.state.modal.content.has_value() ||
            context.state.filePicker.active) {
            return std::nullopt;
        }

        const auto enqueue = context.callbacks.enqueueMail;
        return WelcomeHudView::Config{
            .version = "v" JETSTREAM_VERSION_STR,
            .onNewFlowgraph = [enqueue]() {
                enqueue(MailNewFlowgraph{});
            },
            .onOpenFlowgraph = [enqueue]() {
                enqueue(MailOpenFlowgraph{});
            },
            .onOpenExamples = [enqueue]() {
                enqueue(MailOpenModal{.content = ModalContent::FlowgraphExamples});
            },
            .onOpenWebsite = [enqueue]() {
                enqueue(MailOpenUrl{.url = "https://cyberether.org"});
            },
            .onOpenDocs = [enqueue]() {
                enqueue(MailOpenUrl{.url = "https://cyberether.org/docs"});
            },
            .onOpenSettings = [enqueue]() {
                enqueue(MailOpenModal{.content = ModalContent::Settings, .settings = SettingsSection::General});
            },
            .onOpenAbout = [enqueue]() {
                enqueue(MailOpenModal{.content = ModalContent::About});
            },
        };
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_HUD_WELCOME_HH
