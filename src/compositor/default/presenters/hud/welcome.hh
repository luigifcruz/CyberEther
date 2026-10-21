#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_HUD_WELCOME_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_HUD_WELCOME_HH

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/hud/welcome.hh"

#include "jetstream/config.hh"

#include <optional>

namespace Jetstream {

inline bool WelcomeShown(const DefaultCompositorState& state) {
    return state.flowgraph.items.empty() && !state.interface.settingsTab && !state.modal.content.has_value() &&
           !state.filePicker.active;
}

struct WelcomeHudPresenter {
    const PresenterContext& context;

    explicit WelcomeHudPresenter(const PresenterContext& context) : context(context) {}

    std::optional<WelcomeHudView::Config> build() const {
        if (!WelcomeShown(context.state) || context.state.interface.startTab) {
            return std::nullopt;
        }
        return config();
    }

    std::optional<WelcomeHudView::Config> buildStartTab() const {
        if (!context.state.interface.startTab) {
            return std::nullopt;
        }
        return config();
    }

 private:
    WelcomeHudView::Config config() const {
        const auto enqueue = context.callbacks.enqueueMail;
        return WelcomeHudView::Config{
            .version = "v" JETSTREAM_VERSION_STR,
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
                enqueue(MailOpenSettings{.section = SettingsSection::General});
            },
            .onOpenAbout = [enqueue]() {
                enqueue(MailOpenModal{.content = ModalContent::About});
            },
        };
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_HUD_WELCOME_HH
