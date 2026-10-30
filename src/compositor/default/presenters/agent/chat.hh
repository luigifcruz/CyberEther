#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_AGENT_CHAT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_AGENT_CHAT_HH

#include "labels.hh"
#include "transcript.hh"
#include "../context.hh"
#include "../hud/welcome.hh"
#include "../../views/agent/chat.hh"

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>

namespace Jetstream {

inline std::optional<Sakura::ChatComposer::Usage> AgentChatUsage(U64 tokens, U64 window) {
    if (window == 0) {
        return std::nullopt;
    }
    const F32 used = std::clamp(static_cast<F32>(tokens) / static_cast<F32>(window), 0.0f, 1.0f);
    const U64 percent = static_cast<U64>(std::round(used * 100.0f));
    return Sakura::ChatComposer::Usage{
        .value = used,
        .title = "Context window:",
        .details = {
            jst::fmt::format("{}% used", percent),
            jst::fmt::format("{} / {} tokens used", AgentLabels::Tokens(tokens), AgentLabels::Tokens(window)),
        },
    };
}

struct AgentChatPresenter {
    const PresenterContext& context;
    mutable AgentTranscriptFormatter formatter;

    explicit AgentChatPresenter(const PresenterContext& context) : context(context) {}

    AgentChatView::Config build() const {
        const auto& agent = context.state.agent;
        const auto& settings = agent.settings;
        const auto& conversation = agent.chat.conversation;
        const auto enqueue = context.callbacks.enqueueMail;
        AgentChatView::Config config;
        auto& chat = config.chat;
        chat.snapshot = formatter.format(conversation);
        chat.clearRequest = agent.chat.dockAccepted;
        config.open = agent.ui.visible;
        config.focusRequest = agent.ui.chatFocusRequests;
        config.hidden = !settings.enabled ||
                        (WelcomeShown(context.state) && !agent.ui.visible && conversation.messages.empty());

        const auto modelKey = [](const auto& model) {
            return jst::fmt::format("{}:{}{}", model.provider.size(), model.provider, model.id);
        };
        Sakura::ChatComposer::Selector selector{.placeholder = "No models"};
        U64 contextWindow = 0;
        for (const auto& model : settings.models) {
            const auto key = modelKey(model);
            selector.options.push_back({key, AgentLabels::Model(model), model.id.empty() ? "" : model.provider});
            if (AgentProviders::Same(model, settings.provider, settings.model)) {
                selector.value = key;
                contextWindow = model.contextLength;
                selector.effort = AgentProviders::Effort(model);
                for (const auto& effort : model.reasoningEfforts) {
                    selector.efforts.push_back({effort, AgentLabels::Effort(effort), ""});
                }
                selector.onEffort = [enqueue, provider = model.provider, id = model.id](const std::string& value) {
                    enqueue(MailSetAgentReasoningEffort{.provider = provider, .model = id, .value = value});
                };
            }
        }
        selector.disabled = selector.options.empty();
        selector.onSelect = [enqueue, saved = settings.models, modelKey](const std::string& key) {
            for (const auto& model : saved) {
                if (modelKey(model) == key) {
                    enqueue(MailSelectAgentModel{.provider = model.provider, .model = model.id});
                    return;
                }
            }
        };
        chat.selector = std::move(selector);
        chat.usage = AgentChatUsage(conversation.contextTokens, contextWindow);
        config.onOpen = [enqueue]() {
            enqueue(MailSetAgentChatVisible{.value = true, .focus = true});
        };
        config.onMinimize = [enqueue]() {
            enqueue(MailSetAgentChatVisible{.value = false});
        };
        chat.onSubmit = [enqueue, reveal = !agent.ui.visible](const std::string& text) {
            enqueue(MailSubmitAgentChat{.composer = AgentComposer::Dock, .text = text, .reveal = reveal});
        };
        chat.onCancel = [enqueue]() {
            enqueue(MailCancelAgentChat{});
        };
        chat.onClear = [enqueue]() {
            enqueue(MailClearAgentChat{});
        };
        chat.onVoice = [enqueue]() {
            enqueue(MailSetAgentVoiceVisible{.value = true});
        };
        return config;
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_AGENT_CHAT_HH
