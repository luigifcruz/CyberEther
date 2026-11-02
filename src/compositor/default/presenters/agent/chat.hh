#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_AGENT_CHAT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_AGENT_CHAT_HH

#include "labels.hh"
#include "transcript.hh"
#include "../context.hh"
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
        config.hidden = !settings.enabled;

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
        config.onClose = [enqueue]() {
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
        config.placeholder = placeholder();
        return config;
    }

 private:
    AgentChatPlaceholder::Config placeholder() const {
        const auto enqueue = context.callbacks.enqueueMail;
        if (context.state.agent.settings.models.empty()) {
            return {
                .title = "Connect a model",
                .subtitle = "Add a model provider in Settings to start chatting.",
                .setupLabel = "Open Settings",
                .onSetup = [enqueue]() {
                    enqueue(MailOpenSettings{.section = SettingsSection::Intelligence});
                },
            };
        }
        const auto suggest = [enqueue](const char* icon, std::string label) {
            return AgentChatPlaceholder::Suggestion{
                .icon = icon,
                .label = label,
                .onClick = [enqueue, label]() {
                    enqueue(MailSubmitAgentChat{.composer = AgentComposer::Dock, .text = label, .reveal = false});
                },
            };
        };
        const auto active = context.state.activeFlowgraph();
        if (!active.has_value()) {
            return {
                .title = "What should we build?",
                .subtitle = "Describe a signal chain or ask anything about CyberEther.",
                .suggestions = {
                    suggest(ICON_LC_RADIO, "Build an FM radio receiver."),
                    suggest(ICON_LC_AUDIO_WAVEFORM, "Make a spectrum analyzer for my radio."),
                    suggest(ICON_LC_PLANE, "Track nearby aircraft with ADS-B."),
                    suggest(ICON_LC_CIRCLE_QUESTION_MARK, "What can CyberEther do?"),
                },
            };
        }
        const auto blocks = context.state.flowgraph.blocks.find(*active);
        if (blocks == context.state.flowgraph.blocks.end() || blocks->second.empty()) {
            return {
                .title = "What goes in here?",
                .subtitle = "Describe the signal chain and the assistant will build it in this flowgraph.",
                .suggestions = {
                    suggest(ICON_LC_ACTIVITY, "Generate a test tone and plot its spectrum."),
                    suggest(ICON_LC_RADIO, "Build an FM radio receiver."),
                    suggest(ICON_LC_AUDIO_WAVEFORM, "Make a spectrum analyzer for my radio."),
                    suggest(ICON_LC_FILE_AUDIO, "Open a recording and plot its waterfall."),
                },
            };
        }
        return {
            .title = "What should we change?",
            .subtitle = "Ask how it works, look for problems, or describe an edit.",
            .suggestions = {
                suggest(ICON_LC_BOOK_OPEN, "Explain how this flowgraph works."),
                suggest(ICON_LC_SEARCH_CHECK, "Look for problems in this flowgraph."),
                suggest(ICON_LC_LIGHTBULB, "Suggest what to add next."),
                suggest(ICON_LC_GAUGE, "Make this flowgraph run faster."),
            },
        };
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_AGENT_CHAT_HH
