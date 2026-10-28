#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_AGENT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_AGENT_HH

#include "persist.hh"
#include "workbench.hh"
#include "../model/callbacks.hh"
#include "../model/messages.hh"
#include "../model/state.hh"
#include "../support/agent/providers.hh"
#include "../support/agent/runtime.hh"

#include "jetstream/config.hh"
#include "jetstream/types.hh"
#include "tools/random.hh"

#include <algorithm>
#include <chrono>
#include <atomic>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace Jetstream {

struct AgentActions {
    using Filter = std::tuple<MailSetAgentEnabled,
                              MailSetAgentChatVisible,
                              MailSetAgentVoiceVisible,
                              MailStartAgentVoice,
                              MailSubmitAgentChat,
                              MailCancelAgentChat,
                              MailClearAgentChat,
                              MailSelectAgentModel,
                              MailSetAgentReasoningEffort,
                              MailOpenAgentModelPicker,
                              MailSetAgentModelSearch,
                              MailAddAgentModel,
                              MailRemoveAgentModel,
                              MailOpenAgentProviderEditor,
                              MailSaveAgentProvider,
                              MailRemoveAgentProvider,
                              MailTestAgentProvider,
                              MailSetAgentVoiceBackend,
                              MailSetAgentVoiceModel,
                              MailSetAgentVoiceName>;

    DefaultCompositorState& state;
    DefaultCompositorCallbacks& callbacks;
    AgentRuntime& runtime;

    AgentActions(DefaultCompositorState& state, DefaultCompositorCallbacks& callbacks, AgentRuntime& runtime) :
        state(state), callbacks(callbacks), runtime(runtime) {}

    Result handle(const MailSetAgentEnabled& msg) {
#ifdef JST_OS_BROWSER
        if (msg.value) {
            return Result::SUCCESS;
        }
#endif
        auto& agent = state.agent;
        agent.settings.enabled = msg.value;

        if (!msg.value) {
            agent.ui.visible = false;
            hideVoice();
            runtime.session().cancel();
            cancel();
        }

        return persist();
    }

    Result handle(const MailSetAgentChatVisible& msg) {
        auto& ui = state.agent.ui;
        if (msg.value && !state.agent.settings.enabled) {
            return Result::SUCCESS;
        }

        if (msg.value) {
            hideVoice();
        }
        ui.visible = msg.value;

        if (msg.value && msg.focus) {
            ++ui.chatFocusRequests;
        }

        return Result::SUCCESS;
    }

    Result handle(const MailSetAgentVoiceVisible& msg) {
        setVoiceVisible(msg.value);
        return Result::SUCCESS;
    }

    Result handle(const MailStartAgentVoice&) {
        if (state.agent.ui.voiceVisible && !startVoice()) {
            hideVoice();
        }
        return Result::SUCCESS;
    }

    Result handle(const MailSubmitAgentChat& msg) {
        auto& agent = state.agent;
        if (!agent.settings.enabled || !runtime.submit(msg.text)) {
            return Result::SUCCESS;
        }

        auto& chat = agent.chat;
        if (msg.composer == AgentComposer::Welcome) {
            ++chat.welcomeAccepted;
        } else {
            ++chat.dockAccepted;
        }

        if (msg.reveal) {
            hideVoice();
            agent.ui.visible = true;
            ++agent.ui.chatFocusRequests;
        }

        return Result::SUCCESS;
    }

    Result handle(const MailCancelAgentChat&) {
        runtime.session().cancel();
        return Result::SUCCESS;
    }

    Result handle(const MailClearAgentChat&) {
        runtime.session().clear();
        return Result::SUCCESS;
    }

    Result handle(const MailSelectAgentModel& msg) {
        if (msg.model.empty()) {
            return Result::SUCCESS;
        }

        auto& settings = state.agent.settings;
        if (!msg.provider.empty()) {
            settings.provider = msg.provider;
        }
        settings.model = msg.model;

        return persist();
    }

    Result handle(const MailSetAgentReasoningEffort& msg) {
        auto& models = state.agent.settings.models;
        const auto model = std::find_if(models.begin(), models.end(), [&](const auto& entry) {
            return AgentProviders::Same(entry, msg.provider, msg.model);
        });
        if (model == models.end() || model->reasoningEffort == msg.value) {
            return Result::SUCCESS;
        }

        model->reasoningEffort = msg.value;
        return persist();
    }

    Result handle(const MailOpenAgentModelPicker&) {
        openIntelligencePage(IntelligencePage::AddModel);
        state.agent.catalog.search.clear();
        ensureModels();
        return Result::SUCCESS;
    }

    Result handle(const MailSetAgentModelSearch& msg) {
        state.agent.catalog.search = msg.value;
        return Result::SUCCESS;
    }

    Result handle(const MailAddAgentModel& msg) {
        if (msg.value.provider.empty() || msg.value.id.empty()) {
            callbacks.notify(Sakura::ToastType::Error, 5000, "Models need a provider and a model id.");
            return Result::SUCCESS;
        }

        auto& settings = state.agent.settings;
        const bool exists = std::any_of(settings.models.begin(), settings.models.end(), [&](const auto& entry) {
            return AgentProviders::Same(entry, msg.value);
        });
        if (!exists) {
            settings.models.push_back(msg.value);
        }
        if (settings.models.size() == 1) {
            settings.provider = msg.value.provider;
            settings.model = msg.value.id;
        }

        returnToSettings();
        JST_CHECK(persist());
        callbacks.notify(Sakura::ToastType::Success, 3000, "Model added.");
        return Result::SUCCESS;
    }

    Result handle(const MailRemoveAgentModel& msg) {
        auto& settings = state.agent.settings;
        const auto removed = std::erase_if(settings.models, [&](const auto& entry) {
            return AgentProviders::Same(entry, msg.value);
        });
        if (!removed) {
            return Result::SUCCESS;
        }

        if (AgentProviders::Same(settings.voice.backend, msg.value)) {
            settings.voice.backend = {};
        }

        const bool selected = AgentProviders::Same(msg.value, settings.provider, settings.model);
        if (selected && !selectFallbackModel()) {
            settings.provider.clear();
            settings.model.clear();
        }

        JST_CHECK(persist());
        callbacks.notify(Sakura::ToastType::Success, 3000, "Model removed.");
        return Result::SUCCESS;
    }

    Result handle(const MailOpenAgentProviderEditor& msg) {
        auto& agent = state.agent;
        auto& providers = agent.providers;
        providers.editTarget.reset();

        if (msg.name.has_value()) {
            auto provider = AgentProviders(agent).find(*msg.name);
            if (!provider.name.empty()) {
                providers.editTarget = std::move(provider);
            }
        }

        providers.editSession++;
        resetProviderTest();
        openIntelligencePage(IntelligencePage::Provider);
        return Result::SUCCESS;
    }

    Result handle(const MailSaveAgentProvider& msg) {
        auto& agent = state.agent;
        auto& task = agent.providers.keyTask;
        if (task.running) {
            return Result::SUCCESS;
        }

        auto provider = AgentProviders::Normalize(msg.value);
        if (!AgentProviders::Valid(provider)) {
            callbacks.notify(Sakura::ToastType::Error, 5000, "Providers need a name and a base URL.");
            return Result::SUCCESS;
        }

        const auto previousName = msg.replaceName.value_or(provider.name);
        const bool renamed = previousName != provider.name;
        if (renamed && !AgentProviders(agent).find(provider.name).name.empty()) {
            callbacks.notify(Sakura::ToastType::Error,
                             5000,
                             jst::fmt::format("A provider named '{}' already exists.", provider.name));
            return Result::SUCCESS;
        }

        const auto previous = AgentProviders(agent).find(previousName);
        task.kind = DefaultCompositorState::AgentState::ProviderKeyTask::Kind::Save;
        task.previousName = previousName;
        task.session = agent.providers.editSession;

        if (!msg.apiKey.empty()) {
            if (previous.apiKeyId.empty()) {
                provider.apiKeyId = Random::HexToken();
            } else {
                provider.apiKeyId = previous.apiKeyId;
            }
            task.future = std::async(std::launch::async,
                                     AgentProviders::Store,
                                     provider.apiKeyId,
                                     provider.host,
                                     msg.apiKey);
        } else if (AgentProviders::Transferable(previous, provider)) {
            provider.apiKeyId = previous.apiKeyId;
        } else if (!previous.apiKeyId.empty()) {
            task.future = std::async(std::launch::async, AgentProviders::Erase, previous.apiKeyId);
        }

        if (!task.future.valid()) {
            return saveProvider(std::move(provider), previousName);
        }

        task.running = true;
        task.provider = std::move(provider);
        return Result::SUCCESS;
    }

    Result handle(const MailRemoveAgentProvider& msg) {
        auto& task = state.agent.providers.keyTask;
        if (task.running) {
            return Result::SUCCESS;
        }

        const auto provider = AgentProviders(state.agent).find(msg.name);
        if (provider.name.empty()) {
            return Result::SUCCESS;
        }
        if (provider.apiKeyId.empty()) {
            return removeProvider(msg.name);
        }

        task.running = true;
        task.kind = DefaultCompositorState::AgentState::ProviderKeyTask::Kind::Remove;
        task.provider = provider;
        task.previousName = msg.name;
        task.future = std::async(std::launch::async, AgentProviders::Erase, provider.apiKeyId);
        return Result::SUCCESS;
    }

    Result handle(const MailTestAgentProvider& msg) {
        auto& test = state.agent.providerTest;
        if (test.running) {
            return Result::SUCCESS;
        }

        test.revision = msg.revision;
        auto provider = AgentProviders::Normalize(msg.value);
        if (!AgentProviders::Valid(provider)) {
            test.completed = true;
            test.success = false;
            test.message = "A name and a base URL are required.";
            return Result::SUCCESS;
        }

        std::optional<Settings::Intelligence::Provider> stored;
        if (msg.apiKey.empty()) {
            const auto& target = state.agent.providers.editTarget;
            const std::string& sourceName = target ? target->name : provider.name;
            auto source = AgentProviders(state.agent).find(sourceName);
            if (AgentProviders::Transferable(source, provider)) {
                stored = std::move(source);
            }
        }

        test.running = true;
        test.completed = false;
        test.success = false;
        test.message.clear();
        test.cancel = std::make_shared<std::atomic<bool>>(false);
        test.future = std::async(std::launch::async,
                                 AgentProviders::Test,
                                 std::move(provider),
                                 msg.apiKey,
                                 std::move(stored),
                                 Cancelled(test.cancel));
        return Result::SUCCESS;
    }

    Result handle(const MailSetAgentVoiceBackend& msg) {
        state.agent.settings.voice.backend = msg.value;
        return persist();
    }

    Result handle(const MailSetAgentVoiceModel& msg) {
        state.agent.settings.voice.model = msg.value;
        return persist();
    }

    Result handle(const MailSetAgentVoiceName& msg) {
        state.agent.settings.voice.voice = msg.value;
        return persist();
    }

    void refresh() {
        auto& agent = state.agent;
        auto& catalog = agent.catalog;
        if (!agent.settings.enabled || stopping) {
            return;
        }

        if (catalog.loading) {
            catalog.refreshPending = true;
            return;
        }

        catalog.refreshPending = false;
        catalog.loading = true;
        catalog.error.clear();
        catalog.cancel = std::make_shared<std::atomic<bool>>(false);
        catalog.future = std::async(std::launch::async,
                                    AgentProviders::Models,
                                    AgentProviders(agent).all(),
                                    Cancelled(catalog.cancel));
    }

    void reconcile() {
        checkContextLengths();
        checkSettingsSection();
        collectProviderTest();
        collectProviderKeyTask();
        collectModels();
    }

    void shutdown() {
        stopping = true;
        cancel();

        auto& catalog = state.agent.catalog;
        if (catalog.loading) {
            catalog.future.wait();
            collectModels();
        }
        auto& test = state.agent.providerTest;
        if (test.running) {
            test.future.wait();
            collectProviderTest();
        }

        auto& keyTask = state.agent.providers.keyTask;
        if (keyTask.running) {
            keyTask.future.wait();
            collectProviderKeyTask();
        }
    }

 private:
    void setVoiceVisible(bool visible) {
        if (!visible) {
            hideVoice();
            return;
        }

        if (state.agent.settings.enabled && !state.agent.ui.voiceVisible) {
            startVoice();
        }
    }

    void cancel() {
        auto& agent = state.agent;
        agent.catalog.refreshPending = false;
        if (agent.catalog.cancel) {
            agent.catalog.cancel->store(true);
        }
        if (agent.providerTest.cancel) {
            agent.providerTest.cancel->store(true);
        }
    }

    static Agent::Cancellation Cancelled(const std::shared_ptr<std::atomic<bool>>& flag) {
        return [flag]() {
            return flag->load();
        };
    }

    void hideVoice() {
        state.agent.ui.voiceVisible = false;
        runtime.hangup();
    }

    bool startVoice() {
        auto& agent = state.agent;
        const auto availability = AgentProviders(agent).availability();
        if (availability.state != AgentProviders::VoiceAvailability::State::Ready) {
            callbacks.notify(Sakura::ToastType::Info, 5000, availability.message);
            return false;
        }

        agent.ui.voiceVisible = true;
        agent.ui.visible = false;
        runtime.listen(availability.voice);

        return true;
    }

    void ensureModels() {
        const auto& catalog = state.agent.catalog;
        if (catalog.models.empty() && !catalog.loading) {
            refresh();
        }
    }

    void resetProviderTest() {
        auto& test = state.agent.providerTest;
        if (test.running) {
            return;
        }

        test.completed = false;
        test.success = false;
        test.message.clear();
    }

    bool selectFallbackModel() {
        auto& settings = state.agent.settings;
        if (settings.models.empty()) {
            return false;
        }

        settings.provider = settings.models.front().provider;
        settings.model = settings.models.front().id;
        return true;
    }

    void returnToSettings() {
        state.settings.intelligencePage = IntelligencePage::Overview;
    }

    void openIntelligencePage(IntelligencePage page) {
        WorkbenchActions::OpenSettings(state, SettingsSection::Intelligence);
        state.settings.intelligencePage = page;
    }

    void renameProvider(const std::string& previous, const std::string& next) {
        auto& agent = state.agent;
        auto& settings = agent.settings;
        RenameModels(settings.models, previous, next);
        RenameModels(agent.catalog.models, previous, next);

        if (settings.provider == previous) {
            settings.provider = next;
        }
        if (settings.voice.backend.provider == previous) {
            settings.voice.backend.provider = next;
        }

        std::erase_if(settings.customProviders, [&](const auto& entry) {
            return entry.name == previous;
        });
    }

    Result saveProvider(Settings::Intelligence::Provider provider, const std::string& previousName) {
        auto& agent = state.agent;
        auto& customs = agent.settings.customProviders;
        if (previousName != provider.name) {
            renameProvider(previousName, provider.name);
        }

        const auto existing = std::find_if(customs.begin(), customs.end(), [&](const auto& entry) {
            return entry.name == provider.name;
        });
        const bool replaced = existing != customs.end() || previousName != provider.name;
        if (existing != customs.end()) {
            *existing = provider;
        } else {
            customs.push_back(provider);
        }

        JST_CHECK(persist());
        const char* verb = replaced ? "updated" : "added";
        callbacks.notify(Sakura::ToastType::Success,
                         3000,
                         jst::fmt::format("Provider '{}' {}.", provider.name, verb));

        const bool sameSession = agent.providers.editSession == agent.providers.keyTask.session;
        const bool editorOpen = state.interface.settingsTab &&
                                state.settings.intelligencePage == IntelligencePage::Provider;
        if (sameSession && editorOpen) {
            agent.providers.editTarget.reset();
            returnToSettings();
            resetProviderTest();
        }

        refresh();
        return Result::SUCCESS;
    }

    Result removeProvider(const std::string& name) {
        auto& agent = state.agent;
        auto& settings = agent.settings;
        const auto removed = std::erase_if(settings.customProviders, [&](const auto& entry) {
            return entry.name == name;
        });
        if (!removed) {
            return Result::SUCCESS;
        }

        std::erase_if(settings.models, [&](const auto& entry) {
            return entry.provider == name;
        });
        if (settings.voice.backend.provider == name) {
            settings.voice.backend = {};
        }

        if (settings.provider == name && !selectFallbackModel()) {
            const auto remaining = AgentProviders(agent).all();
            if (remaining.empty()) {
                settings.provider = "";
            } else {
                settings.provider = remaining.front().name;
            }
            settings.model.clear();
        }

        std::erase_if(agent.catalog.models, [&](const auto& entry) {
            return entry.provider == name;
        });

        JST_CHECK(persist());
        callbacks.notify(Sakura::ToastType::Success, 3000, jst::fmt::format("Provider '{}' removed.", name));
        refresh();
        return Result::SUCCESS;
    }

    void collectProviderKeyTask() {
        auto& task = state.agent.providers.keyTask;
        if (!task.running || !FutureReady(task.future)) {
            return;
        }

        const auto result = task.future.get();
        task.running = false;
        if (!result.success) {
            callbacks.notify(Sakura::ToastType::Error, 5000, result.message);
            return;
        }

        Result commit;
        if (task.kind == DefaultCompositorState::AgentState::ProviderKeyTask::Kind::Save) {
            commit = saveProvider(std::move(task.provider), task.previousName);
        } else {
            commit = removeProvider(task.previousName);
        }
        if (commit != Result::SUCCESS) {
            JST_WARN("[AGENT] Failed to apply provider changes.");
        }
    }

    void checkContextLengths() {
        auto& agent = state.agent;
        const auto& models = agent.settings.models;
        if (!agent.settings.enabled || agent.catalog.contextLengthsChecked) {
            return;
        }

        agent.catalog.contextLengthsChecked = true;
        const bool missing = std::any_of(models.begin(), models.end(), [](const auto& model) {
            const bool missingLimits = model.contextLength == 0 || model.outputLength == 0;
            const bool missingDetails = model.name.empty() || model.reasoningEfforts.empty() || !model.priced;
            return missingLimits || missingDetails;
        });
        if (missing && !agent.catalog.loading) {
            refresh();
        }
    }

    void checkSettingsSection() {
        auto& agent = state.agent;
        const bool shown = state.interface.settingsTab && state.settings.section == SettingsSection::Intelligence;
        const bool entered = shown && !agent.ui.settingsShown;
        agent.ui.settingsShown = shown;

        if (entered) {
            ensureModels();
        }
        if (!shown || agent.catalog.loading) {
            return;
        }

        const auto providers = AgentProviders(agent).all();
        const bool unchecked = std::any_of(providers.begin(), providers.end(), [&](const auto& provider) {
            return !agent.providers.health.contains(provider.name);
        });
        if (unchecked) {
            refresh();
        }
    }

    void collectProviderTest() {
        auto& test = state.agent.providerTest;
        if (!test.running || !FutureReady(test.future)) {
            return;
        }

        const auto [success, message] = test.future.get();
        test.running = false;
        if (test.cancel && test.cancel->load()) {
            test.completed = false;
            test.message.clear();
            return;
        }
        test.completed = true;
        test.success = success;
        test.message = message;
    }

    void collectModels() {
        auto& agent = state.agent;
        auto& catalog = agent.catalog;
        if (!catalog.loading || !FutureReady(catalog.future)) {
            return;
        }

        auto result = catalog.future.get();
        catalog.loading = false;
        if (catalog.cancel && catalog.cancel->load()) {
            return;
        }
        if (catalog.refreshPending) {
            refresh();
            return;
        }

        catalog.models = std::move(result.models);
        catalog.error = std::move(result.error);
        agent.providers.health = std::move(result.health);

        if (mergeModelMetadata() && persist() != Result::SUCCESS) {
            JST_WARN("[AGENT] Failed to persist model metadata.");
        }
    }

    bool mergeModelMetadata() {
        bool changed = false;
        for (auto& saved : state.agent.settings.models) {
            for (const auto& model : state.agent.catalog.models) {
                changed |= MergeModel(saved, model);
            }
        }

        return changed;
    }

    Result persist() {
        return PersistSettings([&](Settings& settings) {
            settings.intelligence = state.agent.settings;
        });
    }

    template<class Value>
    static bool FutureReady(const std::future<Value>& future) {
        return future.valid() && future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready;
    }

    static void RenameModels(std::vector<Settings::Intelligence::Model>& models,
                             const std::string& previous, const std::string& next) {
        for (auto& model : models) {
            if (model.provider == previous) {
                model.provider = next;
            }
        }
    }

    static bool MergeModel(Settings::Intelligence::Model& saved, const Settings::Intelligence::Model& model) {
        if (!AgentProviders::Same(model, saved)) {
            return false;
        }

        bool changed = false;
        if (model.contextLength != 0 && model.contextLength != saved.contextLength) {
            saved.contextLength = model.contextLength;
            changed = true;
        }
        if (model.outputLength != 0 && model.outputLength != saved.outputLength) {
            saved.outputLength = model.outputLength;
            changed = true;
        }
        if (!model.name.empty() && model.name != saved.name) {
            saved.name = model.name;
            changed = true;
        }
        if (!model.reasoningEfforts.empty() && model.reasoningEfforts != saved.reasoningEfforts) {
            saved.reasoningEfforts = model.reasoningEfforts;
            changed = true;
        }
        const bool repriced = model.inputCost != saved.inputCost || model.outputCost != saved.outputCost;
        if (model.priced && (!saved.priced || repriced)) {
            saved.priced = true;
            saved.inputCost = model.inputCost;
            saved.outputCost = model.outputCost;
            changed = true;
        }

        return changed;
    }

    bool stopping = false;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_AGENT_HH
