#include "agent.hh"

namespace Jetstream {

AgentActions::AgentActions(DefaultCompositorState& state, DefaultCompositorCallbacks& callbacks, AgentRuntime& runtime) :
    state(state), callbacks(callbacks), runtime(runtime) {}

Result AgentActions::handle(const MailSetAgentEnabled& msg) {
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

Result AgentActions::handle(const MailSetAgentChatVisible& msg) {
    auto& ui = state.agent.ui;
    if (msg.value && !state.agent.settings.enabled) {
        return Result::SUCCESS;
    }

    if (msg.value) {
        hideVoice();
        ui.unread = false;
    }
    ui.visible = msg.value;

    if (msg.value && msg.focus) {
        ++ui.chatFocusRequests;
    }

    return Result::SUCCESS;
}

Result AgentActions::handle(const MailSetAgentVoiceVisible& msg) {
    setVoiceVisible(msg.value);
    return Result::SUCCESS;
}

Result AgentActions::handle(const MailStartAgentVoice&) {
    if (state.agent.ui.voiceVisible && !startVoice()) {
        hideVoice();
    }
    return Result::SUCCESS;
}

Result AgentActions::handle(const MailSetAgentVoiceMuted& msg) {
    if (state.agent.ui.voiceVisible) {
        runtime.setMuted(msg.value);
        state.agent.voice.muted = msg.value;
    }
    return Result::SUCCESS;
}

Result AgentActions::handle(const MailSubmitAgentChat& msg) {
    auto& agent = state.agent;
    const bool steerable = !agent.ui.voiceVisible;
    if (!agent.settings.enabled || !(runtime.submit(msg.text) || (steerable && runtime.session().steer(msg.text)))) {
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

Result AgentActions::handle(const MailCancelAgentChat&) {
    runtime.session().cancel();
    return Result::SUCCESS;
}

Result AgentActions::handle(const MailClearAgentChat&) {
    runtime.session().clear();
    return Result::SUCCESS;
}

Result AgentActions::handle(const MailWithdrawAgentChat& msg) {
    runtime.session().withdraw(msg.key);
    return Result::SUCCESS;
}

Result AgentActions::handle(const MailResumeAgentChat&) {
    if (state.agent.settings.enabled && !state.agent.ui.voiceVisible) {
        runtime.submit({});
    }
    return Result::SUCCESS;
}

Result AgentActions::handle(const MailSelectAgentModel& msg) {
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

Result AgentActions::handle(const MailSetAgentReasoningEffort& msg) {
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

Result AgentActions::handle(const MailOpenAgentModelPicker&) {
    openIntelligencePage(IntelligencePage::AddModel);
    state.agent.catalog.search.clear();
    ensureModels();
    return Result::SUCCESS;
}

Result AgentActions::handle(const MailSetAgentModelSearch& msg) {
    state.agent.catalog.search = msg.value;
    return Result::SUCCESS;
}

Result AgentActions::handle(const MailAddAgentModel& msg) {
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

Result AgentActions::handle(const MailRemoveAgentModel& msg) {
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

Result AgentActions::handle(const MailOpenAgentProviderEditor& msg) {
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
    resetSignIn();
    openIntelligencePage(IntelligencePage::Provider);
    return Result::SUCCESS;
}

Result AgentActions::handle(const MailSaveAgentProvider& msg) {
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
    const auto planBilled = AgentProviders::PlanBilled(provider);
    const auto apiKey = planBilled ? pendingSignIn() : Text::TrimBlank(msg.apiKey);

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
    task.signedIn = planBilled && !apiKey.empty();

    if (planBilled) {
        const bool sameAccount = AgentProviders::PlanBilled(previous);
        provider.account = task.signedIn ? agent.providers.signIn.email : sameAccount ? previous.account : "";
        for (auto& model : agent.settings.models) {
            if (!sameAccount && model.provider == previousName) {
                model.priced = false;
                model.inputCost = 0.0;
                model.outputCost = 0.0;
            }
        }
    }

    const bool planChanged = AgentProviders::PlanBilled(previous) != planBilled;
    if (AgentProviders::External(provider) && !planChanged) {
        provider.apiKeyId = previous.apiKeyId;
    } else if (AgentProviders::External(provider)) {
        if (!previous.apiKeyId.empty()) {
            task.future = std::async(std::launch::async, AgentProviders::Erase, previous.apiKeyId,
                                     AgentProviders::PlanBilled(previous));
        }
    } else if (!apiKey.empty()) {
        if (previous.apiKeyId.empty()) {
            provider.apiKeyId = Random::HexToken();
        } else {
            provider.apiKeyId = previous.apiKeyId;
        }
        task.future = std::async(std::launch::async,
                                 AgentProviders::Store,
                                 provider.apiKeyId,
                                 provider.host,
                                 apiKey,
                                 AgentProviders::PlanBilled(previous));
    } else if (AgentProviders::Transferable(previous, provider)) {
        provider.apiKeyId = previous.apiKeyId;
    } else if (!previous.apiKeyId.empty()) {
        task.future = std::async(std::launch::async, AgentProviders::Erase, previous.apiKeyId,
                                 AgentProviders::PlanBilled(previous));
    }

    if (!task.future.valid()) {
        return saveProvider(std::move(provider), previousName, false);
    }

    task.running = true;
    task.provider = std::move(provider);
    return Result::SUCCESS;
}

Result AgentActions::handle(const MailRemoveAgentProvider& msg) {
    auto& task = state.agent.providers.keyTask;
    if (task.running) {
        return Result::SUCCESS;
    }

    const auto provider = AgentProviders(state.agent).find(msg.name);
    if (provider.name.empty()) {
        return Result::SUCCESS;
    }
    if (state.agent.providers.signIn.replaceName == msg.name) {
        resetSignIn();
    }
    if (provider.apiKeyId.empty()) {
        return removeProvider(msg.name);
    }

    task.running = true;
    task.kind = DefaultCompositorState::AgentState::ProviderKeyTask::Kind::Remove;
    task.provider = provider;
    task.previousName = msg.name;
    task.future = std::async(std::launch::async, AgentProviders::Erase, provider.apiKeyId,
                             AgentProviders::PlanBilled(provider));
    return Result::SUCCESS;
}

Result AgentActions::handle(const MailTestAgentProvider& msg) {
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

    const auto planBilled = AgentProviders::PlanBilled(provider);
    const auto apiKey = planBilled ? pendingSignIn() : Text::TrimBlank(msg.apiKey);
    std::optional<Settings::Intelligence::Provider> stored;
    if (apiKey.empty()) {
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
                             apiKey,
                             std::move(stored),
                             Cancelled(test.cancel));
    return Result::SUCCESS;
}

Result AgentActions::handle(const MailSignInAgentProvider& msg) {
    auto& agent = state.agent;
    auto& signIn = agent.providers.signIn;
    if (signIn.running) {
        return Result::SUCCESS;
    }
    if (!AgentProviders::SignInAvailable()) {
        callbacks.notify(Sakura::ToastType::Error, 5000, "Sign in with ChatGPT needs the system keychain.");
        return Result::SUCCESS;
    }

    signIn.running = true;
    signIn.session = agent.providers.editSession;
    signIn.revision = msg.revision;
    signIn.draft = msg.value;
    signIn.replaceName = msg.replaceName;
    signIn.record.clear();
    signIn.email.clear();
    signIn.message = "Finish signing in with your browser.";
    signIn.cancel = std::make_shared<std::atomic<bool>>(false);
    signIn.link = std::make_shared<DefaultCompositorState::AgentState::SignInLink>();
    signIn.future = std::async(std::launch::async,
                               AgentProviders::SignIn,
                               signIn.link,
                               Cancelled(signIn.cancel));
    resetProviderTest();
    return Result::SUCCESS;
}

Result AgentActions::handle(const MailCancelAgentProviderSignIn&) {
    auto& signIn = state.agent.providers.signIn;
    if (signIn.running && signIn.cancel) {
        signIn.cancel->store(true);
    }
    return Result::SUCCESS;
}

Result AgentActions::handle(const MailSetAgentVoiceBackend& msg) {
    state.agent.settings.voice.backend = msg.value;
    return persist();
}

Result AgentActions::handle(const MailSetAgentVoiceModel& msg) {
    state.agent.settings.voice.model = msg.value;
    return persist();
}

Result AgentActions::handle(const MailSetAgentVoiceName& msg) {
    state.agent.settings.voice.voice = msg.value;
    return persist();
}

Result AgentActions::handle(const MailSetAgentMcp& msg) {
    state.agent.settings.mcp = msg.value;
    return persist();
}

void AgentActions::refresh() {
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

void AgentActions::reconcile() {
    checkUnread();
    checkContextLengths();
    checkSettingsSection();
    collectProviderTest();
    collectSignIn();
    collectProviderKeyTask();
    collectModels();
}

void AgentActions::shutdown() {
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

    auto& signIn = state.agent.providers.signIn;
    if (signIn.running) {
        signIn.future.wait();
        collectSignIn();
    }

    auto& keyTask = state.agent.providers.keyTask;
    if (keyTask.running) {
        keyTask.future.wait();
        collectProviderKeyTask();
    }
}

void AgentActions::setVoiceVisible(bool visible) {
    if (!visible) {
        hideVoice();
        return;
    }

    if (state.agent.settings.enabled && !state.agent.ui.voiceVisible) {
        startVoice();
    }
}

void AgentActions::cancel() {
    auto& agent = state.agent;
    agent.catalog.refreshPending = false;
    if (agent.catalog.cancel) {
        agent.catalog.cancel->store(true);
    }
    if (agent.providerTest.cancel) {
        agent.providerTest.cancel->store(true);
    }
    if (agent.providers.signIn.cancel) {
        agent.providers.signIn.cancel->store(true);
    }
}

Agent::Cancellation AgentActions::Cancelled(const std::shared_ptr<std::atomic<bool>>& flag) {
    return [flag]() {
        return flag->load();
    };
}

void AgentActions::hideVoice() {
    state.agent.ui.voiceVisible = false;
    runtime.hangup();
}

bool AgentActions::startVoice() {
    auto& agent = state.agent;
    const auto availability = AgentProviders(agent).availability();
    if (availability.state != AgentProviders::VoiceAvailability::State::Ready) {
        callbacks.notify(Sakura::ToastType::Info, 5000, availability.message);
        return false;
    }

    agent.ui.voiceVisible = true;
    runtime.listen(availability.voice);

    return true;
}

void AgentActions::ensureModels() {
    const auto& catalog = state.agent.catalog;
    if (catalog.models.empty() && !catalog.loading) {
        refresh();
    }
}

void AgentActions::resetProviderTest() {
    auto& test = state.agent.providerTest;
    if (test.running) {
        return;
    }

    test.completed = false;
    test.success = false;
    test.message.clear();
}

void AgentActions::resetSignIn() {
    auto& signIn = state.agent.providers.signIn;
    if (signIn.running && signIn.cancel) {
        signIn.cancel->store(true);
    }
    signIn.record.clear();
    signIn.email.clear();
    signIn.message.clear();
    signIn.draft.reset();
    signIn.replaceName.reset();
}

std::string AgentActions::pendingSignIn() const {
    const auto& providers = state.agent.providers;
    return providers.signIn.session == providers.editSession ? providers.signIn.record : std::string{};
}

bool AgentActions::selectFallbackModel() {
    auto& settings = state.agent.settings;
    if (settings.models.empty()) {
        return false;
    }

    settings.provider = settings.models.front().provider;
    settings.model = settings.models.front().id;
    return true;
}

void AgentActions::returnToSettings() {
    state.settings.intelligencePage = IntelligencePage::Overview;
}

void AgentActions::openIntelligencePage(IntelligencePage page) {
    WorkbenchActions::OpenSettings(state, SettingsSection::Intelligence);
    state.settings.intelligencePage = page;
}

void AgentActions::renameProvider(const std::string& previous, const std::string& next) {
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

Result AgentActions::saveProvider(Settings::Intelligence::Provider provider, const std::string& previousName,
                                  bool signedIn) {
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
    const bool planBilled = AgentProviders::PlanBilled(provider);
    if (planBilled) {
        agent.providers.signIn.record.clear();
        agent.providers.signIn.email.clear();
    }
    callbacks.notify(Sakura::ToastType::Success,
                     3000,
                     jst::fmt::format("Provider '{}' {}.", provider.name, verb));
    if (signedIn && !state.modal.content) {
        state.modal = {.content = ModalContent::PlanNotice};
    }

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

Result AgentActions::removeProvider(const std::string& name) {
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

void AgentActions::collectProviderKeyTask() {
    auto& task = state.agent.providers.keyTask;
    if (!task.running || !FutureReady(task.future)) {
        return;
    }

    const auto result = task.future.get();
    task.running = false;
    if (!result.success) {
        using Kind = DefaultCompositorState::AgentState::ProviderKeyTask::Kind;
        const bool removingExternal = task.kind == Kind::Remove && AgentProviders::External(task.provider);
        if (!removingExternal) {
            callbacks.notify(Sakura::ToastType::Error, 5000, result.message);
            return;
        }
        JST_WARN("[AGENT] {}", result.message);
        callbacks.notify(Sakura::ToastType::Warning, 5000, result.message);
    }

    Result commit;
    if (task.kind == DefaultCompositorState::AgentState::ProviderKeyTask::Kind::Save) {
        commit = saveProvider(std::move(task.provider), task.previousName, task.signedIn);
    } else {
        commit = removeProvider(task.previousName);
    }
    if (commit != Result::SUCCESS) {
        JST_WARN("[AGENT] Failed to apply provider changes.");
    }
    if (!result.warning.empty()) {
        callbacks.notify(Sakura::ToastType::Warning, 8000, result.warning);
    }
}

void AgentActions::checkUnread() {
    auto& agent = state.agent;
    const auto& conversation = agent.chat.conversation;
    const bool shown = agent.settings.enabled && agent.ui.visible;
    if (streaming && !conversation.streaming && !shown) {
        agent.ui.unread = true;
    }
    if (shown || conversation.messages.empty()) {
        agent.ui.unread = false;
    }
    streaming = conversation.streaming;
}

void AgentActions::checkContextLengths() {
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

void AgentActions::checkSettingsSection() {
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

void AgentActions::collectSignIn() {
    auto& agent = state.agent;
    auto& signIn = agent.providers.signIn;
    if (signIn.running && signIn.link) {
        std::string url;
        {
            std::lock_guard lock(signIn.link->mutex);
            url = std::exchange(signIn.link->url, {});
        }
        if (!url.empty()) {
            callbacks.enqueueMail(MailOpenUrl{.url = std::move(url), .notifyResult = true});
        }
    }
    if (!signIn.running || !FutureReady(signIn.future)) {
        return;
    }

    auto result = signIn.future.get();
    signIn.running = false;
    if (signIn.session != agent.providers.editSession) {
        signIn.message.clear();
        return;
    }
    if (signIn.cancel && signIn.cancel->load()) {
        signIn.message = "Sign-in cancelled.";
        return;
    }

    signIn.message = result.message;
    if (!result.success) {
        return;
    }

    signIn.record = std::move(result.record);
    signIn.email = std::move(result.email);
    auto& test = agent.providerTest;
    if (!test.running) {
        test.completed = true;
        test.success = true;
        test.revision = signIn.revision;
        test.message = result.message;
    }

    if (!signIn.draft) {
        return;
    }
    auto draft = AgentProviders::Normalize(*std::exchange(signIn.draft, std::nullopt));
    const auto& target = agent.providers.editTarget;
    if (!target) {
        const auto base = draft.name;
        for (U64 suffix = 2; !AgentProviders(agent).find(draft.name).name.empty(); ++suffix) {
            draft.name = jst::fmt::format("{} {}", base, suffix);
        }
    }
    const auto previousName = signIn.replaceName.value_or(draft.name);
    const auto existing = AgentProviders(agent).find(draft.name);
    const bool taken = !existing.name.empty() && existing.name != previousName;
    const bool replacing = !target || AgentProviders::PlanBilled(*target);
    if (AgentProviders::Valid(draft) && !taken && replacing) {
        callbacks.enqueueMail(MailSaveAgentProvider{
            .value = draft,
            .replaceName = signIn.replaceName,
        });
    }
}

void AgentActions::collectProviderTest() {
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

void AgentActions::collectModels() {
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

bool AgentActions::mergeModelMetadata() {
    bool changed = false;
    for (auto& saved : state.agent.settings.models) {
        for (const auto& model : state.agent.catalog.models) {
            changed |= MergeModel(saved, model);
        }
    }

    return changed;
}

Result AgentActions::persist() {
    return PersistSettings([&](Settings& settings) {
        settings.intelligence = state.agent.settings;
    });
}

void AgentActions::RenameModels(std::vector<Settings::Intelligence::Model>& models,
                                const std::string& previous, const std::string& next) {
    for (auto& model : models) {
        if (model.provider == previous) {
            model.provider = next;
        }
    }
}

bool AgentActions::MergeModel(Settings::Intelligence::Model& saved, const Settings::Intelligence::Model& model) {
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

}  // namespace Jetstream
