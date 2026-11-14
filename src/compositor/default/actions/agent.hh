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
                              MailSetAgentVoiceMuted,
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
                              MailSetAgentVoiceName,
                              MailSetAgentMcp>;

    DefaultCompositorState& state;
    DefaultCompositorCallbacks& callbacks;
    AgentRuntime& runtime;

    AgentActions(DefaultCompositorState& state, DefaultCompositorCallbacks& callbacks, AgentRuntime& runtime);
    Result handle(const MailSetAgentEnabled& msg);
    Result handle(const MailSetAgentChatVisible& msg);
    Result handle(const MailSetAgentVoiceVisible& msg);
    Result handle(const MailStartAgentVoice&);
    Result handle(const MailSetAgentVoiceMuted& msg);
    Result handle(const MailSubmitAgentChat& msg);
    Result handle(const MailCancelAgentChat&);
    Result handle(const MailClearAgentChat&);
    Result handle(const MailSelectAgentModel& msg);
    Result handle(const MailSetAgentReasoningEffort& msg);
    Result handle(const MailOpenAgentModelPicker&);
    Result handle(const MailSetAgentModelSearch& msg);
    Result handle(const MailAddAgentModel& msg);
    Result handle(const MailRemoveAgentModel& msg);
    Result handle(const MailOpenAgentProviderEditor& msg);
    Result handle(const MailSaveAgentProvider& msg);
    Result handle(const MailRemoveAgentProvider& msg);
    Result handle(const MailTestAgentProvider& msg);
    Result handle(const MailSetAgentVoiceBackend& msg);
    Result handle(const MailSetAgentVoiceModel& msg);
    Result handle(const MailSetAgentVoiceName& msg);
    Result handle(const MailSetAgentMcp& msg);
    void refresh();
    void reconcile();
    void shutdown();

 private:
    void setVoiceVisible(bool visible);
    void cancel();
    static Agent::Cancellation Cancelled(const std::shared_ptr<std::atomic<bool>>& flag);
    void hideVoice();
    bool startVoice();
    void ensureModels();
    void resetProviderTest();
    bool selectFallbackModel();
    void returnToSettings();
    void openIntelligencePage(IntelligencePage page);
    void renameProvider(const std::string& previous, const std::string& next);
    Result saveProvider(Settings::Intelligence::Provider provider, const std::string& previousName);
    Result removeProvider(const std::string& name);
    void collectProviderKeyTask();
    void checkUnread();
    void checkContextLengths();
    void checkSettingsSection();
    void collectProviderTest();
    void collectModels();
    bool mergeModelMetadata();
    Result persist();

    template<class Value>
    static bool FutureReady(const std::future<Value>& future) {
        return future.valid() && future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready;
    }

    static void RenameModels(std::vector<Settings::Intelligence::Model>& models,
                             const std::string& previous, const std::string& next);
    static bool MergeModel(Settings::Intelligence::Model& saved, const Settings::Intelligence::Model& model);

    bool stopping = false;
    bool streaming = false;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_AGENT_HH
