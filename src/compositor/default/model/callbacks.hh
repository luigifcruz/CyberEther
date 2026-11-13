#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_MODEL_CALLBACKS_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_MODEL_CALLBACKS_HH

#include "file_picker.hh"
#include "history.hh"
#include "messages.hh"

#include "jetstream/agent/tools.hh"
#include "jetstream/render/sakura/base.hh"
#include "jetstream/types.hh"

#include <optional>
#include <string>

namespace Jetstream {

struct DefaultCompositorCallbacks {
    Sakura::Callback<void(Mail&&)> enqueueMail;
    Sakura::Callback<void(Sakura::Callback<Result()>, bool)> enqueueCommand;
    Sakura::Callback<void(Sakura::Callback<Result()>, bool, Sakura::Callback<void(Result)>)> enqueueCommandWithCompletion;
    Sakura::Callback<void(FlowgraphEdit)> enqueueFlowgraphEdit;
    Sakura::Callback<void(Sakura::ToastType, I32, const std::string&)> notify;
    Sakura::Callback<void(Result, const std::string&)> notifyResult;
    Sakura::Callback<void(const std::string&)> setClipboardText;
    Sakura::Callback<std::string()> getClipboardText;
    Sakura::Callback<Result(FilePickerRequest)> requestFile;
    Sakura::Callback<void(const std::string&, Agent::Tools::Layout)> layout;
    Sakura::Callback<void()> checkForUpdates;
    Sakura::Callback<void()> downloadUpdate;
    Sakura::Callback<bool()> applyUpdate;
    Sakura::Callback<void(const std::string&)> submitFeedback;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_MODEL_CALLBACKS_HH
