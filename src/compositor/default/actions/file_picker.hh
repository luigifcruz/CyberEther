#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_FILE_PICKER_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_FILE_PICKER_HH

#include "../model/callbacks.hh"
#include "../model/messages.hh"
#include "../model/state.hh"

#include "jetstream/logger.hh"
#include "jetstream/platform.hh"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace Jetstream {

struct FilePickerActions {
    using Filter = std::tuple<MailBrowseConfigPath,
                              MailFilePickerNavigate,
                              MailFilePickerSelect,
                              MailFilePickerSetFilename,
                              MailFilePickerConfirm,
                              MailFilePickerCancel>;

    DefaultCompositorState& state;
    DefaultCompositorCallbacks& callbacks;

    FilePickerActions(DefaultCompositorState& state,
                      DefaultCompositorCallbacks& callbacks);
    Result request(FilePickerRequest request);
    void reconcileAvailability();
    void cancel();
    Result handle(const MailBrowseConfigPath& msg);
    Result handle(const MailFilePickerNavigate& msg);
    Result handle(const MailFilePickerSelect& msg);
    Result handle(const MailFilePickerSetFilename& msg);
    Result handle(const MailFilePickerCancel& msg);
    Result handle(const MailFilePickerConfirm& msg);

 private:
    Sakura::Callback<void(std::string)> completion;

    bool shouldUseServerPicker() const;
    static bool pathIsWithin(const std::filesystem::path& root, const std::filesystem::path& path);
    static std::string lowercase(std::string value);
    void refreshEntries();
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_FILE_PICKER_HH
