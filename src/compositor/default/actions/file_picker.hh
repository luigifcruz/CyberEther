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
#include <functional>
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
                      DefaultCompositorCallbacks& callbacks) :
        state(state),
        callbacks(callbacks) {}

    Result request(FilePickerRequest request) {
        if (!shouldUseServerPicker()) {
            std::string path = request.initialPath;
            if (request.mode == FilePickerMode::Save) {
                return Platform::SaveFile(path, std::move(request.callback));
            }
            return Platform::PickFile(path, request.extensions, std::move(request.callback));
        }

        if (state.filePicker.active) {
            JST_ERROR("[FILE_PICKER] A file picker request is already active.");
            return Result::ERROR;
        }

        std::error_code error;
        const auto root = std::filesystem::canonical(std::filesystem::current_path(error), error);
        if (error) {
            JST_ERROR("[FILE_PICKER] Failed to resolve the working directory: {}", error.message());
            return Result::ERROR;
        }

        auto directory = root;
        std::string filename;
        if (!request.initialPath.empty()) {
            auto initial = Platform::PathFromUtf8(request.initialPath);
            if (initial.is_relative()) initial = root / initial;
            const auto parent = std::filesystem::weakly_canonical(
                std::filesystem::is_directory(initial, error) ? initial : initial.parent_path(), error);
            if (!error && pathIsWithin(root, parent)) {
                directory = parent;
                if (request.mode == FilePickerMode::Save && !std::filesystem::is_directory(initial, error)) {
                    filename = Platform::PathToUtf8(initial.filename());
                }
            }
        }

        auto& picker = state.filePicker;
        picker.active = true;
        picker.overwritePending = false;
        ++picker.generation;
        picker.mode = request.mode;
        picker.root = Platform::PathToUtf8(root);
        picker.directory = Platform::PathToUtf8(directory);
        picker.selectedPath.clear();
        picker.filename = std::move(filename);
        picker.error.clear();
        picker.extensions = std::move(request.extensions);
        completion = std::move(request.callback);
        refreshEntries();
        return Result::INCOMPLETE;
    }

    void reconcileAvailability() {
        if (state.filePicker.active && !shouldUseServerPicker()) {
            cancel();
        }
    }

    void cancel() {
        const auto generation = state.filePicker.generation;
        state.filePicker = {};
        state.filePicker.generation = generation;
        completion = nullptr;
    }

    Result handle(const MailBrowseConfigPath& msg) {
        const auto callback = [onSelect = msg.onSelect](std::string selectedPath) {
            if (onSelect) {
                onSelect(std::move(selectedPath));
            }
        };

        const Result result = request({
            .mode = msg.save ? FilePickerMode::Save : FilePickerMode::Open,
            .initialPath = msg.path,
            .extensions = msg.extensions,
            .callback = callback,
        });
        if (result != Result::SUCCESS && result != Result::INCOMPLETE && !Platform::IsFilePending()) {
            callbacks.notifyResult(result, "");
        }

        return Result::SUCCESS;
    }

    Result handle(const MailFilePickerNavigate& msg) {
        auto& picker = state.filePicker;
        if (!picker.active || msg.generation != picker.generation) return Result::SUCCESS;

        std::error_code error;
        const auto root = std::filesystem::canonical(Platform::PathFromUtf8(picker.root), error);
        const auto target = std::filesystem::canonical(Platform::PathFromUtf8(msg.path), error);
        if (error || !pathIsWithin(root, target) || !std::filesystem::is_directory(target, error)) {
            picker.error = "That directory is outside the allowed server root.";
            return Result::SUCCESS;
        }
        picker.directory = Platform::PathToUtf8(target);
        picker.selectedPath.clear();
        picker.overwritePending = false;
        refreshEntries();
        return Result::SUCCESS;
    }

    Result handle(const MailFilePickerSelect& msg) {
        auto& picker = state.filePicker;
        if (!picker.active || msg.generation != picker.generation) return Result::SUCCESS;

        const auto entry = std::find_if(picker.entries.begin(), picker.entries.end(), [&](const auto& value) {
            return value.path == msg.path;
        });
        if (entry == picker.entries.end()) return Result::SUCCESS;
        picker.selectedPath = entry->path;
        if (picker.mode == FilePickerMode::Save && !entry->directory) picker.filename = entry->name;
        picker.overwritePending = false;
        picker.error.clear();
        return Result::SUCCESS;
    }

    Result handle(const MailFilePickerSetFilename& msg) {
        auto& picker = state.filePicker;
        if (!picker.active || msg.generation != picker.generation) return Result::SUCCESS;

        picker.filename = msg.value;
        picker.overwritePending = false;
        picker.error.clear();
        return Result::SUCCESS;
    }

    Result handle(const MailFilePickerCancel& msg) {
        if (state.filePicker.active && msg.generation == state.filePicker.generation) cancel();
        return Result::SUCCESS;
    }

    Result handle(const MailFilePickerConfirm& msg) {
        auto& picker = state.filePicker;
        if (!picker.active || msg.generation != picker.generation) return Result::SUCCESS;

        std::filesystem::path target;
        if (picker.mode == FilePickerMode::Open) {
            std::error_code error;
            const auto root = std::filesystem::canonical(Platform::PathFromUtf8(picker.root), error);
            target = std::filesystem::canonical(Platform::PathFromUtf8(picker.selectedPath), error);
            if (error || picker.selectedPath.empty() || !pathIsWithin(root, target) ||
                !std::filesystem::is_regular_file(target, error)) {
                picker.error = "Select a file to open.";
                return Result::SUCCESS;
            }
        } else {
            if (picker.filename.empty() || picker.filename == "." || picker.filename == ".." ||
                picker.filename.find('/') != std::string::npos || picker.filename.find('\\') != std::string::npos) {
                picker.error = "Enter a valid filename.";
                return Result::SUCCESS;
            }
            target = Platform::PathFromUtf8(picker.directory) / Platform::PathFromUtf8(picker.filename);
            if (target.extension().empty() && !picker.extensions.empty()) {
                std::string extension = picker.extensions.front();
                if (!extension.empty() && extension.front() != '.') extension.insert(extension.begin(), '.');
                target += Platform::PathFromUtf8(extension);
            }
            std::error_code error;
            const auto root = std::filesystem::canonical(Platform::PathFromUtf8(picker.root), error);
            const auto parent = std::filesystem::canonical(target.parent_path(), error);
            if (error || !pathIsWithin(root, parent)) {
                picker.error = "The save location is outside the allowed server root.";
                return Result::SUCCESS;
            }
            if (std::filesystem::exists(target, error)) {
                const auto resolved = std::filesystem::canonical(target, error);
                if (error || !pathIsWithin(root, resolved) || !std::filesystem::is_regular_file(resolved, error)) {
                    picker.error = "The save target is not a writable file.";
                    return Result::SUCCESS;
                }
                target = resolved;
                if (!picker.overwritePending) {
                    picker.overwritePending = true;
                    return Result::SUCCESS;
                }
            }
        }

        auto callback = std::move(completion);
        cancel();
        if (callback) callback(Platform::PathToUtf8(target));
        return Result::SUCCESS;
    }

 private:
    std::function<void(std::string)> completion;

    bool shouldUseServerPicker() const {
        return state.remote.started && state.remote.clientCount > 0;
    }

    static bool pathIsWithin(const std::filesystem::path& root, const std::filesystem::path& path) {
        const auto relative = path.lexically_relative(root);
        if (relative.empty()) return path == root;
        const auto first = *relative.begin();
        return first != ".." && !relative.is_absolute();
    }

    static std::string lowercase(std::string value) {
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return value;
    }

    void refreshEntries() {
        auto& picker = state.filePicker;
        picker.entries.clear();
        picker.error.clear();

        std::error_code error;
        const auto root = std::filesystem::canonical(Platform::PathFromUtf8(picker.root), error);
        const auto directory = std::filesystem::canonical(Platform::PathFromUtf8(picker.directory), error);
        if (error || !pathIsWithin(root, directory)) {
            picker.error = "Cannot access this directory.";
            return;
        }

        std::vector<std::string> extensions;
        extensions.reserve(picker.extensions.size());
        for (auto extension : picker.extensions) {
            if (!extension.empty() && extension.front() == '.') extension.erase(extension.begin());
            extensions.push_back(lowercase(std::move(extension)));
        }

        for (std::filesystem::directory_iterator it(directory, error), end; !error && it != end; it.increment(error)) {
            const auto& entry = *it;
            const std::string name = Platform::PathToUtf8(entry.path().filename());
            if (name.empty() || name.front() == '.') continue;

            const auto resolved = std::filesystem::canonical(entry.path(), error);
            if (error) {
                error.clear();
                continue;
            }
            if (!pathIsWithin(root, resolved)) continue;

            const bool isDirectory = std::filesystem::is_directory(resolved, error);
            const bool isFile = std::filesystem::is_regular_file(resolved, error);
            if (error || (!isDirectory && !isFile)) {
                error.clear();
                continue;
            }
            if (isFile && picker.mode == FilePickerMode::Open && !extensions.empty()) {
                std::string extension = lowercase(Platform::PathToUtf8(resolved.extension()));
                if (!extension.empty() && extension.front() == '.') extension.erase(extension.begin());
                if (std::find(extensions.begin(), extensions.end(), extension) == extensions.end()) continue;
            }
            picker.entries.push_back({name, Platform::PathToUtf8(resolved), isDirectory});
        }
        if (error) picker.error = "Some files could not be listed.";

        std::sort(picker.entries.begin(), picker.entries.end(), [](const auto& lhs, const auto& rhs) {
            if (lhs.directory != rhs.directory) return lhs.directory > rhs.directory;
            return lowercase(lhs.name) < lowercase(rhs.name);
        });
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_FILE_PICKER_HH
