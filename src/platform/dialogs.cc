#include "jetstream/platform.hh"

#include <algorithm>
#include <array>
#include <filesystem>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#if defined(JST_OS_BROWSER)
#include <emscripten.h>

static bool _filePicking = false;
static std::function<void(std::string)> _fileCallback;

extern "C" {
EMSCRIPTEN_KEEPALIVE
void jst_on_file_picked(const char* path, int status) {
    _filePicking = false;
    if (status == 1 && path && _fileCallback) {
        auto callback = std::move(_fileCallback);
        callback("/storage/" + std::string(path));
    } else {
        _fileCallback = nullptr;
    }
}
}

// clang-format off
EM_JS(void, _jst_start_pick_file, (const char* ext_json), {
    const channel = new BroadcastChannel('jst_file_picker');
    channel.postMessage({ type: 'pickFile', extensions: UTF8ToString(ext_json) });
    channel.close();
});

EM_JS(void, _jst_start_save_file, (), {
    const channel = new BroadcastChannel('jst_file_picker');
    channel.postMessage({ type: 'saveFile' });
    channel.close();
});

EM_JS(void, _jst_start_pick_folder, (), {
    const channel = new BroadcastChannel('jst_file_picker');
    channel.postMessage({ type: 'pickFolder' });
    channel.close();
});
// clang-format on
#elif defined(JST_OS_WINDOWS)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <commdlg.h>
#undef ERROR
#undef FATAL
#endif

namespace Jetstream::Platform {

#if defined(JST_OS_WINDOWS)
namespace {

constexpr std::size_t WindowsDialogPathCapacity = 32768;

void AppendWindowsFilter(std::vector<wchar_t>& filter, const std::wstring& value) {
    filter.insert(filter.end(), value.begin(), value.end());
    filter.push_back(L'\0');
}

std::vector<wchar_t> WindowsOpenFileFilter(const std::vector<std::string>& extensions) {
    std::vector<wchar_t> filter;
    AppendWindowsFilter(filter, L"All Files");
    AppendWindowsFilter(filter, L"*.*");

    if (!extensions.empty()) {
        std::wstring label = L"Selected Files (";
        std::wstring patterns;
        for (std::size_t i = 0; i < extensions.size(); ++i) {
            const auto extension = PathFromUtf8(extensions[i]).native();
            if (i > 0) {
                label += L", ";
                patterns += L";";
            }
            label += L"." + extension;
            patterns += L"*." + extension;
        }
        label += L")";
        AppendWindowsFilter(filter, label);
        AppendWindowsFilter(filter, patterns);
    }

    filter.push_back(L'\0');
    return filter;
}

bool InitializeWindowsDialogPath(const std::string& path,
                                 std::array<wchar_t, WindowsDialogPathCapacity>& buffer) {
    if (path.empty()) {
        return true;
    }

    const auto nativePath = PathFromUtf8(path).native();
    if (nativePath.size() >= buffer.size()) {
        return false;
    }

    std::copy(nativePath.begin(), nativePath.end(), buffer.begin());
    return true;
}

}  // namespace
#endif

#if defined(JST_OS_MAC) || defined(JST_OS_IOS)

// Defined on apple.mm.

#elif defined(JST_OS_BROWSER)

Result PickFile(std::string& path,
                const std::vector<std::string>& extensions,
                std::function<void(std::string)> callback) {
    if (_filePicking) {
        return Result::ERROR;
    }

    std::string extensionJson = "[";
    for (std::size_t i = 0; i < extensions.size(); ++i) {
        if (i > 0) extensionJson += ",";
        extensionJson += "\"" + extensions[i] + "\"";
    }
    extensionJson += "]";

    _filePicking = true;
    _fileCallback = std::move(callback);
    _jst_start_pick_file(extensionJson.c_str());
    return Result::ERROR;
}

#elif defined(JST_OS_LINUX)

Result PickFile(std::string& path,
                const std::vector<std::string>& extensions,
                std::function<void(std::string)> callback) {
    std::vector<std::string> arguments = {"--file-selection"};
    if (!extensions.empty()) {
        std::string filter = "Selected files |";
        for (const auto& extension : extensions) {
            filter += " *." + extension;
        }
        arguments.push_back("--file-filter=" + filter);
    }

    if (RunProcess("zenity", arguments, path) != Result::SUCCESS || path.empty()) {
        JST_ERROR("No file selected or operation cancelled.");
        return Result::ERROR;
    }
    if (path.back() == '\n') path.pop_back();
    if (callback) callback(path);
    return Result::SUCCESS;
}

#elif defined(JST_OS_WINDOWS)

Result PickFile(std::string& path,
                const std::vector<std::string>& extensions,
                std::function<void(std::string)> callback) {
    std::array<wchar_t, WindowsDialogPathCapacity> buffer = {};
    if (!InitializeWindowsDialogPath(path, buffer)) {
        JST_ERROR("Initial file path is too long.");
        return Result::ERROR;
    }

    auto filter = WindowsOpenFileFilter(extensions);
    OPENFILENAMEW dialog = {};
    dialog.lStructSize = sizeof(dialog);
    dialog.lpstrFile = buffer.data();
    dialog.nMaxFile = static_cast<DWORD>(buffer.size());
    dialog.lpstrFilter = filter.data();
    dialog.nFilterIndex = extensions.empty() ? 1 : 2;
    if (!GetOpenFileNameW(&dialog)) {
        JST_ERROR("No file selected or operation cancelled.");
        return Result::ERROR;
    }

    path = PathToUtf8(std::filesystem::path(buffer.data()));
    if (path.empty()) {
        JST_ERROR("No file selected or operation cancelled.");
        return Result::ERROR;
    }
    if (path.back() == '\n') path.pop_back();
    if (callback) callback(path);
    return Result::SUCCESS;
}

#else

Result PickFile(std::string&,
                const std::vector<std::string>&,
                std::function<void(std::string)>) {
    JST_ERROR("Picking files is not supported in this platform.");
    return Result::ERROR;
}

#endif

#if defined(JST_OS_MAC) || defined(JST_OS_IOS)

// Defined on apple.mm.

#elif defined(JST_OS_BROWSER)

Result PickFolder(std::string&, std::function<void(std::string)> callback) {
    if (_filePicking) {
        return Result::ERROR;
    }
    _filePicking = true;
    _fileCallback = std::move(callback);
    _jst_start_pick_folder();
    return Result::ERROR;
}

#elif defined(JST_OS_LINUX)

Result PickFolder(std::string& path, std::function<void(std::string)> callback) {
    if (RunProcess("zenity", {"--file-selection", "--directory"}, path) != Result::SUCCESS ||
        path.empty()) {
        JST_ERROR("No folder selected or operation cancelled.");
        return Result::ERROR;
    }
    if (path.back() == '\n') path.pop_back();
    if (callback) callback(path);
    return Result::SUCCESS;
}

#else

Result PickFolder(std::string&, std::function<void(std::string)>) {
    JST_ERROR("Picking files is not supported in this platform.");
    return Result::ERROR;
}

#endif

#if defined(JST_OS_MAC) || defined(JST_OS_IOS)

// Defined on apple.mm.

#elif defined(JST_OS_BROWSER)

Result SaveFile(std::string&, std::function<void(std::string)> callback) {
    if (_filePicking) {
        return Result::ERROR;
    }
    _filePicking = true;
    _fileCallback = std::move(callback);
    _jst_start_save_file();
    return Result::ERROR;
}

#elif defined(JST_OS_LINUX)

Result SaveFile(std::string& path, std::function<void(std::string)> callback) {
    const std::vector<std::string> arguments = {
        "--file-selection",
        "--save",
        "--confirm-overwrite",
        "--file-filter=YAML files | *.yml *.yaml",
    };
    if (RunProcess("zenity", arguments, path) != Result::SUCCESS || path.empty()) {
        JST_ERROR("No file selected or operation cancelled.");
        return Result::ERROR;
    }
    if (path.back() == '\n') path.pop_back();
    if (callback) callback(path);
    return Result::SUCCESS;
}

#elif defined(JST_OS_WINDOWS)

Result SaveFile(std::string& path, std::function<void(std::string)> callback) {
    std::array<wchar_t, WindowsDialogPathCapacity> buffer = {};
    if (!InitializeWindowsDialogPath(path, buffer)) {
        JST_ERROR("Initial file path is too long.");
        return Result::ERROR;
    }

    static constexpr wchar_t filter[] =
        L"All Files\0*.*\0CyberEther Flowgraphs (.yml, .yaml)\0*.yml;*.yaml\0\0";
    OPENFILENAMEW dialog = {};
    dialog.lStructSize = sizeof(dialog);
    dialog.lpstrFile = buffer.data();
    dialog.nMaxFile = static_cast<DWORD>(buffer.size());
    dialog.lpstrFilter = filter;
    dialog.nFilterIndex = 2;
    if (!GetSaveFileNameW(&dialog)) {
        JST_ERROR("No file selected or operation cancelled.");
        return Result::ERROR;
    }

    path = PathToUtf8(std::filesystem::path(buffer.data()));
    if (path.empty()) {
        JST_ERROR("No file selected or operation cancelled.");
        return Result::ERROR;
    }
    if (path.back() == '\n') path.pop_back();
    if (callback) callback(path);
    return Result::SUCCESS;
}

#else

Result SaveFile(std::string&, std::function<void(std::string)>) {
    JST_ERROR("Saving files is not supported in this platform.");
    return Result::ERROR;
}

#endif

#if defined(JST_OS_BROWSER)
bool IsFilePending() {
    return _filePicking;
}
#else
bool IsFilePending() {
    return false;
}
#endif

}  // namespace Jetstream::Platform
