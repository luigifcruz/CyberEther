#include "jetstream/platform.hh"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <memory>

#if defined(JST_OS_IOS) || defined(JST_OS_MAC)
#include "apple.hh"
#endif

#if defined(JST_OS_BROWSER)

#include "emscripten.h"

#include <functional>
#include <string>

static bool _filePicking = false;
static std::function<void(std::string)> _fileCallback;

extern "C" {
EMSCRIPTEN_KEEPALIVE
void jst_on_file_picked(const char* path, int status) {
    _filePicking = false;
    if (status == 1 && path && _fileCallback) {
        auto cb = std::move(_fileCallback);
        cb("/storage/" + std::string(path));
    } else {
        _fileCallback = nullptr;
    }
}
}

// clang-format off

EM_JS(void, _jst_start_pick_file, (const char* ext_json), {
    const channel = new BroadcastChannel('jst_file_picker');
    channel.postMessage({
        type: 'pickFile',
        extensions: UTF8ToString(ext_json),
    });
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

#endif

#if defined(JST_OS_WINDOWS)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <comutil.h>
#include <commdlg.h>
#include <stdio.h>
#include <shellapi.h>
#undef ERROR
#undef FATAL
#endif

namespace Jetstream::Platform {

//
// Open URL
//

#if defined(JST_OS_MAC) || defined(JST_OS_IOS)

// Defined on apple.mm.

#elif defined(JST_OS_BROWSER)

Result OpenUrl(const std::string& url) {
    emscripten_run_script(jst::fmt::format("window.open('{}', '_blank');", url).c_str());
    return Result::SUCCESS;
}

#elif defined(JST_OS_LINUX)

Result OpenUrl(const std::string& url) {
    const auto res = system(jst::fmt::format("xdg-open ""{}""", url).c_str());
    if (res != 0) {
        JST_ERROR("Failed to open URL: {}", url);
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

#elif defined(JST_OS_WINDOWS)
Result OpenUrl(const std::string& url) {
    INT_PTR res = (INT_PTR)ShellExecuteA(0, 0, url.c_str(), 0, 0 , SW_SHOW );
    if (res < 32) {
        JST_ERROR("Failed to open URL: {}", url);
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

#else

Result OpenUrl(const std::string& url) {
    JST_ERROR("Opening URL is not supported in this platform.");
    return Result::ERROR;
}

#endif

//
// App Path
//

#if defined(JST_OS_MAC) || defined(JST_OS_IOS)

// Defined on apple.mm.

#elif defined(JST_OS_BROWSER)

Result ConfigPath(std::string& path) {
    path = "/storage/cyberether";
    return Result::SUCCESS;
}

Result CachePath(std::string& path) {
    path = "/storage/cyberether/cache";
    return Result::SUCCESS;
}

#elif defined(JST_OS_LINUX)

namespace {

Result ResolveLinuxBasePath(const char* homeEnv, std::string& homePath) {
    const char* home = std::getenv(homeEnv);
    if (!home || std::string(home).empty()) {
        JST_ERROR("Failed to resolve app path because {} is not set.", homeEnv);
        return Result::ERROR;
    }

    homePath = home;
    return Result::SUCCESS;
}

}  // namespace

Result ConfigPath(std::string& path) {
    const char* xdg = std::getenv("XDG_CONFIG_HOME");
    if (xdg && *xdg) {
        const std::filesystem::path basePath(xdg);
        if (basePath.is_absolute()) {
            path = (basePath / "cyberether").string();
            return Result::SUCCESS;
        }

        JST_WARN("XDG_CONFIG_HOME must be an absolute path. Falling back to $HOME/.config.");
    }

    std::string homePath;
    JST_CHECK(ResolveLinuxBasePath("HOME", homePath));
    path = (std::filesystem::path(homePath) / ".config" / "cyberether").string();
    return Result::SUCCESS;
}

Result CachePath(std::string& path) {
    const char* xdg = std::getenv("XDG_CACHE_HOME");
    if (xdg && *xdg) {
        const std::filesystem::path basePath(xdg);
        if (basePath.is_absolute()) {
            path = (basePath / "cyberether").string();
            return Result::SUCCESS;
        }

        JST_WARN("XDG_CACHE_HOME must be an absolute path. Falling back to $HOME/.cache.");
    }

    std::string homePath;
    JST_CHECK(ResolveLinuxBasePath("HOME", homePath));
    path = (std::filesystem::path(homePath) / ".cache" / "cyberether").string();
    return Result::SUCCESS;
}

#elif defined(JST_OS_WINDOWS)

namespace {

std::string WindowsPathToUtf8(const std::filesystem::path& nativePath) {
    const auto utf8Path = nativePath.u8string();

    std::string utf8String;
    utf8String.reserve(utf8Path.size());
    for (const auto ch : utf8Path) {
        utf8String.push_back(static_cast<char>(ch));
    }

    return utf8String;
}

}  // namespace

Result ConfigPath(std::string& path) {
    const wchar_t* appData = _wgetenv(L"APPDATA");
    if (!(appData && *appData)) {
        JST_ERROR("Failed to resolve config path because APPDATA is not set.");
        return Result::ERROR;
    }

    path = WindowsPathToUtf8(std::filesystem::path(appData) / L"CyberEther");
    return Result::SUCCESS;
}

Result CachePath(std::string& path) {
    const wchar_t* appData = _wgetenv(L"APPDATA");
    const wchar_t* localAppData = _wgetenv(L"LOCALAPPDATA");

    if (localAppData && *localAppData) {
        path = WindowsPathToUtf8(std::filesystem::path(localAppData) / L"CyberEther" / L"Cache");
        return Result::SUCCESS;
    }

    if (appData && *appData) {
        path = WindowsPathToUtf8(std::filesystem::path(appData) / L"CyberEther" / L"Cache");
        return Result::SUCCESS;
    }

    JST_ERROR("Failed to resolve cache path because LOCALAPPDATA and APPDATA are not set.");
    return Result::ERROR;
}

#else

Result ConfigPath(std::string& path) {
    path.clear();
    JST_ERROR("Resolving configuration paths is not supported on this platform.");
    return Result::ERROR;
}

Result CachePath(std::string& path) {
    path.clear();
    JST_ERROR("Resolving cache paths is not supported on this platform.");
    return Result::ERROR;
}

#endif

//
// Pick File
//

// TODO: Implement iOS support.

#if defined(JST_OS_MAC) || defined(JST_OS_IOS)

// Defined on apple.mm.

#elif defined(JST_OS_BROWSER)

Result PickFile(std::string& path,
                const std::vector<std::string>& extensions,
                std::function<void(std::string)> callback) {
    if (_filePicking) {
        return Result::ERROR;
    }

    std::string ext_json = "[";
    for (size_t i = 0; i < extensions.size(); ++i) {
        if (i > 0) ext_json += ",";
        ext_json += "\"" + extensions[i] + "\"";
    }
    ext_json += "]";

    _filePicking = true;
    _fileCallback = std::move(callback);
    _jst_start_pick_file(ext_json.c_str());
    return Result::ERROR;
}

#elif defined(JST_OS_LINUX)

Result PickFile(std::string& path,
                const std::vector<std::string>& extensions,
                std::function<void(std::string)> callback) {
    std::array<char, 1024> buffer;
    std::string command = "zenity --file-selection ";

    if (!extensions.empty()) {
        command += "--file-filter='Selected files | ";
        for (size_t i = 0; i < extensions.size(); ++i) {
            if (i > 0) command += " ";
            command += "*." + extensions[i];
        }
        command += "' ";
    }
    command += "2>/dev/null";

    auto pipeDeleter = [](FILE* file) { if (file) pclose(file); };
    std::unique_ptr<FILE, decltype(pipeDeleter)> pipe(popen(command.c_str(), "r"), pipeDeleter);

    if (!pipe) {
        JST_ERROR("Failed to open file selection dialog.");
        return Result::ERROR;
    }

    if (pipe.get() == nullptr) {
        JST_ERROR("No file selected or operation cancelled.");
        return Result::ERROR;
    }

    const auto res = fgets(buffer.data(), buffer.size(), pipe.get());
    if (res == nullptr) {
        JST_ERROR("No file selected or operation cancelled.");
        return Result::ERROR;
    }

    path = buffer.data();

    if (path.empty()) {
        JST_ERROR("No file selected or operation cancelled.");
        return Result::ERROR;
    }

    if (path.back() == '\n') {
        path.pop_back();
    }

    if (callback) {
        callback(path);
    }

    return Result::SUCCESS;
}

#elif defined(JST_OS_WINDOWS)

Result PickFile(std::string& path,
                const std::vector<std::string>& extensions,
                std::function<void(std::string)> callback) {
    // File path buffer
    char buf[256] = {'\0'};
    memcpy_s(buf, 256, path.c_str(), path.length());

    // Create OpenFilenameA Struct
    OPENFILENAMEA ofn;
    ZeroMemory(&ofn, sizeof(OPENFILENAME));

    // Fill struct
    ofn.lStructSize = sizeof(OPENFILENAME);
    ofn.hwndOwner = NULL;
    ofn.lpstrFile = buf;
    ofn.nMaxFile = 256;

    std::string filter = "All Files\0*.*\0";
    if (!extensions.empty()) {
        filter += "Selected Files (";
        for (size_t i = 0; i < extensions.size(); ++i) {
            if (i > 0) filter += ", ";
            filter += "." + extensions[i];
        }
        filter += ")\0";
        for (size_t i = 0; i < extensions.size(); ++i) {
            if (i > 0) filter += ";";
            filter += "*." + extensions[i];
        }
        filter += "\0";
        ofn.nFilterIndex = 2;
    }
    ofn.nFilterIndex = 1;
    ofn.lpstrFilter = filter.c_str();

    bool ret = GetOpenFileNameA(&ofn);
    if (!ret) {
        JST_ERROR("No file selected or operation cancelled.");
        return Result::ERROR;
    }

    path = std::string(ofn.lpstrFile);

    if (path.empty()) {
        JST_ERROR("No file selected or operation cancelled.");
        return Result::ERROR;
    }

    if (path.back() == '\n') {
        path.pop_back();
    }

    if (callback) {
        callback(path);
    }

    return Result::SUCCESS;
}

#else

Result PickFile(std::string& path,
                const std::vector<std::string>& extensions,
                std::function<void(std::string)>) {
    JST_ERROR("Picking files is not supported in this platform.");
    return Result::ERROR;
}

#endif

//
// Pick Folder
//

// TODO: Implement iOS support.

#if defined(JST_OS_MAC) || defined(JST_OS_IOS)

// Defined on apple.mm.

#elif defined(JST_OS_BROWSER)

Result PickFolder(std::string& path,
                  std::function<void(std::string)> callback) {
    if (_filePicking) {
        return Result::ERROR;
    }

    _filePicking = true;
    _fileCallback = std::move(callback);
    _jst_start_pick_folder();
    return Result::ERROR;
}

#elif defined(JST_OS_LINUX)

Result PickFolder(std::string& path,
                  std::function<void(std::string)> callback) {
    std::array<char, 1024> buffer;
    std::string command = "zenity --file-selection --directory 2>/dev/null";

    auto pipeDeleter = [](FILE* file) { if (file) pclose(file); };
    std::unique_ptr<FILE, decltype(pipeDeleter)> pipe(popen(command.c_str(), "r"), pipeDeleter);

    if (!pipe) {
        JST_ERROR("Failed to open folder selection dialog.");
        return Result::ERROR;
    }

    if (pipe.get() == nullptr) {
        JST_ERROR("No folder selected or operation cancelled.");
        return Result::ERROR;
    }

    const auto res = fgets(buffer.data(), buffer.size(), pipe.get());

    if (res == nullptr) {
        JST_ERROR("No folder selected or operation cancelled.");
        return Result::ERROR;
    }

    path = buffer.data();

    if (callback) {
        callback(path);
    }

    return Result::SUCCESS;
}

// TODO: Implement folder picker for Windows.
//#elif defined(JST_OS_WINDOWS)

#else

Result PickFolder(std::string& path,
                  std::function<void(std::string)>) {
    JST_ERROR("Picking files is not supported in this platform.");
    return Result::ERROR;
}

#endif

//
// Save File
//

// TODO: Implement iOS support.

#if defined(JST_OS_MAC) || defined(JST_OS_IOS)

// Defined on apple.mm.

#elif defined(JST_OS_BROWSER)

Result SaveFile(std::string& path,
                std::function<void(std::string)> callback) {
    if (_filePicking) {
        return Result::ERROR;
    }

    _filePicking = true;
    _fileCallback = std::move(callback);
    _jst_start_save_file();
    return Result::ERROR;
}

#elif defined(JST_OS_LINUX)

Result SaveFile(std::string& path,
                std::function<void(std::string)> callback) {
    std::array<char, 1024> buffer;
    std::string command = "zenity --file-selection --save --confirm-overwrite --file-filter='YAML files | *.yml *.yaml' 2>/dev/null";

    auto pipeDeleter = [](FILE* file) { if (file) pclose(file); };
    std::unique_ptr<FILE, decltype(pipeDeleter)> pipe(popen(command.c_str(), "r"), pipeDeleter);

    if (!pipe) {
        JST_ERROR("Failed to open save file dialog.");
        return Result::ERROR;
    }

    if (pipe.get() == nullptr) {
        JST_ERROR("No file selected or operation cancelled.");
        return Result::ERROR;
    }

    const auto res = fgets(buffer.data(), buffer.size(), pipe.get());
    if (res == nullptr) {
        JST_ERROR("No file selected or operation cancelled.");
        return Result::ERROR;
    }

    path = buffer.data();

    if (path.empty()) {
        JST_ERROR("No file selected or operation cancelled.");
        return Result::ERROR;
    }

    if (path.back() == '\n') {
        path.pop_back();
    }

    if (callback) {
        callback(path);
    }

    return Result::SUCCESS;
}

#elif defined(JST_OS_WINDOWS)

Result SaveFile(std::string& path,
                std::function<void(std::string)> callback) {
    /* File path buffer */
    char buf[256] = {'\0'};
    memcpy_s(buf,256,path.c_str(),path.length());

    /* Create OpenFilenameA Struct */
    OPENFILENAMEA ofn;
    ZeroMemory(&ofn, sizeof(OPENFILENAME));

    /* Fill struct */
    ofn.lStructSize = sizeof(OPENFILENAME);
    ofn.hwndOwner = NULL;
    ofn.lpstrFile = buf;
    ofn.nMaxFile = 256;
    ofn.lpstrFilter = "All Files\0*.*\0CyberEther Flowgraphs (.yml, .yaml)\0*.yml;*.yaml\0";
    ofn.nFilterIndex = 2;

    bool ret = GetSaveFileNameA(&ofn);
    if (!ret) {
        JST_ERROR("No file selected or operation cancelled.");
        return Result::ERROR;
    }

    path = std::string(ofn.lpstrFile);

    if (path.empty()) {
        JST_ERROR("No file selected or operation cancelled.");
        return Result::ERROR;
    }

    if (path.back() == '\n') {
        path.pop_back();
    }

    if (callback) {
        callback(path);
    }

    return Result::SUCCESS;
}

#else

Result SaveFile(std::string& path,
                std::function<void(std::string)>) {
    JST_ERROR("Saving files is not supported in this platform.");
    return Result::ERROR;
}

#endif

//
// File Pending
//

#if defined(JST_OS_BROWSER)

bool IsFilePending() {
    return _filePicking;
}

#else

bool IsFilePending() { return false; }

#endif

}  // namespace Jetstream::Platform
