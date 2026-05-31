#include "jetstream/platform.hh"

#include <array>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <utility>

#if !defined(JST_OS_WINDOWS) && !defined(JST_OS_BROWSER)
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

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

struct FileLock::Impl {
    std::string path;
    bool locked = false;

#if defined(JST_OS_WINDOWS)
    HANDLE handle = INVALID_HANDLE_VALUE;
#elif !defined(JST_OS_BROWSER)
    int fd = -1;
#endif

    static Result ensureParentDirectory(const std::string& path);
};

Result FileLock::Impl::ensureParentDirectory(const std::string& path) {
    const auto parent = std::filesystem::u8path(path).parent_path();
    if (parent.empty()) {
        return Result::SUCCESS;
    }

    std::error_code ec;
    std::filesystem::create_directories(parent, ec);
    if (ec) {
        JST_ERROR("Failed to create file lock directory '{}'.", parent.string());
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

FileLock::FileLock() {
    impl = std::make_unique<Impl>();
}

FileLock::~FileLock() {
    release();
}

FileLock::FileLock(FileLock&&) noexcept = default;

FileLock& FileLock::operator=(FileLock&& other) noexcept {
    if (this != &other) {
        release();
        impl = std::move(other.impl);
    }

    return *this;
}

Result FileLock::acquire(const std::string& path, bool wait) {
    if (!impl) {
        impl = std::make_unique<Impl>();
    }

    auto& state = *impl;

    if (state.locked) {
        JST_ERROR("Cannot acquire file lock '{}' because this lock already owns '{}'.", path, state.path);
        return Result::ERROR;
    }

    JST_CHECK(Impl::ensureParentDirectory(path));

#if defined(JST_OS_WINDOWS)
    const auto lockPath = std::filesystem::u8path(path);
    HANDLE handle = CreateFileW(lockPath.c_str(),
                                GENERIC_READ | GENERIC_WRITE,
                                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                nullptr,
                                OPEN_ALWAYS,
                                FILE_ATTRIBUTE_NORMAL,
                                nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        JST_ERROR("Failed to open file lock '{}' [Error: {}].", path, GetLastError());
        return Result::ERROR;
    }

    OVERLAPPED overlapped = {};
    DWORD flags = LOCKFILE_EXCLUSIVE_LOCK;
    if (!wait) {
        flags |= LOCKFILE_FAIL_IMMEDIATELY;
    }

    if (!LockFileEx(handle, flags, 0, MAXDWORD, MAXDWORD, &overlapped)) {
        const auto error = GetLastError();
        CloseHandle(handle);
        if (!wait && error == ERROR_LOCK_VIOLATION) {
            return Result::SKIP;
        }

        JST_ERROR("Failed to acquire file lock '{}' [Error: {}].", path, error);
        return Result::ERROR;
    }

    state.handle = handle;
#elif defined(JST_OS_BROWSER)
    (void)wait;
    JST_ERROR("File locks are not supported in this platform.");
    return Result::ERROR;
#else
    const int fd = open(path.c_str(), O_RDWR | O_CREAT, 0600);
    if (fd < 0) {
        JST_ERROR("Failed to open file lock '{}' [Errno: {}].", path, errno);
        return Result::ERROR;
    }

    const int operation = wait ? LOCK_EX : (LOCK_EX | LOCK_NB);
    while (flock(fd, operation) != 0) {
        if (errno == EINTR && wait) {
            continue;
        }

        const auto error = errno;
        close(fd);
        if (!wait && (error == EWOULDBLOCK || error == EAGAIN)) {
            return Result::SKIP;
        }

        JST_ERROR("Failed to acquire file lock '{}' [Errno: {}].", path, error);
        return Result::ERROR;
    }

    state.fd = fd;
#endif

    state.path = path;
    state.locked = true;
    return Result::SUCCESS;
}

void FileLock::release() {
    if (!impl || !impl->locked) {
        return;
    }

#if defined(JST_OS_WINDOWS)
    OVERLAPPED overlapped = {};
    (void)UnlockFileEx(impl->handle, 0, MAXDWORD, MAXDWORD, &overlapped);
    CloseHandle(impl->handle);
    impl->handle = INVALID_HANDLE_VALUE;
#elif !defined(JST_OS_BROWSER)
    (void)flock(impl->fd, LOCK_UN);
    close(impl->fd);
    impl->fd = -1;
#endif

    impl->path.clear();
    impl->locked = false;
}

bool FileLock::locked() const {
    return impl && impl->locked;
}

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

Result WindowsEnvPath(const wchar_t* name, std::filesystem::path& path) {
    const DWORD requiredSize = GetEnvironmentVariableW(name, nullptr, 0);
    if (requiredSize == 0) {
        return Result::ERROR;
    }

    std::wstring value(requiredSize, L'\0');
    const DWORD writtenSize = GetEnvironmentVariableW(name, value.data(), requiredSize);
    if (writtenSize == 0 || writtenSize >= requiredSize) {
        return Result::ERROR;
    }

    value.resize(writtenSize);
    path = value;
    return Result::SUCCESS;
}

}  // namespace

Result ConfigPath(std::string& path) {
    std::filesystem::path appData;
    if (WindowsEnvPath(L"APPDATA", appData) != Result::SUCCESS) {
        JST_ERROR("Failed to resolve config path because APPDATA is not set.");
        return Result::ERROR;
    }

    path = WindowsPathToUtf8(appData / L"CyberEther");
    return Result::SUCCESS;
}

Result CachePath(std::string& path) {
    std::filesystem::path appData;
    std::filesystem::path localAppData;

    if (WindowsEnvPath(L"LOCALAPPDATA", localAppData) == Result::SUCCESS) {
        path = WindowsPathToUtf8(localAppData / L"CyberEther" / L"Cache");
        return Result::SUCCESS;
    }

    if (WindowsEnvPath(L"APPDATA", appData) == Result::SUCCESS) {
        path = WindowsPathToUtf8(appData / L"CyberEther" / L"Cache");
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
