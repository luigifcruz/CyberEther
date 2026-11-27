#include "jetstream/platform.hh"

#include <atomic>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <utility>

#if defined(JST_OS_WINDOWS)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <lmcons.h>
#undef ERROR
#undef FATAL
#else
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#endif

#if defined(JST_OS_MAC)
#include "apple.hh"
#endif

namespace Jetstream::Platform {

namespace {

constexpr U64 MessageLimit = 2048;
constexpr int TimeoutMilliseconds = 2000;

#if defined(JST_OS_WINDOWS)

std::wstring PipeName(const std::string& scheme) {
    wchar_t user[UNLEN + 1] = {};
    DWORD size = UNLEN + 1;
    if (!GetUserNameW(user, &size)) {
        user[0] = L'\0';
    }
    return L"\\\\.\\pipe\\" + PathFromUtf8(scheme).wstring() + L"-" + user;
}

bool WriteRegistryText(const std::wstring& key, const wchar_t* name, const std::wstring& value) {
    const DWORD bytes = static_cast<DWORD>((value.size() + 1) * sizeof(wchar_t));
    return RegSetKeyValueW(HKEY_CURRENT_USER, key.c_str(), name, REG_SZ, value.c_str(), bytes) == ERROR_SUCCESS;
}

#else

bool Endpoint(const std::string& scheme, sockaddr_un& address) {
    std::filesystem::path path;
    if (EnvironmentPath("XDG_RUNTIME_DIR", path) == Result::SUCCESS && !path.empty()) {
        path /= scheme + ".sock";
    } else {
        if (EnvironmentPath("TMPDIR", path) != Result::SUCCESS || path.empty()) {
            path = "/tmp";
        }
        path /= scheme + "-" + std::to_string(getuid()) + ".sock";
    }

    const std::string text = path.string();
    if (text.size() >= sizeof(address.sun_path)) {
        return false;
    }
    address = {};
    address.sun_family = AF_UNIX;
    text.copy(address.sun_path, text.size());
    return true;
}

int OpenSocket() {
    const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd >= 0) {
        (void)fcntl(fd, F_SETFD, FD_CLOEXEC);
        DisableSocketSigPipe(static_cast<std::uintptr_t>(fd));
    }
    return fd;
}

bool Connect(const int fd, const sockaddr_un& address) {
    int result = 0;
    do {
        result = connect(fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address));
    } while (result != 0 && errno == EINTR);
    return result == 0;
}

#endif

#if defined(JST_OS_LINUX)

std::filesystem::path UserFolder(const char* variable, const char* fallback) {
    std::filesystem::path path;
    if (EnvironmentPath(variable, path) == Result::SUCCESS && path.is_absolute()) {
        return path;
    }
    if (EnvironmentPath("HOME", path) == Result::SUCCESS && path.is_absolute()) {
        return path / fallback;
    }
    return {};
}

bool ReadText(const std::filesystem::path& path, std::string& text) {
    text.clear();
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        return !ec;
    }
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }
    std::ostringstream content;
    content << file.rdbuf();
    text = content.str();
    return !file.bad();
}

bool WriteText(const std::filesystem::path& path, const std::string& current, const std::string& text) {
    if (current == text) {
        return true;
    }
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    auto target = std::filesystem::canonical(path, ec);
    if (ec) {
        target = path;
    }
    auto staged = target;
    staged += ".tmp";
    {
        std::ofstream file(staged, std::ios::binary | std::ios::trunc);
        file << text;
        file.flush();
        if (!file.good()) {
            std::filesystem::remove(staged, ec);
            return false;
        }
    }
    std::filesystem::rename(staged, target, ec);
    if (ec) {
        std::filesystem::remove(staged, ec);
        return false;
    }
    return true;
}

std::string QuoteExec(const std::string& path) {
    if (path.find_first_of(" \t\n\"'\\><~|&;$*?#()`%") == std::string::npos) {
        return path;
    }

    std::string quoted = "\"";
    for (const char c : path) {
        if (c == '"' || c == '`' || c == '$') {
            quoted += "\\\\";
        }
        if (c == '\\') {
            quoted += "\\\\\\";
        }
        if (c == '%') {
            quoted += '%';
        }
        quoted += c;
    }
    quoted += '"';
    return quoted;
}

std::string WithDefaultApplication(const std::string& associations, const std::string& type, const std::string& entry) {
    static constexpr std::string_view header = "[Default Applications]";
    const std::string line = type + "=" + entry;

    std::istringstream input(associations);
    std::string output;
    std::string current;
    bool inside = false;
    bool found = false;
    bool written = false;
    while (std::getline(input, current)) {
        if (current.starts_with("[")) {
            if (inside && !written) {
                output += line + "\n";
                written = true;
            }
            inside = current.starts_with(header);
            found = found || inside;
        } else if (inside && current.starts_with(type + "=")) {
            if (written) {
                continue;
            }
            current = line;
            written = true;
        }
        output += current + "\n";
    }
    if (!found) {
        output += std::string(header) + "\n";
    }
    if (!written) {
        output += line + "\n";
    }
    return output;
}

#endif

}  // namespace

struct UrlScheme::Impl {
    Handler handler;
    std::thread thread;
    std::atomic<bool> stopping = false;
    bool events = false;

#if defined(JST_OS_WINDOWS)
    HANDLE pipe = INVALID_HANDLE_VALUE;
    HANDLE stopEvent = nullptr;
    HANDLE ioEvent = nullptr;

    bool complete(OVERLAPPED& overlapped, BOOL started, DWORD timeout, DWORD& bytes);
#else
    int fd = -1;
    int wake[2] = {-1, -1};
    std::string path;
    FileLock lock;
#endif

    void serve();
};

#if defined(JST_OS_WINDOWS)

bool UrlScheme::Impl::complete(OVERLAPPED& overlapped, const BOOL started, const DWORD timeout, DWORD& bytes) {
    bytes = 0;
    if (!started) {
        const DWORD error = GetLastError();
        if (error == ERROR_PIPE_CONNECTED || error == ERROR_NO_DATA) {
            return true;
        }
        if (error != ERROR_IO_PENDING) {
            return false;
        }
        const HANDLE handles[2] = {stopEvent, ioEvent};
        if (WaitForMultipleObjects(2, handles, FALSE, timeout) != WAIT_OBJECT_0 + 1) {
            (void)CancelIo(pipe);
            (void)GetOverlappedResult(pipe, &overlapped, &bytes, TRUE);
            return false;
        }
    }
    return GetOverlappedResult(pipe, &overlapped, &bytes, FALSE) != 0;
}

void UrlScheme::Impl::serve() {
    while (!stopping) {
        DWORD bytes = 0;
        OVERLAPPED connection = {};
        connection.hEvent = ioEvent;
        (void)ResetEvent(ioEvent);
        if (!complete(connection, ConnectNamedPipe(pipe, &connection), INFINITE, bytes)) {
            if (stopping) {
                break;
            }
            (void)DisconnectNamedPipe(pipe);
            continue;
        }

        std::string message;
        char buffer[512];
        while (true) {
            OVERLAPPED read = {};
            read.hEvent = ioEvent;
            (void)ResetEvent(ioEvent);
            const BOOL started = ReadFile(pipe, buffer, sizeof(buffer), nullptr, &read);
            if (!complete(read, started, TimeoutMilliseconds, bytes) || bytes == 0) {
                break;
            }
            message.append(buffer, bytes);
            if (message.size() > MessageLimit) {
                message.clear();
                break;
            }
        }
        (void)DisconnectNamedPipe(pipe);

        if (!message.empty() && !stopping) {
            handler(std::move(message));
        }
    }
}

#else

void UrlScheme::Impl::serve() {
    while (!stopping) {
        pollfd waiting[2] = {{fd, POLLIN, 0}, {wake[0], POLLIN, 0}};
        if (poll(waiting, 2, -1) < 0) {
            if (errno == EINTR) {
                continue;
            }
            break;
        }
        if (waiting[1].revents != 0 || stopping) {
            break;
        }
        if ((waiting[0].revents & POLLIN) == 0) {
            continue;
        }

        const int connection = accept(fd, nullptr, nullptr);
        if (connection < 0) {
            continue;
        }
        (void)fcntl(connection, F_SETFD, FD_CLOEXEC);

        std::string message;
        char buffer[512];
        while (true) {
            pollfd reading[2] = {{connection, POLLIN, 0}, {wake[0], POLLIN, 0}};
            const int ready = poll(reading, 2, TimeoutMilliseconds);
            if (ready < 0 && errno == EINTR) {
                continue;
            }
            if (ready <= 0 || reading[1].revents != 0) {
                message.clear();
                break;
            }
            const ssize_t received = read(connection, buffer, sizeof(buffer));
            if (received < 0 && errno == EINTR) {
                continue;
            }
            if (received < 0) {
                message.clear();
                break;
            }
            if (received == 0) {
                break;
            }
            message.append(buffer, static_cast<U64>(received));
            if (message.size() > MessageLimit) {
                message.clear();
                break;
            }
        }
        ::close(connection);

        if (!message.empty() && !stopping) {
            handler(std::move(message));
        }
    }
}

#endif

UrlScheme::UrlScheme() {
    impl = std::make_unique<Impl>();
}

UrlScheme::~UrlScheme() {
    close();
}

Result UrlScheme::Register(const std::string& scheme) {
#if defined(JST_OS_WINDOWS)
    std::wstring executable(32768, L'\0');
    const DWORD length = GetModuleFileNameW(nullptr, executable.data(), static_cast<DWORD>(executable.size()));
    if (length == 0 || length >= executable.size()) {
        JST_ERROR("Failed to resolve the executable path for the '{}' URL scheme.", scheme);
        return Result::ERROR;
    }
    executable.resize(length);

    const std::wstring name = PathFromUtf8(scheme).wstring();
    const std::wstring key = L"Software\\Classes\\" + name;
    if (!WriteRegistryText(key, nullptr, L"URL:" + name) ||
        !WriteRegistryText(key, L"URL Protocol", L"") ||
        !WriteRegistryText(key + L"\\shell\\open\\command", nullptr, L"\"" + executable + L"\" \"%1\"")) {
        JST_ERROR("Failed to register the '{}' URL scheme.", scheme);
        return Result::ERROR;
    }

    return Result::SUCCESS;
#elif defined(JST_OS_LINUX)
    std::filesystem::path executable;
    if (EnvironmentPath("APPIMAGE", executable) != Result::SUCCESS || !executable.is_absolute()) {
        std::error_code ec;
        executable = std::filesystem::read_symlink("/proc/self/exe", ec);
        if (ec) {
            JST_ERROR("Failed to resolve the executable path for the '{}' URL scheme.", scheme);
            return Result::ERROR;
        }
    }

    const auto data = UserFolder("XDG_DATA_HOME", ".local/share");
    const auto config = UserFolder("XDG_CONFIG_HOME", ".config");
    if (data.empty() || config.empty()) {
        JST_ERROR("Failed to resolve the user folders for the '{}' URL scheme.", scheme);
        return Result::ERROR;
    }

    const std::string type = "x-scheme-handler/" + scheme;
    const std::string entry = scheme + "-url.desktop";
    const std::string desktop = "[Desktop Entry]\n"
                                "Type=Application\n"
                                "Name=CyberEther\n"
                                "Exec=" + QuoteExec(executable.string()) + " %u\n"
                                "Terminal=false\n"
                                "NoDisplay=true\n"
                                "MimeType=" + type + ";\n";
    const auto launcher = data / "applications" / entry;
    const auto associations = config / "mimeapps.list";
    std::string currentLauncher;
    std::string currentAssociations;
    if (!ReadText(launcher, currentLauncher) || !ReadText(associations, currentAssociations)) {
        JST_ERROR("Failed to read the current handlers for the '{}' URL scheme.", scheme);
        return Result::ERROR;
    }
    if (!WriteText(launcher, currentLauncher, desktop) ||
        !WriteText(associations, currentAssociations, WithDefaultApplication(currentAssociations, type, entry))) {
        JST_ERROR("Failed to register the '{}' URL scheme.", scheme);
        return Result::ERROR;
    }

    return Result::SUCCESS;
#else
    (void)scheme;

    return Result::SUCCESS;
#endif
}

Result UrlScheme::Forward(const std::string& scheme, const std::string& url) {
    if (url.empty() || url.size() > MessageLimit) {
        return Result::ERROR;
    }

#if defined(JST_OS_WINDOWS)
    const std::wstring name = PipeName(scheme);
    HANDLE pipe = CreateFileW(name.c_str(), GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
    if (pipe == INVALID_HANDLE_VALUE && GetLastError() == ERROR_PIPE_BUSY &&
        WaitNamedPipeW(name.c_str(), TimeoutMilliseconds)) {
        pipe = CreateFileW(name.c_str(), GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
    }
    if (pipe == INVALID_HANDLE_VALUE) {
        return Result::SKIP;
    }

    (void)AllowSetForegroundWindow(ASFW_ANY);

    DWORD written = 0;
    const bool sent = WriteFile(pipe, url.data(), static_cast<DWORD>(url.size()), &written, nullptr) &&
                      written == url.size();
    CloseHandle(pipe);
    return sent ? Result::SUCCESS : Result::ERROR;
#else
    sockaddr_un address;
    if (!Endpoint(scheme, address)) {
        return Result::SKIP;
    }
    const int fd = OpenSocket();
    if (fd < 0) {
        return Result::SKIP;
    }
    if (!Connect(fd, address)) {
        ::close(fd);
        return Result::SKIP;
    }

    int flags = 0;
#if defined(MSG_NOSIGNAL)
    flags = MSG_NOSIGNAL;
#endif
    U64 offset = 0;
    while (offset < url.size()) {
        const ssize_t sent = send(fd, url.data() + offset, url.size() - offset, flags);
        if (sent < 0 && errno == EINTR) {
            continue;
        }
        if (sent <= 0) {
            ::close(fd);
            return Result::ERROR;
        }
        offset += static_cast<U64>(sent);
    }
    ::close(fd);

    return Result::SUCCESS;
#endif
}

Result UrlScheme::listen(const std::string& scheme, Handler handler) {
    close();
    impl->handler = std::move(handler);
    impl->stopping = false;

#if defined(JST_OS_MAC)
    InstallUrlEventHandler(impl->handler);
    impl->events = true;
#endif

#if defined(JST_OS_WINDOWS)
    impl->pipe = CreateNamedPipeW(PipeName(scheme).c_str(),
                                  PIPE_ACCESS_INBOUND | FILE_FLAG_OVERLAPPED | FILE_FLAG_FIRST_PIPE_INSTANCE,
                                  PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS,
                                  1,
                                  0,
                                  4096,
                                  0,
                                  nullptr);
    if (impl->pipe == INVALID_HANDLE_VALUE) {
        const DWORD error = GetLastError();
        return error == ERROR_ACCESS_DENIED || error == ERROR_PIPE_BUSY ? Result::SKIP : Result::ERROR;
    }
    impl->stopEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    impl->ioEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (impl->stopEvent == nullptr || impl->ioEvent == nullptr) {
        close();
        return Result::ERROR;
    }
#else
    sockaddr_un address;
    if (!Endpoint(scheme, address)) {
        JST_ERROR("Failed to resolve the endpoint for the '{}' URL scheme.", scheme);
        return Result::ERROR;
    }

    const Result locked = impl->lock.acquire(std::string(address.sun_path) + ".lock", false);
    if (locked != Result::SUCCESS) {
        return locked;
    }

    impl->fd = OpenSocket();
    (void)unlink(address.sun_path);
    if (impl->fd < 0 || bind(impl->fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) {
        close();
        return Result::ERROR;
    }
    impl->path = address.sun_path;
    (void)chmod(address.sun_path, 0600);

    if (::listen(impl->fd, 8) != 0 || pipe(impl->wake) != 0) {
        close();
        return Result::ERROR;
    }
    (void)fcntl(impl->wake[0], F_SETFD, FD_CLOEXEC);
    (void)fcntl(impl->wake[1], F_SETFD, FD_CLOEXEC);
#endif

    impl->thread = std::thread([state = impl.get()] {
        state->serve();
    });

    return Result::SUCCESS;
}

void UrlScheme::close() {
    impl->stopping = true;

#if defined(JST_OS_MAC)
    if (std::exchange(impl->events, false)) {
        InstallUrlEventHandler({});
    }
#endif

#if defined(JST_OS_WINDOWS)
    if (impl->stopEvent != nullptr) {
        (void)SetEvent(impl->stopEvent);
    }
    if (impl->thread.joinable()) {
        impl->thread.join();
    }
    if (impl->pipe != INVALID_HANDLE_VALUE) {
        CloseHandle(impl->pipe);
        impl->pipe = INVALID_HANDLE_VALUE;
    }
    for (HANDLE* event : {&impl->stopEvent, &impl->ioEvent}) {
        if (*event != nullptr) {
            CloseHandle(*event);
            *event = nullptr;
        }
    }
#else
    if (impl->wake[1] >= 0) {
        const char byte = 0;
        (void)!write(impl->wake[1], &byte, 1);
    }
    if (impl->thread.joinable()) {
        impl->thread.join();
    }
    for (int* descriptor : {&impl->fd, &impl->wake[0], &impl->wake[1]}) {
        if (*descriptor >= 0) {
            ::close(*descriptor);
            *descriptor = -1;
        }
    }
    if (!impl->path.empty()) {
        (void)unlink(impl->path.c_str());
        impl->path.clear();
    }
    impl->lock.release();
#endif

    impl->handler = {};
}

}  // namespace Jetstream::Platform
