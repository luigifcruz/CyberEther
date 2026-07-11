#include "jetstream/platform.hh"

#include <algorithm>
#include <chrono>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(JST_OS_WINDOWS)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef ERROR
#undef FATAL
#elif defined(JST_OS_LINUX) || defined(JST_OS_MAC)
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace Jetstream::Platform {

namespace {

constexpr std::size_t kMaxProcessOutputSize = 1024 * 1024;

bool TimedOut(const std::chrono::steady_clock::time_point& start, U64 timeoutMilliseconds) {
    return timeoutMilliseconds > 0 &&
           std::chrono::steady_clock::now() - start >=
               std::chrono::milliseconds(timeoutMilliseconds);
}

#if defined(JST_OS_WINDOWS)

class WindowsHandle {
 public:
    WindowsHandle() = default;
    explicit WindowsHandle(HANDLE handle) : handle(handle) {}

    ~WindowsHandle() {
        reset();
    }

    WindowsHandle(const WindowsHandle&) = delete;
    WindowsHandle& operator=(const WindowsHandle&) = delete;

    HANDLE get() const {
        return handle;
    }

    void reset(HANDLE next = nullptr) {
        if (handle && handle != INVALID_HANDLE_VALUE) {
            CloseHandle(handle);
        }
        handle = next;
    }

 private:
    HANDLE handle = nullptr;
};

bool Utf8ToWide(const std::string& value, std::wstring& wide) {
    if (value.empty()) {
        wide.clear();
        return true;
    }

    const int size = MultiByteToWideChar(
        CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), static_cast<int>(value.size()), nullptr, 0);
    if (size <= 0) {
        return false;
    }

    wide.resize(static_cast<std::size_t>(size));
    return MultiByteToWideChar(CP_UTF8,
                               MB_ERR_INVALID_CHARS,
                               value.data(),
                               static_cast<int>(value.size()),
                               wide.data(),
                               size) == size;
}

std::wstring QuoteWindowsArgument(const std::wstring& argument) {
    std::wstring quoted = L"\"";
    std::size_t backslashes = 0;

    for (const wchar_t character : argument) {
        if (character == L'\\') {
            ++backslashes;
            continue;
        }

        if (character == L'\"') {
            quoted.append(backslashes * 2 + 1, L'\\');
            quoted.push_back(character);
        } else {
            quoted.append(backslashes, L'\\');
            quoted.push_back(character);
        }
        backslashes = 0;
    }

    quoted.append(backslashes * 2, L'\\');
    quoted.push_back(L'\"');
    return quoted;
}

Result RunWindowsProcess(const std::string& executable,
                         const std::vector<std::string>& arguments,
                         std::string& output,
                         U64 timeoutMilliseconds) {
    std::wstring nativeExecutable;
    if (!Utf8ToWide(executable, nativeExecutable)) {
        return Result::ERROR;
    }

    std::wstring commandLine = QuoteWindowsArgument(nativeExecutable);
    for (const auto& argument : arguments) {
        std::wstring nativeArgument;
        if (!Utf8ToWide(argument, nativeArgument)) {
            return Result::ERROR;
        }
        commandLine += L" " + QuoteWindowsArgument(nativeArgument);
    }

    SECURITY_ATTRIBUTES security = {
        .nLength = sizeof(SECURITY_ATTRIBUTES),
        .lpSecurityDescriptor = nullptr,
        .bInheritHandle = TRUE,
    };

    HANDLE rawReadPipe = nullptr;
    HANDLE rawWritePipe = nullptr;
    if (!CreatePipe(&rawReadPipe, &rawWritePipe, &security, 0)) {
        return Result::ERROR;
    }
    WindowsHandle readPipe(rawReadPipe);
    WindowsHandle writePipe(rawWritePipe);

    if (!SetHandleInformation(readPipe.get(), HANDLE_FLAG_INHERIT, 0)) {
        return Result::ERROR;
    }

    WindowsHandle nullDevice(CreateFileW(L"NUL",
                                         GENERIC_READ | GENERIC_WRITE,
                                         FILE_SHARE_READ | FILE_SHARE_WRITE,
                                         &security,
                                         OPEN_EXISTING,
                                         FILE_ATTRIBUTE_NORMAL,
                                         nullptr));
    if (nullDevice.get() == INVALID_HANDLE_VALUE) {
        return Result::ERROR;
    }

    SIZE_T attributeSize = 0;
    (void)InitializeProcThreadAttributeList(nullptr, 1, 0, &attributeSize);
    std::vector<unsigned char> attributeStorage(attributeSize);
    auto* attributeList =
        reinterpret_cast<PPROC_THREAD_ATTRIBUTE_LIST>(attributeStorage.data());
    if (!InitializeProcThreadAttributeList(attributeList, 1, 0, &attributeSize)) {
        return Result::ERROR;
    }

    HANDLE inheritedHandles[] = {writePipe.get(), nullDevice.get()};
    if (!UpdateProcThreadAttribute(attributeList,
                                   0,
                                   PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                                   inheritedHandles,
                                   sizeof(inheritedHandles),
                                   nullptr,
                                   nullptr)) {
        DeleteProcThreadAttributeList(attributeList);
        return Result::ERROR;
    }

    STARTUPINFOEXW startup = {};
    startup.StartupInfo.cb = sizeof(startup);
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    startup.StartupInfo.hStdInput = nullDevice.get();
    startup.StartupInfo.hStdOutput = writePipe.get();
    startup.StartupInfo.hStdError = nullDevice.get();
    startup.lpAttributeList = attributeList;

    WindowsHandle job(CreateJobObjectW(nullptr, nullptr));
    if (!job.get()) {
        DeleteProcThreadAttributeList(attributeList);
        return Result::ERROR;
    }
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION jobLimits = {};
    jobLimits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if (!SetInformationJobObject(job.get(),
                                 JobObjectExtendedLimitInformation,
                                 &jobLimits,
                                 sizeof(jobLimits))) {
        DeleteProcThreadAttributeList(attributeList);
        return Result::ERROR;
    }

    PROCESS_INFORMATION process = {};
    const BOOL created = CreateProcessW(nativeExecutable.c_str(),
                                        commandLine.data(),
                                        nullptr,
                                        nullptr,
                                        TRUE,
                                        CREATE_NO_WINDOW | CREATE_SUSPENDED |
                                            EXTENDED_STARTUPINFO_PRESENT,
                                        nullptr,
                                        nullptr,
                                        &startup.StartupInfo,
                                        &process);
    DeleteProcThreadAttributeList(attributeList);
    nullDevice.reset();
    writePipe.reset();
    if (!created) {
        return Result::ERROR;
    }

    WindowsHandle processHandle(process.hProcess);
    WindowsHandle threadHandle(process.hThread);
    if (!AssignProcessToJobObject(job.get(), processHandle.get()) ||
        ResumeThread(threadHandle.get()) == static_cast<DWORD>(-1)) {
        (void)TerminateProcess(processHandle.get(), 1);
        (void)WaitForSingleObject(processHandle.get(), INFINITE);
        return Result::ERROR;
    }
    std::string captured;
    const auto start = std::chrono::steady_clock::now();
    bool pipeClosed = false;

    try {
        while (true) {
            if (TimedOut(start, timeoutMilliseconds)) {
                job.reset();
                (void)WaitForSingleObject(processHandle.get(), INFINITE);
                return Result::ERROR;
            }

            if (!pipeClosed) {
                DWORD available = 0;
                if (!PeekNamedPipe(readPipe.get(), nullptr, 0, nullptr, &available, nullptr)) {
                    pipeClosed = true;
                } else if (available > 0) {
                    char buffer[1024];
                    DWORD bytesRead = 0;
                    const DWORD requested =
                        std::min<DWORD>(available, static_cast<DWORD>(sizeof(buffer)));
                    if (!ReadFile(readPipe.get(), buffer, requested, &bytesRead, nullptr)) {
                        pipeClosed = true;
                    } else {
                        if (captured.size() + bytesRead > kMaxProcessOutputSize) {
                            job.reset();
                            (void)WaitForSingleObject(processHandle.get(), INFINITE);
                            return Result::ERROR;
                        }
                        captured.append(buffer, bytesRead);
                        continue;
                    }
                }
            }

            if (WaitForSingleObject(processHandle.get(), 10) == WAIT_OBJECT_0) {
                break;
            }
        }
    } catch (...) {
        job.reset();
        (void)WaitForSingleObject(processHandle.get(), INFINITE);
        return Result::ERROR;
    }

    try {
        while (true) {
            DWORD available = 0;
            if (!PeekNamedPipe(readPipe.get(), nullptr, 0, nullptr, &available, nullptr) ||
                available == 0) {
                break;
            }

            char buffer[1024];
            DWORD bytesRead = 0;
            const DWORD requested =
                std::min<DWORD>(available, static_cast<DWORD>(sizeof(buffer)));
            if (!ReadFile(readPipe.get(), buffer, requested, &bytesRead, nullptr)) {
                break;
            }
            if (captured.size() + bytesRead > kMaxProcessOutputSize) {
                return Result::ERROR;
            }
            captured.append(buffer, bytesRead);
        }
    } catch (...) {
        return Result::ERROR;
    }

    DWORD exitCode = 1;
    if (!GetExitCodeProcess(processHandle.get(), &exitCode) || exitCode != 0) {
        return Result::ERROR;
    }

    output = std::move(captured);
    return Result::SUCCESS;
}

#elif defined(JST_OS_LINUX) || defined(JST_OS_MAC)

class FileDescriptor {
 public:
    FileDescriptor() = default;
    explicit FileDescriptor(int fd) : fd(fd) {}

    ~FileDescriptor() {
        reset();
    }

    FileDescriptor(const FileDescriptor&) = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;

    int get() const {
        return fd;
    }

    void reset(int next = -1) {
        if (fd >= 0) {
            close(fd);
        }
        fd = next;
    }

 private:
    int fd = -1;
};

bool PreparePipeDescriptor(FileDescriptor& descriptor) {
    if (descriptor.get() <= STDERR_FILENO) {
        const int relocated = fcntl(descriptor.get(), F_DUPFD_CLOEXEC, STDERR_FILENO + 1);
        if (relocated < 0) {
            return false;
        }
        descriptor.reset(relocated);
        return true;
    }
    return fcntl(descriptor.get(), F_SETFD, FD_CLOEXEC) == 0;
}

bool WaitForProcess(pid_t process, int& status, int options, pid_t& result) {
    do {
        result = waitpid(process, &status, options);
    } while (result < 0 && errno == EINTR);
    return result >= 0;
}

void TerminateProcessGroup(pid_t process) {
    (void)kill(-process, SIGKILL);
    (void)kill(process, SIGKILL);
    int status = 0;
    pid_t result = 0;
    (void)WaitForProcess(process, status, 0, result);
}

Result RunPosixProcess(const std::string& executable,
                       const std::vector<std::string>& arguments,
                       std::string& output,
                       U64 timeoutMilliseconds) {
    std::vector<std::string> processArguments;
    processArguments.reserve(arguments.size() + 1);
    processArguments.push_back(executable);
    processArguments.insert(processArguments.end(), arguments.begin(), arguments.end());

    std::vector<char*> argumentPointers;
    argumentPointers.reserve(processArguments.size() + 1);
    for (auto& argument : processArguments) {
        argumentPointers.push_back(argument.data());
    }
    argumentPointers.push_back(nullptr);

    int rawPipe[2];
    if (pipe(rawPipe) != 0) {
        return Result::ERROR;
    }
    FileDescriptor readPipe(rawPipe[0]);
    FileDescriptor writePipe(rawPipe[1]);
    if (!PreparePipeDescriptor(readPipe) || !PreparePipeDescriptor(writePipe)) {
        return Result::ERROR;
    }

    const pid_t process = fork();
    if (process < 0) {
        return Result::ERROR;
    }

    if (process == 0) {
        readPipe.reset();
        if (setpgid(0, 0) != 0) {
            _exit(127);
        }
        int nullDevice = open("/dev/null", O_RDWR);
        if (nullDevice >= 0 && nullDevice <= STDERR_FILENO) {
            const int relocated = fcntl(nullDevice, F_DUPFD, STDERR_FILENO + 1);
            close(nullDevice);
            nullDevice = relocated;
        }
        if (nullDevice < 0 ||
            dup2(nullDevice, STDIN_FILENO) < 0 ||
            dup2(writePipe.get(), STDOUT_FILENO) < 0 ||
            dup2(nullDevice, STDERR_FILENO) < 0) {
            _exit(127);
        }
        close(nullDevice);
        writePipe.reset();
        execvp(executable.c_str(), argumentPointers.data());
        _exit(127);
    }

    (void)setpgid(process, process);
    writePipe.reset();
    const int flags = fcntl(readPipe.get(), F_GETFL, 0);
    if (flags < 0 || fcntl(readPipe.get(), F_SETFL, flags | O_NONBLOCK) < 0) {
        TerminateProcessGroup(process);
        return Result::ERROR;
    }

    std::string captured;
    const auto start = std::chrono::steady_clock::now();
    bool processExited = false;
    int status = 0;

    try {
        while (!processExited) {
            if (TimedOut(start, timeoutMilliseconds)) {
                TerminateProcessGroup(process);
                return Result::ERROR;
            }

            char buffer[1024];
            const ssize_t bytesRead = read(readPipe.get(), buffer, sizeof(buffer));
            if (bytesRead > 0) {
                if (captured.size() + static_cast<std::size_t>(bytesRead) >
                    kMaxProcessOutputSize) {
                    TerminateProcessGroup(process);
                    return Result::ERROR;
                }
                captured.append(buffer, static_cast<std::size_t>(bytesRead));
                continue;
            }
            if (bytesRead < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
                TerminateProcessGroup(process);
                return Result::ERROR;
            }

            pid_t waitResult = 0;
            if (!WaitForProcess(process, status, WNOHANG, waitResult)) {
                TerminateProcessGroup(process);
                return Result::ERROR;
            }
            processExited = waitResult == process;

            if (!processExited) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }

        while (true) {
            char buffer[1024];
            const ssize_t bytesRead = read(readPipe.get(), buffer, sizeof(buffer));
            if (bytesRead <= 0) {
                break;
            }
            if (captured.size() + static_cast<std::size_t>(bytesRead) >
                kMaxProcessOutputSize) {
                return Result::ERROR;
            }
            captured.append(buffer, static_cast<std::size_t>(bytesRead));
        }
    } catch (...) {
        if (!processExited) {
            TerminateProcessGroup(process);
        }
        return Result::ERROR;
    }

    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        return Result::ERROR;
    }

    output = std::move(captured);
    return Result::SUCCESS;
}

#endif

}  // namespace

Result RunProcess(const std::string& executable,
                  const std::vector<std::string>& arguments,
                  std::string& output,
                  U64 timeoutMilliseconds) {
    if (executable.empty()) {
        return Result::ERROR;
    }

#if defined(JST_OS_WINDOWS)
    return RunWindowsProcess(executable, arguments, output, timeoutMilliseconds);
#elif defined(JST_OS_BROWSER) || defined(JST_OS_IOS) || defined(JST_OS_ANDROID)
    (void)arguments;
    (void)output;
    (void)timeoutMilliseconds;
    return Result::ERROR;
#else
    return RunPosixProcess(executable, arguments, output, timeoutMilliseconds);
#endif
}

}  // namespace Jetstream::Platform
