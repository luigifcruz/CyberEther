#include "jetstream/platform.hh"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <string>
#include <utility>
#include <vector>

#if defined(JST_OS_LINUX) || defined(JST_OS_MAC)
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/wait.h>
#include <termios.h>
#include <unistd.h>
#if defined(JST_OS_MAC)
#include <crt_externs.h>
#else
#include <sys/syscall.h>
extern char** environ;
#endif
#endif

namespace Jetstream::Platform {

#if defined(JST_OS_LINUX) || defined(JST_OS_MAC)

namespace {

constexpr U64 kReadChunk = 64 * 1024;
constexpr int kFallbackDescriptorLimit = 65536;
constexpr int kShutdownPolls = 20;
constexpr useconds_t kShutdownPollInterval = 2500;

bool SetDescriptorFlags(int descriptor) {
    const int status = fcntl(descriptor, F_GETFL, 0);
    if (status < 0 || fcntl(descriptor, F_SETFL, status | O_NONBLOCK) < 0) {
        return false;
    }
    return fcntl(descriptor, F_SETFD, FD_CLOEXEC) == 0;
}

char**& Environment() {
#if defined(JST_OS_MAC)
    return *_NSGetEnviron();
#else
    return environ;
#endif
}

std::vector<std::string> MergeEnvironment(const std::vector<std::pair<std::string, std::string>>& overrides) {
    std::vector<std::string> merged;
    for (char** entry = Environment(); entry && *entry; ++entry) {
        const std::string value(*entry);
        const auto separator = value.find('=');
        const std::string name = value.substr(0, separator);
        const bool replaced = std::any_of(overrides.begin(), overrides.end(), [&](const auto& item) {
            return item.first == name;
        });
        if (!replaced) {
            merged.push_back(value);
        }
    }
    for (const auto& [name, value] : overrides) {
        merged.push_back(name + "=" + value);
    }
    return merged;
}

std::vector<char*> Pointers(std::vector<std::string>& values) {
    std::vector<char*> pointers;
    pointers.reserve(values.size() + 1);
    for (auto& value : values) {
        pointers.push_back(value.data());
    }
    pointers.push_back(nullptr);
    return pointers;
}

}  // namespace

struct PseudoTerminal::Impl {
    int master = -1;
    int interruptRead = -1;
    int interruptWrite = -1;
    pid_t process = -1;
    std::atomic<bool> alive = false;
    bool hungUp = false;
    std::string pending;

    bool connected() const {
        return master >= 0 && !hungUp;
    }

    void hangUp() {
        hungUp = true;
        reap(false);
        alive = false;
    }

    void clearInterrupts() {
        char sink[64];
        while (::read(interruptRead, sink, sizeof(sink)) > 0) {
        }
    }

    void reap(bool block) {
        if (process <= 0) {
            return;
        }
        int status = 0;
        pid_t result = 0;
        do {
            result = waitpid(process, &status, block ? 0 : WNOHANG);
        } while (result < 0 && errno == EINTR);
        if (result == process || (result < 0 && errno == ECHILD)) {
            process = -1;
            alive = false;
        }
    }

    void drain() {
        if (master < 0) {
            return;
        }
        char sink[kReadChunk];
        while (::read(master, sink, sizeof(sink)) > 0) {
        }
    }

    void closeMaster() {
        if (master >= 0) {
            close(master);
            master = -1;
        }
    }

    void closeInterrupts() {
        for (int* descriptor : {&interruptRead, &interruptWrite}) {
            if (*descriptor >= 0) {
                close(*descriptor);
                *descriptor = -1;
            }
        }
    }
};

PseudoTerminal::PseudoTerminal() {
    impl = std::make_unique<Impl>();
}

PseudoTerminal::~PseudoTerminal() {
    stop();
}

bool PseudoTerminal::Supported() {
    return true;
}

Result PseudoTerminal::start(const Config& config) {
    stop();

    if (config.program.empty()) {
        JST_ERROR("[PLATFORM] Pseudo terminal needs a program to run.");
        return Result::ERROR;
    }

#if defined(JST_OS_LINUX)
    const int master = posix_openpt(O_RDWR | O_NOCTTY | O_CLOEXEC);
#else
    const int master = posix_openpt(O_RDWR | O_NOCTTY);
#endif
    if (master < 0) {
        JST_ERROR("[PLATFORM] Failed to open a pseudo terminal: {}.", std::strerror(errno));
        return Result::ERROR;
    }
    if (grantpt(master) != 0 || unlockpt(master) != 0 || !SetDescriptorFlags(master)) {
        JST_ERROR("[PLATFORM] Failed to prepare the pseudo terminal: {}.", std::strerror(errno));
        close(master);
        return Result::ERROR;
    }
    char slaveName[256] = {};
    if (ptsname_r(master, slaveName, sizeof(slaveName)) != 0) {
        JST_ERROR("[PLATFORM] Failed to resolve the pseudo terminal device.");
        close(master);
        return Result::ERROR;
    }
    int interrupts[2] = {-1, -1};
#if defined(JST_OS_LINUX)
    const bool piped = pipe2(interrupts, O_CLOEXEC | O_NONBLOCK) == 0;
#else
    const bool piped = pipe(interrupts) == 0 && SetDescriptorFlags(interrupts[0]) && SetDescriptorFlags(interrupts[1]);
#endif
    if (!piped) {
        JST_ERROR("[PLATFORM] Failed to create the pseudo terminal interrupt pipe: {}.", std::strerror(errno));
        for (const int descriptor : interrupts) {
            if (descriptor >= 0) {
                close(descriptor);
            }
        }
        close(master);
        return Result::ERROR;
    }
    impl->interruptRead = interrupts[0];
    impl->interruptWrite = interrupts[1];
    const int slave = open(slaveName, O_RDWR | O_NOCTTY | O_CLOEXEC);
    if (slave < 0) {
        JST_ERROR("[PLATFORM] Failed to open the pseudo terminal device: {}.", std::strerror(errno));
        impl->closeInterrupts();
        close(master);
        return Result::ERROR;
    }

    winsize size{};
    size.ws_col = std::max<U16>(1, config.columns);
    size.ws_row = std::max<U16>(1, config.rows);
    (void)ioctl(master, TIOCSWINSZ, &size);

    std::vector<std::string> arguments;
    arguments.reserve(config.arguments.size() + 1);
    arguments.push_back(config.program);
    arguments.insert(arguments.end(), config.arguments.begin(), config.arguments.end());
    auto argumentPointers = Pointers(arguments);
    auto environment = MergeEnvironment(config.environment);
    auto environmentPointers = Pointers(environment);
    const std::string directory = config.directory;
    const std::string program = config.program;
    const long descriptorLimit = sysconf(_SC_OPEN_MAX);
    const int inheritedLimit = descriptorLimit > STDERR_FILENO && descriptorLimit < INT_MAX
                                   ? static_cast<int>(descriptorLimit)
                                   : kFallbackDescriptorLimit;

    const pid_t process = fork();
    if (process < 0) {
        JST_ERROR("[PLATFORM] Failed to fork the terminal process: {}.", std::strerror(errno));
        close(slave);
        impl->closeInterrupts();
        close(master);
        return Result::ERROR;
    }

    if (process == 0) {
        close(master);
        if (setsid() < 0) {
            _exit(127);
        }
        (void)ioctl(slave, TIOCSCTTY, 0);
        if (dup2(slave, STDIN_FILENO) < 0 || dup2(slave, STDOUT_FILENO) < 0 || dup2(slave, STDERR_FILENO) < 0) {
            _exit(127);
        }
        if (slave > STDERR_FILENO) {
            close(slave);
        }
#if defined(SYS_close_range)
        if (syscall(SYS_close_range, STDERR_FILENO + 1, ~0U, 0) != 0)
#endif
        {
            for (int descriptor = STDERR_FILENO + 1; descriptor < inheritedLimit; ++descriptor) {
                close(descriptor);
            }
        }
        sigset_t signals;
        sigemptyset(&signals);
        sigprocmask(SIG_SETMASK, &signals, nullptr);
        for (int signal = 1; signal < NSIG; ++signal) {
            if (signal != SIGKILL && signal != SIGSTOP) {
                std::signal(signal, SIG_DFL);
            }
        }
        if (!directory.empty() && chdir(directory.c_str()) != 0) {
            (void)chdir("/");
        }
        Environment() = environmentPointers.data();
        execvp(program.c_str(), argumentPointers.data());
        _exit(127);
    }
    close(slave);

    impl->master = master;
    impl->process = process;
    impl->alive = true;
    impl->hungUp = false;
    impl->pending.clear();
    return Result::SUCCESS;
}

void PseudoTerminal::stop() {
    if (impl->process > 0) {
        (void)kill(-impl->process, SIGHUP);
        (void)kill(impl->process, SIGHUP);
        for (int poll = 0; poll < kShutdownPolls && impl->process > 0; ++poll) {
            impl->drain();
            impl->reap(false);
            if (impl->process > 0) {
                usleep(kShutdownPollInterval);
            }
        }
        if (impl->process > 0) {
            impl->closeMaster();
            (void)kill(-impl->process, SIGKILL);
            (void)kill(impl->process, SIGKILL);
            impl->reap(true);
        }
    }
    impl->closeMaster();
    impl->closeInterrupts();
    impl->process = -1;
    impl->alive = false;
    impl->hungUp = false;
    impl->pending.clear();
}

bool PseudoTerminal::running() const {
    return impl->alive;
}

Result PseudoTerminal::read(std::string& output, U64 limit) {
    if (!impl->connected()) {
        return Result::ERROR;
    }
    char buffer[kReadChunk];
    U64 total = 0;
    while (total < limit) {
        const U64 chunk = std::min<U64>(sizeof(buffer), limit - total);
        const ssize_t count = ::read(impl->master, buffer, chunk);
        if (count > 0) {
            output.append(buffer, static_cast<std::size_t>(count));
            total += static_cast<U64>(count);
            continue;
        }
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            break;
        }
        impl->hangUp();
        break;
    }
    if (impl->alive) {
        impl->reap(false);
    }
    return Result::SUCCESS;
}

Result PseudoTerminal::write(std::string_view data) {
    if (!impl->connected()) {
        return Result::ERROR;
    }
    impl->pending.append(data);
    return flush();
}

Result PseudoTerminal::flush() {
    if (!impl->connected()) {
        impl->pending.clear();
        return Result::ERROR;
    }
    while (!impl->pending.empty()) {
        const ssize_t count = ::write(impl->master, impl->pending.data(), impl->pending.size());
        if (count > 0) {
            impl->pending.erase(0, static_cast<std::size_t>(count));
            continue;
        }
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            return Result::SUCCESS;
        }
        impl->pending.clear();
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

Result PseudoTerminal::wait(std::optional<std::chrono::milliseconds> timeout, bool output) {
    if (!impl->connected()) {
        return Result::ERROR;
    }
    const short events = static_cast<short>((output ? POLLIN : 0) | (impl->pending.empty() ? 0 : POLLOUT));
    pollfd descriptors[2] = {
        {.fd = impl->interruptRead, .events = POLLIN, .revents = 0},
        {.fd = events != 0 ? impl->master : -1, .events = events, .revents = 0},
    };
    const auto deadline = timeout.has_value()
        ? std::optional(std::chrono::steady_clock::now() + std::max(*timeout, std::chrono::milliseconds(0)))
        : std::nullopt;
    int count = 0;
    while (true) {
        int milliseconds = -1;
        if (deadline.has_value()) {
            const auto remaining = std::chrono::ceil<std::chrono::milliseconds>(*deadline -
                                                                                std::chrono::steady_clock::now());
            milliseconds = static_cast<int>(std::clamp<std::chrono::milliseconds::rep>(remaining.count(), 0, INT_MAX));
        }
        count = ::poll(descriptors, 2, milliseconds);
        if (count >= 0 || errno != EINTR) {
            break;
        }
    }
    if (count < 0) {
        return Result::ERROR;
    }
    if (descriptors[0].revents & POLLIN) {
        impl->clearInterrupts();
    }
    if (output && (descriptors[1].revents & (POLLHUP | POLLERR | POLLNVAL)) && !(descriptors[1].revents & POLLIN)) {
        impl->hangUp();
    }
    return Result::SUCCESS;
}

void PseudoTerminal::interrupt() {
    if (impl->interruptWrite < 0) {
        return;
    }
    const char byte = 1;
    ssize_t count = 0;
    do {
        count = ::write(impl->interruptWrite, &byte, 1);
    } while (count < 0 && errno == EINTR);
}

Result PseudoTerminal::resize(U16 columns, U16 rows, U16 widthPixels, U16 heightPixels) {
    if (impl->master < 0) {
        return Result::ERROR;
    }
    winsize size{};
    size.ws_col = std::max<U16>(1, columns);
    size.ws_row = std::max<U16>(1, rows);
    size.ws_xpixel = widthPixels;
    size.ws_ypixel = heightPixels;
    if (ioctl(impl->master, TIOCSWINSZ, &size) != 0) {
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

#else

struct PseudoTerminal::Impl {};

PseudoTerminal::PseudoTerminal() {
    impl = std::make_unique<Impl>();
}

PseudoTerminal::~PseudoTerminal() = default;

bool PseudoTerminal::Supported() {
    return false;
}

Result PseudoTerminal::start(const Config&) {
    JST_ERROR("[PLATFORM] Pseudo terminals are not supported on this platform.");
    return Result::ERROR;
}

void PseudoTerminal::stop() {}

bool PseudoTerminal::running() const {
    return false;
}

Result PseudoTerminal::read(std::string&, U64) {
    return Result::ERROR;
}

Result PseudoTerminal::write(std::string_view) {
    return Result::ERROR;
}

Result PseudoTerminal::flush() {
    return Result::ERROR;
}

Result PseudoTerminal::wait(std::optional<std::chrono::milliseconds>, bool) {
    return Result::ERROR;
}

void PseudoTerminal::interrupt() {}

Result PseudoTerminal::resize(U16, U16, U16, U16) {
    return Result::ERROR;
}

#endif

}  // namespace Jetstream::Platform
