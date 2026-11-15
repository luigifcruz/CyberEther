#include "jetstream/platform.hh"
#include "jetstream/config.hh"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <string>
#include <system_error>
#include <vector>

#include <dlfcn.h>
#include <fcntl.h>
#include <mach-o/dyld.h>
#include <mach-o/loader.h>
#include <pthread.h>
#include <signal.h>
#include <sys/sysctl.h>
#include <sys/ucontext.h>
#include <unistd.h>

namespace Jetstream::Platform {

namespace {

constexpr std::size_t MaxImages = 4096;
constexpr std::size_t MaxFrames = 128;
constexpr std::size_t MaxImageName = 64;
constexpr std::size_t MaxReportPath = 1024;
constexpr std::size_t ReportCapacity = 64 * 1024;
constexpr std::size_t AlternateStackSize = 128 * 1024;
constexpr std::array<int, 6> CrashSignals = {SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGABRT, SIGTRAP};

#if defined(__arm64__)
constexpr char Architecture[] = "arm64";
#else
constexpr char Architecture[] = "x86_64";
#endif

struct Image {
    std::atomic<std::uintptr_t> begin{0};
    std::uintptr_t end = 0;
    std::array<std::uint8_t, 16> uuid = {};
    std::array<char, MaxImageName> name = {};
};

struct Report {
    std::array<char, ReportCapacity> data = {};
    std::size_t size = 0;

    void text(const char* value) noexcept {
        while (*value != '\0' && size < data.size()) {
            data[size++] = *value++;
        }
    }

    void decimal(std::uint64_t value) noexcept {
        std::array<char, 20> digits = {};
        std::size_t count = 0;
        do {
            digits[count++] = static_cast<char>('0' + (value % 10));
            value /= 10;
        } while (value != 0);
        while (count > 0 && size < data.size()) {
            data[size++] = digits[--count];
        }
    }

    void hex(std::uint64_t value, const std::size_t width = 1) noexcept {
        constexpr char Digits[] = "0123456789abcdef";
        std::array<char, 16> digits = {};
        std::size_t count = 0;
        do {
            digits[count++] = Digits[value & 0xf];
            value >>= 4;
        } while (value != 0 || count < width);
        while (count > 0 && size < data.size()) {
            data[size++] = digits[--count];
        }
    }

    void address(const std::uintptr_t value) noexcept {
        text("0x");
        hex(value);
    }
};

std::array<Image, MaxImages> images;
std::atomic<std::size_t> imageCount{0};
static_assert(std::atomic<std::uintptr_t>::is_always_lock_free);
static_assert(std::atomic<std::size_t>::is_always_lock_free);

Report report;
std::size_t reportHeaderSize = 0;
std::array<char, MaxReportPath> reportPath = {};
std::array<char, AlternateStackSize> alternateStack;
std::array<struct sigaction, CrashSignals.size()> previousActions = {};
std::atomic_flag crashed = ATOMIC_FLAG_INIT;
bool crashHandlerInstalled = false;

void AddImage(const mach_header* header, intptr_t) {
    const auto* base = reinterpret_cast<const mach_header_64*>(header);
    if (base->magic != MH_MAGIC_64) {
        return;
    }

    std::uintptr_t size = 0;
    std::array<std::uint8_t, 16> uuid = {};
    const auto* command = reinterpret_cast<const load_command*>(base + 1);
    for (std::uint32_t i = 0; i < base->ncmds; ++i) {
        if (command->cmd == LC_UUID) {
            const auto* uuidCommand = reinterpret_cast<const uuid_command*>(command);
            std::memcpy(uuid.data(), uuidCommand->uuid, uuid.size());
        } else if (command->cmd == LC_SEGMENT_64) {
            const auto* segment = reinterpret_cast<const segment_command_64*>(command);
            if (std::strncmp(segment->segname, SEG_TEXT, sizeof(segment->segname)) == 0) {
                size = static_cast<std::uintptr_t>(segment->vmsize);
            }
        }
        command = reinterpret_cast<const load_command*>(
            reinterpret_cast<const char*>(command) + command->cmdsize);
    }
    if (size == 0) {
        return;
    }

    const char* name = "?";
    Dl_info info = {};
    if (dladdr(header, &info) != 0 && info.dli_fname != nullptr) {
        const char* separator = std::strrchr(info.dli_fname, '/');
        name = separator != nullptr ? separator + 1 : info.dli_fname;
    }

    const std::size_t count = imageCount.load(std::memory_order_relaxed);
    if (count == MaxImages) {
        return;
    }

    auto& image = images[count];
    const auto begin = reinterpret_cast<std::uintptr_t>(header);
    image.end = begin + size;
    image.uuid = uuid;
    std::strncpy(image.name.data(), name, image.name.size() - 1);
    image.begin.store(begin, std::memory_order_release);
    imageCount.store(count + 1, std::memory_order_release);
}

void RemoveImage(const mach_header* header, intptr_t) {
    const auto begin = reinterpret_cast<std::uintptr_t>(header);
    const std::size_t count = imageCount.load(std::memory_order_relaxed);
    for (std::size_t i = 0; i < count; ++i) {
        if (images[i].begin.load(std::memory_order_relaxed) == begin) {
            images[i].begin.store(0, std::memory_order_release);
            return;
        }
    }
}

void WriteImages(const std::array<std::uintptr_t, MaxFrames>& frames, const std::size_t frameCount) noexcept {
    const std::size_t count = imageCount.load(std::memory_order_acquire);
    for (std::size_t i = 0; i < count; ++i) {
        const auto& image = images[i];
        const std::uintptr_t begin = image.begin.load(std::memory_order_acquire);
        if (begin == 0) {
            continue;
        }

        bool referenced = false;
        for (std::size_t j = 0; j < frameCount && !referenced; ++j) {
            referenced = frames[j] >= begin && frames[j] < image.end;
        }
        if (!referenced) {
            continue;
        }

        report.text("image ");
        report.address(begin);
        report.text(" ");
        report.address(image.end);
        report.text(" ");
        for (const auto byte : image.uuid) {
            report.hex(byte, 2);
        }
        report.text(" ");
        report.text(image.name.data());
        report.text("\n");
    }
}

void HandleCrash(const int signal, siginfo_t* info, void* context) noexcept {
    for (std::size_t i = 0; i < CrashSignals.size(); ++i) {
        (void)sigaction(CrashSignals[i], &previousActions[i], nullptr);
    }

    if (crashed.test_and_set(std::memory_order_relaxed)) {
        for (;;) {
            (void)pause();
        }
    }

    const auto* userContext = static_cast<const ucontext_t*>(context);
#if defined(__arm64__)
    const auto pc = static_cast<std::uintptr_t>(__darwin_arm_thread_state64_get_pc(userContext->uc_mcontext->__ss));
    const auto lr = static_cast<std::uintptr_t>(__darwin_arm_thread_state64_get_lr(userContext->uc_mcontext->__ss));
    auto fp = static_cast<std::uintptr_t>(__darwin_arm_thread_state64_get_fp(userContext->uc_mcontext->__ss));
#else
    const auto pc = static_cast<std::uintptr_t>(userContext->uc_mcontext->__ss.__rip);
    auto fp = static_cast<std::uintptr_t>(userContext->uc_mcontext->__ss.__rbp);
#endif

    const pthread_t thread = pthread_self();
    const auto stackTop = reinterpret_cast<std::uintptr_t>(pthread_get_stackaddr_np(thread));
    const std::uintptr_t stackBottom = stackTop - pthread_get_stacksize_np(thread);

    std::array<std::uintptr_t, MaxFrames> frames = {};
    std::size_t frameCount = 0;
    frames[frameCount++] = pc;
    while (frameCount < MaxFrames) {
        if (fp < stackBottom || fp > stackTop - 2 * sizeof(std::uintptr_t) || fp % sizeof(std::uintptr_t) != 0) {
            break;
        }

        const auto* record = reinterpret_cast<const std::uintptr_t*>(fp);
        const std::uintptr_t next = record[0];
        const std::uintptr_t caller = record[1];
        if (caller == 0) {
            break;
        }

        frames[frameCount++] = caller;
        if (next <= fp) {
            break;
        }
        fp = next;
    }

    report.size = reportHeaderSize;
    report.text("time ");
    report.decimal(static_cast<std::uint64_t>(time(nullptr)));
    report.text("\nsignal ");
    report.decimal(static_cast<std::uint64_t>(signal));
    report.text("\ncode ");
    report.decimal(static_cast<std::uint64_t>(info->si_code));
    report.text("\naddress ");
    report.address(reinterpret_cast<std::uintptr_t>(info->si_addr));
#if defined(__arm64__)
    report.text("\nlr ");
    report.address(lr);
#endif
    report.text("\n");
    for (std::size_t i = 0; i < frameCount; ++i) {
        report.text("frame ");
        report.address(frames[i]);
        report.text("\n");
    }
    WriteImages(frames, frameCount);

    const int file = open(reportPath.data(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (file >= 0) {
        std::size_t written = 0;
        while (written < report.size) {
            const ssize_t result = write(file, report.data.data() + written, report.size - written);
            if (result <= 0) {
                break;
            }
            written += static_cast<std::size_t>(result);
        }
        (void)close(file);
    }

    if (signal == SIGABRT) {
        (void)raise(signal);
    }
}

std::string SystemString(const char* name) {
    std::array<char, 64> value = {};
    std::size_t size = value.size() - 1;
    if (sysctlbyname(name, value.data(), &size, nullptr, 0) != 0) {
        return "unknown";
    }
    return value.data();
}

}  // namespace

Result InstallCrashHandler(const std::filesystem::path& directory) {
    if (crashHandlerInstalled) {
        return Result::SUCCESS;
    }

    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error) {
        JST_ERROR("[PLATFORM] Failed to create the crash report directory: {}", error.message());
        return Result::ERROR;
    }

    const auto now = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    const std::string path = PathToUtf8(
        directory / (std::to_string(now) + "-" + std::to_string(getpid()) + ".crash"));
    if (path.size() >= reportPath.size()) {
        JST_ERROR("[PLATFORM] Crash report path is too long.");
        return Result::ERROR;
    }
    std::memcpy(reportPath.data(), path.c_str(), path.size() + 1);

    report.size = 0;
    report.text("CyberEther Crash Report 1\nversion " JETSTREAM_VERSION_STR "-" JETSTREAM_BUILD_TYPE "\nsystem macOS ");
    report.text(SystemString("kern.osproductversion").c_str());
    report.text(" ");
    report.text(SystemString("kern.osversion").c_str());
    report.text("\narchitecture ");
    report.text(Architecture);
    report.text("\n");
    reportHeaderSize = report.size;

    _dyld_register_func_for_add_image(AddImage);
    _dyld_register_func_for_remove_image(RemoveImage);

    stack_t currentStack = {};
    if (sigaltstack(nullptr, &currentStack) == 0 && (currentStack.ss_flags & SS_DISABLE) != 0) {
        stack_t stack = {};
        stack.ss_sp = alternateStack.data();
        stack.ss_size = alternateStack.size();
        (void)sigaltstack(&stack, nullptr);
    }

    struct sigaction action = {};
    action.sa_sigaction = HandleCrash;
    action.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&action.sa_mask);
    for (std::size_t i = 0; i < CrashSignals.size(); ++i) {
        if (sigaction(CrashSignals[i], &action, &previousActions[i]) != 0) {
            for (std::size_t j = 0; j < i; ++j) {
                (void)sigaction(CrashSignals[j], &previousActions[j], nullptr);
            }
            JST_ERROR("[PLATFORM] Failed to install the crash signal handlers.");
            return Result::ERROR;
        }
    }

    crashHandlerInstalled = true;
    return Result::SUCCESS;
}

Result PendingCrashReports(std::vector<std::filesystem::path>& reports) {
    reports.clear();
    if (!crashHandlerInstalled) {
        return Result::SUCCESS;
    }

    const auto current = PathFromUtf8(reportPath.data());
    std::error_code error;
    std::filesystem::directory_iterator iterator(current.parent_path(), error);
    for (; !error && iterator != std::filesystem::directory_iterator(); iterator.increment(error)) {
        const auto& path = iterator->path();
        if (path.extension() == ".crash" && path != current) {
            reports.push_back(path);
        }
    }
    if (error) {
        JST_ERROR("[PLATFORM] Failed to list the crash report directory: {}", error.message());
        return Result::ERROR;
    }

    std::sort(reports.begin(), reports.end());
    return Result::SUCCESS;
}

}  // namespace Jetstream::Platform
