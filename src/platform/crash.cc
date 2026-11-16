#include "jetstream/platform.hh"
#include "jetstream/config.hh"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <system_error>
#include <vector>

#if defined(JST_OS_WINDOWS)
#include <csignal>
#include <cstdlib>
#include <cwchar>
#include <exception>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <sddl.h>
#include <tlhelp32.h>
#include <winternl.h>
#undef ERROR
#undef FATAL
#else
#include <cerrno>
#include <ctime>

#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <unistd.h>

#if defined(JST_OS_MAC)
#include <dlfcn.h>
#include <mach-o/dyld.h>
#include <mach-o/loader.h>
#include <sys/sysctl.h>
#include <sys/ucontext.h>
#else
#include <elf.h>
#include <link.h>
#include <sys/utsname.h>
#include <ucontext.h>
#endif
#endif

namespace Jetstream::Platform {

namespace {

[[maybe_unused]] constexpr std::size_t MaxImages = 4096;
constexpr std::size_t MaxFrames = 128;
constexpr std::size_t ReportCapacity = 64 * 1024;
constexpr std::size_t MaxStoredReports = 16;

#if defined(__aarch64__) || defined(_M_ARM64)
constexpr char Architecture[] = "arm64";
#else
constexpr char Architecture[] = "x86_64";
#endif

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

using Frames = std::array<std::uintptr_t, MaxFrames>;

Report report;
std::size_t reportHeaderSize = 0;
bool crashHandlerInstalled = false;
static_assert(std::atomic<std::uintptr_t>::is_always_lock_free);
static_assert(std::atomic<std::size_t>::is_always_lock_free);

void WriteHeader(const std::string& system) {
    report.size = 0;
    report.text("CyberEther Crash Report 1\nversion " JETSTREAM_VERSION_STR "-" JETSTREAM_BUILD_TYPE "\nsystem ");
    report.text(system.c_str());
    report.text("\narchitecture ");
    report.text(Architecture);
    report.text("\n");
    reportHeaderSize = report.size;
}

void WriteFrames(const Frames& frames, const std::size_t frameCount) noexcept {
    for (std::size_t i = 0; i < frameCount; ++i) {
        report.text("frame ");
        report.address(frames[i]);
        report.text("\n");
    }
}

bool Referenced(const Frames& frames, const std::size_t frameCount,
                const std::uintptr_t begin, const std::uintptr_t end) noexcept {
    for (std::size_t i = 0; i < frameCount; ++i) {
        if (frames[i] >= begin && frames[i] < end) {
            return true;
        }
    }
    return false;
}

#if defined(JST_OS_WINDOWS)

constexpr std::size_t MaxImageName = 128;
constexpr std::size_t MaxReportPath = 32768;
constexpr DWORD FatalAppExit = 0x40000015;
constexpr DWORD CodeViewSignature = 0x53445352;
constexpr DWORD ReportWaitMilliseconds = 2000;
constexpr ULONG ImageLoaded = 1;
constexpr ULONG ImageUnloaded = 2;
constexpr std::uint64_t WindowsEpoch = 116444736000000000ull;

struct Image {
    std::atomic<std::uintptr_t> begin{0};
    std::uintptr_t end = 0;
    GUID pdb = {};
    DWORD age = 0;
    DWORD timestamp = 0;
    std::array<wchar_t, MaxImageName> name = {};
};

struct CodeView {
    DWORD signature;
    GUID guid;
    DWORD age;
};

struct LoadedImage {
    ULONG flags;
    const UNICODE_STRING* fullName;
    const UNICODE_STRING* baseName;
    void* base;
    ULONG size;
};

using Notification = void (CALLBACK*)(ULONG, const LoadedImage*, void*);
using RegisterNotification = LONG (NTAPI*)(ULONG, Notification, void*, void**);
using UnregisterNotification = LONG (NTAPI*)(void*);
using VersionFunction = LONG (WINAPI*)(OSVERSIONINFOW*);
using SignalHandler = void (*)(int);

std::array<Image, MaxImages> images;
std::atomic<std::size_t> imageCount{0};
std::atomic<DWORD> crashThread{0};
static_assert(std::atomic<DWORD>::is_always_lock_free);

std::array<wchar_t, MaxReportPath> reportPath = {};
HANDLE reportFile = INVALID_HANDLE_VALUE;
HANDLE reportComplete = nullptr;
void* notificationCookie = nullptr;
UnregisterNotification unregisterNotification = nullptr;
LPTOP_LEVEL_EXCEPTION_FILTER previousExceptionFilter = nullptr;
std::terminate_handler previousTerminate = nullptr;
SignalHandler previousAbort = SIG_DFL;

U64 ProcessId() {
    return GetCurrentProcessId();
}

std::filesystem::path CurrentReport() {
    return std::filesystem::path(reportPath.data());
}

bool ReadIdentity(Image& image, void* base, const DWORD size) noexcept {
    __try {
        const auto* bytes = static_cast<const std::uint8_t*>(base);
        if (size < sizeof(IMAGE_DOS_HEADER) || size < sizeof(IMAGE_NT_HEADERS)) {
            return false;
        }

        const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(bytes);
        if (dos->e_magic != IMAGE_DOS_SIGNATURE || dos->e_lfanew < 0 ||
            static_cast<DWORD>(dos->e_lfanew) > size - sizeof(IMAGE_NT_HEADERS)) {
            return false;
        }

        const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(bytes + dos->e_lfanew);
        if (nt->Signature != IMAGE_NT_SIGNATURE) {
            return false;
        }

        image.end = reinterpret_cast<std::uintptr_t>(base) + size;
        image.timestamp = nt->FileHeader.TimeDateStamp;

        const auto& debug = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_DEBUG];
        if (debug.VirtualAddress > size || debug.Size > size - debug.VirtualAddress) {
            return true;
        }

        const auto* entries = reinterpret_cast<const IMAGE_DEBUG_DIRECTORY*>(bytes + debug.VirtualAddress);
        for (DWORD i = 0; i < debug.Size / sizeof(IMAGE_DEBUG_DIRECTORY); ++i) {
            const auto& entry = entries[i];
            if (entry.Type != IMAGE_DEBUG_TYPE_CODEVIEW || entry.SizeOfData < sizeof(CodeView) ||
                entry.AddressOfRawData > size || sizeof(CodeView) > size - entry.AddressOfRawData) {
                continue;
            }

            const auto* codeView = reinterpret_cast<const CodeView*>(bytes + entry.AddressOfRawData);
            if (codeView->signature == CodeViewSignature) {
                image.pdb = codeView->guid;
                image.age = codeView->age;
                break;
            }
        }
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

void AddImage(void* base, const DWORD size, const wchar_t* name, std::size_t length) noexcept {
    const std::size_t index = imageCount.fetch_add(1, std::memory_order_relaxed);
    if (index >= images.size()) {
        return;
    }

    auto& image = images[index];
    if (!ReadIdentity(image, base, size)) {
        return;
    }

    length = std::min<std::size_t>(length, image.name.size() - 1);
    for (std::size_t i = 0; i < length; ++i) {
        image.name[i] = name[i];
    }
    image.begin.store(reinterpret_cast<std::uintptr_t>(base), std::memory_order_release);
}

void RemoveImage(void* base) noexcept {
    const auto begin = reinterpret_cast<std::uintptr_t>(base);
    const std::size_t count = std::min<std::size_t>(imageCount.load(std::memory_order_relaxed), images.size());
    for (std::size_t i = 0; i < count; ++i) {
        if (images[i].begin.load(std::memory_order_relaxed) == begin) {
            images[i].begin.store(0, std::memory_order_release);
        }
    }
}

void CALLBACK ImageChanged(const ULONG reason, const LoadedImage* image, void*) noexcept {
    if (reason == ImageLoaded && image->baseName != nullptr) {
        AddImage(image->base, image->size, image->baseName->Buffer, image->baseName->Length / sizeof(wchar_t));
    } else if (reason == ImageUnloaded) {
        RemoveImage(image->base);
    }
}

void WriteName(const wchar_t* value) noexcept {
    while (*value != L'\0' && report.size + 4 <= report.data.size()) {
        std::uint32_t point = *value++;
        if (point >= 0xd800 && point <= 0xdbff && *value >= 0xdc00 && *value <= 0xdfff) {
            point = 0x10000 + ((point - 0xd800) << 10) + (*value++ - 0xdc00);
        } else if (point >= 0xd800 && point <= 0xdfff) {
            point = 0xfffd;
        }

        if (point < 0x80) {
            report.data[report.size++] = static_cast<char>(point);
        } else if (point < 0x800) {
            report.data[report.size++] = static_cast<char>(0xc0 | (point >> 6));
            report.data[report.size++] = static_cast<char>(0x80 | (point & 0x3f));
        } else if (point < 0x10000) {
            report.data[report.size++] = static_cast<char>(0xe0 | (point >> 12));
            report.data[report.size++] = static_cast<char>(0x80 | ((point >> 6) & 0x3f));
            report.data[report.size++] = static_cast<char>(0x80 | (point & 0x3f));
        } else {
            report.data[report.size++] = static_cast<char>(0xf0 | (point >> 18));
            report.data[report.size++] = static_cast<char>(0x80 | ((point >> 12) & 0x3f));
            report.data[report.size++] = static_cast<char>(0x80 | ((point >> 6) & 0x3f));
            report.data[report.size++] = static_cast<char>(0x80 | (point & 0x3f));
        }
    }
}

void WriteImages(const Frames& frames, const std::size_t frameCount) noexcept {
    const std::size_t count = std::min<std::size_t>(imageCount.load(std::memory_order_acquire), images.size());
    for (std::size_t i = 0; i < count; ++i) {
        const auto& image = images[i];
        const std::uintptr_t begin = image.begin.load(std::memory_order_acquire);
        if (begin == 0 || !Referenced(frames, frameCount, begin, image.end)) {
            continue;
        }

        report.text("image ");
        report.address(begin);
        report.text(" ");
        report.address(image.end);
        report.text(" ");
        report.hex(image.pdb.Data1, 8);
        report.hex(image.pdb.Data2, 4);
        report.hex(image.pdb.Data3, 4);
        for (const auto byte : image.pdb.Data4) {
            report.hex(byte, 2);
        }
        report.text("-");
        report.decimal(image.age);
        report.text(" ");
        WriteName(image.name.data());
        report.text("\npe ");
        report.address(begin);
        report.text(" ");
        report.hex(image.timestamp, 8);
        report.text("\n");
    }
}

bool Unwind(CONTEXT& context) noexcept {
    __try {
        DWORD64 base = 0;
        if (const auto entry = RtlLookupFunctionEntry(context.Rip, &base, nullptr)) {
            void* data = nullptr;
            DWORD64 frame = 0;
            RtlVirtualUnwind(UNW_FLAG_NHANDLER, base, context.Rip, entry, &context, &data, &frame, nullptr);
            return true;
        }

        SIZE_T read = 0;
        if (!ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(context.Rsp),
                               &context.Rip, sizeof(context.Rip), &read) || read != sizeof(context.Rip)) {
            return false;
        }
        context.Rsp += sizeof(context.Rip);
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

void WriteCrash(const EXCEPTION_RECORD& exception, const CONTEXT& original, const char* reason) noexcept {
    DWORD expected = 0;
    const DWORD thread = GetCurrentThreadId();
    if (!crashThread.compare_exchange_strong(expected, thread, std::memory_order_relaxed)) {
        if (expected != thread) {
            (void)WaitForSingleObject(reportComplete, ReportWaitMilliseconds);
        }
        return;
    }

    CONTEXT context = original;
    ULONG_PTR stackBottom = 0;
    ULONG_PTR stackTop = 0;
    GetCurrentThreadStackLimits(&stackBottom, &stackTop);

    Frames frames = {};
    std::size_t frameCount = 0;
    while (frameCount < MaxFrames && context.Rip != 0) {
        const DWORD64 pc = context.Rip;
        const DWORD64 sp = context.Rsp;
        frames[frameCount++] = static_cast<std::uintptr_t>(pc);
        if (sp < stackBottom || sp >= stackTop || !Unwind(context) || context.Rsp < sp ||
            (context.Rip == pc && context.Rsp == sp)) {
            break;
        }
    }

    const bool memoryFault = (exception.ExceptionCode == EXCEPTION_ACCESS_VIOLATION ||
                              exception.ExceptionCode == EXCEPTION_IN_PAGE_ERROR) &&
                             exception.NumberParameters >= 2;

    FILETIME time = {};
    GetSystemTimeAsFileTime(&time);
    const std::uint64_t ticks = (static_cast<std::uint64_t>(time.dwHighDateTime) << 32) | time.dwLowDateTime;

    report.size = reportHeaderSize;
    report.text("time ");
    report.decimal((ticks - WindowsEpoch) / 10000000);
    report.text("\nreason ");
    report.text(reason);
    report.text("\nexception ");
    report.address(exception.ExceptionCode);
    if (memoryFault) {
        report.text("\naccess ");
        report.decimal(exception.ExceptionInformation[0]);
    }
    report.text("\naddress ");
    report.address(memoryFault ? static_cast<std::uintptr_t>(exception.ExceptionInformation[1])
                               : reinterpret_cast<std::uintptr_t>(exception.ExceptionAddress));
    report.text("\n");
    WriteFrames(frames, frameCount);
    WriteImages(frames, frameCount);

    std::size_t written = 0;
    while (written < report.size) {
        DWORD result = 0;
        if (!WriteFile(reportFile, report.data.data() + written, static_cast<DWORD>(report.size - written),
                       &result, nullptr) || result == 0) {
            break;
        }
        written += result;
    }
    (void)FlushFileBuffers(reportFile);
    (void)SetEvent(reportComplete);
}

LONG WINAPI HandleException(EXCEPTION_POINTERS* exception) noexcept {
    if (exception != nullptr && exception->ExceptionRecord != nullptr && exception->ContextRecord != nullptr) {
        WriteCrash(*exception->ExceptionRecord, *exception->ContextRecord, "unhandled_exception");
    }
    return previousExceptionFilter != nullptr ? previousExceptionFilter(exception) : EXCEPTION_CONTINUE_SEARCH;
}

[[noreturn]] void Terminate(const char* reason) noexcept {
    CONTEXT context = {};
    RtlCaptureContext(&context);
    EXCEPTION_RECORD exception = {};
    exception.ExceptionCode = FatalAppExit;
    exception.ExceptionAddress = reinterpret_cast<void*>(context.Rip);
    WriteCrash(exception, context, reason);
    (void)TerminateProcess(GetCurrentProcess(), FatalAppExit);
    std::_Exit(EXIT_FAILURE);
}

void HandleTerminate() noexcept {
    Terminate("terminate");
}

void HandleAbort(int) noexcept {
    Terminate("abort");
}

void Uninstall() {
    if (!crashHandlerInstalled) {
        return;
    }

    const auto currentFilter = SetUnhandledExceptionFilter(nullptr);
    (void)SetUnhandledExceptionFilter(currentFilter == HandleException ? previousExceptionFilter : currentFilter);
    if (std::get_terminate() == HandleTerminate) {
        (void)std::set_terminate(previousTerminate);
    }
    const auto currentAbort = std::signal(SIGABRT, previousAbort);
    if (currentAbort != HandleAbort && currentAbort != SIG_ERR) {
        (void)std::signal(SIGABRT, currentAbort);
    }
    if (notificationCookie != nullptr && unregisterNotification != nullptr) {
        (void)unregisterNotification(notificationCookie);
    }

    if (reportFile != INVALID_HANDLE_VALUE) {
        (void)CloseHandle(reportFile);
    }
    if (crashThread.load(std::memory_order_relaxed) == 0) {
        (void)DeleteFileW(reportPath.data());
    }
    if (reportComplete != nullptr) {
        (void)CloseHandle(reportComplete);
    }

    notificationCookie = nullptr;
    reportFile = INVALID_HANDLE_VALUE;
    reportComplete = nullptr;
    crashHandlerInstalled = false;
}

std::string SystemString() {
    const auto function = reinterpret_cast<VersionFunction>(
        GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "RtlGetVersion"));
    OSVERSIONINFOW version = {};
    version.dwOSVersionInfoSize = sizeof(version);
    if (function == nullptr || function(&version) != 0) {
        return "Windows unknown";
    }
    return "Windows " + std::to_string(version.dwMajorVersion) + "." + std::to_string(version.dwMinorVersion) +
           "." + std::to_string(version.dwBuildNumber);
}

Result Install(const std::filesystem::path& path) {
    const auto& native = path.native();
    if (native.size() >= reportPath.size()) {
        JST_ERROR("[PLATFORM] Crash report path is too long.");
        return Result::ERROR;
    }
    std::copy(native.begin(), native.end(), reportPath.begin());
    reportPath[native.size()] = L'\0';

    PSECURITY_DESCRIPTOR descriptor = nullptr;
    if (!ConvertStringSecurityDescriptorToSecurityDescriptorW(
            L"D:P(A;;GA;;;OW)", SDDL_REVISION_1, &descriptor, nullptr)) {
        JST_ERROR("[PLATFORM] Failed to build the crash report access rules: {}", GetLastError());
        return Result::ERROR;
    }
    SECURITY_ATTRIBUTES security = {sizeof(SECURITY_ATTRIBUTES), descriptor, FALSE};
    reportFile = CreateFileW(reportPath.data(), GENERIC_WRITE, FILE_SHARE_READ, &security,
                             CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    const DWORD fileError = GetLastError();
    (void)LocalFree(descriptor);
    if (reportFile == INVALID_HANDLE_VALUE) {
        JST_ERROR("[PLATFORM] Failed to open the crash report file: {}", fileError);
        return Result::ERROR;
    }

    reportComplete = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (reportComplete == nullptr) {
        JST_ERROR("[PLATFORM] Failed to create the crash report event: {}", GetLastError());
        (void)CloseHandle(reportFile);
        reportFile = INVALID_HANDLE_VALUE;
        (void)DeleteFileW(reportPath.data());
        return Result::ERROR;
    }

    WriteHeader(SystemString());

    const HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    const auto registerNotification = reinterpret_cast<RegisterNotification>(
        GetProcAddress(ntdll, "LdrRegisterDllNotification"));
    unregisterNotification = reinterpret_cast<UnregisterNotification>(
        GetProcAddress(ntdll, "LdrUnregisterDllNotification"));
    if (registerNotification != nullptr && unregisterNotification != nullptr) {
        (void)registerNotification(0, ImageChanged, nullptr, &notificationCookie);
    }

    const HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, GetCurrentProcessId());
    if (snapshot != INVALID_HANDLE_VALUE) {
        MODULEENTRY32W module = {};
        module.dwSize = sizeof(module);
        if (Module32FirstW(snapshot, &module)) {
            do {
                AddImage(module.modBaseAddr, module.modBaseSize, module.szModule, std::wcslen(module.szModule));
            } while (Module32NextW(snapshot, &module));
        }
        (void)CloseHandle(snapshot);
    }

    previousTerminate = std::set_terminate(HandleTerminate);
    previousAbort = std::signal(SIGABRT, HandleAbort);
    previousExceptionFilter = SetUnhandledExceptionFilter(HandleException);

    crashHandlerInstalled = true;
    if (std::atexit(Uninstall) != 0) {
        JST_ERROR("[PLATFORM] Failed to register the crash handler cleanup.");
        Uninstall();
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

#else

constexpr std::size_t MaxReportPath = 1024;
constexpr std::size_t AlternateStackSize = 128 * 1024;
constexpr std::array<int, 6> CrashSignals = {SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGABRT, SIGTRAP};

struct Registers {
    std::uintptr_t pc = 0;
    std::uintptr_t fp = 0;
    std::uintptr_t lr = 0;
};

#if defined(__aarch64__)
constexpr std::uintptr_t AddressMask = (std::uintptr_t{1} << 48) - 1;
#else
constexpr std::uintptr_t AddressMask = ~std::uintptr_t{0};
#endif

struct FrameRecord {
    std::uintptr_t next = 0;
    std::uintptr_t caller = 0;
};

std::array<char, MaxReportPath> reportPath = {};
std::array<char, AlternateStackSize> alternateStack;
std::array<struct sigaction, CrashSignals.size()> previousActions = {};
std::atomic_flag crashed = ATOMIC_FLAG_INIT;

U64 ProcessId() {
    return static_cast<U64>(getpid());
}

std::filesystem::path CurrentReport() {
    return PathFromUtf8(reportPath.data());
}

#if defined(JST_OS_MAC)

constexpr std::size_t MaxImageName = 64;

struct Image {
    std::atomic<std::uintptr_t> begin{0};
    std::uintptr_t end = 0;
    std::array<std::uint8_t, 16> uuid = {};
    std::array<char, MaxImageName> name = {};
};

std::array<Image, MaxImages> images;
std::atomic<std::size_t> imageCount{0};

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

void WriteImages(const Frames& frames, const std::size_t frameCount) noexcept {
    const std::size_t count = imageCount.load(std::memory_order_acquire);
    for (std::size_t i = 0; i < count; ++i) {
        const auto& image = images[i];
        const std::uintptr_t begin = image.begin.load(std::memory_order_acquire);
        if (begin == 0 || !Referenced(frames, frameCount, begin, image.end)) {
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

Registers ReadRegisters(const ucontext_t& context) noexcept {
#if defined(__aarch64__)
    return {
        .pc = static_cast<std::uintptr_t>(__darwin_arm_thread_state64_get_pc(context.uc_mcontext->__ss)),
        .fp = static_cast<std::uintptr_t>(__darwin_arm_thread_state64_get_fp(context.uc_mcontext->__ss)),
        .lr = static_cast<std::uintptr_t>(__darwin_arm_thread_state64_get_lr(context.uc_mcontext->__ss)),
    };
#else
    return {
        .pc = static_cast<std::uintptr_t>(context.uc_mcontext->__ss.__rip),
        .fp = static_cast<std::uintptr_t>(context.uc_mcontext->__ss.__rbp),
    };
#endif
}

bool ReadFrameRecord(const std::uintptr_t fp, FrameRecord& record) noexcept {
    const pthread_t thread = pthread_self();
    const auto stackTop = reinterpret_cast<std::uintptr_t>(pthread_get_stackaddr_np(thread));
    const std::uintptr_t stackBottom = stackTop - pthread_get_stacksize_np(thread);
    if (fp < stackBottom || fp > stackTop - sizeof(record)) {
        return false;
    }

    std::memcpy(&record, reinterpret_cast<const void*>(fp), sizeof(record));
    return true;
}

std::string SystemValue(const char* name) {
    std::array<char, 64> value = {};
    std::size_t size = value.size() - 1;
    if (sysctlbyname(name, value.data(), &size, nullptr, 0) != 0) {
        return "unknown";
    }
    return value.data();
}

std::string SystemString() {
    return "macOS " + SystemValue("kern.osproductversion") + " " + SystemValue("kern.osversion");
}

Result PrepareImages() {
    _dyld_register_func_for_add_image(AddImage);
    _dyld_register_func_for_remove_image(RemoveImage);
    return Result::SUCCESS;
}

#else

constexpr std::size_t MaxMapLine = 1024;
constexpr std::size_t MapChunk = 4096;

struct Mapping {
    std::uintptr_t begin = 0;
    std::uintptr_t end = 0;
    std::uintptr_t headerEnd = 0;
    std::uintptr_t bias = 0;
    std::uintptr_t imageEnd = 0;
    const ElfW(Phdr)* segments = nullptr;
    std::size_t segmentCount = 0;
    bool header = false;
    bool parsed = false;
    std::array<char, MaxMapLine> path = {};
};

Mapping mapping;
std::array<char, MaxMapLine> mapLine = {};
std::array<char, MapChunk> mapChunk = {};
std::array<int, 2> stackProbe = {-1, -1};

std::size_t AlignNote(const std::size_t size, const std::size_t alignment) noexcept {
    return (size + alignment - 1) & ~(alignment - 1);
}

bool WriteBuildId() noexcept {
    for (std::size_t i = 0; i < mapping.segmentCount; ++i) {
        const auto& segment = mapping.segments[i];
        const auto start = static_cast<std::uintptr_t>(mapping.bias + segment.p_vaddr);
        if (segment.p_type != PT_NOTE || start < mapping.begin || start > mapping.end ||
            segment.p_memsz > mapping.end - start) {
            continue;
        }

        const std::size_t alignment = segment.p_align == 8 ? 8 : 4;
        const auto* cursor = reinterpret_cast<const std::uint8_t*>(start);
        const auto* limit = cursor + segment.p_memsz;
        while (cursor + sizeof(ElfW(Nhdr)) <= limit) {
            const auto* note = reinterpret_cast<const ElfW(Nhdr)*>(cursor);
            const auto* name = cursor + sizeof(ElfW(Nhdr));
            const auto* description = name + AlignNote(note->n_namesz, alignment);
            const auto* next = description + AlignNote(note->n_descsz, alignment);
            if (next > limit || next <= cursor) {
                break;
            }

            if (note->n_type == NT_GNU_BUILD_ID && note->n_namesz == 4 && note->n_descsz > 0 &&
                std::memcmp(name, "GNU", 4) == 0) {
                for (std::size_t j = 0; j < note->n_descsz; ++j) {
                    report.hex(description[j], 2);
                }
                return true;
            }
            cursor = next;
        }
    }
    return false;
}

void ParseMapping() noexcept {
    if (mapping.parsed) {
        return;
    }
    mapping.parsed = true;
    mapping.bias = mapping.begin;
    mapping.imageEnd = 0;
    mapping.segments = nullptr;
    mapping.segmentCount = 0;

    const std::uintptr_t headerSize = mapping.headerEnd - mapping.begin;
    const auto* header = reinterpret_cast<const ElfW(Ehdr)*>(mapping.begin);
    if (!mapping.header || headerSize < sizeof(ElfW(Ehdr)) ||
        std::memcmp(header->e_ident, ELFMAG, SELFMAG) != 0 || header->e_phentsize != sizeof(ElfW(Phdr)) ||
        header->e_phoff > headerSize || header->e_phnum * sizeof(ElfW(Phdr)) > headerSize - header->e_phoff) {
        return;
    }

    const auto* segments = reinterpret_cast<const ElfW(Phdr)*>(mapping.begin + header->e_phoff);
    bool loaded = false;
    for (std::size_t i = 0; i < header->e_phnum; ++i) {
        const auto& segment = segments[i];
        if (segment.p_type != PT_LOAD) {
            continue;
        }

        if (!loaded) {
            mapping.bias = mapping.begin - static_cast<std::uintptr_t>(segment.p_vaddr - segment.p_offset);
            loaded = true;
        }
        mapping.imageEnd = std::max(mapping.imageEnd,
                                    mapping.bias + static_cast<std::uintptr_t>(segment.p_vaddr + segment.p_memsz));
    }
    if (!loaded) {
        return;
    }

    mapping.segments = segments;
    mapping.segmentCount = header->e_phnum;
}

void WriteMapping(const Frames& frames, const std::size_t frameCount) noexcept {
    if (mapping.path[0] == '\0' || !Referenced(frames, frameCount, mapping.begin, mapping.end)) {
        return;
    }

    ParseMapping();
    mapping.end = std::max(mapping.end, mapping.imageEnd);

    const char* separator = std::strrchr(mapping.path.data(), '/');

    report.text("image ");
    report.address(mapping.begin);
    report.text(" ");
    report.address(mapping.end);
    report.text(" ");
    if (!WriteBuildId()) {
        report.text("0");
    }
    report.text(" ");
    report.text(separator != nullptr ? separator + 1 : mapping.path.data());
    report.text("\nbias ");
    report.address(mapping.begin);
    report.text(" ");
    report.address(mapping.bias);
    report.text("\n");
}

bool ReadHex(const char*& cursor, std::uintptr_t& value) noexcept {
    const char* start = cursor;
    value = 0;
    for (;; ++cursor) {
        if (*cursor >= '0' && *cursor <= '9') {
            value = (value << 4) | static_cast<std::uintptr_t>(*cursor - '0');
        } else if (*cursor >= 'a' && *cursor <= 'f') {
            value = (value << 4) | static_cast<std::uintptr_t>(*cursor - 'a' + 10);
        } else {
            break;
        }
    }
    return cursor != start;
}

void SkipField(const char*& cursor) noexcept {
    while (*cursor != ' ' && *cursor != '\0') {
        ++cursor;
    }
    while (*cursor == ' ') {
        ++cursor;
    }
}

void ReadMapping(const char* line, const Frames& frames, const std::size_t frameCount) noexcept {
    const char* cursor = line;
    std::uintptr_t begin = 0;
    std::uintptr_t end = 0;
    std::uintptr_t offset = 0;
    if (!ReadHex(cursor, begin) || *cursor != '-') {
        return;
    }
    ++cursor;
    if (!ReadHex(cursor, end) || *cursor != ' ') {
        return;
    }
    ++cursor;
    const bool readable = *cursor == 'r';
    SkipField(cursor);
    if (!ReadHex(cursor, offset)) {
        return;
    }
    SkipField(cursor);
    SkipField(cursor);
    SkipField(cursor);

    if (*cursor != '/' && std::strcmp(cursor, "[vdso]") != 0) {
        return;
    }
    if (std::strcmp(cursor, mapping.path.data()) == 0) {
        if (offset != 0) {
            mapping.end = std::max(mapping.end, end);
            return;
        }

        ParseMapping();
        if (begin < mapping.imageEnd) {
            mapping.end = std::max(mapping.end, end);
            return;
        }
    }

    WriteMapping(frames, frameCount);
    mapping.begin = begin;
    mapping.end = end;
    mapping.headerEnd = end;
    mapping.header = offset == 0 && readable;
    mapping.parsed = false;
    std::strncpy(mapping.path.data(), cursor, mapping.path.size() - 1);
    mapping.path[mapping.path.size() - 1] = '\0';
}

void WriteImages(const Frames& frames, const std::size_t frameCount) noexcept {
    const int file = open("/proc/self/maps", O_RDONLY | O_CLOEXEC);
    if (file < 0) {
        return;
    }

    mapping.path[0] = '\0';
    std::size_t length = 0;
    for (;;) {
        const ssize_t count = read(file, mapChunk.data(), mapChunk.size());
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            break;
        }

        for (ssize_t i = 0; i < count; ++i) {
            if (mapChunk[i] != '\n') {
                if (length < mapLine.size() - 1) {
                    mapLine[length++] = mapChunk[i];
                }
                continue;
            }

            mapLine[length] = '\0';
            length = 0;
            ReadMapping(mapLine.data(), frames, frameCount);
        }
    }
    WriteMapping(frames, frameCount);
    mapping.path[0] = '\0';
    (void)close(file);
}

Registers ReadRegisters(const ucontext_t& context) noexcept {
#if defined(__aarch64__)
    return {
        .pc = static_cast<std::uintptr_t>(context.uc_mcontext.pc),
        .fp = static_cast<std::uintptr_t>(context.uc_mcontext.regs[29]),
        .lr = static_cast<std::uintptr_t>(context.uc_mcontext.regs[30]),
    };
#else
    return {
        .pc = static_cast<std::uintptr_t>(context.uc_mcontext.gregs[REG_RIP]),
        .fp = static_cast<std::uintptr_t>(context.uc_mcontext.gregs[REG_RBP]),
    };
#endif
}

bool ReadFrameRecord(const std::uintptr_t fp, FrameRecord& record) noexcept {
    const auto size = static_cast<ssize_t>(sizeof(record));
    if (write(stackProbe[1], reinterpret_cast<const void*>(fp), sizeof(record)) != size) {
        return false;
    }
    return read(stackProbe[0], &record, sizeof(record)) == size;
}

std::string SystemString() {
    struct utsname kernel = {};
    if (uname(&kernel) != 0) {
        return "Linux unknown";
    }
    return std::string("Linux ") + kernel.release;
}

Result PrepareImages() {
    if (stackProbe[0] < 0 && pipe2(stackProbe.data(), O_CLOEXEC | O_NONBLOCK) != 0) {
        JST_ERROR("[PLATFORM] Failed to create the crash handler stack probe.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

#endif

void HandleCrash(const int signal, siginfo_t* info, void* context) noexcept {
    for (std::size_t i = 0; i < CrashSignals.size(); ++i) {
        (void)sigaction(CrashSignals[i], &previousActions[i], nullptr);
    }

    if (crashed.test_and_set(std::memory_order_relaxed)) {
        for (;;) {
            (void)pause();
        }
    }

    const Registers registers = ReadRegisters(*static_cast<const ucontext_t*>(context));

    Frames frames = {};
    std::size_t frameCount = 0;
    frames[frameCount++] = registers.pc;
    std::uintptr_t fp = registers.fp;
    while (frameCount < MaxFrames) {
        FrameRecord record;
        if (fp % sizeof(std::uintptr_t) != 0 || !ReadFrameRecord(fp, record) || record.caller == 0) {
            break;
        }

        frames[frameCount++] = record.caller & AddressMask;
        if (record.next <= fp) {
            break;
        }
        fp = record.next;
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
#if defined(__aarch64__)
    report.text("\nlr ");
    report.address(registers.lr & AddressMask);
#endif
    report.text("\n");
    WriteFrames(frames, frameCount);
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

Result Install(const std::filesystem::path& path) {
    const std::string native = PathToUtf8(path);
    if (native.size() >= reportPath.size()) {
        JST_ERROR("[PLATFORM] Crash report path is too long.");
        return Result::ERROR;
    }
    std::memcpy(reportPath.data(), native.c_str(), native.size() + 1);

    WriteHeader(SystemString());
    JST_CHECK(PrepareImages());

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

#endif

void PruneReports(const std::filesystem::path& directory) {
    std::vector<std::filesystem::path> reports;
    std::error_code error;
    std::filesystem::directory_iterator iterator(directory, error);
    for (; !error && iterator != std::filesystem::directory_iterator(); iterator.increment(error)) {
        std::error_code entryError;
        if (iterator->path().extension() != ".crash" || !iterator->is_regular_file(entryError) || entryError) {
            continue;
        }

        if (iterator->file_size(entryError) == 0 && !entryError) {
            std::filesystem::remove(iterator->path(), entryError);
        } else {
            reports.push_back(iterator->path());
        }
    }
    if (reports.size() <= MaxStoredReports) {
        return;
    }

    std::sort(reports.begin(), reports.end());
    reports.resize(reports.size() - MaxStoredReports);
    for (const auto& report : reports) {
        std::filesystem::remove(report, error);
    }
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
    PruneReports(directory);

    const auto now = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    const auto path = std::filesystem::absolute(
        directory / (std::to_string(now) + "-" + std::to_string(ProcessId()) + ".crash"), error);
    if (error) {
        JST_ERROR("[PLATFORM] Failed to resolve the crash report path: {}", error.message());
        return Result::ERROR;
    }

    return Install(path);
}

Result PendingCrashReports(std::vector<std::filesystem::path>& reports) {
    reports.clear();
    if (!crashHandlerInstalled) {
        return Result::SUCCESS;
    }

    const auto current = CurrentReport();
    std::error_code error;
    std::filesystem::directory_iterator iterator(current.parent_path(), error);
    for (; !error && iterator != std::filesystem::directory_iterator(); iterator.increment(error)) {
        const auto& path = iterator->path();
        if (path.extension() != ".crash" || path == current) {
            continue;
        }

        std::error_code entryError;
        if (!iterator->is_symlink(entryError) && iterator->is_regular_file(entryError) &&
            iterator->file_size(entryError) > 0 && !entryError) {
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
