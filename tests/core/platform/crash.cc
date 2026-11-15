#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "jetstream/platform.hh"

#if defined(JST_OS_WINDOWS)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <dbghelp.h>
#include <aclapi.h>
#undef ERROR
#undef FATAL
#else
#include <dlfcn.h>
#include <mach-o/dyld.h>
#include <mach-o/loader.h>
#include <signal.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;
#endif

using namespace Jetstream;

namespace {

using Compute = void (*)();

volatile U32 completed = 0;

#if defined(JST_OS_WINDOWS)
__declspec(noinline)
#else
__attribute__((noinline))
#endif
void ComputeChildModule(Compute compute) {
    compute();
    completed = completed + 1;
}

int Fixture(const std::string& mode, const std::filesystem::path& directory) {
#if defined(JST_OS_WINDOWS)
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
#endif

    if (mode == "recover") {
        const auto blocked = directory / "blocked";
        std::ofstream(blocked) << "not a directory";
        if (Platform::InstallCrashHandler(blocked / "crashes") != Result::ERROR) {
            return 10;
        }
    }
    if (Platform::InstallCrashHandler(directory) != Result::SUCCESS ||
        Platform::InstallCrashHandler(directory) != Result::SUCCESS) {
        return 11;
    }
    if (mode == "normal" || mode == "recover") {
        return 0;
    }

    if (mode == "pending") {
        std::vector<std::filesystem::path> reports;
        if (Platform::PendingCrashReports(reports) != Result::SUCCESS) {
            return 12;
        }
        std::ofstream output(directory / "pending.txt", std::ios::binary);
        for (const auto& path : reports) {
            output << Platform::PathToUtf8(path.filename()) << '\n';
        }
        return 0;
    }

    std::string error;
    void* library = Platform::OpenDynamicLibrary(JETSTREAM_TEST_CRASH_MODULE,
                                                 Platform::DynamicLibraryVisibility::Local,
                                                 error);
    if (library == nullptr) {
        return 13;
    }
    const auto compute = reinterpret_cast<Compute>(
        Platform::LoadDynamicLibrarySymbol(library, "CrashModuleCompute", error));
    if (compute == nullptr) {
        return 14;
    }

    if (mode == "worker") {
        std::thread worker([compute] { ComputeChildModule(compute); });
        worker.join();
    } else {
        ComputeChildModule(compute);
    }
    return 15;
}

struct Directory {
    std::filesystem::path path;

    Directory() {
        static std::atomic<U64> sequence{0};
        path = std::filesystem::temp_directory_path() /
               Platform::PathFromUtf8("cyberether-crash-" + std::to_string(sequence.fetch_add(1)) + "-" +
                                      std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id())) +
                                      "-\xe4\xb8\x96\xe7\x95\x8c space");
        std::error_code error;
        std::filesystem::remove_all(path, error);
        REQUIRE(std::filesystem::create_directory(path));
    }

    ~Directory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }

    std::vector<std::filesystem::path> reports() const {
        std::vector<std::filesystem::path> result;
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".crash") {
                result.push_back(entry.path());
            }
        }
        return result;
    }
};

struct Exit {
    bool crashed = false;
    I64 code = 0;
};

#if defined(JST_OS_WINDOWS)

Exit Run(const std::filesystem::path& directory, const std::string& mode) {
    std::array<wchar_t, 32768> executable{};
    REQUIRE(GetModuleFileNameW(nullptr, executable.data(), static_cast<DWORD>(executable.size())) > 0);
    std::wstring command = L"\"" + std::wstring(executable.data()) + L"\" --crash-fixture " +
                           Platform::PathFromUtf8(mode).native() + L" \"" + directory.native() + L"\"";

    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    REQUIRE(CreateProcessW(executable.data(), command.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW,
                           nullptr, nullptr, &startup, &process));
    const DWORD wait = WaitForSingleObject(process.hProcess, 15000);
    if (wait != WAIT_OBJECT_0) {
        TerminateProcess(process.hProcess, 127);
        WaitForSingleObject(process.hProcess, 5000);
    }

    DWORD code = STILL_ACTIVE;
    GetExitCodeProcess(process.hProcess, &code);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    REQUIRE(wait == WAIT_OBJECT_0);
    return {.crashed = code == EXCEPTION_ACCESS_VIOLATION, .code = static_cast<I64>(code)};
}

#else

Exit Run(const std::filesystem::path& directory, const std::string& mode) {
    std::array<char, 4096> executable{};
    U32 size = static_cast<U32>(executable.size());
    REQUIRE(_NSGetExecutablePath(executable.data(), &size) == 0);

    std::string fixture = "--crash-fixture";
    std::string modeArgument = mode;
    std::string directoryArgument = Platform::PathToUtf8(directory);
    std::array<char*, 5> arguments = {executable.data(), fixture.data(), modeArgument.data(),
                                      directoryArgument.data(), nullptr};

    pid_t process = 0;
    REQUIRE(posix_spawn(&process, executable.data(), nullptr, nullptr, arguments.data(), environ) == 0);
    int status = 0;
    REQUIRE(waitpid(process, &status, 0) == process);
    if (WIFSIGNALED(status)) {
        const int signal = WTERMSIG(status);
        return {.crashed = signal == SIGSEGV || signal == SIGBUS, .code = signal};
    }
    return {.crashed = false, .code = WEXITSTATUS(status)};
}

#endif

std::string Read(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

struct SavedReport {
    struct Image {
        U64 begin = 0;
        U64 end = 0;
        std::string identity;
        std::string name;
    };

    std::string text;
    U64 lr = 0;
    std::vector<U64> frames;
    std::vector<Image> images;

    explicit SavedReport(const std::filesystem::path& path) : text(Read(path)) {
        std::istringstream input(text);
        for (std::string line; std::getline(input, line);) {
            std::istringstream fields(line);
            std::string key;
            fields >> key;
            if (key == "frame") {
                U64 address = 0;
                fields >> std::hex >> address;
                frames.push_back(address);
            } else if (key == "lr") {
                fields >> std::hex >> lr;
            } else if (key == "image") {
                Image image;
                fields >> std::hex >> image.begin >> image.end >> image.identity;
                fields >> std::ws;
                std::getline(fields, image.name);
                images.push_back(std::move(image));
            }
        }
    }
};

#if defined(JST_OS_WINDOWS)

constexpr const char* FaultLines = "exception 0xc0000005\naccess 1\naddress 0x1\n";

struct Symbols {
    HANDLE process = GetCurrentProcess();

    Symbols() {
        SymSetOptions(SYMOPT_UNDNAME | SYMOPT_LOAD_LINES | SYMOPT_FAIL_CRITICAL_ERRORS |
                      SYMOPT_NO_PROMPTS | SYMOPT_EXACT_SYMBOLS);
        REQUIRE(SymInitialize(process, nullptr, FALSE));
    }

    ~Symbols() {
        SymCleanup(process);
    }
};

std::string Identity(const GUID& guid, const DWORD age) {
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(8) << guid.Data1 << std::setw(4) << guid.Data2
           << std::setw(4) << guid.Data3;
    for (const auto byte : guid.Data4) {
        output << std::setw(2) << static_cast<unsigned>(byte);
    }
    output << '-' << std::dec << age;
    return output.str();
}

void CheckSymbols(const SavedReport& saved, const SavedReport::Image& module,
                  const std::filesystem::path& modulePath) {
    Symbols symbols;
    REQUIRE(SymLoadModuleExW(symbols.process, nullptr, modulePath.c_str(), nullptr, module.begin,
                             static_cast<DWORD>(module.end - module.begin), nullptr, 0) == module.begin);
    IMAGEHLP_MODULEW64 info{};
    info.SizeOfStruct = sizeof(info);
    REQUIRE(SymGetModuleInfoW64(symbols.process, module.begin, &info));
    CHECK(info.SymType == SymPdb);
    CHECK(module.identity == Identity(info.PdbSig70, info.PdbAge));

    alignas(SYMBOL_INFO) std::array<char, sizeof(SYMBOL_INFO) + MAX_SYM_NAME> storage{};
    auto* symbol = reinterpret_cast<SYMBOL_INFO*>(storage.data());
    symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
    symbol->MaxNameLen = MAX_SYM_NAME;
    DWORD64 displacement = 0;
    REQUIRE(SymFromAddr(symbols.process, saved.frames.front(), &displacement, symbol));
    CHECK(std::string(symbol->Name).find("CrashModuleFault") != std::string::npos);

    IMAGEHLP_LINE64 line{};
    line.SizeOfStruct = sizeof(line);
    DWORD offset = 0;
    REQUIRE(SymGetLineFromAddr64(symbols.process, saved.frames.front(), &offset, &line));
    CHECK(std::string(line.FileName).find("crash_module.cc") != std::string::npos);
    CHECK(line.LineNumber > 0);

    REQUIRE(saved.frames.size() >= 2);
    REQUIRE(SymFromAddr(symbols.process, saved.frames[1], &displacement, symbol));
    CHECK(std::string(symbol->Name).find("CrashModuleCompute") != std::string::npos);
}

void CheckPrivate(const std::filesystem::path& report) {
    PSECURITY_DESCRIPTOR descriptor = nullptr;
    PACL acl = nullptr;
    REQUIRE(GetNamedSecurityInfoW(report.c_str(), SE_FILE_OBJECT, DACL_SECURITY_INFORMATION,
                                  nullptr, nullptr, &acl, nullptr, &descriptor) == ERROR_SUCCESS);
    SECURITY_DESCRIPTOR_CONTROL control = 0;
    DWORD revision = 0;
    CHECK(GetSecurityDescriptorControl(descriptor, &control, &revision));
    CHECK((control & SE_DACL_PROTECTED) != 0);
    CHECK((acl != nullptr && acl->AceCount == 1));
    LocalFree(descriptor);
}

#else

constexpr const char* FaultLines = "\naddress 0x1\n";

std::string Symbol(const void* address) {
    Dl_info info = {};
    if (dladdr(address, &info) == 0 || info.dli_sname == nullptr) {
        return {};
    }
    return info.dli_sname;
}

void CheckSymbols(const SavedReport& saved, const SavedReport::Image& module,
                  const std::filesystem::path& modulePath) {
    void* library = dlopen(Platform::PathToUtf8(modulePath).c_str(), RTLD_NOW | RTLD_LOCAL);
    REQUIRE(library != nullptr);
    Dl_info info = {};
    REQUIRE(dladdr(dlsym(library, "CrashModuleCompute"), &info) != 0);
    const auto* base = static_cast<const char*>(info.dli_fbase);

    std::ostringstream identity;
    const auto* header = static_cast<const mach_header_64*>(info.dli_fbase);
    const auto* command = reinterpret_cast<const load_command*>(header + 1);
    for (U32 i = 0; i < header->ncmds; ++i) {
        if (command->cmd == LC_UUID) {
            for (const auto byte : reinterpret_cast<const uuid_command*>(command)->uuid) {
                identity << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned>(byte);
            }
        }
        command = reinterpret_cast<const load_command*>(reinterpret_cast<const char*>(command) + command->cmdsize);
    }
    CHECK(module.identity == identity.str());

    CHECK(Symbol(base + (saved.frames.front() - module.begin)).find("CrashModuleFault") != std::string::npos);
#if defined(__arm64__)
    REQUIRE(saved.lr >= module.begin);
    REQUIRE(saved.lr < module.end);
    CHECK(Symbol(base + (saved.lr - module.begin)) == "CrashModuleCompute");
#endif
    dlclose(library);
}

void CheckPrivate(const std::filesystem::path& report) {
    struct stat status = {};
    REQUIRE(stat(Platform::PathToUtf8(report).c_str(), &status) == 0);
    CHECK((status.st_mode & 0777) == 0600);
}

#endif

}  // namespace

TEST_CASE("Crash handler leaves no report on a healthy exit and recovers from directory errors",
          "[core][platform][crash_report]") {
    const auto mode = GENERATE(std::string("normal"), std::string("recover"));
    const Directory directory;
    const auto exit = Run(directory.path, mode);
    CHECK_FALSE(exit.crashed);
    CHECK(exit.code == 0);
    CHECK(directory.reports().empty());
}

TEST_CASE("Crash report discovery ignores empty files and entries that are not reports",
          "[core][platform][crash_report]") {
    const Directory directory;
    std::ofstream(directory.path / "valid.crash", std::ios::binary) << "CyberEther Crash Report 1\n";
    std::ofstream(directory.path / "empty.crash");
    std::ofstream(directory.path / "other.txt") << "not a crash";
    std::filesystem::create_directory(directory.path / "directory.crash");

    CHECK(Run(directory.path, "pending").code == 0);
    CHECK(Read(directory.path / "pending.txt") == "valid.crash\n");
    CHECK(Read(directory.path / "valid.crash") == "CyberEther Crash Report 1\n");
    CHECK_FALSE(std::filesystem::exists(directory.path / "empty.crash"));
    CHECK(std::filesystem::exists(directory.path / "other.txt"));
}

TEST_CASE("Crash handler keeps only the newest stored reports", "[core][platform][crash_report]") {
    const Directory directory;
    for (U64 i = 0; i < 20; ++i) {
        std::ofstream(directory.path / (std::to_string(1000000000 + i) + "-1.crash"), std::ios::binary)
            << "CyberEther Crash Report 1\n";
    }

    CHECK(Run(directory.path, "pending").code == 0);
    std::string expected;
    for (U64 i = 4; i < 20; ++i) {
        expected += std::to_string(1000000000 + i) + "-1.crash\n";
    }
    CHECK(Read(directory.path / "pending.txt") == expected);
    CHECK(directory.reports().size() == 16);
}

TEST_CASE("Module faults save the faulting thread stack and the matching image identity",
          "[core][platform][crash_report]") {
    const auto mode = GENERATE(std::string("main"), std::string("worker"));
    const Directory directory;
    REQUIRE(Run(directory.path, mode).crashed);

    const auto paths = directory.reports();
    REQUIRE(paths.size() == 1);
    const SavedReport saved(paths.front());
    CHECK(saved.text.starts_with("CyberEther Crash Report 1\n"));
    CHECK(saved.text.find("\nsystem ") != std::string::npos);
    CHECK(saved.text.find(FaultLines) != std::string::npos);
    CHECK(saved.text.size() < 64 * 1024);
    CHECK(saved.text.find(Platform::PathToUtf8(directory.path)) == std::string::npos);
    REQUIRE(saved.frames.size() >= 3);

    const auto modulePath = Platform::PathFromUtf8(JETSTREAM_TEST_CRASH_MODULE);
    const auto name = Platform::PathToUtf8(modulePath.filename());
    const auto module = std::find_if(saved.images.begin(), saved.images.end(), [&](const auto& image) {
        return image.name == name;
    });
    REQUIRE(module != saved.images.end());
    CHECK(saved.frames.front() >= module->begin);
    CHECK(saved.frames.front() < module->end);

    CheckSymbols(saved, *module, modulePath);
    CheckPrivate(paths.front());

    CHECK(Run(directory.path, "pending").code == 0);
    CHECK(Read(directory.path / "pending.txt") == Platform::PathToUtf8(paths.front().filename()) + "\n");
    CHECK(Read(paths.front()) == saved.text);
}

#if defined(JST_OS_WINDOWS)
int wmain(int argc, wchar_t** argv) {
    if (argc == 4 && std::wstring_view(argv[1]) == L"--crash-fixture") {
        return Fixture(Platform::PathToUtf8(std::filesystem::path(argv[2])), argv[3]);
    }

    std::vector<std::string> arguments;
    for (int i = 0; i < argc; ++i) {
        arguments.push_back(Platform::PathToUtf8(std::filesystem::path(argv[i])));
    }
    std::vector<char*> pointers;
    for (auto& argument : arguments) {
        pointers.push_back(argument.data());
    }
    return Catch::Session().run(argc, pointers.data());
}
#else
int main(int argc, char* argv[]) {
    if (argc == 4 && std::string_view(argv[1]) == "--crash-fixture") {
        return Fixture(argv[2], Platform::PathFromUtf8(argv[3]));
    }

    return Catch::Session().run(argc, argv);
}
#endif
