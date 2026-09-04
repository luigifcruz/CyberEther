#include "runtime/python/dependencies/base.hh"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "jetstream/config.hh"
#include "jetstream/logger.hh"
#include "jetstream/platform.hh"
#include "runtime/python/bridge/cpython/base.hh"

#if defined(JETSTREAM_LOADER_OPENSSL_AVAILABLE)
#include <openssl/sha.h>
#endif

namespace Jetstream {

namespace {

constexpr U64 kInstallTimeoutMilliseconds = 600000;

struct TemporaryDirectory {
    ~TemporaryDirectory() {
        if (!path.empty()) {
            std::error_code ec;
            std::filesystem::remove_all(path, ec);
        }
    }

    std::filesystem::path path;
};

Result WriteFile(const std::filesystem::path& path, const std::string& contents) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Cannot open '{}'.",
                  Platform::PathToUtf8(path));
        return Result::ERROR;
    }
    file.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    file.close();
    if (!file) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Cannot write '{}'.",
                  Platform::PathToUtf8(path));
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

bool ReadFile(const std::filesystem::path& path, std::string& contents) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }
    contents.assign(std::istreambuf_iterator<char>(file),
                    std::istreambuf_iterator<char>());
    return static_cast<bool>(file);
}

void AppendKeyPart(std::string& payload, const std::string& value) {
    payload += std::to_string(value.size()) + ":" + value;
}

Result AppendFileIdentity(std::string& payload, const std::string& value) {
    std::error_code ec;
    const auto selectedPath = std::filesystem::absolute(
        Platform::PathFromUtf8(value), ec).lexically_normal();
    if (ec) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Cannot identify runtime file '{}'.",
                  value);
        return Result::ERROR;
    }

    const auto path = std::filesystem::weakly_canonical(
        selectedPath, ec);
    if (ec || !std::filesystem::is_regular_file(path, ec) || ec) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Cannot identify runtime file '{}'.",
                  value);
        return Result::ERROR;
    }

    const auto size = std::filesystem::file_size(path, ec);
    if (ec) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Cannot inspect runtime file '{}'.",
                  value);
        return Result::ERROR;
    }
    const auto modified = std::filesystem::last_write_time(path, ec);
    if (ec) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Cannot inspect runtime file '{}'.",
                  value);
        return Result::ERROR;
    }

    AppendKeyPart(payload, Platform::PathToUtf8(selectedPath));
    AppendKeyPart(payload, Platform::PathToUtf8(path));
    AppendKeyPart(payload, std::to_string(size));
    AppendKeyPart(payload, std::to_string(static_cast<long long>(
                               modified.time_since_epoch().count())));
    return Result::SUCCESS;
}

Result EnvironmentKey(const std::vector<std::string>& requirements,
                      std::string& key) {
#if defined(JETSTREAM_LOADER_OPENSSL_AVAILABLE)
    std::string payload;
    AppendKeyPart(payload, "1");
    JST_CHECK(AppendFileIdentity(payload, CPython::Py_ProgramPath()));
    JST_CHECK(AppendFileIdentity(payload, CPython::Py_LibraryPath()));
    for (const auto& requirement : requirements) {
        AppendKeyPart(payload, requirement);
    }

    unsigned char digest[SHA256_DIGEST_LENGTH];
    if (!SHA256(reinterpret_cast<const unsigned char*>(payload.data()),
                payload.size(), digest)) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Cannot hash the Python environment.");
        return Result::ERROR;
    }

    key = jst::fmt::format("{:02x}", jst::fmt::join(digest, ""));
    return Result::SUCCESS;
#else
    (void)requirements;
    (void)key;
    JST_ERROR("[RUNTIME_CONTEXT_PYTHON] OpenSSL is unavailable.");
    return Result::ERROR;
#endif
}

std::string RequirementsText(const std::vector<std::string>& requirements) {
    std::string contents;
    for (const auto& requirement : requirements) {
        contents += requirement + "\n";
    }
    return contents;
}

bool IsCachedEnvironment(const std::filesystem::path& path,
                         const std::string& requirements, std::string& sitePackages) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path / "complete", ec) || ec ||
        !std::filesystem::is_directory(path / "site-packages", ec) || ec) {
        return false;
    }

    std::string cachedRequirements;
    if (!ReadFile(path / "requirements.txt", cachedRequirements) ||
        cachedRequirements != requirements) {
        return false;
    }
    sitePackages = Platform::PathToUtf8(path / "site-packages");
    return true;
}

Result CreateTemporaryDirectory(const std::filesystem::path& root,
                                const std::string& key,
                                TemporaryDirectory& temporary) {
    static std::atomic<U64> sequence{0};
    for (U64 attempt = 0; attempt < 1024; ++attempt) {
        const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
        auto path = root / ("." + key + ".tmp-" + std::to_string(nonce) + "-" +
                            std::to_string(sequence.fetch_add(1)));
        std::error_code ec;
        if (std::filesystem::create_directory(path, ec)) {
            temporary.path = std::move(path);
            return Result::SUCCESS;
        }
        if (ec && ec != std::errc::file_exists) {
            break;
        }
    }
    JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Cannot create a temporary Python "
              "environment directory.");
    return Result::ERROR;
}

}  // namespace

Result PreparePythonDependencyEnvironment(const std::vector<std::string>& requestedRequirements,
                                          bool installIfMissing,
                                          PythonDependencyEnvironment& environment,
                                          std::function<void(std::string_view)> onOutput) {
    if (requestedRequirements.empty()) {
        environment = {};
        return Result::SUCCESS;
    }

    auto requirements = requestedRequirements;
    std::ranges::sort(requirements);
    requirements.erase(std::unique(requirements.begin(), requirements.end()),
                       requirements.end());

    JST_CHECK(CPython::Py_Load());
    const auto& program = CPython::Py_ProgramPath();
    if (program.empty()) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] The selected Python runtime has no "
                  "executable.");
        return Result::ERROR;
    }

    std::string key;
    JST_CHECK(EnvironmentKey(requirements, key));

    std::string cacheValue;
    JST_CHECK(Platform::CachePath(cacheValue));
    const auto root = Platform::PathFromUtf8(cacheValue) / "python-environments";
    const auto path = root / key;
    const auto requirementsText = RequirementsText(requirements);

    std::string sitePackages;
    if (IsCachedEnvironment(path, requirementsText, sitePackages)) {
        environment = {
            .key = std::move(key),
            .sitePackagesPath = std::move(sitePackages),
        };
        return Result::SUCCESS;
    }
    if (!installIfMissing) {
        return Result::INCOMPLETE;
    }

    std::error_code ec;
    std::filesystem::create_directories(root, ec);
    if (ec) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Cannot create the Python environment "
                  "cache directory.");
        return Result::ERROR;
    }

    Platform::FileLock lock;
    JST_CHECK(lock.acquire(Platform::PathToUtf8(root / (key + ".lock"))));
    if (IsCachedEnvironment(path, requirementsText, sitePackages)) {
        environment = {
            .key = std::move(key),
            .sitePackagesPath = std::move(sitePackages),
        };
        return Result::SUCCESS;
    }

    std::filesystem::remove_all(path, ec);
    if (ec) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Cannot remove an incomplete Python "
                  "environment.");
        return Result::ERROR;
    }

    TemporaryDirectory temporary;
    JST_CHECK(CreateTemporaryDirectory(root, key, temporary));
    const auto temporarySitePackages = temporary.path / "site-packages";
    std::filesystem::create_directory(temporarySitePackages, ec);
    if (ec) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Cannot create the Python site-packages "
                  "directory.");
        return Result::ERROR;
    }
    const auto requirementsPath = temporary.path / "requirements.txt";
    JST_CHECK(WriteFile(requirementsPath, requirementsText));

    const std::vector<std::string> arguments = {
        "-m", "pip", "install", "--disable-pip-version-check", "--no-input",
        "--ignore-installed", "--target",
        Platform::PathToUtf8(temporarySitePackages), "--requirement",
        Platform::PathToUtf8(requirementsPath),
    };
    std::string output;
    if (Platform::RunProcess(program, arguments, output, kInstallTimeoutMilliseconds,
                             true, std::move(onOutput)) != Result::SUCCESS) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] pip failed to install Python "
                  "dependencies.\n{}",
                  output);
        return Result::ERROR;
    }
    JST_CHECK(WriteFile(temporary.path / "complete", "1\n"));

    std::filesystem::rename(temporary.path, path, ec);
    if (ec) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Cannot publish the Python dependency "
                  "environment.");
        return Result::ERROR;
    }
    temporary.path.clear();

    environment = {
        .key = std::move(key),
        .sitePackagesPath = Platform::PathToUtf8(path / "site-packages"),
    };
    return Result::SUCCESS;
}

}  // namespace Jetstream
