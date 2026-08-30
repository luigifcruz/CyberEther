#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <thread>
#include <unordered_set>
#include <utility>

#include "jetstream/backend/base.hh"
#include "jetstream/detail/module_impl.hh"
#include "jetstream/logger.hh"
#include "jetstream/module_context.hh"
#include "jetstream/platform.hh"
#include "jetstream/runtime_context_python.hh"
#include "jetstream/scheduler_context.hh"
#include "runtime/python/context.hh"
#include "runtime/python/dependencies/base.hh"

namespace {

using namespace Jetstream;

class PythonTempDirectory {
 public:
    explicit PythonTempDirectory(const std::string& label) {
        static std::atomic<U64> sequence{0};
        for (U64 attempt = 0; attempt < 1024; ++attempt) {
            const auto timestamp =
                std::chrono::steady_clock::now().time_since_epoch().count();
            root = std::filesystem::temp_directory_path() /
                   ("cyberether-python-" + label + "-" + std::to_string(timestamp) +
                    "-" + std::to_string(sequence.fetch_add(1)));
            std::error_code ec;
            if (std::filesystem::create_directory(root, ec)) {
                return;
            }
            if (ec && ec != std::errc::file_exists) {
                break;
            }
        }
        throw std::runtime_error("failed to create Python test directory");
    }

    ~PythonTempDirectory() {
        std::error_code ec;
        std::filesystem::remove_all(root, ec);
    }

    PythonTempDirectory(const PythonTempDirectory&) = delete;
    PythonTempDirectory& operator=(const PythonTempDirectory&) = delete;

    std::filesystem::path root;
};

#if defined(_WIN32)
class PythonEnvironmentGuard {
 public:
    explicit PythonEnvironmentGuard(const wchar_t* name) : name_(name) {
        if (const wchar_t* value = _wgetenv(name)) {
            previous_ = value;
        }
    }

    ~PythonEnvironmentGuard() {
        (void)_wputenv_s(name_.c_str(), previous_ ? previous_->c_str() : L"");
    }

    bool set(const std::wstring& value) const {
        return _wputenv_s(name_.c_str(), value.c_str()) == 0;
    }

 private:
    std::wstring name_;
    std::optional<std::wstring> previous_;
};
#else
class PythonEnvironmentGuard {
 public:
    explicit PythonEnvironmentGuard(const char* name) : name_(name) {
        if (const char* value = std::getenv(name)) {
            previous_ = value;
        }
    }

    ~PythonEnvironmentGuard() {
        if (previous_) {
            (void)setenv(name_.c_str(), previous_->c_str(), 1);
        } else {
            (void)unsetenv(name_.c_str());
        }
    }

    bool set(const std::string& value) const {
        return setenv(name_.c_str(), value.c_str(), 1) == 0;
    }

 private:
    std::string name_;
    std::optional<std::string> previous_;
};
#endif

class PythonCacheSandbox {
 public:
    PythonCacheSandbox()
        : directory_("cache"),
#if defined(_WIN32)
          environment_(L"LOCALAPPDATA")
#elif defined(__APPLE__)
          environment_("CFFIXED_USER_HOME")
#else
          environment_("XDG_CACHE_HOME")
#endif
    {
#if defined(_WIN32)
        const bool configured = environment_.set(directory_.root.wstring());
#else
        const bool configured = environment_.set(directory_.root.string());
#endif
        if (!configured) {
            throw std::runtime_error("failed to configure Python cache sandbox");
        }
    }

 private:
    PythonTempDirectory directory_;
    PythonEnvironmentGuard environment_;
};

void EnsurePythonCacheSandbox() {
    static const PythonCacheSandbox sandbox;
    (void)sandbox;
}

void WritePythonTestFile(const std::filesystem::path& path,
                         const std::string& contents) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file << contents;
    if (!file) {
        throw std::runtime_error("failed to write Python test file");
    }
}

std::string PythonFileUrl(const std::filesystem::path& path) {
    auto value = std::filesystem::absolute(path).generic_string();
#if defined(_WIN32)
    return "file:///" + value;
#else
    return "file://" + value;
#endif
}

TEST_CASE("Python runtime parses PEP 723 script metadata",
          "[core][runtime][python][pep723]") {
    PythonDependencyMetadata metadata;

    SECTION("source without metadata") {
        REQUIRE(ParsePythonDependencyMetadata("def compute(ctx):\n    pass\n", metadata) ==
                Result::SUCCESS);
        CHECK(metadata.requirements.empty());
        CHECK(metadata.requiresPython.empty());
    }

    SECTION("multiline dependency array") {
        const std::string source =
            "# /// script\r\n"
            "# requires-python = \">=3.11\"\r\n"
            "# dependencies = [\r\n"
            "#   \"numpy>=2\", # numerical arrays\r\n"
            "#   'scipy==1.14.0',\r\n"
            "# ]\r\n"
            "# ///\r\n"
            "def compute(ctx):\r\n"
            "    pass\r\n";

        REQUIRE(ParsePythonDependencyMetadata(source, metadata) == Result::SUCCESS);
        CHECK(metadata.requirements ==
              std::vector<std::string>{"numpy>=2", "scipy==1.14.0"});
        CHECK(metadata.requiresPython == ">=3.11");
    }

    SECTION("TOML escapes") {
        const std::string source =
            "# /// script\n"
            "# requires-python = \"\\u003e=3.11\"\n"
            "# dependencies = [\"demo\\u002dpkg\"]\n"
            "# ///\n";

        REQUIRE(ParsePythonDependencyMetadata(source, metadata) == Result::SUCCESS);
        CHECK(metadata.requirements == std::vector<std::string>{"demo-pkg"});
        CHECK(metadata.requiresPython == ">=3.11");
    }

    SECTION("optional fields") {
        REQUIRE(ParsePythonDependencyMetadata(
                    "# /// script\n# dependencies = []\n# ///\n", metadata) ==
                Result::SUCCESS);
        CHECK(metadata.requirements.empty());
        CHECK(metadata.requiresPython.empty());

        REQUIRE(ParsePythonDependencyMetadata(
                    "# /// script\n# requires-python = \">=3.12\"\n# ///\n", metadata) ==
                Result::SUCCESS);
        CHECK(metadata.requirements.empty());
        CHECK(metadata.requiresPython == ">=3.12");
    }
}

TEST_CASE("Python runtime follows PEP 723 block boundaries",
          "[core][runtime][python][pep723]") {
    PythonDependencyMetadata metadata;

    SECTION("unclosed candidates are ignored") {
        const std::string source =
            "# /// script\n"
            "dependencies = [\"ignored\"]\n"
            "# /// script\n"
            "# dependencies = [\"numpy\"]\n"
            "# ///\n";

        REQUIRE(ParsePythonDependencyMetadata(source, metadata) == Result::SUCCESS);
        CHECK(metadata.requirements == std::vector<std::string>{"numpy"});
    }

    SECTION("unknown blocks are not read") {
        const std::string source =
            "# /// future-metadata\n"
            "# /// script\n"
            "# dependencies = [\"ignored\"]\n"
            "# ///\n"
            "# still in the unknown block\n"
            "# ///\n";

        REQUIRE(ParsePythonDependencyMetadata(source, metadata) == Result::SUCCESS);
        CHECK(metadata.requirements.empty());
    }

    SECTION("delimiter text can appear in multiline TOML strings") {
        const std::string source =
            "# /// script\n"
            "# dependencies = [\"numpy\"]\n"
            "# description = \"\"\"\n"
            "# ///\n"
            "# still metadata\n"
            "# \"\"\"\n"
            "# ///\n";

        REQUIRE(ParsePythonDependencyMetadata(source, metadata) == Result::SUCCESS);
        CHECK(metadata.requirements == std::vector<std::string>{"numpy"});
    }

    SECTION("duplicate script blocks are rejected") {
        const std::string source =
            "# /// script\n# dependencies = [\"numpy\"]\n# ///\n\n"
            "# /// script\n# dependencies = [\"scipy\"]\n# ///\n";

        JST_LOG_LAST_ERROR().clear();
        CHECK(ParsePythonDependencyMetadata(source, metadata) == Result::ERROR);
        CHECK(JST_LOG_LAST_ERROR().find("Multiple PEP 723") != std::string::npos);
        CHECK(metadata.requirements.empty());
    }
}

TEST_CASE("Python runtime rejects malformed PEP 723 metadata",
          "[core][runtime][python][pep723]") {
    const std::vector<std::string> sources = {
        "# /// script\n# dependencies = [numpy]\n# ///\n",
        "# /// script\n# dependencies = [\"numpy\" \"scipy\"]\n# ///\n",
        "# /// script\n# requires-python = 3.11\n# ///\n",
        "# /// script\n# dependencies = \"numpy\"\n# ///\n",
        "# /// script\n# dependencies = [1]\n# ///\n",
        "# /// script\n# dependencies = [\"\"]\n# ///\n",
        "# /// script\n# dependencies = []\n# dependencies = []\n# ///\n",
        "# /// script\n# dependencies = [] trailing\n# ///\n",
    };

    for (const auto& source : sources) {
        PythonDependencyMetadata metadata;
        metadata.requirements = {"unchanged"};
        CAPTURE(source);
        JST_LOG_LAST_ERROR().clear();
        CHECK(ParsePythonDependencyMetadata(source, metadata) == Result::ERROR);
        CHECK(metadata.requirements.empty());
        CHECK(JST_LOG_LAST_ERROR().starts_with("[RUNTIME_CONTEXT_PYTHON]"));
    }
}

TEST_CASE("Python runtime rejects unsafe PEP 723 requirement strings",
          "[core][runtime][python][pep723]") {
    const std::vector<std::string> dependencies = {
        std::string("demo\0--target", 13),
        "demo\r--target",
        "demo\n--target",
    };

    for (const auto& dependency : dependencies) {
        CAPTURE(dependency.size());
        JST_LOG_LAST_ERROR().clear();
        CHECK(ValidatePythonDependencyMetadata({.requirements = {dependency}}) == Result::ERROR);
        CHECK(JST_LOG_LAST_ERROR().find("prohibited NUL, CR, or LF") != std::string::npos);
    }

    JST_LOG_LAST_ERROR().clear();
    CHECK(ValidatePythonDependencyMetadata({.requiresPython = ">=3.9\n"}) == Result::ERROR);
    CHECK(JST_LOG_LAST_ERROR().find("prohibited NUL, CR, or LF") != std::string::npos);

    const std::string source =
        "# /// script\n"
        "# dependencies = [\"\"\"demo\n"
        "# --target\"\"\"]\n"
        "# ///\n"
        "def compute(ctx):\n"
        "    pass\n";
    const Module::Interface::EntryList order;
    const TensorMap tensors;
    PythonRuntimeContext context;
    JST_LOG_LAST_ERROR().clear();
    CHECK(context.createCompute(source, {}, order, tensors, order, tensors) == Result::ERROR);
    CHECK(JST_LOG_LAST_ERROR().find("prohibited NUL, CR, or LF") != std::string::npos);
}

TEST_CASE("Python runtime validates PEP 723 requirements with the selected executable",
          "[core][runtime][python][pep723]") {
    PythonDependencyMetadata valid = {
        .requirements = {
            "numpy>=2,<3",
            "requests[security]>=2.31; python_version >= '3.9'",
            "demo @ https://example.com/demo-1.0-py3-none-any.whl",
            "platformdirs; python_version < '0'",
            "demo; extra == 'security'",
        },
        .requiresPython = ">=3.9",
    };
    JST_LOG_LAST_ERROR().clear();
    if (ValidatePythonDependencyMetadata(valid) != Result::SUCCESS) {
        const auto& error = JST_LOG_LAST_ERROR();
        if (error.find("No libpython was found") != std::string::npos ||
            error.find("No loadable libpython was found") != std::string::npos ||
            error.find("Auto could not find") != std::string::npos ||
            error.find("provides neither packaging") != std::string::npos) {
            SKIP("Optional Python requirement preflight is unavailable: " << error);
        }
    }
    REQUIRE(ValidatePythonDependencyMetadata(valid) == Result::SUCCESS);

    const std::vector<std::string> invalidRequirements = {
        "not a valid requirement !!!",
        "demo[broken",
        "demo; python_version >>> '3.9'",
    };
    for (const auto& dependency : invalidRequirements) {
        CAPTURE(dependency);
        JST_LOG_LAST_ERROR().clear();
        CHECK(ValidatePythonDependencyMetadata({.requirements = {dependency}}) == Result::ERROR);
        CHECK(JST_LOG_LAST_ERROR().find("Invalid PEP 508 requirement") != std::string::npos);
    }

    JST_LOG_LAST_ERROR().clear();
    CHECK(ValidatePythonDependencyMetadata({.requiresPython = "<3"}) == Result::ERROR);
    CHECK(JST_LOG_LAST_ERROR().find("does not satisfy PEP 723 requires-python") !=
          std::string::npos);

    JST_LOG_LAST_ERROR().clear();
    CHECK(ValidatePythonDependencyMetadata({.requiresPython = "=>3.9"}) == Result::ERROR);
    CHECK(JST_LOG_LAST_ERROR().find("Invalid PEP 723 requires-python") !=
          std::string::npos);
}

TEST_CASE("Python runtime parses dependency installation policies",
          "[core][runtime][python][pep723]") {
    PythonDependencyPolicy policy = PythonDependencyPolicy::Deny;

    SECTION("canonical values") {
        const std::vector<std::pair<std::string, PythonDependencyPolicy>> values = {
            {"prompt", PythonDependencyPolicy::Prompt},
            {"allow", PythonDependencyPolicy::Allow},
            {"deny", PythonDependencyPolicy::Deny},
        };

        for (const auto& [value, expected] : values) {
            CAPTURE(value);
            REQUIRE(ParsePythonDependencyPolicy(value, policy) == Result::SUCCESS);
            CHECK(policy == expected);
        }
    }

    SECTION("values are matched exactly") {
        for (const auto& value : {"DeNy", " never", "allow "}) {
            CAPTURE(value);
            JST_LOG_LAST_ERROR().clear();
            CHECK(ParsePythonDependencyPolicy(value, policy) == Result::ERROR);
            CHECK(JST_LOG_LAST_ERROR().find("Invalid dependency policy") != std::string::npos);
        }
    }

    SECTION("invalid values are rejected") {
        policy = PythonDependencyPolicy::Allow;
        for (const auto& value : {"", "never", "installation", "a llow", "allow,prompt"}) {
            CAPTURE(value);
            JST_LOG_LAST_ERROR().clear();
            CHECK(ParsePythonDependencyPolicy(value, policy) == Result::ERROR);
            CHECK(JST_LOG_LAST_ERROR().find("Invalid dependency policy") != std::string::npos);
        }
        CHECK(policy == PythonDependencyPolicy::Allow);
    }
}

TEST_CASE("Python runtime resolves the PEP 723 dependency policy",
          "[core][runtime][python][pep723]") {
    const PythonDependencyMetadata declared = {
        .requirements = {"numpy>=2"},
        .requiresPython = ">=3.11",
    };

    SECTION("scripts without metadata bypass the policy") {
        for (const auto policy : {PythonDependencyPolicy::Prompt,
                                  PythonDependencyPolicy::Allow,
                                  PythonDependencyPolicy::Deny}) {
            CAPTURE(static_cast<int>(policy));
            const auto decision = ResolvePythonDependencyPolicy({}, policy);
            CHECK(decision.policy == policy);
            CHECK_FALSE(decision.installAllowed);
            CHECK_FALSE(decision.consentRequired);
        }
    }

    SECTION("allow installs automatically") {
        const auto decision = ResolvePythonDependencyPolicy(declared, PythonDependencyPolicy::Allow);
        CHECK(decision.policy == PythonDependencyPolicy::Allow);
        CHECK(decision.installAllowed);
        CHECK_FALSE(decision.consentRequired);
    }

    SECTION("prompt installs after consent") {
        const auto decision = ResolvePythonDependencyPolicy(declared, PythonDependencyPolicy::Prompt);
        CHECK(decision.policy == PythonDependencyPolicy::Prompt);
        CHECK(decision.installAllowed);
        CHECK(decision.consentRequired);
    }

    SECTION("deny never installs") {
        const auto decision = ResolvePythonDependencyPolicy(declared, PythonDependencyPolicy::Deny);
        CHECK(decision.policy == PythonDependencyPolicy::Deny);
        CHECK_FALSE(decision.installAllowed);
        CHECK_FALSE(decision.consentRequired);
    }

    SECTION("a requires-python constraint alone bypasses the policy") {
        const PythonDependencyMetadata versionOnly = {.requiresPython = ">=3.11"};
        const auto decision = ResolvePythonDependencyPolicy(versionOnly, PythonDependencyPolicy::Deny);
        CHECK(decision.policy == PythonDependencyPolicy::Deny);
        CHECK_FALSE(decision.installAllowed);
        CHECK_FALSE(decision.consentRequired);
    }
}

TEST_CASE("Python runtime resolves the configured dependency policy",
          "[core][runtime][python][pep723]") {
    const PythonDependencyMetadata declared = {
        .requirements = {"numpy>=2"},
    };

    // The CPU backend snapshots its configuration at initialization, so start
    // from a clean slate and reconfigure before every resolution.
    Backend::DestroyAll();

    SECTION("falls back to prompt without a configured policy") {
        const auto decision = ResolvePythonDependencyPolicy(declared);
        CHECK(decision.policy == PythonDependencyPolicy::Prompt);
        CHECK(decision.installAllowed);
        CHECK(decision.consentRequired);
    }

    SECTION("deny propagates from the backend configuration") {
        REQUIRE(Backend::Configure<DeviceType::CPU>(Backend::Config{
                    .dependencyPolicy = "deny"}) == Result::SUCCESS);
        const auto decision = ResolvePythonDependencyPolicy(declared);
        CHECK(decision.policy == PythonDependencyPolicy::Deny);
        CHECK_FALSE(decision.installAllowed);
        CHECK_FALSE(decision.consentRequired);
    }

    SECTION("allow propagates from the backend configuration") {
        REQUIRE(Backend::Configure<DeviceType::CPU>(Backend::Config{
                    .dependencyPolicy = "allow"}) == Result::SUCCESS);
        const auto decision = ResolvePythonDependencyPolicy(declared);
        CHECK(decision.policy == PythonDependencyPolicy::Allow);
        CHECK(decision.installAllowed);
        CHECK_FALSE(decision.consentRequired);
    }

    SECTION("invalid configured values fall back to prompt") {
        REQUIRE(Backend::Configure<DeviceType::CPU>(Backend::Config{
                    .dependencyPolicy = "never"}) == Result::SUCCESS);
        const auto decision = ResolvePythonDependencyPolicy(declared);
        CHECK(decision.policy == PythonDependencyPolicy::Prompt);
        CHECK(decision.installAllowed);
        CHECK(decision.consentRequired);
    }

    Backend::DestroyAll();
}

TEST_CASE("Python runtime combines scheduled dependency requirements",
          "[core][runtime][python][pep723]") {
    PythonRuntimeContext first;
    PythonRuntimeContext second;

    REQUIRE(StagePythonDependencies(&first,
                                    {"scipy>=1", "numpy>=2", "numpy>=2"}) ==
            Result::SUCCESS);
    REQUIRE(StagePythonDependencies(&second, {"requests>=2", "numpy>=2"}) ==
            Result::SUCCESS);
    CHECK(SnapshotPythonDependencies().requirements.empty());

    REQUIRE(SchedulePythonDependencies(&first) == Result::SUCCESS);
    CHECK(SnapshotPythonDependencies().requirements ==
          std::vector<std::string>{"numpy>=2", "scipy>=1"});

    REQUIRE(SchedulePythonDependencies(&second) == Result::SUCCESS);
    const auto combined = SnapshotPythonDependencies();
    CHECK(combined.requirements ==
          std::vector<std::string>{"numpy>=2", "requests>=2", "scipy>=1"});

    REQUIRE(StagePythonDependencies(&first, {"pandas>=2"}) == Result::SUCCESS);
    const auto updated = SnapshotPythonDependencies();
    CHECK(updated.generation > combined.generation);
    CHECK(updated.requirements ==
          std::vector<std::string>{"numpy>=2", "pandas>=2", "requests>=2"});

    REQUIRE(UnschedulePythonDependencies(&second) == Result::SUCCESS);
    CHECK(SnapshotPythonDependencies().requirements ==
          std::vector<std::string>{"pandas>=2"});

    REQUIRE(RemovePythonDependencies(&first) == Result::SUCCESS);
    REQUIRE(RemovePythonDependencies(&second) == Result::SUCCESS);
    CHECK(SnapshotPythonDependencies().requirements.empty());
}

TEST_CASE("Python runtime caches pip dependency environments",
          "[core][runtime][python][pep723]") {
    EnsurePythonCacheSandbox();

    PythonTempDirectory package("cache-package");
    WritePythonTestFile(
        package.root / "setup.py",
        "from setuptools import setup\n"
        "setup(name='cyberether-cache-fixture', version='1.0', "
        "py_modules=['cyberether_cache_fixture'])\n");
    WritePythonTestFile(package.root / "cyberether_cache_fixture.py", "VALUE = 17\n");

    const std::string packageRequirement =
        "cyberether-cache-fixture @ " + PythonFileUrl(package.root);
    const std::string ignoredRequirement =
        "cyberether-cache-ignored; python_version < '0'";
    const std::vector<std::string> requirements = {
        ignoredRequirement,
        packageRequirement,
        packageRequirement,
    };

    PythonDependencyEnvironment missing = {
        .requirements = {"unchanged"},
        .key = "unchanged",
        .sitePackagesPath = "unchanged",
    };
    const auto missResult = PreparePythonDependencyEnvironment(
        requirements, false, missing);
    if (missResult == Result::ERROR) {
        INFO(JST_LOG_LAST_ERROR());
        SKIP("Optional Python environment cache runtime is unavailable");
    }
    REQUIRE(missResult == Result::INCOMPLETE);
    CHECK(missing.key == "unchanged");

    PythonDependencyEnvironment installed;
    const auto installResult = PreparePythonDependencyEnvironment(
        requirements, true, installed);
    const auto& installError = JST_LOG_LAST_ERROR();
    if (installResult != Result::SUCCESS &&
        (installError.find("No module named pip") != std::string::npos ||
         installError.find("No module named 'setuptools'") != std::string::npos)) {
        SKIP("The selected optional Python runtime cannot install local packages: "
             << installError);
    }
    INFO(installError);
    REQUIRE(installResult == Result::SUCCESS);
    CHECK(installed.requirements ==
          std::vector<std::string>{packageRequirement, ignoredRequirement});
    CHECK(installed.key.size() == 64);

    const auto sitePackages = Platform::PathFromUtf8(installed.sitePackagesPath);
    REQUIRE(std::filesystem::is_directory(sitePackages));
    REQUIRE(std::filesystem::is_regular_file(
        sitePackages / "cyberether_cache_fixture.py"));

    PythonDependencyEnvironment cached;
    REQUIRE(PreparePythonDependencyEnvironment(
                {packageRequirement, ignoredRequirement}, false, cached) ==
            Result::SUCCESS);
    CHECK(cached.key == installed.key);
    CHECK(cached.sitePackagesPath == installed.sitePackagesPath);

    std::string cacheValue;
    REQUIRE(Platform::CachePath(cacheValue) == Result::SUCCESS);
    const auto environmentPath = Platform::PathFromUtf8(cacheValue) /
                                 "python-environments" / installed.key;
    const auto complete = environmentPath / "complete";
    REQUIRE(std::filesystem::is_regular_file(environmentPath / "requirements.txt"));
    REQUIRE(std::filesystem::is_regular_file(complete));

    PythonDependencyEnvironment unchanged = {
        .requirements = {"unchanged"},
        .key = "unchanged",
        .sitePackagesPath = "unchanged",
    };
    std::error_code ec;
    std::filesystem::remove(complete, ec);
    REQUIRE_FALSE(ec);
    REQUIRE(PreparePythonDependencyEnvironment(requirements, false, unchanged) ==
            Result::INCOMPLETE);
    CHECK(unchanged.key == "unchanged");
    REQUIRE(PreparePythonDependencyEnvironment(requirements, true, installed) ==
            Result::SUCCESS);

    WritePythonTestFile(environmentPath / "requirements.txt", "tampered\n");
    REQUIRE(PreparePythonDependencyEnvironment(requirements, false, unchanged) ==
            Result::INCOMPLETE);
    REQUIRE(PreparePythonDependencyEnvironment(requirements, true, installed) ==
            Result::SUCCESS);

    std::filesystem::remove_all(environmentPath, ec);
    REQUIRE_FALSE(ec);
    PythonDependencyEnvironment concurrentFirst;
    PythonDependencyEnvironment concurrentSecond;
    Result firstResult = Result::ERROR;
    Result secondResult = Result::ERROR;
    std::thread first([&] {
        firstResult = PreparePythonDependencyEnvironment(
            requirements, true, concurrentFirst);
    });
    std::thread second([&] {
        secondResult = PreparePythonDependencyEnvironment(
            requirements, true, concurrentSecond);
    });
    first.join();
    second.join();
    REQUIRE(firstResult == Result::SUCCESS);
    REQUIRE(secondResult == Result::SUCCESS);
    CHECK(concurrentFirst.sitePackagesPath == concurrentSecond.sitePackagesPath);

    std::filesystem::remove_all(package.root, ec);
    REQUIRE_FALSE(ec);
    REQUIRE(PreparePythonDependencyEnvironment(requirements, false, cached) ==
            Result::SUCCESS);

    const std::string missingRequirement =
        "cyberether-cache-missing @ " + PythonFileUrl(package.root);
    unchanged.key = "unchanged";
    REQUIRE(PreparePythonDependencyEnvironment(
                {missingRequirement}, false, unchanged) == Result::INCOMPLETE);
    CHECK(unchanged.key == "unchanged");
    REQUIRE(PreparePythonDependencyEnvironment(
                {missingRequirement}, true, unchanged) == Result::ERROR);
    CHECK(unchanged.key == "unchanged");
    CHECK_FALSE(JST_LOG_LAST_ERROR().empty());
}

struct SyntheticPythonState {
    std::unordered_map<std::string, U64> initializes;
    std::unordered_map<std::string, U64> deinitializes;
    std::unordered_map<std::string, Result> deinitializeResults;
    std::vector<std::string> initializationOrder;
    std::vector<std::string> deinitializationOrder;
    U64 initializeCount = 0;
    U64 failInitializeAt = 0;

    void reset() {
        initializes.clear();
        deinitializes.clear();
        deinitializeResults.clear();
        initializationOrder.clear();
        deinitializationOrder.clear();
        initializeCount = 0;
        failInitializeAt = 0;
    }

    Result initialize(const std::string& name) {
        initializes[name] += 1;
        initializationOrder.push_back(name);
        initializeCount += 1;
        return initializeCount == failInitializeAt ? Result::ERROR : Result::SUCCESS;
    }

    Result deinitialize(const std::string& name) {
        deinitializes[name] += 1;
        deinitializationOrder.push_back(name);
        const auto result = deinitializeResults.find(name);
        return result == deinitializeResults.end() ? Result::SUCCESS : result->second;
    }

    U64 deinitializeCount() const {
        U64 count = 0;
        for (const auto& [_, value] : deinitializes) {
            count += value;
        }
        return count;
    }
};

SyntheticPythonState& syntheticPythonState() {
    static SyntheticPythonState state;
    return state;
}

struct SyntheticPythonConfig : Module::Config {
    JST_MODULE_TYPE(synthetic_python_runtime)

    Result serialize(Parser::Map&) const override {
        return Result::SUCCESS;
    }

    Result deserialize(const Parser::Map&) override {
        return Result::SUCCESS;
    }

    std::size_t hash() const override {
        return 0;
    }
};

struct SyntheticPythonModule : Module::Impl,
                               DynamicConfig<SyntheticPythonConfig>,
                               PythonRuntimeContext,
                               Scheduler::Context {
    Result define() override {
        JST_CHECK(defineInterfaceOutput("out"));
        return Result::SUCCESS;
    }

    Result create() override {
        JST_CHECK(output.create(DeviceType::CPU, DataType::F32, {1}));
        outputs()["out"].produced(name(), "out", output);
        return Result::SUCCESS;
    }

    Result computeInitialize() override {
        return syntheticPythonState().initialize(name());
    }

    Result computeSubmit() override {
        return Result::SUCCESS;
    }

    Result computeDeinitialize() override {
        return syntheticPythonState().deinitialize(name());
    }

    Tensor output;
};

struct PythonRuntimeSmokeConfig : Module::Config {
    JST_MODULE_TYPE(python_runtime_smoke)

    Result serialize(Parser::Map&) const override {
        return Result::SUCCESS;
    }

    Result deserialize(const Parser::Map&) override {
        return Result::SUCCESS;
    }

    std::size_t hash() const override {
        return 0;
    }
};

struct PythonRuntimeSmokeModule : Module::Impl,
                                  DynamicConfig<PythonRuntimeSmokeConfig>,
                                  PythonRuntimeContext,
                                  Scheduler::Context {
    explicit PythonRuntimeSmokeModule(std::string source = R"PY(
_seen = None

def compute(ctx):
    global _seen
    current = (id(ctx), id(ctx.inputs[0]), id(ctx.outputs[0]))
    if _seen is None:
        _seen = current
    elif current != _seen:
        raise RuntimeError("Python context arrays changed")
    ctx.outputs[0][...] = ctx.inputs[0] * 3.0
)PY",
                                      std::unordered_map<std::string, std::string> pieces = {})
        : source(std::move(source)), pieces(std::move(pieces)) {}

    Result define() override {
        JST_CHECK(defineInterfaceInput("in"));
        JST_CHECK(defineInterfaceOutput("out"));
        return Result::SUCCESS;
    }

    Result create() override {
        input = inputs().at("in").tensor;
        JST_CHECK(output.create(DeviceType::CPU, DataType::F32, input.shape()));
        outputs()["out"].produced(name(), "out", output);

        return createCompute(source,
                             pieces,
                             {"in"},
                             inputs(),
                             {"out"},
                             outputs());
    }

    std::string source;
    std::unordered_map<std::string, std::string> pieces;
    Tensor input;
    Tensor output;
};

std::shared_ptr<Module> makePythonRuntimeSmokeModule(std::string source = {},
                                                     std::unordered_map<std::string, std::string> pieces = {}) {
    auto impl = source.empty() && pieces.empty()
                    ? std::make_shared<PythonRuntimeSmokeModule>()
                    : std::make_shared<PythonRuntimeSmokeModule>(std::move(source), std::move(pieces));
    auto runtimeContext = std::static_pointer_cast<Runtime::Context>(impl);
    auto schedulerContext = std::static_pointer_cast<Scheduler::Context>(impl);
    auto context = std::make_shared<Module::Context>(runtimeContext,
                                                     schedulerContext,
                                                     nullptr,
                                                     nullptr);
    auto stagedConfig = std::static_pointer_cast<Module::Config>(impl);
    auto candidateConfig = std::static_pointer_cast<Module::Config>(impl->candidate());

    return std::make_shared<Module>(DeviceType::CPU,
                                    RuntimeType::PYTHON,
                                    "generic",
                                    impl,
                                    context,
                                    stagedConfig,
                                    candidateConfig);
}

std::shared_ptr<Module> makeSyntheticPythonModule(const std::string& name) {
    auto impl = std::make_shared<SyntheticPythonModule>();
    auto runtimeContext = std::static_pointer_cast<Runtime::Context>(impl);
    auto schedulerContext = std::static_pointer_cast<Scheduler::Context>(impl);
    auto context = std::make_shared<Module::Context>(runtimeContext,
                                                     schedulerContext,
                                                     nullptr,
                                                     nullptr);
    auto stagedConfig = std::static_pointer_cast<Module::Config>(impl);
    auto candidateConfig = std::static_pointer_cast<Module::Config>(impl->candidate());
    auto module = std::make_shared<Module>(DeviceType::CPU,
                                           RuntimeType::PYTHON,
                                           "generic",
                                           impl,
                                           context,
                                           stagedConfig,
                                           candidateConfig);
    Parser::Map config;
    if (module->create(name, config, {}) != Result::SUCCESS) {
        throw std::runtime_error("failed to create synthetic Python module: " + name);
    }

    return module;
}

bool optionalPythonRuntimeUnavailable() {
    const auto& error = JST_LOG_LAST_ERROR();
    return error.find("Can't load Python library") != std::string::npos ||
           error.find("Can't initialize Python runtime helpers") != std::string::npos ||
           error.find("Can't load Python symbol") != std::string::npos ||
           error.find("Auto could not find a valid Python runtime") != std::string::npos ||
           error.find("No libpython was found") != std::string::npos ||
           error.find("No loadable libpython was found") != std::string::npos;
}

bool pythonRuntimeAvailable(const Result createResult) {
    if (createResult == Result::SUCCESS) {
        return true;
    }

    INFO(JST_LOG_LAST_ERROR());
    REQUIRE(optionalPythonRuntimeUnavailable());
    return false;
}

bool equivalentPaths(const std::filesystem::path& lhs, const std::filesystem::path& rhs) {
    std::error_code ec;
    return lhs == rhs || (std::filesystem::equivalent(lhs, rhs, ec) && !ec);
}

void destroyPythonCompute(const std::shared_ptr<Module>& module) {
    auto pythonContext = std::dynamic_pointer_cast<PythonRuntimeContext>(module->context()->runtime());
    REQUIRE(pythonContext != nullptr);
    REQUIRE(pythonContext->destroyCompute() == Result::SUCCESS);
}

void destroyUnavailablePythonModule(const std::shared_ptr<Module>& module,
                                    const Result createResult) {
    destroyPythonCompute(module);
    if (createResult != Result::ERROR) {
        REQUIRE(module->destroy() == Result::SUCCESS);
    }
}

}  // namespace

TEST_CASE("Python runtime rolls back partial initialization", "[core][runtime][python]") {
    auto& state = syntheticPythonState();
    state.reset();
    state.failInitializeAt = 2;

    auto first = makeSyntheticPythonModule("python_rollback_first");
    auto second = makeSyntheticPythonModule("python_rollback_second");

    Runtime runtime("python_rollback", DeviceType::CPU, RuntimeType::PYTHON);
    REQUIRE(runtime.create({
        {"python_rollback_first", first},
        {"python_rollback_second", second},
    }) == Result::ERROR);
    REQUIRE(state.initializeCount == 2);
    REQUIRE(runtime.destroy() == Result::SUCCESS);

    const auto deinitializeCount = state.deinitializeCount();
    REQUIRE(second->destroy() == Result::SUCCESS);
    REQUIRE(first->destroy() == Result::SUCCESS);

    REQUIRE(deinitializeCount == 2);
}

TEST_CASE("Python runtime tears modules down in reverse initialization order",
          "[core][runtime][python]") {
    auto& state = syntheticPythonState();
    state.reset();

    auto first = makeSyntheticPythonModule("python_teardown_first");
    auto second = makeSyntheticPythonModule("python_teardown_second");
    auto third = makeSyntheticPythonModule("python_teardown_third");

    Runtime runtime("python_teardown", DeviceType::CPU, RuntimeType::PYTHON);
    REQUIRE(runtime.create({
        {"python_teardown_first", first},
        {"python_teardown_second", second},
        {"python_teardown_third", third},
    }) == Result::SUCCESS);

    auto expected = state.initializationOrder;
    std::ranges::reverse(expected);
    REQUIRE(runtime.destroy() == Result::SUCCESS);
    REQUIRE(state.deinitializationOrder == expected);

    REQUIRE(third->destroy() == Result::SUCCESS);
    REQUIRE(second->destroy() == Result::SUCCESS);
    REQUIRE(first->destroy() == Result::SUCCESS);
}

TEST_CASE("Python runtime continues teardown after deinitialization failures",
          "[core][runtime][python]") {
    auto& state = syntheticPythonState();
    state.reset();

    auto first = makeSyntheticPythonModule("python_teardown_error_first");
    auto second = makeSyntheticPythonModule("python_teardown_error_second");
    auto third = makeSyntheticPythonModule("python_teardown_error_third");

    Runtime runtime("python_teardown_error", DeviceType::CPU, RuntimeType::PYTHON);
    REQUIRE(runtime.create({
        {"python_teardown_error_first", first},
        {"python_teardown_error_second", second},
        {"python_teardown_error_third", third},
    }) == Result::SUCCESS);

    state.deinitializeResults["python_teardown_error_first"] = Result::ERROR;
    state.deinitializeResults["python_teardown_error_second"] = Result::ERROR;
    state.deinitializeResults["python_teardown_error_third"] = Result::ERROR;
    state.deinitializeResults[state.initializationOrder.back()] = Result::WARNING;

    REQUIRE(runtime.destroy() == Result::WARNING);
    REQUIRE(state.deinitializeCount() == 3);
    REQUIRE(runtime.destroy() == Result::SUCCESS);
    REQUIRE(state.deinitializeCount() == 3);

    REQUIRE(third->destroy() == Result::SUCCESS);
    REQUIRE(second->destroy() == Result::SUCCESS);
    REQUIRE(first->destroy() == Result::SUCCESS);
}

TEST_CASE("Python runtime discovery removes executable aliases", "[core][runtime][python]") {
    const auto candidates = PythonRuntimeContext::DiscoverRuntimes();
#if defined(_WIN32)
    REQUIRE_FALSE(candidates.empty());
#endif
    for (U64 i = 0; i < candidates.size(); ++i) {
        const auto validation = PythonRuntimeContext::ValidateRuntimePath(candidates[i].path);
        CAPTURE(candidates[i].path);
        CHECK(validation.valid);
        CHECK(equivalentPaths(validation.libraryPath, candidates[i].libraryPath));
        CHECK_FALSE(validation.programPath.empty());

        const auto libraryValidation =
            PythonRuntimeContext::ValidateRuntimePath(candidates[i].libraryPath);
        CHECK(libraryValidation.valid);
        CHECK_FALSE(libraryValidation.programPath.empty());
        const auto programValidation =
            PythonRuntimeContext::ValidateRuntimePath(libraryValidation.programPath);
        CHECK(programValidation.valid);
        CHECK(equivalentPaths(programValidation.libraryPath, candidates[i].libraryPath));
#if defined(_WIN32)
        CHECK_FALSE(validation.programPath.empty());
        auto programName = std::filesystem::path(validation.programPath).filename().string();
        std::ranges::transform(programName, programName.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        CHECK(programName != "py.exe");
#endif

        for (U64 j = i + 1; j < candidates.size(); ++j) {
            const auto firstDirectory = std::filesystem::path(candidates[i].path).parent_path();
            const auto secondDirectory = std::filesystem::path(candidates[j].path).parent_path();
            const bool aliases = equivalentPaths(firstDirectory, secondDirectory) &&
                                 equivalentPaths(candidates[i].libraryPath, candidates[j].libraryPath);
            CAPTURE(candidates[i].path, candidates[j].path);
            CHECK_FALSE(aliases);
        }
    }
}

TEST_CASE("Python runtime executes compute() with tensor inputs and outputs", "[core][runtime][python]") {
    Tensor input;
    REQUIRE(input.create(DeviceType::CPU, DataType::F32, {4}) == Result::SUCCESS);

    for (Index i = 0; i < input.size(); ++i) {
        input.at<F32>(i) = static_cast<F32>(i + 1);
    }

    TensorMap inputs;
    inputs["in"].produced("source", "out", input);

    auto module = makePythonRuntimeSmokeModule();
    Parser::Map config;
    const auto createResult = module->create("python_runtime_smoke", config, inputs);

    if (!pythonRuntimeAvailable(createResult)) {
        destroyUnavailablePythonModule(module, createResult);
        SKIP("Optional Python runtime is unavailable: " << JST_LOG_LAST_ERROR());
    }

    REQUIRE(createResult == Result::SUCCESS);

    Runtime runtime("python", DeviceType::CPU, RuntimeType::PYTHON);
    REQUIRE(runtime.create({{"python_runtime_smoke", module}}) == Result::SUCCESS);

    std::unordered_set<std::string> skippedModules;
    std::unordered_set<std::string> failedModules;
    REQUIRE(runtime.compute({"python_runtime_smoke"}, skippedModules, failedModules) == Result::SUCCESS);
    REQUIRE(runtime.compute({"python_runtime_smoke"}, skippedModules, failedModules) == Result::SUCCESS);
    REQUIRE(skippedModules.empty());
    REQUIRE(failedModules.empty());

    Tensor output = module->outputs().at("out").tensor;
    const auto* data = output.data<F32>();
    REQUIRE(data != nullptr);

    for (Index i = 0; i < output.size(); ++i) {
        REQUIRE(std::abs(data[i] - static_cast<F32>((i + 1) * 3)) < 1e-5f);
    }

    REQUIRE(runtime.destroy() == Result::SUCCESS);
    destroyPythonCompute(module);
    REQUIRE(module->destroy() == Result::SUCCESS);
}

TEST_CASE("Python runtime expands source pieces before compiling compute()", "[core][runtime][python]") {
    Tensor input;
    REQUIRE(input.create(DeviceType::CPU, DataType::F32, {4}) == Result::SUCCESS);

    for (Index i = 0; i < input.size(); ++i) {
        input.at<F32>(i) = static_cast<F32>(i + 1);
    }

    TensorMap inputs;
    inputs["in"].produced("source", "out", input);

    auto module = makePythonRuntimeSmokeModule(R"PY(
def compute(ctx):
    <<<BODY>>>
)PY", {{"BODY", "ctx.outputs[0][...] = ctx.inputs[0] + 5.0"}});

    Parser::Map config;
    const auto createResult = module->create("python_runtime_piece_smoke", config, inputs);

    if (!pythonRuntimeAvailable(createResult)) {
        destroyUnavailablePythonModule(module, createResult);
        SKIP("Optional Python runtime is unavailable: " << JST_LOG_LAST_ERROR());
    }

    REQUIRE(createResult == Result::SUCCESS);

    Runtime runtime("python", DeviceType::CPU, RuntimeType::PYTHON);
    REQUIRE(runtime.create({{"python_runtime_piece_smoke", module}}) == Result::SUCCESS);

    std::unordered_set<std::string> skippedModules;
    std::unordered_set<std::string> failedModules;
    REQUIRE(runtime.compute({"python_runtime_piece_smoke"}, skippedModules, failedModules) == Result::SUCCESS);
    REQUIRE(skippedModules.empty());
    REQUIRE(failedModules.empty());

    Tensor output = module->outputs().at("out").tensor;
    const auto* data = output.data<F32>();
    REQUIRE(data != nullptr);

    for (Index i = 0; i < output.size(); ++i) {
        REQUIRE(std::abs(data[i] - static_cast<F32>(i + 6)) < 1e-5f);
    }

    REQUIRE(runtime.destroy() == Result::SUCCESS);
    destroyPythonCompute(module);
    REQUIRE(module->destroy() == Result::SUCCESS);
}

TEST_CASE("Python runtime compute can run on a different thread than creation", "[core][runtime][python]") {
    Tensor input;
    REQUIRE(input.create(DeviceType::CPU, DataType::F32, {4}) == Result::SUCCESS);

    for (Index i = 0; i < input.size(); ++i) {
        input.at<F32>(i) = static_cast<F32>(i + 1);
    }

    TensorMap inputs;
    inputs["in"].produced("source", "out", input);

    auto module = makePythonRuntimeSmokeModule();
    Parser::Map config;
    const auto createResult = module->create("python_runtime_smoke", config, inputs);

    if (!pythonRuntimeAvailable(createResult)) {
        destroyUnavailablePythonModule(module, createResult);
        SKIP("Optional Python runtime is unavailable: " << JST_LOG_LAST_ERROR());
    }

    REQUIRE(createResult == Result::SUCCESS);

    auto runtime = std::make_shared<Runtime>("python", DeviceType::CPU, RuntimeType::PYTHON);
    REQUIRE(runtime->create({{"python_runtime_smoke", module}}) == Result::SUCCESS);

    Result computeResult = Result::ERROR;
    std::unordered_set<std::string> skippedModules;
    std::unordered_set<std::string> failedModules;
    std::thread computeThread([&] {
        computeResult = runtime->compute({"python_runtime_smoke"}, skippedModules, failedModules);
    });
    computeThread.join();
    REQUIRE(computeResult == Result::SUCCESS);
    REQUIRE(skippedModules.empty());
    REQUIRE(failedModules.empty());

    Tensor output = module->outputs().at("out").tensor;
    const auto* data = output.data<F32>();
    REQUIRE(data != nullptr);

    for (Index i = 0; i < output.size(); ++i) {
        REQUIRE(std::abs(data[i] - static_cast<F32>((i + 1) * 3)) < 1e-5f);
    }

    REQUIRE(runtime->destroy() == Result::SUCCESS);
    destroyPythonCompute(module);
    REQUIRE(module->destroy() == Result::SUCCESS);
}

TEST_CASE("Python runtime handles NumPy buffering on a worker thread", "[core][runtime][python]") {
    Tensor input;
    REQUIRE(input.create(DeviceType::CPU, DataType::F32, {16}) == Result::SUCCESS);

    for (Index i = 0; i < input.size(); ++i) {
        input.at<F32>(i) = static_cast<F32>(i + 1);
    }

    TensorMap inputs;
    inputs["in"].produced("source", "out", input);

    auto module = makePythonRuntimeSmokeModule(R"PY(
import gc as _jetstream_gc
import numpy as _jetstream_np

_state = globals().setdefault('_numpy_buffer_state', {
    'buffer': _jetstream_np.zeros(0, dtype=_jetstream_np.float32),
    'last': 0.0,
})

def compute(ctx):
    mono = _jetstream_np.asarray(ctx.inputs[0], dtype=_jetstream_np.float32).reshape(-1)
    if mono.size == 0:
        return

    for _ in range(16):
        owned = mono.copy()
        decimated = owned[::2]
        _state['buffer'] = _jetstream_np.concatenate([_state['buffer'], decimated])

        if _state['buffer'].size >= 8:
            chunk = _state['buffer'][:8]
            _state['buffer'] = _state['buffer'][8:]
            peak = float(_jetstream_np.max(_jetstream_np.abs(chunk))) if chunk.size else 0.0
            if peak > 1e-6:
                chunk = chunk / max(1.0, peak)
            _state['last'] = float(_jetstream_np.sum(chunk))

    ctx.outputs[0][...] = ctx.inputs[0] * 7.0
    _jetstream_gc.collect()
)PY");
    Parser::Map config;
    const auto createResult = module->create("python_runtime_numpy_buffer", config, inputs);

    if (!pythonRuntimeAvailable(createResult)) {
        destroyUnavailablePythonModule(module, createResult);
        SKIP("Optional Python runtime is unavailable: " << JST_LOG_LAST_ERROR());
    }

    REQUIRE(createResult == Result::SUCCESS);

    auto runtime = std::make_shared<Runtime>("python", DeviceType::CPU, RuntimeType::PYTHON);
    REQUIRE(runtime->create({{"python_runtime_numpy_buffer", module}}) == Result::SUCCESS);

    Result computeResult = Result::ERROR;
    std::thread computeThread([&] {
        for (int iteration = 0; iteration < 16; ++iteration) {
            std::unordered_set<std::string> skippedModules;
            std::unordered_set<std::string> failedModules;
            const auto result = runtime->compute({"python_runtime_numpy_buffer"},
                                                 skippedModules,
                                                 failedModules);
            if (result != Result::SUCCESS || !skippedModules.empty() || !failedModules.empty()) {
                computeResult = result == Result::SUCCESS ? Result::ERROR : result;
                return;
            }
        }

        computeResult = Result::SUCCESS;
    });
    computeThread.join();
    REQUIRE(computeResult == Result::SUCCESS);

    Tensor output = module->outputs().at("out").tensor;
    const auto* data = output.data<F32>();
    REQUIRE(data != nullptr);

    for (Index i = 0; i < output.size(); ++i) {
        REQUIRE(std::abs(data[i] - static_cast<F32>((i + 1) * 7)) < 1e-5f);
    }

    REQUIRE(runtime->destroy() == Result::SUCCESS);
    destroyPythonCompute(module);
    REQUIRE(module->destroy() == Result::SUCCESS);
}

TEST_CASE("Python runtime runs cleanup hook during compute destruction", "[core][runtime][python]") {
    Tensor input;
    REQUIRE(input.create(DeviceType::CPU, DataType::F32, {4}) == Result::SUCCESS);

    for (Index i = 0; i < input.size(); ++i) {
        input.at<F32>(i) = static_cast<F32>(i + 1);
    }

    TensorMap inputs;
    inputs["in"].produced("source", "out", input);

    auto module = makePythonRuntimeSmokeModule(R"PY(
def compute(ctx):
    ctx.outputs[0][...] = ctx.inputs[0] * 11.0

def cleanup():
    print("__JETSTREAM_CLEANUP_CALLED__")
)PY");
    Parser::Map config;
    const auto createResult = module->create("python_runtime_cleanup", config, inputs);

    if (!pythonRuntimeAvailable(createResult)) {
        destroyUnavailablePythonModule(module, createResult);
        SKIP("Optional Python runtime is unavailable: " << JST_LOG_LAST_ERROR());
    }

    REQUIRE(createResult == Result::SUCCESS);

    Runtime runtime("python", DeviceType::CPU, RuntimeType::PYTHON);
    REQUIRE(runtime.create({{"python_runtime_cleanup", module}}) == Result::SUCCESS);

    std::unordered_set<std::string> skippedModules;
    std::unordered_set<std::string> failedModules;
    REQUIRE(runtime.compute({"python_runtime_cleanup"}, skippedModules, failedModules) == Result::SUCCESS);
    REQUIRE(skippedModules.empty());
    REQUIRE(failedModules.empty());

    auto pythonContext = std::dynamic_pointer_cast<PythonRuntimeContext>(module->context()->runtime());
    REQUIRE(pythonContext != nullptr);
    REQUIRE(pythonContext->destroyCompute() == Result::SUCCESS);

    const auto diagnostic = pythonContext->diagnostic();
    bool cleanupCalled = false;
    for (const auto& line : diagnostic.console) {
        cleanupCalled = cleanupCalled || line.find("__JETSTREAM_CLEANUP_CALLED__") != std::string::npos;
    }
    REQUIRE(cleanupCalled);

    REQUIRE(runtime.destroy() == Result::SUCCESS);
    REQUIRE(module->destroy() == Result::SUCCESS);
}

TEST_CASE("Python runtime cleans multiprocessing resources on compute destruction", "[core][runtime][python]") {
    Tensor input;
    REQUIRE(input.create(DeviceType::CPU, DataType::F32, {4}) == Result::SUCCESS);

    for (Index i = 0; i < input.size(); ++i) {
        input.at<F32>(i) = static_cast<F32>(i + 1);
    }

    TensorMap inputs;
    inputs["in"].produced("source", "out", input);

    auto module = makePythonRuntimeSmokeModule(R"PY(
try:
    import multiprocessing as mp
    mp_semaphore = mp.get_context("spawn").Semaphore(1)
except Exception:
    mp_semaphore = None

def compute(ctx):
    ctx.outputs[0][...] = ctx.inputs[0] * 13.0
)PY");
    Parser::Map config;
    const auto createResult = module->create("python_runtime_multiprocessing_cleanup", config, inputs);

    if (!pythonRuntimeAvailable(createResult)) {
        destroyUnavailablePythonModule(module, createResult);
        SKIP("Optional Python runtime is unavailable: " << JST_LOG_LAST_ERROR());
    }

    REQUIRE(createResult == Result::SUCCESS);

    Runtime runtime("python", DeviceType::CPU, RuntimeType::PYTHON);
    REQUIRE(runtime.create({{"python_runtime_multiprocessing_cleanup", module}}) == Result::SUCCESS);

    std::unordered_set<std::string> skippedModules;
    std::unordered_set<std::string> failedModules;
    REQUIRE(runtime.compute({"python_runtime_multiprocessing_cleanup"}, skippedModules, failedModules) == Result::SUCCESS);
    REQUIRE(skippedModules.empty());
    REQUIRE(failedModules.empty());

    REQUIRE(runtime.destroy() == Result::SUCCESS);
    destroyPythonCompute(module);
    REQUIRE(module->destroy() == Result::SUCCESS);

    auto verifier = makePythonRuntimeSmokeModule(R"PY(
def compute(ctx):
    import sys as _jetstream_sys
    import multiprocessing.util as _jetstream_mp_util

    pending = [
        key for key in _jetstream_mp_util._finalizer_registry
        if key[0] is not None and key[0] >= 0
    ]

    tracker = _jetstream_sys.modules.get("multiprocessing.resource_tracker")
    tracker_fd = -1 if tracker is None or tracker._resource_tracker._fd is None else tracker._resource_tracker._fd

    ctx.outputs[0][...] = 0.0
    ctx.outputs[0][0] = len(pending)
    ctx.outputs[0][1] = 0.0 if tracker_fd < 0 else 1.0
)PY");
    const auto verifyCreateResult = verifier->create("python_runtime_multiprocessing_verify", config, inputs);

    REQUIRE(verifyCreateResult == Result::SUCCESS);

    Runtime verifyRuntime("python", DeviceType::CPU, RuntimeType::PYTHON);
    REQUIRE(verifyRuntime.create({{"python_runtime_multiprocessing_verify", verifier}}) == Result::SUCCESS);

    skippedModules.clear();
    failedModules.clear();
    const auto verifyComputeResult = verifyRuntime.compute({"python_runtime_multiprocessing_verify"},
                                                           skippedModules,
                                                           failedModules);
    const auto verifyDiagnostic = verifier->context()->runtime()->diagnostic();
    for (const auto& line : verifyDiagnostic.console) {
        INFO(line);
    }
    REQUIRE(verifyComputeResult == Result::SUCCESS);
    REQUIRE(skippedModules.empty());
    REQUIRE(failedModules.empty());

    Tensor verifyOutput = verifier->outputs().at("out").tensor;
    const auto* verifyData = verifyOutput.data<F32>();
    REQUIRE(verifyData != nullptr);
    REQUIRE(verifyData[0] == 0.0f);
    REQUIRE(verifyData[1] == 0.0f);

    REQUIRE(verifyRuntime.destroy() == Result::SUCCESS);
    destroyPythonCompute(verifier);
    REQUIRE(verifier->destroy() == Result::SUCCESS);
}
