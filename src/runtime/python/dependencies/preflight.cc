#include "runtime/python/dependencies/base.hh"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "jetstream/logger.hh"
#include "jetstream/platform.hh"
#include "runtime/python/bridge/cpython/base.hh"
#include "runtime/python/dependencies/preflight.hh"

namespace Jetstream {

namespace {

constexpr std::string_view kResultPrefix = "JETSTREAM_PEP723_RESULT=";
constexpr U64 kPreflightTimeoutMilliseconds = 10000;
constexpr std::size_t kMaximumValidationCacheEntries = 64;

struct ValidationResult {
    Result result = Result::ERROR;
    std::string error;
    std::vector<std::string> watchPaths;
};

struct ValidationCacheEntry {
    ValidationResult validation;
    std::string fingerprint;
};

std::mutex& ValidationCacheMutex() {
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<std::string, ValidationCacheEntry>& ValidationCache() {
    static std::unordered_map<std::string, ValidationCacheEntry> cache;
    return cache;
}

bool ContainsProhibitedCharacter(const std::string& value) {
    return value.find('\0') != std::string::npos ||
           value.find('\r') != std::string::npos ||
           value.find('\n') != std::string::npos;
}

void AppendKeyPart(std::string& key, const std::string& value) {
    key.append(std::to_string(value.size()));
    key.push_back(':');
    key.append(value);
}

std::string ValidationCacheKey(const std::string& programPath,
                               const PythonDependencyMetadata& metadata) {
    std::string key;
    AppendKeyPart(key, programPath);
    AppendKeyPart(key, metadata.requiresPython);
    for (const auto& requirement : metadata.requirements) {
        AppendKeyPart(key, requirement);
    }
    for (const char* name : {
             "PYTHONNOUSERSITE",
             "PYTHONUSERBASE",
             "HOME",
             "USERPROFILE",
             "APPDATA",
             "__PYVENV_LAUNCHER__",
         }) {
        std::string value;
        const bool isSet =
            Platform::EnvironmentVariable(name, value) == Result::SUCCESS;
        AppendKeyPart(key, isSet ? "set" : "unset");
        if (isSet) {
            AppendKeyPart(key, value);
        }
    }
    return key;
}

std::optional<std::string> EnvironmentFingerprint(const std::string& programPath,
                                                  const std::vector<std::string>& watchPaths) {
    auto paths = watchPaths;
    paths.push_back(programPath);
    std::ranges::sort(paths);
    paths.erase(std::unique(paths.begin(), paths.end()), paths.end());

    std::string fingerprint;
    for (const auto& value : paths) {
        std::error_code ec;
        const auto path = std::filesystem::weakly_canonical(
            Platform::PathFromUtf8(value), ec);
        if (ec) {
            return std::nullopt;
        }

        const auto status = std::filesystem::status(path, ec);
        if (ec || (!std::filesystem::is_regular_file(status) &&
                   !std::filesystem::is_directory(status))) {
            return std::nullopt;
        }

        const auto modified = std::filesystem::last_write_time(path, ec);
        if (ec) {
            return std::nullopt;
        }

        AppendKeyPart(fingerprint, Platform::PathToUtf8(path));
        AppendKeyPart(fingerprint,
                      std::filesystem::is_directory(status) ? "directory" : "file");
        AppendKeyPart(fingerprint,
                      std::to_string(static_cast<long long>(
                          modified.time_since_epoch().count())));
        if (std::filesystem::is_regular_file(status)) {
            const auto size = std::filesystem::file_size(path, ec);
            if (ec) {
                return std::nullopt;
            }
            AppendKeyPart(fingerprint, std::to_string(size));
        }
    }

    return fingerprint;
}

Result ReportValidation(const ValidationResult& validation) {
    if (validation.result != Result::SUCCESS) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] {}", validation.error);
    }
    return validation.result;
}

Result RunValidation(const std::string& programPath,
                     const PythonDependencyMetadata& metadata,
                     ValidationResult& validation) {
    std::vector<std::string> arguments = {
        "-I",
        "-X",
        "utf8",
        "-c",
        kPythonDependencyPreflight,
        metadata.requiresPython,
    };
    arguments.insert(arguments.end(),
                     metadata.requirements.begin(),
                     metadata.requirements.end());

    std::string output;
    if (Platform::RunProcess(programPath,
                             arguments,
                             output,
                             kPreflightTimeoutMilliseconds,
                             true) != Result::SUCCESS) {
        validation.error = "Failed to run Python requirement preflight";
        if (!output.empty()) {
            validation.error += ": " + output;
        } else {
            validation.error += ".";
        }
        return Result::ERROR;
    }

    bool resultFound = false;
    std::string payload;
    std::size_t lineStart = 0;
    while (lineStart < output.size()) {
        const auto lineEnd = output.find('\n', lineStart);
        auto line = std::string_view(output).substr(
            lineStart,
            lineEnd == std::string::npos ? std::string::npos : lineEnd - lineStart);
        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1);
        }

        if (line.starts_with(kResultPrefix)) {
            if (resultFound) {
                validation.error = "Python requirement preflight returned multiple results.";
                return Result::ERROR;
            }
            resultFound = true;
            payload = line.substr(kResultPrefix.size());
        }

        if (lineEnd == std::string::npos) {
            break;
        }
        lineStart = lineEnd + 1;
    }

    if (!resultFound) {
        validation.error = "Python requirement preflight returned no result.";
        return Result::ERROR;
    }

    try {
        const auto response = nlohmann::json::parse(payload);
        validation.watchPaths = response.at("watch").get<std::vector<std::string>>();
        if (!response.at("valid").get<bool>()) {
            validation.error = response.value("error", "Python requirement preflight failed.");
            return Result::ERROR;
        }
    } catch (const std::exception& exception) {
        validation.error = jst::fmt::format(
            "Python requirement preflight returned an invalid result: {}.",
            exception.what());
        return Result::ERROR;
    }

    validation.result = Result::SUCCESS;
    return Result::SUCCESS;
}

}  // namespace

Result ValidatePythonDependencyMetadata(const PythonDependencyMetadata& metadata) {
    if (metadata.requirements.empty() && metadata.requiresPython.empty()) {
        return Result::SUCCESS;
    }

    for (std::size_t i = 0; i < metadata.requirements.size(); ++i) {
        if (metadata.requirements[i].empty()) {
            JST_ERROR("[RUNTIME_CONTEXT_PYTHON] The PEP 723 dependency at index {} is empty.", i);
            return Result::ERROR;
        }
        if (ContainsProhibitedCharacter(metadata.requirements[i])) {
            JST_ERROR("[RUNTIME_CONTEXT_PYTHON] The PEP 723 dependency at index {} contains "
                      "a prohibited NUL, CR, or LF character.", i);
            return Result::ERROR;
        }
    }

    if (ContainsProhibitedCharacter(metadata.requiresPython)) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] The PEP 723 requires-python value contains a prohibited "
                  "NUL, CR, or LF character.");
        return Result::ERROR;
    }

    JST_CHECK(CPython::Py_Load());
    const auto& programPath = CPython::Py_ProgramPath();
    if (programPath.empty()) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] The selected Python runtime has no executable "
                  "for PEP 723 requirement validation.");
        return Result::ERROR;
    }

    const auto key = ValidationCacheKey(programPath, metadata);
    std::optional<ValidationCacheEntry> cached;
    {
        std::lock_guard lock(ValidationCacheMutex());
        if (const auto entry = ValidationCache().find(key);
            entry != ValidationCache().end()) {
            cached = entry->second;
        }
    }

    if (cached) {
        const auto fingerprint = EnvironmentFingerprint(
            programPath, cached->validation.watchPaths);
        if (fingerprint && *fingerprint == cached->fingerprint) {
            return ReportValidation(cached->validation);
        }

        std::lock_guard lock(ValidationCacheMutex());
        if (const auto entry = ValidationCache().find(key);
            entry != ValidationCache().end() &&
            entry->second.fingerprint == cached->fingerprint) {
            ValidationCache().erase(entry);
        }
    }

    ValidationResult validation;
    RunValidation(programPath, metadata, validation);
    if (!validation.watchPaths.empty()) {
        const auto fingerprintBefore = EnvironmentFingerprint(
            programPath, validation.watchPaths);
        ValidationResult confirmedValidation;
        RunValidation(programPath, metadata, confirmedValidation);
        const auto fingerprintAfter = EnvironmentFingerprint(
            programPath, confirmedValidation.watchPaths);
        if (fingerprintBefore && fingerprintAfter &&
            validation.watchPaths == confirmedValidation.watchPaths &&
            *fingerprintBefore == *fingerprintAfter) {
            std::lock_guard lock(ValidationCacheMutex());
            auto& cache = ValidationCache();
            if (!cache.contains(key) &&
                cache.size() >= kMaximumValidationCacheEntries) {
                cache.erase(cache.begin());
            }
            cache.insert_or_assign(
                key,
                ValidationCacheEntry{
                    .validation = confirmedValidation,
                    .fingerprint = *fingerprintAfter,
                });
        }
        validation = std::move(confirmedValidation);
    }
    return ReportValidation(validation);
}

}  // namespace Jetstream
