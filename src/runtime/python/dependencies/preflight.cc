#include "runtime/python/dependencies/base.hh"

#include <nlohmann/json.hpp>

#include <string>
#include <string_view>
#include <vector>

#include "jetstream/logger.hh"
#include "jetstream/platform.hh"
#include "runtime/python/bridge/cpython/base.hh"
#include "runtime/python/dependencies/preflight.hh"

namespace Jetstream {

namespace {

constexpr std::string_view kResultPrefix = "JETSTREAM_PEP723_RESULT=";
constexpr U64 kPreflightTimeoutMilliseconds = 10000;

bool ContainsProhibitedCharacter(const std::string& value) {
    return value.find('\0') != std::string::npos ||
           value.find('\r') != std::string::npos ||
           value.find('\n') != std::string::npos;
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
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Failed to run Python requirement preflight. {}", output);
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
                JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Python requirement preflight returned multiple results.");
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
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Python requirement preflight returned no result.");
        return Result::ERROR;
    }

    try {
        const auto response = nlohmann::json::parse(payload);
        if (!response.at("valid").get<bool>()) {
            JST_ERROR("[RUNTIME_CONTEXT_PYTHON] {}",
                      response.value("error", "Python requirement preflight failed."));
            return Result::ERROR;
        }
    } catch (const std::exception& exception) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Python requirement preflight returned an invalid result: {}.",
                  exception.what());
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream
