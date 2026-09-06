#include "runtime/python/dependencies/base.hh"

#include <toml++/toml.hpp>

#include <exception>
#include <string_view>
#include <utility>
#include <vector>

#include "jetstream/logger.hh"

namespace Jetstream {

namespace {

bool IsMetadataTypeCharacter(const unsigned char character) {
    return (character >= 'a' && character <= 'z') ||
           (character >= 'A' && character <= 'Z') ||
           (character >= '0' && character <= '9') ||
           character == '-';
}

bool MetadataBlockType(const std::string_view line, std::string_view& type) {
    constexpr std::string_view prefix = "# /// ";
    if (!line.starts_with(prefix)) {
        return false;
    }

    type = line.substr(prefix.size());
    if (type.empty()) {
        return false;
    }

    for (const unsigned char character : type) {
        if (!IsMetadataTypeCharacter(character)) {
            return false;
        }
    }

    return true;
}

bool IsEmbeddedContentLine(const std::string_view line) {
    return line == "#" || line.starts_with("# ");
}

std::vector<std::string_view> SplitLines(const std::string& source) {
    std::vector<std::string_view> lines;

    std::size_t start = 0;
    while (start < source.size()) {
        const auto end = source.find('\n', start);
        auto line = std::string_view(source).substr(
            start,
            end == std::string::npos ? std::string::npos : end - start);
        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1);
        }
        lines.push_back(line);

        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }

    return lines;
}

Result ExtractScriptBlock(const std::string& source, std::string& scriptBlock) {
    const auto lines = SplitLines(source);
    bool found = false;

    for (std::size_t i = 0; i < lines.size();) {
        std::string_view type;
        if (!MetadataBlockType(lines[i], type)) {
            ++i;
            continue;
        }

        std::string content;
        bool complete = false;
        std::size_t end = i + 1;
        for (; end < lines.size(); ++end) {
            const auto line = lines[end];
            const bool nextLineIsContent = end + 1 < lines.size() &&
                                           IsEmbeddedContentLine(lines[end + 1]);
            if (line == "# ///" && !nextLineIsContent) {
                complete = true;
                break;
            }
            if (!IsEmbeddedContentLine(line)) {
                break;
            }

            if (line.size() > 1) {
                content.append(line.substr(2));
            }
            content.push_back('\n');
        }

        if (!complete) {
            ++i;
            continue;
        }

        i = end + 1;
        if (type != "script") {
            continue;
        }
        if (found) {
            JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Multiple PEP 723 script metadata "
                      "blocks were found.");
            return Result::ERROR;
        }

        found = true;
        scriptBlock = std::move(content);
    }

    return Result::SUCCESS;
}

Result ReadScriptMetadata(const toml::table& table, PythonDependencyMetadata& metadata) {
    PythonDependencyMetadata parsed;

    if (const auto* requiresPython = table.get("requires-python")) {
        const auto* value = requiresPython->as_string();
        if (!value) {
            JST_ERROR("[RUNTIME_CONTEXT_PYTHON] The PEP 723 requires-python value must be "
                      "a TOML string.");
            return Result::ERROR;
        }
        parsed.requiresPython = value->get();
    }

    if (const auto* dependencies = table.get("dependencies")) {
        const auto* array = dependencies->as_array();
        if (!array) {
            JST_ERROR("[RUNTIME_CONTEXT_PYTHON] The PEP 723 dependencies value must be a "
                      "TOML array of strings.");
            return Result::ERROR;
        }

        parsed.requirements.reserve(array->size());
        for (const auto& dependency : *array) {
            const auto* value = dependency.as_string();
            if (!value) {
                JST_ERROR("[RUNTIME_CONTEXT_PYTHON] The PEP 723 dependencies array must "
                          "contain only strings.");
                return Result::ERROR;
            }
            if (value->get().empty()) {
                JST_ERROR("[RUNTIME_CONTEXT_PYTHON] The PEP 723 dependencies array cannot "
                          "contain an empty requirement.");
                return Result::ERROR;
            }
            parsed.requirements.push_back(value->get());
        }
    }

    metadata = std::move(parsed);
    return Result::SUCCESS;
}

}  // namespace

Result ParsePythonDependencyMetadata(const std::string& source,
                                     PythonDependencyMetadata& metadata) {
    metadata = {};

    std::string scriptBlock;
    JST_CHECK(ExtractScriptBlock(source, scriptBlock));
    if (scriptBlock.empty()) {
        return Result::SUCCESS;
    }

    try {
        const auto table = toml::parse(scriptBlock);
        return ReadScriptMetadata(table, metadata);
    } catch (const std::exception& exception) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Invalid PEP 723 TOML: {}", exception.what());
    } catch (...) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Invalid PEP 723 TOML.");
    }

    return Result::ERROR;
}

}  // namespace Jetstream
