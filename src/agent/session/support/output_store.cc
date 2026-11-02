#include "base.hh"

#include "jetstream/config.hh"
#include "tools/random.hh"
#include "tools/text.hh"

#include <algorithm>
#include <string_view>

#include <nlohmann/json.hpp>

namespace Jetstream::Agent::Detail {

namespace {

constexpr U64 kArtifactBytes = 16 * 1024 * 1024;
constexpr U64 kArtifactCount = 64;

Tools::Output Failure(const char* json) {
    return {json, {}, true};
}

}  // namespace

OutputStore::OutputStore() : identity(Random::HexToken()) {}

Provider::Function OutputStore::Schema() {
    return {
        "read_tool_result",
        "Read a retained oversized tool result by ID. Offsets and limits are "
        "UTF-8 byte offsets; use next_offset to continue. Results expire when "
        "context is cleared or the bounded store evicts them; then inspect "
        "the original resource with narrower filters.",
        R"({"type":"object","required":["id"],"properties":{)"
        R"("id":{"type":"string"},"offset":{"type":"integer","minimum":0},)"
        R"("limit":{"type":"integer","minimum":1,"maximum":8192}},)"
        R"("additionalProperties":false})",
    };
}

std::string OutputStore::bound(std::string content, U64 limit, bool failed) {
    limit = std::max<U64>(1024, limit);
    if (content.size() <= limit) {
        return content;
    }

    using Json = nlohmann::json;

    const U64 previewBytes = (limit - 512) / 6;
    Json result = {
        {"truncated", true},
        {"original_bytes", content.size()},
        {"preview", std::string(Text::Utf8Prefix(content, previewBytes))},
        {"note", "Use read_tool_result with result_id, or inspect the original "
                 "resource with narrower filters."},
    };

    if (failed) {
        const auto original = Json::parse(content, nullptr, false);
        const bool partial = original.is_object() && original.value("status", std::string{}) == "partial";
        if (partial) {
            result["status"] = "partial";
        } else {
            result["error"] = "Tool failed; read the retained result for details.";
        }
    }

    if (content.size() > kArtifactBytes) {
        result["note"] = "Result exceeds retained storage capacity. "
                         "Re-run the original inspection with narrower filters.";
        return result.dump(-1, ' ', false, Json::error_handler_t::replace);
    }

    while (!entries.empty()) {
        const bool overBytes = bytes + content.size() > kArtifactBytes;
        const bool overCount = entries.size() >= kArtifactCount;
        if (!overBytes && !overCount) {
            break;
        }

        bytes -= entries.front().second.size();
        entries.pop_front();
    }

    auto id = "result-" + identity + "-" + std::to_string(++next);
    bytes += content.size();
    entries.emplace_back(id, std::move(content));
    result["result_id"] = std::move(id);

    return result.dump(-1, ' ', false, Json::error_handler_t::replace);
}

Tools::Output OutputStore::read(const std::string& args, U64 maxBytes) const {
    using Json = nlohmann::json;

    const auto input = Json::parse(args, nullptr, false);
    if (!input.is_object() || !input.contains("id") || !input["id"].is_string()) {
        return Failure(R"({"error":"A string id is required."})");
    }

    for (const char* key : {"offset", "limit"}) {
        if (!input.contains(key)) {
            continue;
        }

        const bool zeroLimit = std::string_view(key) == "limit" && input[key] == 0;
        if (!input[key].is_number_unsigned() || zeroLimit) {
            return Failure(R"({"error":"offset must be non-negative; limit must be positive."})");
        }
    }

    const auto id = input["id"].get<std::string>();
    const auto found = std::find_if(entries.begin(), entries.end(), [&](const auto& entry) {
        return entry.first == id;
    });
    if (found == entries.end()) {
        return Failure(R"({"error":"Result expired or unknown."})");
    }

    const auto& text = found->second;
    U64 offset = std::min<U64>(input.value("offset", U64{0}), text.size());
    while (offset < text.size() && (static_cast<unsigned char>(text[offset]) & 0xc0) == 0x80) {
        ++offset;
    }

    const U64 ceiling = std::clamp<U64>((std::max<U64>(maxBytes, 1024) - 512) / 6, 4, 8192);
    const U64 limit = std::clamp<U64>(input.value("limit", U64{4096}), 4, ceiling);
    auto excerpt = std::string(Text::Utf8Prefix(std::string_view(text).substr(offset), limit));

    Json result = {
        {"id", found->first},
        {"offset", offset},
        {"total_bytes", text.size()},
        {"text", excerpt},
    };
    if (offset + excerpt.size() < text.size()) {
        result["next_offset"] = offset + excerpt.size();
    }

    return {result.dump(-1, ' ', false, Json::error_handler_t::replace)};
}

}  // namespace Jetstream::Agent::Detail
