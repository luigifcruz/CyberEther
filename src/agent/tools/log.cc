#include "support/base.hh"
#include "jetstream/logger.hh"
#include "tools/text.hh"

#include <array>

namespace Jetstream::Agent::Detail {
namespace {

struct Level {
    const char* name;
    const char* tag;
};

constexpr std::array<Level, 6> kLevels = {{
    {"trace", "[TRACE]"},
    {"debug", "[DEBUG]"},
    {"info", "[INFO]"},
    {"warn", "[WARN]"},
    {"error", "[ERROR]"},
    {"fatal", "[FATAL]"},
}};

std::optional<U64> Rank(const std::string& name) {
    for (U64 i = 0; i < kLevels.size(); ++i) {
        if (name == kLevels[i].name) {
            return i;
        }
    }
    return std::nullopt;
}

std::optional<std::pair<U64, std::string>> Parse(const std::string& line) {
    for (U64 i = 0; i < kLevels.size(); ++i) {
        if (!line.starts_with(kLevels[i].tag)) {
            continue;
        }
        const auto separator = line.find("| ");
        std::string message = separator == std::string::npos ? line.substr(std::string(kLevels[i].tag).size())
                                                              : line.substr(separator + 2);
        return std::pair{i, Text::TrimBlank(message)};
    }
    return std::nullopt;
}

Json ReadLog(const Invocation& call) {
    const auto& args = call.args;
    const std::string level = Text::ToLower(args.value("level", std::string{"warn"}));
    const auto minimum = Rank(level);
    if (!minimum) {
        return Response::Error(jst::fmt::format("Unknown level '{}'. Use trace, debug, info, warn, error, or fatal.",
                                                level));
    }

    const std::string query = Text::ToLower(args.value("query", std::string{}));
    const U64 limit = Arguments::Clamp(args, "limit", 40, 1, 256);

    std::vector<Json> matches;
    for (const auto& line : JST_LOG_RECENT()) {
        const auto parsed = Parse(line);
        if (!parsed || parsed->first < *minimum) {
            continue;
        }
        if (!query.empty() && Text::ToLower(parsed->second).find(query) == std::string::npos) {
            continue;
        }
        matches.push_back({{"level", kLevels[parsed->first].name}, {"message", parsed->second}});
    }

    const U64 total = matches.size();
    const U64 first = total > limit ? total - limit : 0;

    Json entries = Json::array();
    for (U64 i = first; i < total; ++i) {
        entries.push_back(std::move(matches[i]));
    }

    Json body = {{"entries", std::move(entries)}, {"matched", total}};
    if (first > 0) {
        body["omitted_older"] = first;
    }
    return body;
}

}  // namespace

std::vector<Entry> Toolbox::Log() {
    return {
        Entry::FromJson({
            "read_log",
            "Read recent CyberEther application log entries, oldest first, at or above a severity level. Covers failures that are not attached to a block, such as devices, files, sockets, and remote sessions. Only recent entries are kept.",
            R"({"type":"object","properties":{"level":{"type":"string","enum":["trace","debug","info","warn","error","fatal"],"description":"Minimum severity, default warn."},"query":{"type":"string","description":"Optional case-insensitive text filter."},"limit":{"type":"integer","minimum":1,"maximum":256,"description":"Maximum newest entries returned, default 40."}},"additionalProperties":false})",
            {},
        }, ReadLog),
    };
}

}  // namespace Jetstream::Agent::Detail
