#include "support/base.hh"

#include "jetstream/platform.hh"
#include "tools/text.hh"

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <optional>
#include <set>
#include <string_view>
#include <system_error>

namespace Jetstream::Agent::Detail {
namespace {

constexpr U64 kMaxListed = 200;
constexpr U64 kMaxBlocksShown = 12;
constexpr U64 kMaxPathBytes = 4096;
constexpr U64 kMaxSuggested = 64;
constexpr std::uintmax_t kMaxFileBytes = 16 * 1024 * 1024;
constexpr std::array<std::string_view, 6> kDocumentKeys = {
    "title:", "summary:", "author:", "license:", "description:", "meta:",
};

struct Saved {
    std::string path;
    std::string name;
    std::string folder;
    std::string title;
    std::string summary;
    std::vector<std::string> blocks;
    U64 blockCount = 0;
};

// File names and file contents are not guaranteed to be UTF-8, and a single bad byte
// would make the whole reply unserializable.
bool ValidUtf8(const std::string& text) {
    const Json value(text);
    return value.dump(-1, ' ', false, Json::error_handler_t::ignore) ==
           value.dump(-1, ' ', false, Json::error_handler_t::replace);
}

std::string Clean(const std::string& text) {
    if (ValidUtf8(text)) {
        return text;
    }
    const auto replaced = Json(text).dump(-1, ' ', false, Json::error_handler_t::replace);
    return Json::parse(replaced).get<std::string>();
}

std::string Value(const std::string& line, const U64 offset) {
    std::string value = line.substr(offset);
    const auto first = value.find_first_not_of(" \t");
    const auto last = value.find_last_not_of(" \t");
    if (first == std::string::npos) {
        return {};
    }

    value = value.substr(first, last - first + 1);
    const bool quoted = value.size() >= 2 && (value.front() == '\'' || value.front() == '"') &&
                        value.back() == value.front();
    return Clean(quoted ? value.substr(1, value.size() - 2) : value);
}

bool Integer(const std::string& value) {
    return !value.empty() && std::all_of(value.begin(), value.end(), [](const unsigned char c) {
        return std::isdigit(c);
    });
}

// A flowgraph document has a top-level integer `version:` and a `graph:` key that opens a
// sequence. Only the document's own keys and block names are read, so listing never loads
// a graph and a file that merely resembles one reveals nothing else.
bool Describe(const std::filesystem::path& file, Saved& saved) {
    std::ifstream input(file);
    if (!input) {
        return false;
    }

    bool versioned = false;
    bool graph = false;
    bool inGraph = false;
    std::optional<U64> itemIndent;
    std::string line;
    for (U64 number = 0; std::getline(input, line); ++number) {
        if (number == 0 && line.starts_with("\xEF\xBB\xBF")) {
            line.erase(0, 3);
        }
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        const auto indent = line.find_first_not_of(' ');
        if (indent == std::string::npos || line[indent] == '#') {
            continue;
        }

        if (indent != 0) {
            const bool item = inGraph && line.compare(indent, 2, "- ") == 0;
            if (!item || indent != itemIndent.value_or(indent)) {
                continue;
            }
            itemIndent = indent;
            ++saved.blockCount;
            if (line.compare(indent, 7, "- name:") == 0 && saved.blocks.size() < kMaxBlocksShown) {
                saved.blocks.push_back(Value(line, indent + 7));
            }
            continue;
        }

        if (line.starts_with("- ") && inGraph && itemIndent.value_or(0) == 0) {
            itemIndent = 0;
            ++saved.blockCount;
            if (line.starts_with("- name:") && saved.blocks.size() < kMaxBlocksShown) {
                saved.blocks.push_back(Value(line, 7));
            }
            continue;
        }

        inGraph = false;
        if (line.starts_with("---") || line.starts_with("%")) {
            continue;
        }
        if (line.starts_with("version:")) {
            if (!Integer(Value(line, 8))) {
                return false;
            }
            versioned = true;
        } else if (line.starts_with("graph:")) {
            if (!Value(line, 6).empty()) {
                return false;
            }
            graph = true;
            inGraph = true;
        } else if (line.starts_with("title:")) {
            saved.title = Value(line, 6);
        } else if (line.starts_with("summary:")) {
            saved.summary = Value(line, 8);
        } else if (!versioned && !graph &&
                   std::none_of(kDocumentKeys.begin(), kDocumentKeys.end(), [&](const std::string_view key) {
                       return line.starts_with(key);
                   })) {
            return false;
        }
    }

    return versioned && graph;
}

std::vector<Saved> Collect(Host& host, std::vector<std::string>& folders) {
    std::vector<Saved> found;
    std::set<std::string> seen;

    for (const auto& configured : host.flowgraphFolders()) {
        std::error_code error;
        const auto folder = std::filesystem::canonical(Platform::PathFromUtf8(configured), error);
        if (error || !std::filesystem::is_directory(folder, error)) {
            continue;
        }
        folders.push_back(Platform::PathToUtf8(folder));

        for (std::filesystem::directory_iterator it(folder, error), end; !error && it != end; it.increment(error)) {
            std::error_code status;
            if (!it->is_regular_file(status) || it->file_size(status) > kMaxFileBytes || status) {
                continue;
            }

            const auto extension = Text::ToLower(Platform::PathToUtf8(it->path().extension()));
            if (!extension.empty() && extension != ".yml" && extension != ".yaml") {
                continue;
            }

            const auto path = std::filesystem::canonical(it->path(), status);
            const bool usable = !status && path.parent_path() == folder && ValidUtf8(Platform::PathToUtf8(path));
            if (!usable || !seen.insert(Platform::PathToUtf8(path)).second) {
                continue;
            }

            Saved saved{
                .path = Platform::PathToUtf8(path),
                .name = Platform::PathToUtf8(path.stem()),
                .folder = folders.back(),
            };
            if (Describe(path, saved)) {
                found.push_back(std::move(saved));
            }
        }
    }

    std::sort(found.begin(), found.end(), [](const Saved& a, const Saved& b) {
        return a.path < b.path;
    });
    return found;
}

std::optional<Saved> Find(Host& host, const std::string& path, std::string& error) {
    if (path.size() > kMaxPathBytes || !ValidUtf8(path)) {
        error = "That is not a saved flowgraph path. Use an exact path from list_flowgraphs.";
        return std::nullopt;
    }

    std::vector<std::string> folders;
    const auto saved = Collect(host, folders);

    // Suggestions compare file names only: comparing whole paths costs far more and the
    // folder part is the same for every candidate.
    std::vector<std::string> names;
    for (const auto& entry : saved) {
        if (entry.path == path) {
            return entry;
        }
        if (names.size() < kMaxSuggested) {
            names.push_back(entry.name);
        }
    }

    const auto wanted = Platform::PathToUtf8(Platform::PathFromUtf8(path).stem());
    error = jst::fmt::format("Saved flowgraph '{}' was not found. Use an exact path from list_flowgraphs.{}",
                             path, Lookup::Similar(wanted, names, "saved flowgraph names"));
    return std::nullopt;
}

Json ListFlowgraphs(const Invocation& call) {
    std::vector<std::string> folders;
    const auto saved = Collect(call.host, folders);

    Json entries = Json::array();
    for (const auto& entry : saved) {
        if (entries.size() >= kMaxListed) {
            break;
        }

        Json item = {
            {"path", entry.path},
            {"name", entry.name},
            {"block_count", entry.blockCount},
            {"blocks", entry.blocks},
        };
        if (!entry.title.empty()) {
            item["title"] = entry.title;
        }
        if (!entry.summary.empty()) {
            item["summary"] = entry.summary;
        }
        entries.push_back(std::move(item));
    }

    Json body = {
        {"folders", folders},
        {"count", saved.size()},
        {"flowgraphs", std::move(entries)},
    };
    if (saved.size() > kMaxListed) {
        body["truncated"] = true;
    }
    if (folders.empty()) {
        body["note"] = "No flowgraph folders are known yet. A folder is remembered when the user opens or "
                       "saves a flowgraph there, and more can be listed under intelligence.flowgraphFolders "
                       "in the settings file.";
    }
    return body;
}

std::string GraphName(const std::string& name) {
    std::string result;
    for (const char c : name) {
        result += std::isalnum(static_cast<unsigned char>(c)) ? c : '_';
    }
    return result.empty() ? "flowgraph" : result;
}

Entry::NewGraph SavedGraph(Host& host, const Json& args) {
    if (!args.is_object() || !args.contains("path") || !args["path"].is_string()) {
        return {{}, "open_flowgraph requires a string 'path' from list_flowgraphs."};
    }

    std::string missing;
    const auto saved = Find(host, args["path"].get<std::string>(), missing);
    if (!saved.has_value()) {
        return {{}, missing};
    }
    return {GraphName(saved->name), {}};
}

Tools::Output OpenFlowgraph(const Invocation& call) {
    std::string missing;
    const auto saved = Find(call.host, call.args.value("path", std::string{}), missing);
    if (!saved.has_value()) {
        return Response::Failure(missing);
    }

    const auto imported = call.flowgraph->importFromFile(saved->path);

    Json body = {
        {"opened", call.created},
        {"path", saved->path},
        {"name", saved->name},
    };
    try {
        body["flowgraph"] = Serialize::FlowgraphDetail(call.flowgraph);
    } catch (const std::exception& error) {
        body["flowgraph"] = Response::Error(jst::fmt::format("Could not summarize the graph: {}", error.what()));
    }
    if (imported == Result::ERROR) {
        body["error"] = jst::fmt::format("The file could not be loaded, so the new graph '{}' is empty. "
                                         "The file itself is unchanged. Tell the user to close that graph "
                                         "without saving, because saving it would overwrite the file.",
                                         call.created);
    } else if (imported != Result::SUCCESS) {
        body["error"] = "The flowgraph opened with errors, so some of its blocks may not be ready.";
    }

    return Response::Reply(body);
}

}  // namespace

std::vector<Entry> Toolbox::Saved() {
    Entry open{
        {
            "open_flowgraph",
            "Open one of the user's saved flowgraph files in a new graph and select it for subsequent "
            "calls. path must be an exact path returned by list_flowgraphs; other files cannot be opened. "
            "Returns the name of the new graph and its blocks and connections.",
            R"({"type":"object","required":["path"],"properties":{"path":{"type":"string","description":"Exact path from list_flowgraphs."}},"additionalProperties":false})",
            {.mutates = true},
        },
        OpenFlowgraph,
    };
    open.newGraph = SavedGraph;

    return {
        Entry::FromJson({
            "list_flowgraphs",
            "List the user's own saved flowgraph files, such as dashboards and projects they built. "
            "Looks in the folders where the user has opened or saved flowgraphs. Returns each file's "
            "path, name, block count and first block names, plus title and summary when the file has "
            "them. Match the user's wording against names and block names, then open the file with "
            "open_flowgraph. Bundled examples are separate and come from list_examples.",
            R"({"type":"object","properties":{},"additionalProperties":false})",
        }, ListFlowgraphs),
        std::move(open),
    };
}

}  // namespace Jetstream::Agent::Detail
