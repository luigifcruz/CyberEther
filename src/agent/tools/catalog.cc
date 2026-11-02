#include "support/base.hh"
#include "tools/text.hh"

#include <cctype>

namespace Jetstream::Agent::Detail {
namespace {

std::optional<Registry::FlowgraphRegistration> FindExample(const std::string& key, std::string& error) {
    if (!key.empty()) {
        const auto examples = Registry::ListAvailableFlowgraphs(key);
        if (!examples.empty()) {
            return examples.front();
        }
    }

    std::vector<std::string> keys;
    for (const auto& example : Registry::ListAvailableFlowgraphs()) {
        keys.push_back(example.key);
    }

    error = jst::fmt::format("Example '{}' not found.{}", key, Lookup::Similar(key, keys, "example keys"));
    return std::nullopt;
}

std::vector<std::string> QueryTerms(const std::string& query) {
    std::vector<std::string> terms;
    for (const auto& term : Parser::SplitString(query, " ")) {
        if (!term.empty()) {
            terms.push_back(term);
        }
    }
    return terms;
}

std::shared_ptr<Block> AddDefaultWidth(Host& host, const std::string& type, Json& target) {
    std::shared_ptr<Block> prototype;
    if (Registry::BuildBlock(type, prototype) != Result::SUCCESS) {
        return nullptr;
    }

    if (const auto width = host.width(prototype->config().nodeSize())) {
        target["default_width"] = *width;
    }

    return prototype;
}

bool MatchesFilter(const Json& implementation, const char* key, const std::string& filter) {
    if (filter.empty()) {
        return true;
    }
    return Text::ToLower(implementation.value(key, std::string{})) == filter;
}

I64 TermScore(const std::string& type, const std::string& title, const std::string& term) {
    if (type.find(term) != std::string::npos) {
        return 8;
    }
    if (title.find(term) != std::string::npos) {
        return 5;
    }
    return 2;
}

Json SearchBlocks(const Invocation& call) {
    const auto& args = call.args;
    const std::string query = Text::ToLower(args.value("query", std::string{}));
    const std::string domain = Text::ToLower(args.value("domain", std::string{}));
    const std::string device = Text::ToLower(args.value("device", std::string{}));
    const std::string runtime = Text::ToLower(args.value("runtime", std::string{}));
    const std::string provider = Text::ToLower(args.value("provider", std::string{}));
    const U64 limit = Arguments::Clamp(args, "limit", 15, 1, 50);
    const auto terms = QueryTerms(query);
    const bool filtered = !device.empty() || !runtime.empty() || !provider.empty();

    std::vector<std::pair<I64, Json>> matches;
    for (const auto& block : Registry::ListAvailableBlocks()) {
        if (!domain.empty() && Text::ToLower(block.domain).find(domain) == std::string::npos) {
            continue;
        }

        Json implementations = Json::array();
        for (const auto& implementation : Serialize::Implementations(block.type)) {
            if (!MatchesFilter(implementation, "device", device) ||
                !MatchesFilter(implementation, "runtime", runtime) ||
                !MatchesFilter(implementation, "provider", provider)) {
                continue;
            }
            implementations.push_back(implementation);
        }
        if (filtered && implementations.empty()) {
            continue;
        }

        const std::string type = Text::ToLower(block.type);
        const std::string title = Text::ToLower(block.title);
        const std::string haystack = jst::fmt::format("{} {} {} {} {}",
                                                      type,
                                                      title,
                                                      Text::ToLower(block.domain),
                                                      Text::ToLower(block.summary),
                                                      Text::ToLower(block.description));

        I64 score = terms.empty() ? 1 : 0;
        for (const auto& term : terms) {
            if (haystack.find(term) != std::string::npos) {
                score += TermScore(type, title, term);
            }
        }
        if (!terms.empty() && score == 0) {
            continue;
        }

        matches.push_back({score, {
            {"type", block.type},
            {"title", block.title},
            {"domain", block.domain},
            {"summary", block.summary},
            {"implementations", std::move(implementations)},
        }});
    }

    std::sort(matches.begin(), matches.end(), [](const auto& a, const auto& b) {
        if (a.first != b.first) {
            return a.first > b.first;
        }
        return a.second.value("type", "") < b.second.value("type", "");
    });

    Json results = Json::array();
    for (U64 i = 0; i < std::min<U64>(limit, matches.size()); ++i) {
        auto& match = matches[i].second;
        AddDefaultWidth(call.host, match["type"].get<std::string>(), match);
        results.push_back(std::move(match));
    }

    return Json({{"query", query}, {"matches", std::move(results)}});
}

Json GetBlockDocs(const Invocation& call) {
    const auto& flowgraph = call.flowgraph;
    const auto& args = call.args;

    const std::string type = args.value("type", std::string{});
    if (type.empty()) {
        return Response::Error("'type' is required.");
    }

    const auto registrations = Registry::ListAvailableBlocks(type);
    if (registrations.empty()) {
        return Response::Error(jst::fmt::format("Block type '{}' not found.{}",
                                                type, Lookup::Similar(type, Lookup::Types(), "block types")));
    }

    const auto& block = registrations.front();
    Json j = {
        {"type", block.type},
        {"domain", block.domain},
        {"title", block.title},
        {"summary", block.summary},
        {"description", block.description},
    };

    Parser::Map defaultConfig;
    const auto prototype = AddDefaultWidth(call.host, type, j);
    if (prototype && prototype->config().serialize(defaultConfig) == Result::SUCCESS) {
        j["default_config"] = Serialize::CompactMap(defaultConfig);
    }
    j["implementations"] = Serialize::Implementations(block.type);

    Json observed = Json::array();
    std::vector<Flowgraph::View::BlockData> blocks;
    if (flowgraph && Serialize::Collect(flowgraph, blocks) == Result::SUCCESS) {
        for (const auto& data : blocks) {
            if (data.type != type) {
                continue;
            }

            if (observed.empty()) {
                j["config_fields"] = Serialize::Interface(data.interfaceConfigs);
            }

            observed.push_back({
                {"block", data.name},
                {"config", Serialize::CompactMap(data.config)},
                {"input_ports", Serialize::PortNames(data.interfaceInputs)},
                {"output_ports", Serialize::PortNames(data.interfaceOutputs)},
            });
        }
    }
    j["observed_interfaces"] = std::move(observed);

    return j;
}

Json ListExamples(const Graph&, const Json& args) {
    const std::string query = Text::ToLower(args.value("query", std::string{}));
    const auto terms = QueryTerms(query);
    const auto available = Registry::ListAvailableFlowgraphs();

    Json examples = Json::array();
    for (const auto& example : available) {
        const std::string about = jst::fmt::format("{} {} {} {}",
                                                   Text::ToLower(example.key),
                                                   Text::ToLower(example.title),
                                                   Text::ToLower(example.summary),
                                                   Text::ToLower(example.description));
        const std::string content = Text::ToLower(example.content);

        const auto inAbout = [&](const auto& term) {
            return about.find(term) != std::string::npos;
        };
        const auto found = [&](const auto& term) {
            return inAbout(term) || content.find(term) != std::string::npos;
        };
        if (!std::all_of(terms.begin(), terms.end(), found)) {
            continue;
        }

        Json item = {
            {"key", example.key},
            {"title", example.title},
            {"summary", example.summary},
        };
        if (!std::all_of(terms.begin(), terms.end(), inAbout)) {
            item["matched_in"] = "content";
        }
        examples.push_back(std::move(item));
    }

    return Json({
        {"query", query},
        {"total", available.size()},
        {"examples", std::move(examples)},
    });
}

Json GetExample(const Graph&, const Json& args) {
    const std::string key = args.value("key", std::string{});
    if (key.empty()) {
        return Response::Error("'key' is required.");
    }

    std::string error;
    const auto example = FindExample(key, error);
    if (!example.has_value()) {
        return Response::Error(error);
    }

    Json result = {
        {"key", example->key},
        {"title", example->title},
        {"summary", example->summary},
    };

    const bool ranged = args.contains("pattern") || args.contains("offset") || args.contains("limit");
    if (!ranged) {
        result["description"] = example->description;
        result["content"] = example->content;
        return result;
    }

    auto lines = BlockText::Range(example->content, args, error);
    if (!error.empty()) {
        return Response::Error(error);
    }

    result.update(lines);
    return result;
}

std::string ExampleGraphName(const std::string& key) {
    std::string name;
    for (const char c : key.substr(0, key.find_last_of('.'))) {
        name += std::isalnum(static_cast<unsigned char>(c)) ? c : '_';
    }
    return name.empty() ? "example" : name;
}

Entry::NewGraph ExampleGraph(const Json& args) {
    if (!args.is_object() || !args.contains("key") || !args["key"].is_string()) {
        return {{}, "open_example requires a string 'key' from list_examples."};
    }

    const auto key = args["key"].get<std::string>();
    if (key.empty()) {
        return {{}, "open_example requires a non-empty 'key' from list_examples."};
    }

    std::string missing;
    if (!FindExample(key, missing).has_value()) {
        return {{}, missing};
    }

    return {ExampleGraphName(key), {}};
}

Json ExampleSummary(const Graph& flowgraph) {
    try {
        return Serialize::FlowgraphDetail(flowgraph);
    } catch (const std::exception& error) {
        return Response::Error(jst::fmt::format("tool 'inspect_flowgraph' failed: {}", error.what()));
    } catch (...) {
        return Response::Error("tool 'inspect_flowgraph' failed with an unknown error");
    }
}

Tools::Output OpenExample(const Invocation& call) {
    std::string missing;
    const auto example = FindExample(call.args.value("key", std::string{}), missing);
    if (!example.has_value()) {
        return Response::Failure(missing);
    }

    const std::vector<char> blob(example->content.begin(), example->content.end());
    const auto imported = call.flowgraph->importFromBlob(blob);

    Json body = {
        {"opened", call.created},
        {"example", example->key},
        {"title", example->title},
        {"flowgraph", ExampleSummary(call.flowgraph)},
    };
    if (imported != Result::SUCCESS) {
        body["error"] = "The example opened with errors, so some of its blocks may not be ready.";
    }

    return Response::Reply(body);
}

}  // namespace

std::vector<Entry> Toolbox::Catalog() {
    return {
        Entry::FromJson({
            "search_blocks",
            "Find block types for a task. Searches type, title, domain, summary, description, device, runtime, and provider, then returns ranked implementation options and each type's default_width in graph units for planning layout. default_width is the node width before resizing, and height grows with config fields and rendered surfaces.",
            R"({"type":"object","properties":{"query":{"type":"string","description":"Words describing the block or task."},"domain":{"type":"string"},"device":{"type":"string"},"runtime":{"type":"string"},"provider":{"type":"string"},"limit":{"type":"integer","minimum":1,"maximum":50}},"additionalProperties":false})",
        }, SearchBlocks),
        Entry::FromJson({
            "get_block_docs",
            "Inspect one registered block type. Returns documentation, supported implementations, default_width, and default_config in the base values expected by apply_flowgraph_patch, with long strings and arrays shortened. Consult the documentation for units. observed_interfaces lists existing instances of the type with their compact config and port names, and config_fields gives the first instance's config fields with help and options. Interfaces are runtime-defined, so when no instance exists, create or inspect one to see its active config fields and ports.",
            R"({"type":"object","required":["type"],"properties":{"type":{"type":"string","description":"Exact block type from search_blocks, inspect_flowgraph, or inspect_block."}},"additionalProperties":false})",
            {.needsGraph = true},
        }, GetBlockDocs),
        Entry::FromJson({
            "list_examples",
            "List the bundled example flowgraphs with their key, title, and summary. Optional query words must all appear in the key, title, summary, description, or YAML content. matched_in is 'content' when some words appear only in the YAML, which get_example with pattern can locate.",
            R"({"type":"object","properties":{"query":{"type":"string","description":"Optional words to filter examples."}},"additionalProperties":false})",
        }, ListExamples),
        Entry::FromJson({
            "get_example",
            "Read one bundled example flowgraph by key, useful as a reference for block choices, configuration, wiring, and layout. Without pattern, offset, or limit it returns the description and full YAML content. With pattern it returns matching YAML lines (marked match=true) plus context, and with offset or limit a range of lines, each with its line number. pattern is a literal substring unless regex is true. Does not change any graph.",
            R"({"type":"object","required":["key"],"properties":{"key":{"type":"string","description":"Exact example key from list_examples."},"pattern":{"type":"string","description":"Text to find on a YAML line."},"regex":{"type":"boolean","description":"Treat pattern as an ECMAScript regular expression."},"context":{"type":"integer","minimum":0,"maximum":20,"description":"Lines of context around each match. Default 2."},"offset":{"type":"integer","minimum":1,"description":"First line to return when pattern is absent. Default 1."},"limit":{"type":"integer","minimum":1,"maximum":500,"description":"Maximum lines to return. Default 200."}},"additionalProperties":false})",
        }, GetExample),
        {{
            "open_example",
            "Open a bundled example as a new flowgraph and select it for subsequent tools. Returns the new flowgraph name and its summary. Use only when the user wants to open or start from an example; get_example reads it without changing the graph.",
            R"({"type":"object","required":["key"],"properties":{"key":{"type":"string","description":"Exact example key from list_examples."}},"additionalProperties":false})",
        }, OpenExample, ExampleGraph},
    };
}

}  // namespace Jetstream::Agent::Detail
