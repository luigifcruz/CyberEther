#include "support/base.hh"

namespace Jetstream::Agent::Detail {
namespace {

constexpr const char* kUnavailable = "Documentation is not available in this host.";

Json SearchDocs(const Invocation& call) {
    const std::string query = call.args.value("query", std::string{});
    const U64 limit = Arguments::Clamp(call.args, "limit", 8, 1, 30);

    const auto matches = call.host.searchDocs(query, limit);
    if (!matches) {
        return Response::Error(kUnavailable);
    }

    Json results = Json::array();
    for (const auto& match : *matches) {
        Json entry = {
            {"page", match.slug},
            {"title", match.title},
        };
        if (!match.heading.empty()) {
            entry["section"] = match.heading;
        }
        if (!match.snippet.empty()) {
            entry["snippet"] = match.snippet;
        }
        results.push_back(std::move(entry));
    }

    return Json({{"query", query}, {"matches", std::move(results)}});
}

Json ReadDocs(const Invocation& call) {
    const std::string slug = call.args.value("page", std::string{});
    if (slug.empty()) {
        return Response::Error("'page' is required.");
    }

    const auto page = call.host.readDocs(slug);
    if (!page) {
        const auto pages = call.host.searchDocs({}, 0);
        if (!pages) {
            return Response::Error(kUnavailable);
        }

        std::vector<std::string> slugs;
        for (const auto& match : *pages) {
            slugs.push_back(match.slug);
        }
        return Response::Error(jst::fmt::format("Documentation page '{}' not found.{}",
                                            slug, Lookup::Similar(slug, slugs, "pages")));
    }

    return Json({
        {"page", page->slug},
        {"title", page->title},
        {"description", page->description},
        {"body", page->body},
    });
}

}  // namespace

std::vector<Entry> Toolbox::Docs() {
    return {
        Entry::FromJson({
            "search_docs",
            "Search the bundled CyberEther documentation, which covers the interface, flowgraph files, remote sessions, plugins, the Python block, tensors, platforms, and installation. Returns matching pages with the section and a snippet. An empty query lists every page. Use get_block_docs for a specific block type.",
            R"({"type":"object","properties":{"query":{"type":"string","description":"Words to search for. Empty lists all pages."},"limit":{"type":"integer","minimum":1,"maximum":30,"description":"Maximum search matches, default 8. Ignored for an empty query."}},"additionalProperties":false})",
            {},
        }, SearchDocs),
        Entry::FromJson({
            "read_docs",
            "Read one bundled CyberEther documentation page as Markdown.",
            R"({"type":"object","required":["page"],"properties":{"page":{"type":"string","description":"Page id returned by search_docs."}},"additionalProperties":false})",
            {},
        }, ReadDocs),
    };
}

}  // namespace Jetstream::Agent::Detail
