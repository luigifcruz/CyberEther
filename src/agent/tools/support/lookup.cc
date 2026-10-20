#include "base.hh"
#include "tools/text.hh"

#include <algorithm>
#include <iterator>

namespace Jetstream::Agent::Detail {
namespace {

constexpr U64 kSimilarNames = 5;

U64 EditDistance(const std::string& a, const std::string& b) {
    std::vector<U64> row(b.size() + 1);
    for (U64 j = 0; j <= b.size(); ++j) {
        row[j] = j;
    }

    for (U64 i = 1; i <= a.size(); ++i) {
        U64 diagonal = row[0];
        row[0] = i;

        for (U64 j = 1; j <= b.size(); ++j) {
            const U64 above = row[j];
            const U64 substitution = diagonal + (a[i - 1] == b[j - 1] ? 0 : 1);
            row[j] = std::min({row[j] + 1, row[j - 1] + 1, substitution});
            diagonal = above;
        }
    }

    return row[b.size()];
}

}  // namespace

std::string Lookup::Similar(const std::string& query,
                            const std::vector<std::string>& candidates,
                            const std::string& noun) {
    const std::string needle = Text::ToLower(query);

    std::vector<std::pair<U64, std::string>> ranked;
    for (const auto& candidate : candidates) {
        const std::string name = Text::ToLower(candidate);

        const bool contains = name.find(needle) != std::string::npos ||
                              needle.find(name) != std::string::npos;
        if (!needle.empty() && contains) {
            ranked.push_back({0, candidate});
            continue;
        }

        const U64 distance = EditDistance(needle, name);
        const U64 threshold = std::max<U64>(2, std::max(needle.size(), name.size()) / 3);
        if (distance <= threshold) {
            ranked.push_back({distance, candidate});
        }
    }

    if (ranked.empty()) {
        return jst::fmt::format(" No similar {}.", noun);
    }

    std::sort(ranked.begin(), ranked.end());

    std::string list;
    const U64 shown = std::min<U64>(kSimilarNames, ranked.size());
    for (U64 i = 0; i < shown; ++i) {
        jst::fmt::format_to(std::back_inserter(list), "{}{}", i > 0 ? ", " : "", ranked[i].second);
    }

    return jst::fmt::format(" Similar {}: {}.", noun, list);
}

std::vector<std::string> Lookup::Blocks(const Graph& flowgraph) {
    std::vector<std::string> names;
    if (flowgraph) {
        flowgraph->view().keys(names);
    }
    return names;
}

std::string Lookup::Missing(const Graph& flowgraph, const std::string& name, const std::string& label) {
    const auto similar = Lookup::Similar(name, Lookup::Blocks(flowgraph), "block names");
    return jst::fmt::format("{} '{}' not found.{}", label, name, similar);
}

std::vector<std::string> Lookup::Types() {
    std::vector<std::string> types;
    for (const auto& block : Registry::ListAvailableBlocks()) {
        types.push_back(block.type);
    }
    return types;
}

}  // namespace Jetstream::Agent::Detail
