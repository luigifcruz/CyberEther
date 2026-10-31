#include "docs.hh"

#include "tools/text.hh"

#include <algorithm>
#include <cctype>
#include <functional>
#include <optional>
#include <sstream>
#include <unordered_map>

namespace Jetstream {

namespace {

constexpr const char* kInternalCategory = "Internal";
constexpr U64 kSnippetLead = 60;
constexpr U64 kSnippetLength = 160;

std::string Unquote(std::string value) {
    value = Text::TrimBlank(value);
    if (value.size() >= 2 && (value.front() == '"' || value.front() == '\'') && value.back() == value.front()) {
        return value.substr(1, value.size() - 2);
    }
    return value;
}

std::string StartCase(const std::string& slug) {
    std::string title;
    bool word = true;
    for (const char c : slug) {
        if (c == '-' || c == '_') {
            title += ' ';
            word = true;
            continue;
        }
        title += word ? static_cast<char>(std::toupper(static_cast<unsigned char>(c))) : c;
        word = false;
    }
    return title;
}

Docs::Page ParsePage(std::string_view slug, std::string_view source) {
    Docs::Page page{.slug = std::string(slug)};
    std::string body(source);
    if (body.rfind("---", 0) == 0) {
        const auto close = body.find("\n---", 3);
        if (close != std::string::npos) {
            for (const auto& line : Text::SplitLines(body.substr(3, close - 3))) {
                const auto colon = line.find(':');
                if (colon == std::string::npos) {
                    continue;
                }
                const std::string key = Text::TrimBlank(line.substr(0, colon));
                const std::string value = Unquote(line.substr(colon + 1));
                if (key == "title") {
                    page.title = value;
                } else if (key == "description") {
                    page.description = value;
                } else if (key == "category") {
                    page.category = value;
                } else if (key == "parent") {
                    page.parent = value;
                } else if (key == "order") {
                    try {
                        page.order = std::stoll(value);
                    } catch (...) {
                    }
                }
            }
            const auto end = body.find('\n', close + 4);
            body = end == std::string::npos ? std::string() : body.substr(end + 1);
        }
    }
    page.body = Text::TrimBlank(body);
    if (page.title.empty()) {
        page.title = StartCase(page.slug);
    }
    if (page.category.empty()) {
        page.category = "Overview";
    }
    return page;
}

std::string PlainInline(std::string_view line) {
    std::string plain;
    bool code = false;
    for (U64 i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '`') {
            code = !code;
            continue;
        }
        if (code) {
            plain += c;
            continue;
        }
        if (c == '!' && i + 1 < line.size() && line[i + 1] == '[') {
            continue;
        }
        if (c == '[') {
            const auto close = line.find("](", i);
            const bool link = close != std::string_view::npos && line.find(']', i) == close;
            const auto end = link ? line.find(')', close) : std::string_view::npos;
            if (end != std::string_view::npos) {
                plain += PlainInline(line.substr(i + 1, close - i - 1));
                i = end;
                continue;
            }
        }
        if (c == '<' && i + 1 < line.size() && (std::isalpha(static_cast<unsigned char>(line[i + 1])) ||
                                                line[i + 1] == '/')) {
            const auto end = line.find('>', i);
            if (end != std::string_view::npos) {
                i = end;
                continue;
            }
        }
        if (c == '*' || c == '|') {
            plain += c == '|' ? " " : "";
            continue;
        }
        plain += c;
    }
    return plain;
}

std::string StripBlockMarkers(const std::string& line) {
    U64 start = 0;
    while (start < line.size() && (line[start] == '>' || line[start] == ' ')) {
        ++start;
    }
    if (start + 1 < line.size() && (line[start] == '-' || line[start] == '*' || line[start] == '+') &&
        line[start + 1] == ' ') {
        start += 2;
    }
    U64 digits = start;
    while (digits < line.size() && std::isdigit(static_cast<unsigned char>(line[digits]))) {
        ++digits;
    }
    if (digits > start && digits + 1 < line.size() && line[digits] == '.' && line[digits + 1] == ' ') {
        start = digits + 2;
    }
    return line.substr(start);
}

std::string CollapseSpaces(const std::string& text) {
    std::string collapsed;
    for (const char c : text) {
        const bool space = std::isspace(static_cast<unsigned char>(c));
        if (space && (collapsed.empty() || collapsed.back() == ' ')) {
            continue;
        }
        collapsed += space ? ' ' : c;
    }
    while (!collapsed.empty() && collapsed.back() == ' ') {
        collapsed.pop_back();
    }
    return collapsed;
}

std::vector<std::pair<std::string, std::string>> SplitSections(const std::string& body) {
    std::vector<std::pair<std::string, std::string>> sections = {{"", ""}};
    bool fenced = false;
    for (const auto& line : Text::SplitLines(body)) {
        const std::string trimmed = Text::TrimBlank(line);
        if (trimmed.rfind("```", 0) == 0 || trimmed.rfind("~~~", 0) == 0) {
            fenced = !fenced;
            continue;
        }
        if (fenced) {
            sections.back().second += trimmed + " ";
            continue;
        }
        const auto hashes = trimmed.find_first_not_of('#');
        if (hashes > 0 && hashes != std::string::npos && trimmed[hashes] == ' ') {
            sections.push_back({CollapseSpaces(PlainInline(trimmed.substr(hashes + 1))), ""});
            continue;
        }
        if (trimmed.find_first_not_of("|-: ") == std::string::npos) {
            continue;
        }
        sections.back().second += PlainInline(StripBlockMarkers(trimmed)) + " ";
    }
    for (auto& section : sections) {
        section.second = CollapseSpaces(section.second);
    }
    std::erase_if(sections, [](const auto& section) {
        return section.first.empty() && section.second.empty();
    });
    return sections;
}

std::vector<std::string> SearchTerms(const std::string& query) {
    std::vector<std::string> terms;
    std::istringstream stream(Text::ToLower(query));
    for (std::string term; stream >> term;) {
        terms.push_back(std::move(term));
    }
    return terms;
}

bool ContainsAll(const std::string& text, const std::vector<std::string>& terms) {
    return std::all_of(terms.begin(), terms.end(), [&](const std::string& term) {
        return text.find(term) != std::string::npos;
    });
}

U64 CountMatches(const std::string& text, const std::vector<std::string>& terms) {
    U64 count = 0;
    for (const auto& term : terms) {
        for (auto at = text.find(term); at != std::string::npos; at = text.find(term, at + term.size())) {
            ++count;
        }
    }
    return count;
}

std::pair<U64, U64> SnippetRange(const std::string& lower, const std::vector<std::string>& terms) {
    U64 first = lower.size();
    for (const auto& term : terms) {
        first = std::min<U64>(first, lower.find(term) == std::string::npos ? lower.size() : lower.find(term));
    }
    if (first == lower.size()) {
        first = 0;
    }
    U64 start = first > kSnippetLead ? first - kSnippetLead : 0;
    if (start > 0) {
        const auto space = lower.find(' ', start);
        start = space != std::string::npos && space < first ? space + 1 : start;
    }
    U64 end = std::min<U64>(lower.size(), start + kSnippetLength);
    if (end < lower.size()) {
        const auto space = lower.rfind(' ', end);
        end = space != std::string::npos && space > first ? space : end;
    }
    while (end < lower.size() && (static_cast<unsigned char>(lower[end]) & 0xC0) == 0x80) {
        ++end;
    }
    return {start, end};
}

std::vector<std::pair<U64, U64>> MatchMarks(const std::string& lower, const std::vector<std::string>& terms) {
    std::vector<std::pair<U64, U64>> marks;
    for (const auto& term : terms) {
        for (auto at = lower.find(term); at != std::string::npos; at = lower.find(term, at + term.size())) {
            U64 begin = at;
            U64 end = at + term.size();
            while (begin > 0 && lower[begin - 1] != ' ') {
                --begin;
            }
            while (end < lower.size() && lower[end] != ' ') {
                ++end;
            }
            marks.push_back({begin, end});
        }
    }
    std::sort(marks.begin(), marks.end());
    std::vector<std::pair<U64, U64>> merged;
    for (const auto& mark : marks) {
        if (!merged.empty() && mark.first <= merged.back().second) {
            merged.back().second = std::max(merged.back().second, mark.second);
        } else {
            merged.push_back(mark);
        }
    }
    return merged;
}

}  // namespace

const Docs& Docs::Bundled() {
    static const Docs docs = Parse(Resources::BundledDocs());
    return docs;
}

Docs Docs::Parse(const std::vector<std::pair<std::string_view, std::string_view>>& files) {
    std::vector<Page> pages;
    std::unordered_map<std::string, U64> bySlug;
    for (const auto& [slug, source] : files) {
        auto page = ParsePage(slug, source);
        if (page.category == kInternalCategory) {
            continue;
        }
        bySlug[page.slug] = pages.size();
        pages.push_back(std::move(page));
    }

    const auto root = [&](U64 index) -> std::optional<U64> {
        std::vector<U64> chain = {index};
        while (true) {
            const auto& parent = pages[chain.back()].parent;
            const auto it = bySlug.find(parent);
            if (parent.empty() || it == bySlug.end()) {
                return chain.back();
            }
            if (std::find(chain.begin(), chain.end(), it->second) != chain.end()) {
                return std::nullopt;
            }
            chain.push_back(it->second);
        }
    };
    for (U64 index = 0; index < pages.size(); ++index) {
        pages[index].category = pages[root(index).value_or(index)].category;
    }

    std::unordered_map<std::string, I64> categoryOrder;
    std::unordered_map<std::string, std::vector<U64>> children;
    std::vector<U64> tops;
    for (U64 index = 0; index < pages.size(); ++index) {
        const auto& page = pages[index];
        const auto top = root(index);
        if (top.has_value() && *top != index) {
            children[page.parent].push_back(index);
            continue;
        }
        tops.push_back(index);
        const auto it = categoryOrder.find(page.category);
        if (it == categoryOrder.end() || page.order < it->second) {
            categoryOrder[page.category] = page.order;
        }
    }

    const auto sibling = [&](U64 a, U64 b) {
        return pages[a].order != pages[b].order ? pages[a].order < pages[b].order : pages[a].title < pages[b].title;
    };
    std::sort(tops.begin(), tops.end(), [&](U64 a, U64 b) {
        const I64 left = categoryOrder[pages[a].category];
        const I64 right = categoryOrder[pages[b].category];
        return left != right ? left < right : sibling(a, b);
    });

    Docs docs;
    std::function<void(U64, U64)> visit = [&](U64 index, U64 depth) {
        auto page = pages[index];
        page.depth = depth;
        if (docs.grouped.empty() || docs.grouped.back().title != page.category) {
            docs.grouped.push_back({.title = page.category});
        }
        docs.grouped.back().pages.push_back(docs.ordered.size());
        docs.ordered.push_back(std::move(page));
        auto kids = children[pages[index].slug];
        std::sort(kids.begin(), kids.end(), sibling);
        for (const U64 child : kids) {
            visit(child, depth + 1);
        }
    };
    for (const U64 top : tops) {
        visit(top, 0);
    }
    for (U64 index = 0; index < docs.ordered.size(); ++index) {
        const auto& page = docs.ordered[index];
        for (auto& [heading, text] : SplitSections(page.description + "\n\n" + page.body)) {
            const std::string searchable = Text::ToLower(page.title + "\n" + heading + "\n" + text);
            docs.sections.push_back({
                .page = index,
                .heading = std::move(heading),
                .text = std::move(text),
                .searchable = searchable,
            });
        }
    }
    return docs;
}

std::optional<Docs::Link> Docs::Resolve(const std::string& url) {
    std::string path = url;
    const std::string web = WebBase;
    if (path.rfind(web, 0) == 0) {
        path = LinkPrefix + path.substr(web.size());
    }
    if (path.rfind(LinkPrefix, 0) != 0) {
        return std::nullopt;
    }
    path = path.substr(std::string(LinkPrefix).size());
    Link link;
    const auto hash = path.find('#');
    link.slug = path.substr(0, hash);
    if (hash != std::string::npos) {
        link.anchor = path.substr(hash + 1);
    }
    while (!link.slug.empty() && link.slug.back() == '/') {
        link.slug.pop_back();
    }
    return link;
}

std::string Docs::WebUrl(const Link& link) {
    std::string url = std::string(WebBase) + link.slug;
    if (!link.anchor.empty()) {
        url += "#" + link.anchor;
    }
    return url;
}

std::string Docs::Anchor(const std::string& heading) {
    std::string anchor;
    bool dash = false;
    for (const char c : heading) {
        const auto byte = static_cast<unsigned char>(c);
        if (std::isalnum(byte)) {
            if (dash && !anchor.empty()) {
                anchor += '-';
            }
            anchor += static_cast<char>(std::tolower(byte));
            dash = false;
        } else if (c == ' ' || c == '-' || c == '_') {
            dash = true;
        }
    }
    return anchor;
}

const std::vector<Docs::Page>& Docs::pages() const {
    return ordered;
}

const std::vector<Docs::Group>& Docs::groups() const {
    return grouped;
}

std::optional<U64> Docs::find(const std::string& slug) const {
    for (U64 index = 0; index < ordered.size(); ++index) {
        if (ordered[index].slug == slug) {
            return index;
        }
    }
    return std::nullopt;
}

std::vector<Docs::Hit> Docs::search(const std::string& query, U64 limit) const {
    const auto terms = SearchTerms(query);
    if (terms.empty()) {
        return {};
    }
    struct Ranked {
        U64 section = 0;
        U64 weight = 0;
        U64 count = 0;
    };
    std::vector<Ranked> ranked;
    for (U64 index = 0; index < sections.size(); ++index) {
        const auto& section = sections[index];
        if (!ContainsAll(section.searchable, terms)) {
            continue;
        }
        U64 weight = 0;
        if (ContainsAll(Text::ToLower(section.heading), terms)) {
            weight = 2;
        } else if (ContainsAll(Text::ToLower(ordered[section.page].title), terms)) {
            weight = 1;
        }
        ranked.push_back({.section = index, .weight = weight, .count = CountMatches(section.searchable, terms)});
    }
    std::stable_sort(ranked.begin(), ranked.end(), [](const Ranked& a, const Ranked& b) {
        return a.weight != b.weight ? a.weight > b.weight : a.count > b.count;
    });
    ranked.resize(std::min<U64>(ranked.size(), limit));

    std::vector<Hit> hits;
    for (const auto& entry : ranked) {
        const auto& section = sections[entry.section];
        const std::string lower = Text::ToLower(section.text);
        const auto [start, end] = SnippetRange(lower, terms);
        hits.push_back({
            .page = section.page,
            .heading = section.heading,
            .anchor = section.heading.empty() ? std::string() : Anchor(section.heading),
            .snippet = section.text.substr(start, end - start),
            .marks = MatchMarks(lower.substr(start, end - start), terms),
        });
    }
    return hits;
}

}  // namespace Jetstream
