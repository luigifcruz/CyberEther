#include "docs.hh"

#include "tools/text.hh"

#include <algorithm>
#include <cctype>
#include <functional>
#include <optional>
#include <unordered_map>

namespace Jetstream {

namespace {

constexpr const char* kInternalCategory = "Internal";

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

}  // namespace Jetstream
