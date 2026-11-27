#include "home_gallery_remote.hh"

#include "jetstream/config.hh"
#include "jetstream/logger.hh"
#include "jetstream/parser.hh"
#include "jetstream/platform.hh"
#include "tools/catalog.hh"

#include <nlohmann/json.hpp>

#if defined(JETSTREAM_LOADER_OPENSSL_AVAILABLE)
#include <openssl/sha.h>
#endif

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace Jetstream {

namespace {

constexpr U64 kTimeoutSeconds = 10;
constexpr U64 kDigestLength = 64;
constexpr const char* kCacheFolder = "gallery";
constexpr const char* kIndexFile = "index.json";

#if defined(JETSTREAM_LOADER_OPENSSL_AVAILABLE)
constexpr bool kVerifiable = true;
#else
constexpr bool kVerifiable = false;
#endif

struct Entry {
    std::string slug;
    std::string digest;
    std::string url;
    U64 size = 0;
};

std::string CurrentPlatform() {
#if defined(JST_OS_MAC)
    return "macos";
#elif defined(JST_OS_LINUX)
    return "linux";
#elif defined(JST_OS_WINDOWS)
    return "windows";
#elif defined(JST_OS_BROWSER)
    return "web";
#else
    return "";
#endif
}

std::string CurrentArch() {
#if defined(__aarch64__) || defined(_M_ARM64)
    return "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#else
    return "";
#endif
}

bool ValidSlug(std::string_view slug) {
    return !slug.empty() && std::all_of(slug.begin(), slug.end(), [](const char c) {
        return std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_';
    });
}

std::optional<std::string> TrailingSlug(std::string_view rest) {
    rest = rest.substr(0, rest.find_first_of("?#"));
    if (rest.ends_with('/')) {
        rest.remove_suffix(1);
    }
    if (!ValidSlug(rest)) {
        return std::nullopt;
    }
    return std::string(rest);
}

std::optional<std::string> Digest(const std::string& content) {
#if defined(JETSTREAM_LOADER_OPENSSL_AVAILABLE)
    unsigned char digest[SHA256_DIGEST_LENGTH];
    if (!SHA256(reinterpret_cast<const unsigned char*>(content.data()), content.size(), digest)) {
        return std::nullopt;
    }
    constexpr const char* kHexDigits = "0123456789abcdef";
    std::string hex;
    hex.reserve(kDigestLength);
    for (const unsigned char byte : digest) {
        hex += kHexDigits[byte >> 4];
        hex += kHexDigits[byte & 0x0F];
    }
    return hex;
#else
    (void)content;
    return std::nullopt;
#endif
}

bool Verified(const std::string& content, const std::string& digest) {
    return Digest(content) == digest;
}

Platform::Http::Client::Response Request(const Home::Gallery::Remote::Source& source,
                                         const std::string& host,
                                         const std::string& path,
                                         const std::function<bool()>& cancelled) {
    if (!source.send) {
        return {};
    }
    Platform::Http::Client::Request request;
    request.host = host;
    request.path = path;
    request.timeoutSeconds = kTimeoutSeconds;
    request.cancelled = cancelled;
    return source.send(request);
}

std::optional<std::filesystem::path> CacheFolder() {
    std::string root;
    if (Platform::CachePath(root) != Result::SUCCESS || root.empty()) {
        return std::nullopt;
    }
    return Platform::PathFromUtf8(root) / kCacheFolder;
}

std::optional<std::string> ReadFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return std::nullopt;
    }
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

bool WriteFile(const std::filesystem::path& path, const std::string& content) {
    std::error_code error;
    std::filesystem::create_directories(path.parent_path(), error);
    auto staging = path;
    staging += ".tmp";
    {
        std::ofstream file(staging, std::ios::binary | std::ios::trunc);
        if (!file || !file.write(content.data(), static_cast<std::streamsize>(content.size()))) {
            JST_WARN("[GALLERY] Could not write '{}' to the cache.", Platform::PathToUtf8(path));
            return false;
        }
    }
    std::filesystem::rename(staging, path, error);
    if (error) {
        std::filesystem::remove(staging, error);
        return false;
    }
    return true;
}

std::optional<Home::Gallery::Flowgraph> Decode(const std::string& slug, std::string content) {
    Home::Gallery::Flowgraph flowgraph;
    if (Home::Gallery::Flowgraph::Parse(slug, std::move(content), flowgraph) != Result::SUCCESS) {
        return std::nullopt;
    }
    if (flowgraph.title.empty()) {
        flowgraph.title = slug;
    }
    return flowgraph;
}

std::optional<Entry> ReadEntry(const nlohmann::json& item) {
    if (!item.is_object()) {
        return std::nullopt;
    }
    Entry entry;
    if (!Catalog::ReadText(item, "slug", entry.slug) || !Catalog::ReadText(item, "flowgraphSha256", entry.digest) ||
        !Catalog::ReadText(item, "url", entry.url) || !Catalog::ReadSize(item, entry.size) || !ValidSlug(entry.slug)) {
        return std::nullopt;
    }
    return entry;
}

std::optional<std::vector<Entry>> Listing(const nlohmann::json& listing, bool& skipped) {
    skipped = false;
    if (!listing.is_object()) {
        return std::nullopt;
    }
    const auto items = listing.find("flowgraphs");
    if (items == listing.end() || !items->is_array()) {
        return std::nullopt;
    }
    std::vector<Entry> entries;
    for (const auto& item : *items) {
        auto entry = ReadEntry(item);
        if (!entry.has_value()) {
            JST_WARN("[GALLERY] Skipping a malformed flowgraph entry.");
            skipped = true;
            continue;
        }
        entries.push_back(std::move(*entry));
    }
    return entries;
}

void Prune(const std::filesystem::path& folder, const std::vector<Entry>& keep) {
    std::unordered_set<std::string> digests;
    for (const auto& entry : keep) {
        digests.insert(entry.digest);
    }
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(folder, error)) {
        const auto path = entry.path();
        if (path.extension() == ".yml" && !digests.contains(Platform::PathToUtf8(path.stem()))) {
            std::filesystem::remove(path, error);
        }
    }
}

nlohmann::json QueryJson(const Home::Gallery::Remote::Query& query) {
    return {
        {"owner", query.owner},
        {"version", query.version},
        {"platform", query.platform},
        {"arch", query.arch},
    };
}

std::string IndexText(const Home::Gallery::Remote::Query& query, const std::vector<Entry>& entries) {
    nlohmann::json items = nlohmann::json::array();
    for (const auto& entry : entries) {
        items.push_back({
            {"slug", entry.slug},
            {"flowgraphSha256", entry.digest},
            {"url", entry.url},
            {"size", entry.size},
        });
    }
    return nlohmann::json{{"query", QueryJson(query)}, {"flowgraphs", std::move(items)}}.dump();
}

std::optional<std::vector<Entry>> CachedListing(const std::filesystem::path& folder,
                                                const Home::Gallery::Remote::Query& query) {
    const auto index = ReadFile(folder / kIndexFile);
    if (!index.has_value()) {
        return std::nullopt;
    }
    const auto listing = nlohmann::json::parse(*index, nullptr, false);
    if (!listing.is_object() || listing.value("query", nlohmann::json()) != QueryJson(query)) {
        return std::nullopt;
    }
    bool skipped = false;
    return Listing(listing, skipped);
}

std::optional<Home::Gallery::Flowgraph> Restore(const std::filesystem::path& folder, const Entry& entry) {
    if (!Catalog::ValidDigest(entry.digest)) {
        return std::nullopt;
    }
    auto content = ReadFile(folder / (entry.digest + ".yml"));
    if (!content.has_value() || !Verified(*content, entry.digest)) {
        return std::nullopt;
    }
    return Decode(entry.slug, std::move(*content));
}

std::optional<std::string> Download(const Home::Gallery::Remote::Source& source,
                                    const Entry& entry,
                                    bool verifiable,
                                    const std::function<bool()>& cancelled) {
    std::string host;
    std::string path;
    auto file = Catalog::SplitUrl(entry.url, host, path) ? Request(source, host, path, cancelled)
                                                          : Platform::Http::Client::Response{};
    if (file.error || file.status != 200 || (entry.size > 0 && file.body.size() != entry.size)) {
        JST_WARN("[GALLERY] Skipping flowgraph '{}' because its file could not be downloaded.", entry.slug);
        return std::nullopt;
    }
    if (verifiable && !Verified(file.body, entry.digest)) {
        JST_WARN("[GALLERY] Skipping flowgraph '{}' because its file does not match its digest.", entry.slug);
        return std::nullopt;
    }
    return std::move(file.body);
}

}  // namespace

Home::Gallery::Remote::Source Home::Gallery::Remote::Source::Default() {
    return {
        .send = &Platform::Http::Client::Send,
        .folder = CacheFolder(),
    };
}

Home::Gallery::Remote::Query Home::Gallery::Remote::Current() {
    return {
        .owner = Origin,
        .version = JETSTREAM_VERSION_STR,
        .platform = CurrentPlatform(),
        .arch = CurrentArch(),
    };
}

std::string Home::Gallery::Remote::Path(const Query& query) {
    std::string path = Endpoint;
    char separator = '?';
    const auto add = [&](const char* name, const std::string& value) {
        if (value.empty()) {
            return;
        }
        path += separator;
        path += name;
        path += '=';
        path += value;
        separator = '&';
    };
    add("owner", query.owner);
    add("version", query.version);
    add("platform", query.platform);
    add("arch", query.arch);
    return path;
}

std::optional<std::string> Home::Gallery::Remote::Slug(const std::string& url) {
    std::string_view rest = url;
    for (const std::string_view scheme : {"https://", "http://"}) {
        if (rest.starts_with(scheme)) {
            rest.remove_prefix(scheme.size());
            break;
        }
    }
    if (rest.starts_with("www.")) {
        rest.remove_prefix(4);
    }
    const std::string_view host = std::string_view(Host).substr(std::string_view(Host).find("://") + 3);
    if (!rest.starts_with(host)) {
        return std::nullopt;
    }
    rest.remove_prefix(host.size());
    if (!rest.starts_with(PagePath)) {
        return std::nullopt;
    }
    rest.remove_prefix(std::string_view(PagePath).size());
    return TrailingSlug(rest);
}

bool Home::Gallery::Remote::ValidOwner(std::string_view owner) {
    return ValidSlug(owner);
}

std::optional<Home::Gallery::Remote::Reference> Home::Gallery::Remote::Link(const std::string& url) {
    std::string_view rest = url;
    if (!rest.starts_with(LinkPrefix)) {
        return std::nullopt;
    }
    rest.remove_prefix(std::string_view(LinkPrefix).size());
    rest = rest.substr(0, rest.find_first_of("?#"));
    if (rest.ends_with('/')) {
        rest.remove_suffix(1);
    }

    std::string_view owner = Origin;
    if (const auto separator = rest.find('/'); separator != std::string_view::npos) {
        owner = rest.substr(0, separator);
        rest.remove_prefix(separator + 1);
    }
    if (!ValidOwner(owner) || !ValidSlug(rest)) {
        return std::nullopt;
    }

    return Reference{.owner = std::string(owner), .slug = std::string(rest)};
}

std::vector<Home::Gallery::Section> Home::Gallery::Remote::Sections(const Flowgraphs& flowgraphs) {
    struct Known {
        const char* title;
        const char* description;
    };
    static const Known known[] = {
        {"Basics", "Start here to learn the editor, sources, and plots."},
        {"DSP", "Signal processing techniques, taken apart block by block."},
        {"Receivers", "Tune, demodulate, and decode signals off the air."},
        {"Visualization", "Turn numbers into color, shape, and motion."},
    };
    std::vector<Section> sections;
    for (const auto& flowgraph : flowgraphs) {
        const std::string title = flowgraph.category.empty() ? "Other" : flowgraph.category;
        auto section = std::find_if(sections.begin(), sections.end(), [&](const auto& item) {
            return item.title == title;
        });
        if (section == sections.end()) {
            const auto info = std::find_if(std::begin(known), std::end(known), [&](const auto& item) {
                return title == item.title;
            });
            sections.push_back({
                .key = std::string(Origin) + "/" + title,
                .title = title,
                .description = info != std::end(known) ? info->description
                                                       : "Complete pipelines, ready to run in Web or Native.",
            });
            section = std::prev(sections.end());
        }
        section->flowgraphs.push_back(flowgraph);
    }
    return sections;
}

Result Home::Gallery::Remote::Load(Flowgraphs& flowgraphs) {
    return LoadFrom(Source::Default(), Current(), flowgraphs);
}

Result Home::Gallery::Remote::Fetch(const Query& query,
                                    const std::function<bool()>& cancelled,
                                    Flowgraphs& flowgraphs) {
    return FetchFrom(Source::Default(), query, cancelled, flowgraphs);
}

Result Home::Gallery::Remote::LoadFrom(const Source& source, const Query& query, Flowgraphs& flowgraphs) {
    flowgraphs.clear();
    if (!source.folder.has_value()) {
        return Result::ERROR;
    }
    const auto listing = CachedListing(*source.folder, query);
    if (!listing.has_value()) {
        return Result::ERROR;
    }
    std::unordered_set<std::string> resolved;
    for (const auto& entry : *listing) {
        if (resolved.contains(entry.slug)) {
            continue;
        }
        if (auto flowgraph = Restore(*source.folder, entry)) {
            resolved.insert(entry.slug);
            flowgraphs.push_back(std::move(*flowgraph));
        }
    }
    return Result::SUCCESS;
}

Result Home::Gallery::Remote::FetchFrom(const Source& source,
                                        const Query& query,
                                        const std::function<bool()>& cancelled,
                                        Flowgraphs& flowgraphs) {
    flowgraphs.clear();
    const auto response = Request(source, Host, Path(query), cancelled);
    if (response.error || response.status != 200) {
        JST_DEBUG("[GALLERY] Flowgraph list unavailable (status {}, {}).", response.status, response.error.message);
        return Result::ERROR;
    }
    bool skipped = false;
    const auto listing = Listing(nlohmann::json::parse(response.body, nullptr, false), skipped);
    if (!listing.has_value()) {
        JST_WARN("[GALLERY] Flowgraph list response is malformed.");
        return Result::ERROR;
    }

    const auto& folder = source.folder;
    std::unordered_map<std::string, Entry> previous;
    if (folder.has_value()) {
        for (auto& entry : CachedListing(*folder, query).value_or(std::vector<Entry>{})) {
            previous.emplace(entry.slug, std::move(entry));
        }
    }

    std::vector<Entry> kept;
    std::unordered_set<std::string> resolved;
    bool complete = !skipped;
    for (const auto& entry : *listing) {
        if (cancelled && cancelled()) {
            return Result::ERROR;
        }
        if (resolved.contains(entry.slug)) {
            continue;
        }

        const bool verifiable = kVerifiable && Catalog::ValidDigest(entry.digest);
        const bool cacheable = verifiable && folder.has_value();
        auto flowgraph = cacheable ? Restore(*folder, entry) : std::nullopt;
        bool cached = flowgraph.has_value();
        if (!flowgraph.has_value()) {
            if (auto content = Download(source, entry, verifiable, cancelled)) {
                flowgraph = Decode(entry.slug, std::move(*content));
                if (!flowgraph.has_value()) {
                    JST_WARN("[GALLERY] Skipping flowgraph '{}' because its file could not be parsed.", entry.slug);
                } else if (cacheable) {
                    cached = WriteFile(*folder / (entry.digest + ".yml"), flowgraph->content);
                }
            }
        }
        const auto fallback = previous.find(entry.slug);
        if (flowgraph.has_value()) {
            if (cached) {
                kept.push_back(entry);
            } else if (cacheable && fallback != previous.end()) {
                kept.push_back(fallback->second);
            }
            resolved.insert(entry.slug);
            flowgraphs.push_back(std::move(*flowgraph));
            continue;
        }

        complete = false;
        if (fallback == previous.end()) {
            continue;
        }
        if (auto restored = Restore(*folder, fallback->second)) {
            JST_WARN("[GALLERY] Keeping the cached copy of flowgraph '{}'.", entry.slug);
            kept.push_back(fallback->second);
            resolved.insert(entry.slug);
            flowgraphs.push_back(std::move(*restored));
        }
    }

    if (!complete && flowgraphs.empty()) {
        return Result::INCOMPLETE;
    }
    if (folder.has_value() && WriteFile(*folder / kIndexFile, IndexText(query, kept))) {
        Prune(*folder, kept);
    }
    return complete ? Result::SUCCESS : Result::INCOMPLETE;
}

}  // namespace Jetstream
