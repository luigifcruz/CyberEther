#include "gallery.hh"

#include "jetstream/config.hh"
#include "jetstream/logger.hh"
#include "jetstream/parser.hh"
#include "jetstream/platform.hh"

#include <nlohmann/json.hpp>

#if defined(JETSTREAM_LOADER_OPENSSL_AVAILABLE)
#include <openssl/sha.h>
#endif

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
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

bool SplitUrl(const std::string& url, std::string& host, std::string& path) {
    const U64 scheme = url.find("://");
    if (scheme == std::string::npos) {
        return false;
    }
    const U64 slash = url.find('/', scheme + 3);
    host = url.substr(0, slash);
    path = slash == std::string::npos ? "/" : url.substr(slash);
    return !host.empty();
}

bool ValidDigest(const std::string& digest) {
    return digest.size() == kDigestLength && std::all_of(digest.begin(), digest.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
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

Platform::Http::Client::Response Get(const Gallery::Source& source,
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

std::optional<Registry::FlowgraphRegistration> Register(const std::string& slug, std::string content) {
    Registry::FlowgraphRegistration registration;
    Parser::Map data;
    if (Parser::YamlDecode(content, data) != Result::SUCCESS || registration.deserialize(data) != Result::SUCCESS) {
        return std::nullopt;
    }
    registration.key = slug;
    registration.content = std::move(content);
    if (registration.title.empty()) {
        registration.title = slug;
    }
    return registration;
}

bool ReadText(const nlohmann::json& item, const char* name, std::string& value) {
    const auto field = item.find(name);
    if (field == item.end()) {
        return true;
    }
    if (!field->is_string()) {
        return false;
    }
    value = field->get<std::string>();
    return true;
}

bool ReadSize(const nlohmann::json& item, U64& value) {
    const auto field = item.find("size");
    if (field == item.end()) {
        return true;
    }
    if (!field->is_number_unsigned()) {
        return false;
    }
    value = field->get<U64>();
    return true;
}

std::optional<Entry> ReadEntry(const nlohmann::json& item) {
    if (!item.is_object()) {
        return std::nullopt;
    }
    Entry entry;
    if (!ReadText(item, "slug", entry.slug) || !ReadText(item, "flowgraphSha256", entry.digest) ||
        !ReadText(item, "url", entry.url) || !ReadSize(item, entry.size) || entry.slug.empty()) {
        return std::nullopt;
    }
    return entry;
}

std::optional<std::vector<Entry>> Listing(const nlohmann::json& listing) {
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

nlohmann::json QueryJson(const Gallery::Query& query) {
    return {
        {"owner", query.owner},
        {"version", query.version},
        {"platform", query.platform},
        {"arch", query.arch},
    };
}

std::string IndexText(const Gallery::Query& query, const std::vector<Entry>& entries) {
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

std::optional<std::vector<Entry>> CachedListing(const std::filesystem::path& folder, const Gallery::Query& query) {
    const auto index = ReadFile(folder / kIndexFile);
    if (!index.has_value()) {
        return std::nullopt;
    }
    const auto listing = nlohmann::json::parse(*index, nullptr, false);
    if (!listing.is_object() || listing.value("query", nlohmann::json()) != QueryJson(query)) {
        return std::nullopt;
    }
    return Listing(listing);
}

std::optional<Registry::FlowgraphRegistration> Restore(const std::filesystem::path& folder, const Entry& entry) {
    if (!ValidDigest(entry.digest)) {
        return std::nullopt;
    }
    auto content = ReadFile(folder / (entry.digest + ".yml"));
    if (!content.has_value() || !Verified(*content, entry.digest)) {
        return std::nullopt;
    }
    return Register(entry.slug, std::move(*content));
}

std::optional<std::string> Download(const Gallery::Source& source,
                                    const Entry& entry,
                                    bool verifiable,
                                    const std::function<bool()>& cancelled) {
    std::string host;
    std::string path;
    auto file = SplitUrl(entry.url, host, path) ? Get(source, host, path, cancelled) : Platform::Http::Client::Response{};
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

Gallery::Source Gallery::Source::Default() {
    return {
        .send = &Platform::Http::Client::Send,
        .folder = CacheFolder(),
    };
}

Gallery::Query Gallery::Current() {
    return {
        .owner = Owner,
        .version = JETSTREAM_VERSION_STR,
        .platform = CurrentPlatform(),
        .arch = CurrentArch(),
    };
}

std::string Gallery::Path(const Query& query) {
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

Result Gallery::Load(std::vector<Registry::FlowgraphRegistration>& flowgraphs) {
    return LoadFrom(Source::Default(), Current(), flowgraphs);
}

Result Gallery::Fetch(const Query& query,
                      const std::function<bool()>& cancelled,
                      std::vector<Registry::FlowgraphRegistration>& flowgraphs) {
    return FetchFrom(Source::Default(), query, cancelled, flowgraphs);
}

Result Gallery::LoadFrom(const Source& source,
                         const Query& query,
                         std::vector<Registry::FlowgraphRegistration>& flowgraphs) {
    flowgraphs.clear();
    if (!source.folder.has_value()) {
        return Result::ERROR;
    }
    const auto listing = CachedListing(*source.folder, query);
    if (!listing.has_value()) {
        return Result::ERROR;
    }
    for (const auto& entry : *listing) {
        if (auto registration = Restore(*source.folder, entry)) {
            flowgraphs.push_back(std::move(*registration));
        }
    }
    return Result::SUCCESS;
}

Result Gallery::FetchFrom(const Source& source,
                          const Query& query,
                          const std::function<bool()>& cancelled,
                          std::vector<Registry::FlowgraphRegistration>& flowgraphs) {
    flowgraphs.clear();
    const auto response = Get(source, Host, Path(query), cancelled);
    if (response.error || response.status != 200) {
        JST_DEBUG("[GALLERY] Flowgraph list unavailable (status {}, {}).", response.status, response.error.message);
        return Result::ERROR;
    }
    const auto listing = Listing(nlohmann::json::parse(response.body, nullptr, false));
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
    bool complete = true;
    for (const auto& entry : *listing) {
        if (cancelled && cancelled()) {
            return Result::ERROR;
        }

        const bool verifiable = kVerifiable && ValidDigest(entry.digest);
        const bool cacheable = verifiable && folder.has_value();
        auto registration = cacheable ? Restore(*folder, entry) : std::nullopt;
        bool cached = registration.has_value();
        if (!registration.has_value()) {
            if (auto content = Download(source, entry, verifiable, cancelled)) {
                registration = Register(entry.slug, std::move(*content));
                if (!registration.has_value()) {
                    JST_WARN("[GALLERY] Skipping flowgraph '{}' because its file could not be parsed.", entry.slug);
                } else if (cacheable) {
                    cached = WriteFile(*folder / (entry.digest + ".yml"), registration->content);
                }
            }
        }
        const auto fallback = previous.find(entry.slug);
        if (registration.has_value()) {
            if (cached) {
                kept.push_back(entry);
            } else if (cacheable && fallback != previous.end()) {
                kept.push_back(fallback->second);
            }
            flowgraphs.push_back(std::move(*registration));
            continue;
        }

        complete = false;
        if (fallback == previous.end()) {
            continue;
        }
        if (auto restored = Restore(*folder, fallback->second)) {
            JST_WARN("[GALLERY] Keeping the cached copy of flowgraph '{}'.", entry.slug);
            kept.push_back(fallback->second);
            flowgraphs.push_back(std::move(*restored));
        }
    }

    if (folder.has_value() && WriteFile(*folder / kIndexFile, IndexText(query, kept))) {
        Prune(*folder, kept);
    }
    return complete ? Result::SUCCESS : Result::INCOMPLETE;
}

}  // namespace Jetstream
