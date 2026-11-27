#include "plugin_gallery.hh"

#include "home_gallery_remote.hh"

#include "jetstream/config.hh"
#include "jetstream/logger.hh"
#include "tools/catalog.hh"
#include "tools/text.hh"

#include <nlohmann/json.hpp>

#if defined(JETSTREAM_LOADER_OPENSSL_AVAILABLE)
#include <openssl/evp.h>
#endif

#include <algorithm>
#include <cctype>
#include <fstream>
#include <memory>
#include <string_view>
#include <system_error>
#include <utility>

namespace Jetstream {

namespace {

constexpr U64 kListTimeoutSeconds = 10;
constexpr U64 kDownloadTimeoutSeconds = 300;

bool ReadBlocks(const nlohmann::json& item, std::vector<std::string>& blocks) {
    const auto field = item.find("blocks");
    if (field == item.end() || field->is_null()) {
        return true;
    }
    if (!field->is_array()) {
        return false;
    }
    for (const auto& block : *field) {
        if (!block.is_string()) {
            return false;
        }
        blocks.push_back(block.get<std::string>());
    }
    return true;
}

std::optional<Plugin::Gallery::Entry> ReadEntry(const nlohmann::json& item) {
    if (!item.is_object()) {
        return std::nullopt;
    }

    Plugin::Gallery::Entry entry;
    if (!Catalog::ReadText(item, "id", entry.id) || !Catalog::ReadText(item, "title", entry.title) ||
        !Catalog::ReadText(item, "summary", entry.summary) ||
        !Catalog::ReadText(item, "description", entry.description) ||
        !Catalog::ReadText(item, "author", entry.author) ||
        !Catalog::ReadText(item, "license", entry.license) || !Catalog::ReadText(item, "version", entry.version) ||
        !Catalog::ReadText(item, "url", entry.url) || !Catalog::ReadText(item, "cepSha256", entry.digest) ||
        !Catalog::ReadSize(item, entry.size) || !ReadBlocks(item, entry.blocks) ||
        !Plugin::Gallery::ValidId(entry.id)) {
        return std::nullopt;
    }

    if (entry.title.empty()) {
        entry.title = entry.id;
    }
    return entry;
}

#if defined(JETSTREAM_LOADER_OPENSSL_AVAILABLE)
struct DigestContext {
    DigestContext() : context(EVP_MD_CTX_new(), &EVP_MD_CTX_free) {
        valid = context && EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) == 1;
    }

    bool update(std::string_view data) {
        valid = valid && EVP_DigestUpdate(context.get(), data.data(), data.size()) == 1;
        return valid;
    }

    std::optional<std::string> finish() {
        unsigned char digest[EVP_MAX_MD_SIZE];
        unsigned int length = 0;
        if (!valid || EVP_DigestFinal_ex(context.get(), digest, &length) != 1) {
            return std::nullopt;
        }

        constexpr const char* kHexDigits = "0123456789abcdef";
        std::string hex;
        hex.reserve(length * 2);
        for (unsigned int i = 0; i < length; ++i) {
            hex += kHexDigits[digest[i] >> 4];
            hex += kHexDigits[digest[i] & 0x0F];
        }
        return hex;
    }

    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context;
    bool valid = false;
};
#endif

}  // namespace

bool Plugin::Gallery::ValidId(std::string_view id) {
    if (id.empty() || id.size() > 32 || !std::islower(static_cast<unsigned char>(id.front()))) {
        return false;
    }

    if (ReservedName(id)) {
        return false;
    }

    return std::all_of(id.begin(), id.end(), [](unsigned char character) {
        return std::islower(character) || std::isdigit(character);
    });
}

std::optional<Plugin::Gallery::Reference> Plugin::Gallery::Link(const std::string& url) {
    std::string_view rest = url;
    if (!rest.starts_with(LinkPrefix)) {
        return std::nullopt;
    }
    rest.remove_prefix(std::string_view(LinkPrefix).size());
    rest = rest.substr(0, rest.find_first_of("?#"));
    if (rest.ends_with('/')) {
        rest.remove_suffix(1);
    }

    std::string_view owner = Home::Gallery::Origin;
    if (const auto separator = rest.find('/'); separator != std::string_view::npos) {
        owner = rest.substr(0, separator);
        rest.remove_prefix(separator + 1);
    }
    if (!Home::Gallery::Remote::ValidOwner(owner) || !ValidId(rest)) {
        return std::nullopt;
    }

    return Reference{.owner = std::string(owner), .id = std::string(rest)};
}

bool Plugin::Gallery::ReservedName(std::string_view name) {
    static constexpr std::string_view reserved[] = {
        "aux", "con", "nul", "prn",
        "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8", "com9",
        "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
        "com\xc2\xb9", "com\xc2\xb2", "com\xc2\xb3", "lpt\xc2\xb9", "lpt\xc2\xb2", "lpt\xc2\xb3",
    };
    const auto base = Text::ToLower(Text::TrimBlank(std::string(name.substr(0, name.find('.')))));
    return std::find(std::begin(reserved), std::end(reserved), base) != std::end(reserved);
}

std::string Plugin::Gallery::Host() {
    std::string host;
    if (Platform::EnvironmentVariable(HostVariable, host) != Result::SUCCESS || host.empty()) {
        return Home::Gallery::Remote::Host;
    }
    return host;
}

Plugin::Gallery::Source Plugin::Gallery::Source::Default() {
    return {.send = &Platform::Http::Client::Send};
}

Plugin::Gallery::Query Plugin::Gallery::Current(std::vector<std::string> ids) {
    const auto base = Home::Gallery::Remote::Current();
    return {
        .owner = base.owner,
        .version = base.version,
        .platform = base.platform,
        .arch = base.arch,
        .build = JETSTREAM_BUILD_ID,
        .ids = std::move(ids),
    };
}

std::string Plugin::Gallery::Path(const Query& query) {
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
    add("build", query.build);
    add("ids", Text::Join(query.ids, ","));
    return path;
}

Result Plugin::Gallery::Fetch(const Source& source,
                              const Query& query,
                              const std::function<bool()>& cancelled,
                              Entries& entries) {
    entries.clear();
    if (!source.send) {
        return Result::ERROR;
    }

    Platform::Http::Client::Request request;
    request.host = Host();
    request.path = Path(query);
    request.timeoutSeconds = kListTimeoutSeconds;
    request.cancelled = cancelled;

    const auto response = source.send(request);
    if (response.error || response.status != 200) {
        JST_ERROR("[PLUGIN] The CyberEther gallery is unreachable (status {}).", response.status);
        return Result::ERROR;
    }

    const auto listing = nlohmann::json::parse(response.body, nullptr, false);
    const auto items = listing.is_object() ? listing.find("plugins") : listing.end();
    if (!listing.is_object() || items == listing.end() || !items->is_array()) {
        JST_ERROR("[PLUGIN] The CyberEther gallery sent a malformed plugin list.");
        return Result::ERROR;
    }

    for (const auto& item : *items) {
        auto entry = ReadEntry(item);
        if (!entry.has_value()) {
            JST_WARN("[PLUGIN] Skipping a malformed plugin entry from the gallery.");
            continue;
        }
        const auto duplicate = std::any_of(entries.begin(), entries.end(), [&](const auto& known) {
            return known.id == entry->id;
        });
        if (!duplicate) {
            entries.push_back(std::move(*entry));
        }
    }

    return Result::SUCCESS;
}

Result Plugin::Gallery::Download(const Source& source,
                                 const Entry& entry,
                                 const std::filesystem::path& destination,
                                 const std::function<bool()>& cancelled) {
#if defined(JETSTREAM_LOADER_OPENSSL_AVAILABLE)
    std::string host;
    std::string path;
    if (!entry.available() || !Catalog::SplitUrl(entry.url, host, path) || !source.send) {
        JST_ERROR("[PLUGIN] The gallery entry for plugin '{}' has no download.", entry.id);
        return Result::ERROR;
    }

    if (!Catalog::ValidDigest(entry.digest)) {
        JST_ERROR("[PLUGIN] The gallery entry for plugin '{}' has no valid checksum.", entry.id);
        return Result::ERROR;
    }

    std::error_code ec;
    std::filesystem::create_directories(destination.parent_path(), ec);
    std::ofstream file(destination, std::ios::binary | std::ios::trunc);
    if (!file) {
        JST_ERROR("[PLUGIN] Failed to create '{}'.", Platform::PathToUtf8(destination));
        return Result::ERROR;
    }

    DigestContext digest;
    U64 received = 0;
    bool written = true;

    Platform::Http::Client::Request request;
    request.host = host;
    request.path = path;
    request.timeoutSeconds = kDownloadTimeoutSeconds;
    request.cancelled = cancelled;
    request.onResponse = [](I32 status, const Platform::Http::Headers&) {
        return status == 200;
    };
    request.onData = [&](std::string_view data) {
        received += data.size();
        if (entry.size > 0 && received > entry.size) {
            return false;
        }
        file.write(data.data(), static_cast<std::streamsize>(data.size()));
        written = written && static_cast<bool>(file) && digest.update(data);
        return written;
    };

    const auto response = source.send(request);
    file.close();

    auto fail = [&](const std::string& reason) {
        std::filesystem::remove(destination, ec);
        JST_ERROR("[PLUGIN] Failed to download plugin '{}': {}.", entry.id, reason);
        return Result::ERROR;
    };

    if (response.error || response.status != 200) {
        return fail("the gallery is unreachable");
    }
    if (!written || !file) {
        return fail("the file could not be written");
    }
    if (entry.size > 0 && received != entry.size) {
        return fail("the file has the wrong size");
    }
    if (digest.finish() != entry.digest) {
        return fail("the file does not match its checksum");
    }

    return Result::SUCCESS;
#else
    (void)source;
    (void)destination;
    (void)cancelled;
    JST_ERROR("[PLUGIN] Cannot download plugin '{}' because this build cannot verify checksums.", entry.id);
    return Result::ERROR;
#endif
}

}  // namespace Jetstream
