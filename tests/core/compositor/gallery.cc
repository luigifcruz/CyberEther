#include <catch2/catch_test_macros.hpp>

#include "compositor/default/actions/gallery.hh"
#include "environment.hh"

#include "jetstream/config.hh"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Jetstream;

namespace {

Registry::FlowgraphRegistration Published(const std::string& key, const std::string& title) {
    Registry::FlowgraphRegistration registration;
    registration.key = key;
    registration.title = title;
    registration.category = "Basics";
    registration.content = "---\nversion: 2\ngraph: []\n";
    return registration;
}

Result NoCache(GalleryActions::Flowgraphs&) {
    return Result::ERROR;
}

void Settle(GalleryActions& actions, DefaultCompositorState& state, GalleryActions::Clock::time_point now) {
    actions.reconcile(now);
    REQUIRE(state.registry.gallery.loading);
    state.registry.gallery.future.wait();
    actions.reconcile(now + GalleryActions::ProgressMinimum);
    REQUIRE_FALSE(state.registry.gallery.loading);
}

const std::string kAlpha = "---\nversion: 2\ntitle: Alpha\ngraph: []\n";
const std::string kAlphaDigest = "c7811bde9fb80e2fd0be7b83c543ff25de34758f22b93935128a2118403088e7";
const std::string kAlphaUrl = "https://files.test/alpha.yml";
const std::string kBeta = "---\nversion: 2\ntitle: Beta\ngraph: []\n";
const std::string kBetaDigest = "fb3ad7e63119a598cdf24b189c6c673e74a332eb8e66724fb75bab9ab1382b2d";
const std::string kBetaUrl = "https://files.test/beta.yml";

class GalleryServer {
 public:
    std::map<std::string, std::string> files;
    std::vector<std::string> requests;

    void list(const std::string& body, const Gallery::Query& query = {}) {
        files[std::string(Gallery::Host) + Gallery::Path(query)] = body;
    }

    Gallery::Source source(std::optional<std::filesystem::path> folder = std::nullopt) {
        return {
            .send = [this](const Platform::Http::Client::Request& request) {
                requests.push_back(request.host + request.path);
                Platform::Http::Client::Response response;
                const auto file = files.find(request.host + request.path);
                response.status = file == files.end() ? 404 : 200;
                if (file != files.end()) {
                    response.body = file->second;
                }
                return response;
            },
            .folder = std::move(folder),
        };
    }

    bool requested(const std::string& url) const {
        return std::find(requests.begin(), requests.end(), url) != requests.end();
    }

};

std::vector<std::string> Keys(const GalleryActions::Flowgraphs& flowgraphs) {
    std::vector<std::string> keys;
    for (const auto& flowgraph : flowgraphs) {
        keys.push_back(flowgraph.key);
    }
    return keys;
}

void WriteText(const std::filesystem::path& path, const std::string& content) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file << content;
}

std::string ReadText(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

}  // namespace

TEST_CASE("Gallery queries name the owner, version, platform and architecture and skip unknown values",
          "[core][compositor][gallery]") {
    CHECK(Gallery::Path({.owner = "cyberether", .version = "1.16.2", .platform = "linux", .arch = "arm64"}) ==
          "/api/v1/flowgraphs?owner=cyberether&version=1.16.2&platform=linux&arch=arm64");
    CHECK(Gallery::Path({.owner = "cyberether", .version = "1.16.2"}) ==
          "/api/v1/flowgraphs?owner=cyberether&version=1.16.2");

    const auto current = Gallery::Current();
    CHECK(current.owner == "cyberether");
    CHECK(current.version == JETSTREAM_VERSION_STR);
    const std::vector<std::string> platforms = {"", "linux", "macos", "windows", "web"};
    const std::vector<std::string> archs = {"", "x86_64", "arm64"};
    CHECK(std::find(platforms.begin(), platforms.end(), current.platform) != platforms.end());
    CHECK(std::find(archs.begin(), archs.end(), current.arch) != archs.end());
#if defined(JST_OS_MAC)
    CHECK(current.platform == "macos");
#elif defined(JST_OS_LINUX)
    CHECK(current.platform == "linux");
#elif defined(JST_OS_WINDOWS)
    CHECK(current.platform == "windows");
#endif
}

TEST_CASE("Gallery page links resolve to their flowgraph slug and other links do not", "[core][compositor][gallery]") {
    CHECK(Gallery::Slug("https://cyberether.org/gallery/flowgraphs/biq-python-roundtrip") == "biq-python-roundtrip");
    CHECK(Gallery::Slug("http://www.cyberether.org/gallery/flowgraphs/python-qpsk/") == "python-qpsk");
    CHECK(Gallery::Slug("https://cyberether.org/gallery/flowgraphs/first_steps?ref=note#top") == "first_steps");
    CHECK_FALSE(Gallery::Slug("https://cyberether.org/gallery/flowgraphs/").has_value());
    CHECK_FALSE(Gallery::Slug("https://cyberether.org/gallery/flowgraphs/a/b").has_value());
    CHECK_FALSE(Gallery::Slug("https://cyberether.org/docs/quick-start").has_value());
    CHECK_FALSE(Gallery::Slug("https://example.com/gallery/flowgraphs/python-qpsk").has_value());
    CHECK_FALSE(Gallery::Slug("https://cyberether.org.evil.com/gallery/flowgraphs/python-qpsk").has_value());
    CHECK_FALSE(Gallery::Slug("/gallery/flowgraphs/python-qpsk").has_value());
}

TEST_CASE("Published gallery flowgraphs lead the catalog and replace bundled ones with the same key",
          "[core][compositor][gallery]") {
    DefaultCompositorState state;
    GalleryActions(state).refreshRegistry();
    const auto bundled = state.registry.flowgraphs;
    REQUIRE_FALSE(bundled.empty());
    const auto replaced = bundled.front().key;

    Gallery::Query seen;
    GalleryActions actions(state, [&](const Gallery::Query& query, const std::function<bool()>&,
                                      GalleryActions::Flowgraphs& flowgraphs) {
        seen = query;
        flowgraphs = {Published("remote-only", "Remote Only"), Published(replaced, "Published Copy")};
        return Result::SUCCESS;
    }, NoCache);
    const auto now = GalleryActions::Clock::now();
    Settle(actions, state, now);
    CHECK(seen.version == JETSTREAM_VERSION_STR);
    CHECK(state.registry.stale);

    actions.refreshRegistry();
    const auto& flowgraphs = state.registry.flowgraphs;
    REQUIRE(flowgraphs.size() == bundled.size() + 1);
    CHECK(flowgraphs[0].key == "remote-only");
    CHECK(flowgraphs[1].key == replaced);
    CHECK(flowgraphs[1].title == "Published Copy");
    CHECK(std::count_if(flowgraphs.begin(), flowgraphs.end(), [&](const auto& entry) {
              return entry.key == replaced;
          }) == 1);

    actions.reconcile(now + std::chrono::minutes(30));
    CHECK_FALSE(state.registry.gallery.loading);
    actions.reconcile(now + GalleryActions::RefreshInterval + std::chrono::seconds(5));
    CHECK(state.registry.gallery.loading);
    actions.shutdown();
    CHECK_FALSE(state.registry.gallery.loading);
}

TEST_CASE("Published gallery flowgraphs never hide or replace a plugin example with the same key",
          "[core][compositor][gallery]") {
    auto example = Published("demo", "Plugin Demo");
    auto bundled = Published("starter", "Bundled Starter");
    const GalleryActions::Flowgraphs published = {Published("demo", "Gallery Demo"),
                                                  Published("starter", "Published Starter")};
    const std::vector<Plugin::Info> plugins = {{.path = "/plugins/demo.cep", .examples = {"demo"}}};

    const auto catalog = GalleryActions::Catalog(published, {example, bundled}, plugins);
    CHECK(Keys(catalog) == std::vector<std::string>{std::string(GalleryActions::PublishedPrefix) + "demo", "starter",
                                                    "demo"});
    REQUIRE(catalog.size() == 3);
    CHECK(catalog[0].title == "Gallery Demo");
    CHECK(catalog[1].title == "Published Starter");
    CHECK(catalog[2].title == "Plugin Demo");

    CHECK(Keys(GalleryActions::Catalog(published, {example, bundled}, {})) ==
          std::vector<std::string>{"demo", "starter"});
}

TEST_CASE("Gallery fetches retry while offline and keep the bundled catalog", "[core][compositor][gallery]") {
    DefaultCompositorState state;
    GalleryActions(state).refreshRegistry();
    const auto bundled = state.registry.flowgraphs.size();

    U64 attempts = 0;
    GalleryActions actions(state, [&](const Gallery::Query&, const std::function<bool()>&,
                                      GalleryActions::Flowgraphs&) {
        ++attempts;
        return Result::ERROR;
    }, NoCache);
    const auto now = GalleryActions::Clock::now();
    Settle(actions, state, now);
    CHECK(attempts == 1);
    CHECK_FALSE(state.registry.stale);
    CHECK(state.registry.gallery.flowgraphs.empty());

    actions.reconcile(now + std::chrono::seconds(10));
    CHECK_FALSE(state.registry.gallery.loading);
    Settle(actions, state, now + GalleryActions::RetryInterval + std::chrono::seconds(5));
    CHECK(attempts == 2);
    actions.refreshRegistry();
    CHECK(state.registry.flowgraphs.size() == bundled);
}

TEST_CASE("Cached gallery flowgraphs show at startup before the refresh finishes", "[core][compositor][gallery]") {
    DefaultCompositorState state;
    GalleryActions(state).refreshRegistry();

    U64 loads = 0;
    GalleryActions actions(
        state,
        [&](const Gallery::Query&, const std::function<bool()>&, GalleryActions::Flowgraphs& flowgraphs) {
            flowgraphs = {Published("fresh", "Fresh")};
            return Result::SUCCESS;
        },
        [&](GalleryActions::Flowgraphs& flowgraphs) {
            ++loads;
            flowgraphs = {Published("cached", "Cached")};
            return Result::SUCCESS;
        });
    const auto now = GalleryActions::Clock::now();
    actions.reconcile(now);
    CHECK(loads == 1);
    CHECK(state.registry.gallery.loading);
    CHECK(state.registry.gallery.cached);
    CHECK(state.registry.stale);
    actions.refreshRegistry();
    REQUIRE_FALSE(state.registry.flowgraphs.empty());
    CHECK(state.registry.flowgraphs.front().key == "cached");

    state.registry.gallery.future.wait();
    actions.reconcile(now + GalleryActions::ProgressMinimum);
    CHECK(loads == 1);
    actions.refreshRegistry();
    CHECK(state.registry.flowgraphs.front().key == "fresh");
    CHECK(std::none_of(state.registry.flowgraphs.begin(), state.registry.flowgraphs.end(), [](const auto& entry) {
        return entry.key == "cached";
    }));
}

TEST_CASE("Registry snapshot refreshes only while it is marked stale", "[core][compositor][gallery]") {
    TestSupport::SettingsSandbox sandbox("compositor-registry");
    sandbox.require();
    DefaultCompositorState state;
    GalleryActions actions(state);
    REQUIRE(state.registry.stale);

    actions.refreshRegistry();
    CHECK_FALSE(state.registry.stale);
    CHECK_FALSE(state.registry.blocks.empty());
    CHECK_FALSE(state.registry.flowgraphs.empty());

    state.registry.flowgraphs.clear();
    actions.refreshRegistry();
    CHECK(state.registry.flowgraphs.empty());

    state.registry.stale = true;
    actions.refreshRegistry();
    CHECK_FALSE(state.registry.flowgraphs.empty());
}

TEST_CASE("Registry snapshot stays stale and retries while the settings file cannot be read",
          "[core][compositor][gallery]") {
    TestSupport::SettingsSandbox sandbox("compositor-registry");
    sandbox.require();
    std::string configPath;
    REQUIRE(Platform::ConfigPath(configPath) == Result::SUCCESS);
    const auto settingsPath = Platform::PathFromUtf8(configPath) / "settings.yaml";
    std::filesystem::create_directories(settingsPath.parent_path());
    {
        std::ofstream file(settingsPath, std::ios::binary | std::ios::trunc);
        file << "graphics: [unterminated\n";
    }

    DefaultCompositorState state;
    GalleryActions actions(state);
    actions.refreshRegistry();
    CHECK(state.registry.stale);
    CHECK_FALSE(state.registry.blocks.empty());
    CHECK_FALSE(state.registry.flowgraphs.empty());
    CHECK(state.registry.registered.empty());

    {
        std::ofstream file(settingsPath, std::ios::binary | std::ios::trunc);
        file << "registry:\n  plugins:\n    - /plugins/missing.cep\n";
    }
    actions.refreshRegistry();
    CHECK_FALSE(state.registry.stale);
    REQUIRE(state.registry.registered.size() == 1);
    CHECK(state.registry.registered.front().path == "/plugins/missing.cep");
    CHECK_FALSE(state.registry.registered.front().exists);
    CHECK_FALSE(state.registry.registered.front().info.has_value());
}

TEST_CASE("Registry snapshot lists plugins behind an inaccessible folder as missing",
          "[core][compositor][gallery]") {
    TestSupport::SettingsSandbox sandbox("compositor-registry");
    sandbox.require();
    std::string configPath;
    REQUIRE(Platform::ConfigPath(configPath) == Result::SUCCESS);
    const auto settingsPath = Platform::PathFromUtf8(configPath) / "settings.yaml";
    const auto locked = sandbox.root() / "locked";
    const auto plugin = Platform::PathToUtf8(locked / "plugin.cep");
    WriteText(locked / "plugin.cep", "");
    WriteText(settingsPath, "registry:\n  plugins:\n    - " + plugin + "\n");
    std::filesystem::permissions(locked, std::filesystem::perms::none);
    std::error_code denied;
    if (std::filesystem::exists(locked / "plugin.cep", denied) || !denied) {
        std::filesystem::permissions(locked, std::filesystem::perms::owner_all);
        SKIP("Folder permissions do not deny access on this machine");
    }

    DefaultCompositorState state;
    GalleryActions actions(state);
    CHECK_NOTHROW(actions.refreshRegistry());
    std::filesystem::permissions(locked, std::filesystem::perms::owner_all);
    CHECK_FALSE(state.registry.stale);
    REQUIRE(state.registry.registered.size() == 1);
    CHECK(state.registry.registered.front().path == plugin);
    CHECK_FALSE(state.registry.registered.front().exists);
}

TEST_CASE("Registry snapshot keeps block metadata without holding registered factories",
          "[core][compositor][gallery]") {
    TestSupport::SettingsSandbox sandbox("compositor-registry");
    sandbox.require();
    auto token = std::make_shared<int>(0);
    REQUIRE(Registry::RegisterBlock("snapshot_probe", "Probe", "Snapshot Probe", "Probe summary.", "",
                                    [token]() -> std::shared_ptr<Block> {
                                        return nullptr;
                                    }) == Result::SUCCESS);
    const auto registered = token.use_count();

    DefaultCompositorState state;
    GalleryActions(state).refreshRegistry();
    CHECK(token.use_count() == registered);

    const auto& blocks = state.registry.blocks;
    const auto probe = std::find_if(blocks.begin(), blocks.end(), [](const auto& block) {
        return block.type == "snapshot_probe";
    });
    REQUIRE(probe != blocks.end());
    CHECK(probe->domain == "Probe");
    CHECK(probe->title == "Snapshot Probe");
    CHECK(probe->summary == "Probe summary.");

    REQUIRE(Registry::UnregisterBlock("snapshot_probe") == Result::SUCCESS);
    CHECK(token.use_count() == 1);
}

TEST_CASE("The first gallery fetch keeps loading for the minimum progress time", "[core][compositor][gallery]") {
    DefaultCompositorState state;
    U64 attempts = 0;
    GalleryActions actions(state, [&](const Gallery::Query&, const std::function<bool()>&,
                                      GalleryActions::Flowgraphs& flowgraphs) {
        ++attempts;
        flowgraphs = {Published("fresh", "Fresh")};
        return Result::SUCCESS;
    }, NoCache);
    auto& gallery = state.registry.gallery;
    const auto now = GalleryActions::Clock::now();
    actions.reconcile(now);
    REQUIRE(gallery.loading);
    CHECK_FALSE(gallery.cached);

    gallery.future.wait();
    actions.reconcile(now + std::chrono::milliseconds(500));
    CHECK(gallery.loading);
    CHECK_FALSE(gallery.attempted);
    actions.reconcile(now + GalleryActions::ProgressMinimum);
    CHECK_FALSE(gallery.loading);
    CHECK(gallery.attempted);
    CHECK_FALSE(gallery.failed);
    CHECK(Keys(gallery.flowgraphs) == std::vector<std::string>{"fresh"});

    const auto later = now + GalleryActions::RefreshInterval + std::chrono::seconds(5);
    actions.reconcile(later);
    REQUIRE(gallery.loading);
    gallery.future.wait();
    actions.reconcile(later);
    CHECK_FALSE(gallery.loading);
    CHECK(attempts == 2);
}

TEST_CASE("Incomplete gallery fetches publish what arrived and retry soon", "[core][compositor][gallery]") {
    DefaultCompositorState state;
    U64 attempts = 0;
    GalleryActions actions(state, [&](const Gallery::Query&, const std::function<bool()>&,
                                      GalleryActions::Flowgraphs& flowgraphs) {
        ++attempts;
        flowgraphs = {Published("partial", "Partial")};
        return Result::INCOMPLETE;
    }, NoCache);
    const auto now = GalleryActions::Clock::now();
    Settle(actions, state, now);
    CHECK(attempts == 1);
    CHECK_FALSE(state.registry.gallery.failed);
    CHECK(state.registry.stale);
    CHECK(Keys(state.registry.gallery.flowgraphs) == std::vector<std::string>{"partial"});

    actions.reconcile(now + std::chrono::seconds(10));
    CHECK_FALSE(state.registry.gallery.loading);
    Settle(actions, state, now + GalleryActions::RetryInterval + std::chrono::seconds(5));
    CHECK(attempts == 2);
}

TEST_CASE("Gallery fetches that throw count as failed attempts", "[core][compositor][gallery]") {
    DefaultCompositorState state;
    GalleryActions actions(state, [](const Gallery::Query&, const std::function<bool()>&,
                                     GalleryActions::Flowgraphs&) -> Result {
        throw std::runtime_error("malformed listing");
    }, NoCache);
    Settle(actions, state, GalleryActions::Clock::now());
    CHECK(state.registry.gallery.attempted);
    CHECK(state.registry.gallery.failed);
    CHECK(state.registry.gallery.flowgraphs.empty());
}

TEST_CASE("Gallery fetches skip malformed entries and files that fail their size", "[core][compositor][gallery]") {
    GalleryServer server;
    server.files[kBetaUrl] = kBeta;
    server.files[kAlphaUrl] = kAlpha;
    server.list(R"({"flowgraphs": [
        7,
        {"slug": null, "url": "https://files.test/beta.yml"},
        {"slug": 7, "url": "https://files.test/beta.yml"},
        {"slug": "sized", "url": "https://files.test/beta.yml", "size": "37"},
        {"slug": "negative", "url": "https://files.test/beta.yml", "size": -1},
        {"slug": "linked", "url": 5},
        {"slug": "short", "url": "https://files.test/alpha.yml", "size": 10},
        {"slug": "missing", "url": "https://files.test/missing.yml"},
        {"slug": "plain", "url": "https://files.test/beta.yml", "size": 37}
    ]})");
    GalleryActions::Flowgraphs flowgraphs;
    REQUIRE(Gallery::FetchFrom(server.source(), {}, {}, flowgraphs) == Result::INCOMPLETE);
    CHECK(Keys(flowgraphs) == std::vector<std::string>{"plain"});
    REQUIRE(flowgraphs.size() == 1);
    CHECK(flowgraphs.front().title == "Beta");
    CHECK(flowgraphs.front().content == kBeta);

    server.list(R"({"flowgraphs": {"slug": "plain"}})");
    CHECK(Gallery::FetchFrom(server.source(), {}, {}, flowgraphs) == Result::ERROR);
    CHECK(flowgraphs.empty());
    server.list("not json");
    CHECK(Gallery::FetchFrom(server.source(), {}, {}, flowgraphs) == Result::ERROR);
    server.files.clear();
    CHECK(Gallery::FetchFrom(server.source(), {}, {}, flowgraphs) == Result::ERROR);
    CHECK(Gallery::FetchFrom({}, {}, {}, flowgraphs) == Result::ERROR);
}

TEST_CASE("Gallery cache restores skip malformed index entries", "[core][compositor][gallery]") {
    TestSupport::SettingsSandbox sandbox("compositor-gallery");
    const auto folder = sandbox.root() / "gallery";
    GalleryServer server;
    GalleryActions::Flowgraphs flowgraphs;
    CHECK(Gallery::LoadFrom(server.source(), {}, flowgraphs) == Result::ERROR);
    CHECK(Gallery::LoadFrom(server.source(folder), {}, flowgraphs) == Result::ERROR);

    WriteText(folder / (kAlphaDigest + ".yml"), kAlpha);
    WriteText(folder / "index.json", R"({"query": {"owner": "", "version": "", "platform": "", "arch": ""},
    "flowgraphs": [
        {"slug": null, "flowgraphSha256": "c7811bde9fb80e2fd0be7b83c543ff25de34758f22b93935128a2118403088e7"},
        {"slug": "alpha", "flowgraphSha256": 5},
        {"slug": "alpha", "flowgraphSha256": "not-a-digest"}
    ]})");
    REQUIRE(Gallery::LoadFrom(server.source(folder), {}, flowgraphs) == Result::SUCCESS);
    CHECK(flowgraphs.empty());
}

#if defined(JETSTREAM_LOADER_OPENSSL_AVAILABLE)
TEST_CASE("Gallery fetches verify downloads, cache them by digest and restore them from the index",
          "[core][compositor][gallery]") {
    TestSupport::SettingsSandbox sandbox("compositor-gallery");
    const auto folder = sandbox.root() / "gallery";
    const auto cached = folder / (kAlphaDigest + ".yml");
    GalleryServer server;
    server.files[kAlphaUrl] = kAlpha;
    server.list(R"({"flowgraphs": [{"slug": "alpha", "flowgraphSha256": ")" + kAlphaDigest +
                R"(", "url": "https://files.test/alpha.yml", "size": 38}]})");
    WriteText(folder / "stale.yml", kBeta);

    GalleryActions::Flowgraphs flowgraphs;
    REQUIRE(Gallery::FetchFrom(server.source(folder), {}, {}, flowgraphs) == Result::SUCCESS);
    CHECK(Keys(flowgraphs) == std::vector<std::string>{"alpha"});
    CHECK(ReadText(cached) == kAlpha);
    CHECK(std::filesystem::exists(folder / "index.json"));
    CHECK_FALSE(std::filesystem::exists(folder / "stale.yml"));
    CHECK(server.requested(kAlphaUrl));

    server.requests.clear();
    server.files.erase(kAlphaUrl);
    REQUIRE(Gallery::FetchFrom(server.source(folder), {}, {}, flowgraphs) == Result::SUCCESS);
    CHECK(Keys(flowgraphs) == std::vector<std::string>{"alpha"});
    CHECK_FALSE(server.requested(kAlphaUrl));

    WriteText(cached, kBeta);
    server.files[kAlphaUrl] = kAlpha;
    REQUIRE(Gallery::FetchFrom(server.source(folder), {}, {}, flowgraphs) == Result::SUCCESS);
    CHECK(Keys(flowgraphs) == std::vector<std::string>{"alpha"});
    CHECK(flowgraphs.front().title == "Alpha");
    CHECK(server.requested(kAlphaUrl));
    CHECK(ReadText(cached) == kAlpha);

    REQUIRE(Gallery::LoadFrom(server.source(folder), {}, flowgraphs) == Result::SUCCESS);
    CHECK(Keys(flowgraphs) == std::vector<std::string>{"alpha"});
    WriteText(cached, kBeta);
    REQUIRE(Gallery::LoadFrom(server.source(folder), {}, flowgraphs) == Result::SUCCESS);
    CHECK(flowgraphs.empty());
}

TEST_CASE("Gallery fetches skip downloads that do not match their digest and never cache them",
          "[core][compositor][gallery]") {
    TestSupport::SettingsSandbox sandbox("compositor-gallery");
    const auto folder = sandbox.root() / "gallery";
    GalleryServer server;
    server.files[kAlphaUrl] = kAlpha;
    server.files[kBetaUrl] = kBeta;
    server.list(R"({"flowgraphs": [
        {"slug": "forged", "flowgraphSha256": ")" + kBetaDigest + R"(", "url": "https://files.test/alpha.yml"},
        {"slug": "beta", "flowgraphSha256": ")" + kBetaDigest + R"(", "url": "https://files.test/beta.yml"}
    ]})");

    GalleryActions::Flowgraphs flowgraphs;
    REQUIRE(Gallery::FetchFrom(server.source(folder), {}, {}, flowgraphs) == Result::INCOMPLETE);
    CHECK(Keys(flowgraphs) == std::vector<std::string>{"beta"});
    CHECK(ReadText(folder / (kBetaDigest + ".yml")) == kBeta);
    CHECK_FALSE(std::filesystem::exists(folder / (kAlphaDigest + ".yml")));
}

TEST_CASE("Gallery refreshes keep the cached copy of a flowgraph whose new version fails to download",
          "[core][compositor][gallery]") {
    TestSupport::SettingsSandbox sandbox("compositor-gallery");
    const auto folder = sandbox.root() / "gallery";
    GalleryServer server;
    server.files[kAlphaUrl] = kAlpha;
    server.list(R"({"flowgraphs": [{"slug": "demo", "flowgraphSha256": ")" + kAlphaDigest +
                R"(", "url": "https://files.test/alpha.yml"}]})");
    GalleryActions::Flowgraphs flowgraphs;
    REQUIRE(Gallery::FetchFrom(server.source(folder), {}, {}, flowgraphs) == Result::SUCCESS);
    REQUIRE(Keys(flowgraphs) == std::vector<std::string>{"demo"});

    server.list(R"({"flowgraphs": [
        {"slug": "demo", "flowgraphSha256": ")" + kBetaDigest + R"(", "url": "https://files.test/beta.yml"},
        {"slug": "fresh", "flowgraphSha256": ")" + kBetaDigest + R"(", "url": "https://files.test/beta.yml"}
    ]})");
    CHECK(Gallery::FetchFrom(server.source(folder), {}, {}, flowgraphs) == Result::INCOMPLETE);
    REQUIRE(Keys(flowgraphs) == std::vector<std::string>{"demo"});
    CHECK(flowgraphs.front().title == "Alpha");
    CHECK(ReadText(folder / (kAlphaDigest + ".yml")) == kAlpha);
    REQUIRE(Gallery::LoadFrom(server.source(folder), {}, flowgraphs) == Result::SUCCESS);
    REQUIRE(Keys(flowgraphs) == std::vector<std::string>{"demo"});
    CHECK(flowgraphs.front().title == "Alpha");

    CHECK(Gallery::FetchFrom(server.source(folder), {}, {}, flowgraphs) == Result::INCOMPLETE);
    CHECK(Keys(flowgraphs) == std::vector<std::string>{"demo"});

    server.files[kBetaUrl] = kBeta;
    REQUIRE(Gallery::FetchFrom(server.source(folder), {}, {}, flowgraphs) == Result::SUCCESS);
    CHECK(Keys(flowgraphs) == std::vector<std::string>{"demo", "fresh"});
    CHECK(flowgraphs.front().title == "Beta");
    CHECK_FALSE(std::filesystem::exists(folder / (kAlphaDigest + ".yml")));
}

TEST_CASE("Gallery refreshes keep the cached files the saved index needs when the new index cannot be saved",
          "[core][compositor][gallery]") {
    TestSupport::SettingsSandbox sandbox("compositor-gallery");
    const auto folder = sandbox.root() / "gallery";
    GalleryServer server;
    server.files[kAlphaUrl] = kAlpha;
    server.files[kBetaUrl] = kBeta;
    server.list(R"({"flowgraphs": [{"slug": "demo", "flowgraphSha256": ")" + kAlphaDigest +
                R"(", "url": "https://files.test/alpha.yml"}]})");
    GalleryActions::Flowgraphs flowgraphs;
    REQUIRE(Gallery::FetchFrom(server.source(folder), {}, {}, flowgraphs) == Result::SUCCESS);

    std::filesystem::create_directories(folder / "index.json.tmp");
    server.list(R"({"flowgraphs": [{"slug": "demo", "flowgraphSha256": ")" + kBetaDigest +
                R"(", "url": "https://files.test/beta.yml"}]})");
    REQUIRE(Gallery::FetchFrom(server.source(folder), {}, {}, flowgraphs) == Result::SUCCESS);
    CHECK(flowgraphs.front().title == "Beta");

    CHECK(ReadText(folder / (kAlphaDigest + ".yml")) == kAlpha);
    REQUIRE(Gallery::LoadFrom(server.source(folder), {}, flowgraphs) == Result::SUCCESS);
    REQUIRE(Keys(flowgraphs) == std::vector<std::string>{"demo"});
    CHECK(flowgraphs.front().title == "Alpha");
}

TEST_CASE("Gallery refreshes keep the previous cached copy when the new version cannot be saved",
          "[core][compositor][gallery]") {
    TestSupport::SettingsSandbox sandbox("compositor-gallery");
    const auto folder = sandbox.root() / "gallery";
    GalleryServer server;
    server.files[kAlphaUrl] = kAlpha;
    server.files[kBetaUrl] = kBeta;
    server.list(R"({"flowgraphs": [{"slug": "demo", "flowgraphSha256": ")" + kAlphaDigest +
                R"(", "url": "https://files.test/alpha.yml"}]})");
    GalleryActions::Flowgraphs flowgraphs;
    REQUIRE(Gallery::FetchFrom(server.source(folder), {}, {}, flowgraphs) == Result::SUCCESS);

    std::filesystem::create_directories(folder / (kBetaDigest + ".yml.tmp"));
    server.list(R"({"flowgraphs": [{"slug": "demo", "flowgraphSha256": ")" + kBetaDigest +
                R"(", "url": "https://files.test/beta.yml"}]})");
    REQUIRE(Gallery::FetchFrom(server.source(folder), {}, {}, flowgraphs) == Result::SUCCESS);
    CHECK(flowgraphs.front().title == "Beta");

    CHECK(ReadText(folder / (kAlphaDigest + ".yml")) == kAlpha);
    REQUIRE(Gallery::LoadFrom(server.source(folder), {}, flowgraphs) == Result::SUCCESS);
    REQUIRE(Keys(flowgraphs) == std::vector<std::string>{"demo"});
    CHECK(flowgraphs.front().title == "Alpha");
}

TEST_CASE("Gallery cache restores ignore an index written for another query", "[core][compositor][gallery]") {
    TestSupport::SettingsSandbox sandbox("compositor-gallery");
    const auto folder = sandbox.root() / "gallery";
    const Gallery::Query current = {.owner = "cyberether", .version = "2.0.0", .platform = "linux", .arch = "arm64"};
    GalleryServer server;
    server.files[kAlphaUrl] = kAlpha;
    server.list(R"({"flowgraphs": [{"slug": "alpha", "flowgraphSha256": ")" + kAlphaDigest +
                R"(", "url": "https://files.test/alpha.yml"}]})", current);
    GalleryActions::Flowgraphs flowgraphs;
    REQUIRE(Gallery::FetchFrom(server.source(folder), current, {}, flowgraphs) == Result::SUCCESS);

    REQUIRE(Gallery::LoadFrom(server.source(folder), current, flowgraphs) == Result::SUCCESS);
    CHECK(Keys(flowgraphs) == std::vector<std::string>{"alpha"});
    auto other = current;
    other.version = "1.0.0";
    CHECK(Gallery::LoadFrom(server.source(folder), other, flowgraphs) == Result::ERROR);
    CHECK(flowgraphs.empty());
    other = current;
    other.platform = "windows";
    CHECK(Gallery::LoadFrom(server.source(folder), other, flowgraphs) == Result::ERROR);
    other = current;
    other.arch = "x86_64";
    CHECK(Gallery::LoadFrom(server.source(folder), other, flowgraphs) == Result::ERROR);

    WriteText(folder / "index.json", R"({"flowgraphs": [{"slug": "alpha", "flowgraphSha256": ")" + kAlphaDigest +
                                         R"("}]})");
    CHECK(Gallery::LoadFrom(server.source(folder), current, flowgraphs) == Result::ERROR);
}
#endif
