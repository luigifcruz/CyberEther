#include <catch2/catch_test_macros.hpp>

#include "environment.hh"
#include "home_gallery_remote.hh"

#include "jetstream/config.hh"
#include "jetstream/home.hh"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <optional>
#include <string>
#include <vector>

using namespace Jetstream;

using Remote = Home::Gallery::Remote;

namespace {

Home::Gallery::Flowgraph Published(const std::string& key,
                                   const std::string& title,
                                   const std::string& category = "Basics") {
    Home::Gallery::Flowgraph flowgraph;
    flowgraph.key = key;
    flowgraph.title = title;
    flowgraph.category = category;
    flowgraph.content = "---\nversion: 2\ngraph: []\n";
    return flowgraph;
}

Home::Gallery::Section Section(const std::string& key, std::vector<Home::Gallery::Flowgraph> flowgraphs) {
    return {.key = key, .title = key, .flowgraphs = std::move(flowgraphs)};
}

std::vector<std::string> Titles(const std::vector<Home::Gallery::Section>& sections) {
    std::vector<std::string> titles;
    for (const auto& section : sections) {
        titles.push_back(section.title);
    }
    return titles;
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

    void list(const std::string& body, const Remote::Query& query = {}) {
        files[std::string(Remote::Host) + Remote::Path(query)] = body;
    }

    Remote::Source source(std::optional<std::filesystem::path> folder = std::nullopt) {
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

std::vector<std::string> Keys(const Remote::Flowgraphs& flowgraphs) {
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

TEST_CASE("Gallery remote queries name the owner, version, platform and architecture and skip unknown values",
          "[core][home][gallery]") {
    CHECK(Remote::Path({.owner = "cyberether", .version = "1.16.2", .platform = "linux", .arch = "arm64"}) ==
          "/api/v1/flowgraphs?owner=cyberether&version=1.16.2&platform=linux&arch=arm64");
    CHECK(Remote::Path({.owner = "cyberether", .version = "1.16.2"}) ==
          "/api/v1/flowgraphs?owner=cyberether&version=1.16.2");

    const auto current = Remote::Current();
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

TEST_CASE("Gallery remote page links resolve to their flowgraph slug and other links do not", "[core][home][gallery]") {
    CHECK(Remote::Slug("https://cyberether.org/gallery/flowgraphs/biq-python-roundtrip") == "biq-python-roundtrip");
    CHECK(Remote::Slug("http://www.cyberether.org/gallery/flowgraphs/python-qpsk/") == "python-qpsk");
    CHECK(Remote::Slug("https://cyberether.org/gallery/flowgraphs/first_steps?ref=note#top") == "first_steps");
    CHECK_FALSE(Remote::Slug("https://cyberether.org/gallery/flowgraphs/").has_value());
    CHECK_FALSE(Remote::Slug("https://cyberether.org/gallery/flowgraphs/a/b").has_value());
    CHECK_FALSE(Remote::Slug("https://cyberether.org/docs/quick-start").has_value());
    CHECK_FALSE(Remote::Slug("https://example.com/gallery/flowgraphs/python-qpsk").has_value());
    CHECK_FALSE(Remote::Slug("https://cyberether.org.evil.com/gallery/flowgraphs/python-qpsk").has_value());
    CHECK_FALSE(Remote::Slug("/gallery/flowgraphs/python-qpsk").has_value());
}

TEST_CASE("Home gallery lists the CyberEther sections first and plugin sections in publish order",
          "[core][home][gallery]") {
    Home::Gallery gallery;
    CHECK(gallery.sections().empty());
    CHECK(gallery.revision() == 0);

    REQUIRE(gallery.publish("/plugins/one.cep", {Section("One", {Published("one/demo", "One Demo")})}) ==
            Result::SUCCESS);
    REQUIRE(gallery.publish("/plugins/two.cep", {Section("Two", {Published("two/demo", "Two Demo")})}) ==
            Result::SUCCESS);
    REQUIRE(gallery.publish(Home::Gallery::Origin, {Section("Basics", {Published("first-steps", "First Steps")}),
                                                    Section("DSP", {Published("fold", "Fold")})}) ==
            Result::SUCCESS);
    CHECK(Titles(gallery.sections()) == std::vector<std::string>{"Basics", "DSP", "One", "Two"});
    CHECK(gallery.revision() == 3);

    auto starred = Published("three/starred", "Starred");
    starred.featured = true;
    auto spotlight = Published("spotlight", "Spotlight");
    spotlight.featured = true;
    REQUIRE(gallery.publish("/plugins/three.cep", {Section("Three", {starred})}) == Result::SUCCESS);
    REQUIRE(gallery.publish(Home::Gallery::Origin, {Section("Basics", {Published("first-steps", "First Steps"),
                                                                       spotlight}),
                                                    Section("DSP", {Published("fold", "Fold")})}) ==
            Result::SUCCESS);
    CHECK_FALSE(gallery.flowgraph("three/starred")->featured);
    CHECK(gallery.flowgraph("spotlight")->featured);
    REQUIRE(gallery.publish("/plugins/three.cep", {}) == Result::SUCCESS);
    CHECK(gallery.revision() == 6);

    REQUIRE(gallery.flowgraph("two/demo").has_value());
    CHECK(gallery.flowgraph("two/demo")->title == "Two Demo");
    CHECK_FALSE(gallery.flowgraph("missing").has_value());

    REQUIRE(gallery.publish("/plugins/one.cep", {Section("One", {Published("one/other", "One Other")})}) ==
            Result::SUCCESS);
    CHECK(Titles(gallery.sections()) == std::vector<std::string>{"Basics", "DSP", "One", "Two"});
    CHECK_FALSE(gallery.flowgraph("one/demo").has_value());
    CHECK(gallery.flowgraph("one/other").has_value());

    REQUIRE(gallery.publish("/plugins/one.cep", {}) == Result::SUCCESS);
    CHECK(Titles(gallery.sections()) == std::vector<std::string>{"Basics", "DSP", "Two"});
    CHECK(gallery.revision() == 8);
    REQUIRE(gallery.publish("/plugins/one.cep", {}) == Result::SUCCESS);
    CHECK(gallery.revision() == 8);
}

TEST_CASE("Home gallery rejects empty origins and keys that are empty, repeated or owned by another origin",
          "[core][home][gallery]") {
    Home::Gallery gallery;
    REQUIRE(gallery.publish("/plugins/one.cep", {Section("One", {Published("one/demo", "One Demo")})}) ==
            Result::SUCCESS);
    const auto revision = gallery.revision();

    CHECK(gallery.publish("", {Section("Empty", {Published("empty", "Empty")})}) == Result::ERROR);
    CHECK(gallery.publish("/plugins/two.cep", {Section("", {Published("two/demo", "Two Demo")})}) == Result::ERROR);
    CHECK(gallery.publish("/plugins/two.cep", {Section("Two", {Published("", "Unnamed")})}) == Result::ERROR);
    CHECK(gallery.publish("/plugins/two.cep", {Section("One", {Published("two/demo", "Two Demo")})}) ==
          Result::ERROR);
    CHECK(gallery.publish("/plugins/two.cep", {Section("Two", {Published("one/demo", "Stolen")})}) ==
          Result::ERROR);
    CHECK(gallery.publish("/plugins/two.cep", {Section("Two", {Published("two/demo", "A"),
                                                               Published("two/demo", "B")})}) == Result::ERROR);
    CHECK(gallery.publish("/plugins/two.cep", {Section("Two", {}), Section("Two", {})}) == Result::ERROR);
    CHECK(gallery.revision() == revision);
    CHECK(Titles(gallery.sections()) == std::vector<std::string>{"One"});
    CHECK(gallery.flowgraph("one/demo")->title == "One Demo");
}

TEST_CASE("Home gallery flowgraphs read their metadata from the YAML root and default without it",
          "[core][home][gallery]") {
    const std::string yaml = R"(---
version: 2
revision: 1
title: Gallery Example
summary: Short summary.
category: Receivers
accent: '#5fd1b4'
featured: true
diagram: |-
  sdr[SDR] --> fm[FM]
  fm --> audio[Audio]
compatibility: '>=1.16.0'
graph: []
meta:
  layout:
    order: 1
)";
    Home::Gallery::Flowgraph flowgraph;
    REQUIRE(Home::Gallery::Flowgraph::Parse("gallery-example", yaml, flowgraph) == Result::SUCCESS);
    CHECK(flowgraph.key == "gallery-example");
    CHECK(flowgraph.title == "Gallery Example");
    CHECK(flowgraph.summary == "Short summary.");
    CHECK(flowgraph.category == "Receivers");
    CHECK(flowgraph.accent == "#5fd1b4");
    CHECK(flowgraph.featured);
    CHECK(flowgraph.diagram == "sdr[SDR] --> fm[FM]\nfm --> audio[Audio]");
    CHECK(flowgraph.content == yaml);

    Home::Gallery::Flowgraph plain;
    REQUIRE(Home::Gallery::Flowgraph::Parse("plain", "---\nversion: 2\ngraph: []\n", plain) == Result::SUCCESS);
    CHECK(plain.title.empty());
    CHECK(plain.summary == "No summary.");
    CHECK(plain.category.empty());
    CHECK(plain.accent.empty());
    CHECK(plain.diagram.empty());
    CHECK_FALSE(plain.featured);

    CHECK(Home::Gallery::Flowgraph::Parse("broken", "graph: [unterminated\n", plain) == Result::ERROR);
}

TEST_CASE("Gallery remote listings become one section per category in the order they first appear",
          "[core][home][gallery]") {
    const auto sections = Remote::Sections({Published("radio", "Radio", "Receivers"),
                                            Published("odd", "Odd", "Odd Stuff"),
                                            Published("first", "First", "Basics"),
                                            Published("loose", "Loose", ""),
                                            Published("fm", "FM", "Receivers")});
    CHECK(Titles(sections) == std::vector<std::string>{"Receivers", "Odd Stuff", "Basics", "Other"});
    REQUIRE(sections.size() == 4);
    CHECK(Keys(sections[0].flowgraphs) == std::vector<std::string>{"radio", "fm"});
    CHECK(sections[0].key == "cyberether/Receivers");
    CHECK(sections[0].description == "Tune, demodulate, and decode signals off the air.");
    CHECK(sections[1].description == "Complete pipelines, ready to run in Web or Native.");
    CHECK(sections[2].description == "Start here to learn the editor, sources, and plots.");
    CHECK(Keys(sections[3].flowgraphs) == std::vector<std::string>{"loose"});
    CHECK(Remote::Sections({}).empty());
}

TEST_CASE("Gallery remote fetches skip malformed entries and files that fail their size", "[core][home][gallery]") {
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
        {"slug": "nested/demo", "url": "https://files.test/beta.yml"},
        {"slug": "late", "url": "https://files.test/missing.yml"},
        {"slug": "late", "url": "https://files.test/alpha.yml"},
        {"slug": "plain", "url": "https://files.test/beta.yml", "size": 37},
        {"slug": "plain", "url": "https://files.test/alpha.yml"}
    ]})");
    Remote::Flowgraphs flowgraphs;
    REQUIRE(Remote::FetchFrom(server.source(), {}, {}, flowgraphs) == Result::INCOMPLETE);
    CHECK(Keys(flowgraphs) == std::vector<std::string>{"late", "plain"});
    REQUIRE(flowgraphs.size() == 2);
    CHECK(flowgraphs[0].title == "Alpha");
    CHECK(flowgraphs[1].title == "Beta");
    CHECK(flowgraphs[1].content == kBeta);

    server.list(R"({"flowgraphs": {"slug": "plain"}})");
    CHECK(Remote::FetchFrom(server.source(), {}, {}, flowgraphs) == Result::ERROR);
    CHECK(flowgraphs.empty());
    server.list("not json");
    CHECK(Remote::FetchFrom(server.source(), {}, {}, flowgraphs) == Result::ERROR);
    server.files.clear();
    CHECK(Remote::FetchFrom(server.source(), {}, {}, flowgraphs) == Result::ERROR);
    CHECK(Remote::FetchFrom({}, {}, {}, flowgraphs) == Result::ERROR);
}

TEST_CASE("Gallery remote cache restores skip malformed index entries", "[core][home][gallery]") {
    TestSupport::SettingsSandbox sandbox("compositor-gallery");
    const auto folder = sandbox.root() / "gallery";
    GalleryServer server;
    Remote::Flowgraphs flowgraphs;
    CHECK(Remote::LoadFrom(server.source(), {}, flowgraphs) == Result::ERROR);
    CHECK(Remote::LoadFrom(server.source(folder), {}, flowgraphs) == Result::ERROR);

    WriteText(folder / (kAlphaDigest + ".yml"), kAlpha);
    WriteText(folder / "index.json", R"({"query": {"owner": "", "version": "", "platform": "", "arch": ""},
    "flowgraphs": [
        {"slug": null, "flowgraphSha256": "c7811bde9fb80e2fd0be7b83c543ff25de34758f22b93935128a2118403088e7"},
        {"slug": "alpha", "flowgraphSha256": 5},
        {"slug": "alpha", "flowgraphSha256": "not-a-digest"}
    ]})");
    REQUIRE(Remote::LoadFrom(server.source(folder), {}, flowgraphs) == Result::SUCCESS);
    CHECK(flowgraphs.empty());
}

#if defined(JETSTREAM_LOADER_OPENSSL_AVAILABLE)
TEST_CASE("Gallery remote fetches verify downloads, cache them by digest and restore them from the index",
          "[core][home][gallery]") {
    TestSupport::SettingsSandbox sandbox("compositor-gallery");
    const auto folder = sandbox.root() / "gallery";
    const auto cached = folder / (kAlphaDigest + ".yml");
    GalleryServer server;
    server.files[kAlphaUrl] = kAlpha;
    server.list(R"({"flowgraphs": [{"slug": "alpha", "flowgraphSha256": ")" + kAlphaDigest +
                R"(", "url": "https://files.test/alpha.yml", "size": 38}]})");
    WriteText(folder / "stale.yml", kBeta);

    Remote::Flowgraphs flowgraphs;
    REQUIRE(Remote::FetchFrom(server.source(folder), {}, {}, flowgraphs) == Result::SUCCESS);
    CHECK(Keys(flowgraphs) == std::vector<std::string>{"alpha"});
    CHECK(ReadText(cached) == kAlpha);
    CHECK(std::filesystem::exists(folder / "index.json"));
    CHECK_FALSE(std::filesystem::exists(folder / "stale.yml"));
    CHECK(server.requested(kAlphaUrl));

    server.requests.clear();
    server.files.erase(kAlphaUrl);
    REQUIRE(Remote::FetchFrom(server.source(folder), {}, {}, flowgraphs) == Result::SUCCESS);
    CHECK(Keys(flowgraphs) == std::vector<std::string>{"alpha"});
    CHECK_FALSE(server.requested(kAlphaUrl));

    WriteText(cached, kBeta);
    server.files[kAlphaUrl] = kAlpha;
    REQUIRE(Remote::FetchFrom(server.source(folder), {}, {}, flowgraphs) == Result::SUCCESS);
    CHECK(Keys(flowgraphs) == std::vector<std::string>{"alpha"});
    CHECK(flowgraphs.front().title == "Alpha");
    CHECK(server.requested(kAlphaUrl));
    CHECK(ReadText(cached) == kAlpha);

    REQUIRE(Remote::LoadFrom(server.source(folder), {}, flowgraphs) == Result::SUCCESS);
    CHECK(Keys(flowgraphs) == std::vector<std::string>{"alpha"});
    WriteText(cached, kBeta);
    REQUIRE(Remote::LoadFrom(server.source(folder), {}, flowgraphs) == Result::SUCCESS);
    CHECK(flowgraphs.empty());
}

TEST_CASE("Gallery remote refreshes whose entries are all rejected keep the cached index and its files",
          "[core][home][gallery]") {
    TestSupport::SettingsSandbox sandbox("home-gallery");
    const auto folder = sandbox.root() / "gallery";
    GalleryServer server;
    server.files[kAlphaUrl] = kAlpha;
    server.list(R"({"flowgraphs": [{"slug": "alpha", "flowgraphSha256": ")" + kAlphaDigest +
                R"(", "url": "https://files.test/alpha.yml"}]})");
    Remote::Flowgraphs flowgraphs;
    REQUIRE(Remote::FetchFrom(server.source(folder), {}, {}, flowgraphs) == Result::SUCCESS);

    server.list(R"({"flowgraphs": [{"slug": "alpha.v2", "url": "https://files.test/alpha.yml"}]})");
    CHECK(Remote::FetchFrom(server.source(folder), {}, {}, flowgraphs) == Result::INCOMPLETE);
    CHECK(flowgraphs.empty());
    CHECK(ReadText(folder / (kAlphaDigest + ".yml")) == kAlpha);
    REQUIRE(Remote::LoadFrom(server.source(folder), {}, flowgraphs) == Result::SUCCESS);
    CHECK(Keys(flowgraphs) == std::vector<std::string>{"alpha"});
}

TEST_CASE("Gallery remote fetches skip downloads that do not match their digest and never cache them",
          "[core][home][gallery]") {
    TestSupport::SettingsSandbox sandbox("compositor-gallery");
    const auto folder = sandbox.root() / "gallery";
    GalleryServer server;
    server.files[kAlphaUrl] = kAlpha;
    server.files[kBetaUrl] = kBeta;
    server.list(R"({"flowgraphs": [
        {"slug": "forged", "flowgraphSha256": ")" + kBetaDigest + R"(", "url": "https://files.test/alpha.yml"},
        {"slug": "beta", "flowgraphSha256": ")" + kBetaDigest + R"(", "url": "https://files.test/beta.yml"}
    ]})");

    Remote::Flowgraphs flowgraphs;
    REQUIRE(Remote::FetchFrom(server.source(folder), {}, {}, flowgraphs) == Result::INCOMPLETE);
    CHECK(Keys(flowgraphs) == std::vector<std::string>{"beta"});
    CHECK(ReadText(folder / (kBetaDigest + ".yml")) == kBeta);
    CHECK_FALSE(std::filesystem::exists(folder / (kAlphaDigest + ".yml")));
}

TEST_CASE("Gallery remote refreshes keep the cached copy of a flowgraph whose new version fails to download",
          "[core][home][gallery]") {
    TestSupport::SettingsSandbox sandbox("compositor-gallery");
    const auto folder = sandbox.root() / "gallery";
    GalleryServer server;
    server.files[kAlphaUrl] = kAlpha;
    server.list(R"({"flowgraphs": [{"slug": "demo", "flowgraphSha256": ")" + kAlphaDigest +
                R"(", "url": "https://files.test/alpha.yml"}]})");
    Remote::Flowgraphs flowgraphs;
    REQUIRE(Remote::FetchFrom(server.source(folder), {}, {}, flowgraphs) == Result::SUCCESS);
    REQUIRE(Keys(flowgraphs) == std::vector<std::string>{"demo"});

    server.list(R"({"flowgraphs": [
        {"slug": "demo", "flowgraphSha256": ")" + kBetaDigest + R"(", "url": "https://files.test/beta.yml"},
        {"slug": "fresh", "flowgraphSha256": ")" + kBetaDigest + R"(", "url": "https://files.test/beta.yml"}
    ]})");
    CHECK(Remote::FetchFrom(server.source(folder), {}, {}, flowgraphs) == Result::INCOMPLETE);
    REQUIRE(Keys(flowgraphs) == std::vector<std::string>{"demo"});
    CHECK(flowgraphs.front().title == "Alpha");
    CHECK(ReadText(folder / (kAlphaDigest + ".yml")) == kAlpha);
    REQUIRE(Remote::LoadFrom(server.source(folder), {}, flowgraphs) == Result::SUCCESS);
    REQUIRE(Keys(flowgraphs) == std::vector<std::string>{"demo"});
    CHECK(flowgraphs.front().title == "Alpha");

    CHECK(Remote::FetchFrom(server.source(folder), {}, {}, flowgraphs) == Result::INCOMPLETE);
    CHECK(Keys(flowgraphs) == std::vector<std::string>{"demo"});

    server.files[kBetaUrl] = kBeta;
    REQUIRE(Remote::FetchFrom(server.source(folder), {}, {}, flowgraphs) == Result::SUCCESS);
    CHECK(Keys(flowgraphs) == std::vector<std::string>{"demo", "fresh"});
    CHECK(flowgraphs.front().title == "Beta");
    CHECK_FALSE(std::filesystem::exists(folder / (kAlphaDigest + ".yml")));
}

TEST_CASE("Gallery remote refreshes keep the cached files the saved index needs when the new index cannot be saved",
          "[core][home][gallery]") {
    TestSupport::SettingsSandbox sandbox("compositor-gallery");
    const auto folder = sandbox.root() / "gallery";
    GalleryServer server;
    server.files[kAlphaUrl] = kAlpha;
    server.files[kBetaUrl] = kBeta;
    server.list(R"({"flowgraphs": [{"slug": "demo", "flowgraphSha256": ")" + kAlphaDigest +
                R"(", "url": "https://files.test/alpha.yml"}]})");
    Remote::Flowgraphs flowgraphs;
    REQUIRE(Remote::FetchFrom(server.source(folder), {}, {}, flowgraphs) == Result::SUCCESS);

    std::filesystem::create_directories(folder / "index.json.tmp");
    server.list(R"({"flowgraphs": [{"slug": "demo", "flowgraphSha256": ")" + kBetaDigest +
                R"(", "url": "https://files.test/beta.yml"}]})");
    REQUIRE(Remote::FetchFrom(server.source(folder), {}, {}, flowgraphs) == Result::SUCCESS);
    CHECK(flowgraphs.front().title == "Beta");

    CHECK(ReadText(folder / (kAlphaDigest + ".yml")) == kAlpha);
    REQUIRE(Remote::LoadFrom(server.source(folder), {}, flowgraphs) == Result::SUCCESS);
    REQUIRE(Keys(flowgraphs) == std::vector<std::string>{"demo"});
    CHECK(flowgraphs.front().title == "Alpha");
}

TEST_CASE("Gallery remote refreshes keep the previous cached copy when the new version cannot be saved",
          "[core][home][gallery]") {
    TestSupport::SettingsSandbox sandbox("compositor-gallery");
    const auto folder = sandbox.root() / "gallery";
    GalleryServer server;
    server.files[kAlphaUrl] = kAlpha;
    server.files[kBetaUrl] = kBeta;
    server.list(R"({"flowgraphs": [{"slug": "demo", "flowgraphSha256": ")" + kAlphaDigest +
                R"(", "url": "https://files.test/alpha.yml"}]})");
    Remote::Flowgraphs flowgraphs;
    REQUIRE(Remote::FetchFrom(server.source(folder), {}, {}, flowgraphs) == Result::SUCCESS);

    std::filesystem::create_directories(folder / (kBetaDigest + ".yml.tmp"));
    server.list(R"({"flowgraphs": [{"slug": "demo", "flowgraphSha256": ")" + kBetaDigest +
                R"(", "url": "https://files.test/beta.yml"}]})");
    REQUIRE(Remote::FetchFrom(server.source(folder), {}, {}, flowgraphs) == Result::SUCCESS);
    CHECK(flowgraphs.front().title == "Beta");

    CHECK(ReadText(folder / (kAlphaDigest + ".yml")) == kAlpha);
    REQUIRE(Remote::LoadFrom(server.source(folder), {}, flowgraphs) == Result::SUCCESS);
    REQUIRE(Keys(flowgraphs) == std::vector<std::string>{"demo"});
    CHECK(flowgraphs.front().title == "Alpha");
}

TEST_CASE("Gallery remote cache restores ignore an index written for another query", "[core][home][gallery]") {
    TestSupport::SettingsSandbox sandbox("compositor-gallery");
    const auto folder = sandbox.root() / "gallery";
    const Remote::Query current = {.owner = "cyberether", .version = "2.0.0", .platform = "linux", .arch = "arm64"};
    GalleryServer server;
    server.files[kAlphaUrl] = kAlpha;
    server.list(R"({"flowgraphs": [{"slug": "alpha", "flowgraphSha256": ")" + kAlphaDigest +
                R"(", "url": "https://files.test/alpha.yml"}]})", current);
    Remote::Flowgraphs flowgraphs;
    REQUIRE(Remote::FetchFrom(server.source(folder), current, {}, flowgraphs) == Result::SUCCESS);

    REQUIRE(Remote::LoadFrom(server.source(folder), current, flowgraphs) == Result::SUCCESS);
    CHECK(Keys(flowgraphs) == std::vector<std::string>{"alpha"});
    auto other = current;
    other.version = "1.0.0";
    CHECK(Remote::LoadFrom(server.source(folder), other, flowgraphs) == Result::ERROR);
    CHECK(flowgraphs.empty());
    other = current;
    other.platform = "windows";
    CHECK(Remote::LoadFrom(server.source(folder), other, flowgraphs) == Result::ERROR);
    other = current;
    other.arch = "x86_64";
    CHECK(Remote::LoadFrom(server.source(folder), other, flowgraphs) == Result::ERROR);

    WriteText(folder / "index.json", R"({"flowgraphs": [{"slug": "alpha", "flowgraphSha256": ")" + kAlphaDigest +
                                         R"("}]})");
    CHECK(Remote::LoadFrom(server.source(folder), current, flowgraphs) == Result::ERROR);
}
#endif
