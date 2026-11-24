#include <catch2/catch_test_macros.hpp>

#include "compositor/default/actions/gallery.hh"
#include "environment.hh"

#include "jetstream/config.hh"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Jetstream;

using Remote = GalleryActions::Remote;

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

Result NoCache(GalleryActions::Flowgraphs&) {
    return Result::ERROR;
}

void Settle(GalleryActions& actions, DefaultCompositorState& state, GalleryActions::Clock::time_point now) {
    actions.reconcile(now);
    REQUIRE(state.registry.fetch.loading);
    state.registry.fetch.future.wait();
    actions.reconcile(now + GalleryActions::ProgressMinimum);
    REQUIRE_FALSE(state.registry.fetch.loading);
}

}  // namespace

TEST_CASE("Fetched gallery flowgraphs publish into the home gallery and refresh the registry snapshot",
          "[core][compositor][gallery]") {
    DefaultCompositorState state;
    Home::Gallery gallery;
    REQUIRE(gallery.publish("/plugins/demo.cep", {Section("Demo", {Published("demo/example", "Plugin Demo")})}) ==
            Result::SUCCESS);

    Remote::Query seen;
    GalleryActions actions(state, gallery, [&](const Remote::Query& query, const std::function<bool()>&,
                                               GalleryActions::Flowgraphs& flowgraphs) {
        seen = query;
        flowgraphs = {Published("remote-only", "Remote Only"), Published("radio", "Radio", "Receivers")};
        return Result::SUCCESS;
    }, NoCache);
    const auto now = GalleryActions::Clock::now();
    Settle(actions, state, now);
    CHECK(seen.version == JETSTREAM_VERSION_STR);
    CHECK(seen.owner == Home::Gallery::Origin);
    CHECK(state.registry.stale);

    actions.refreshRegistry();
    CHECK(state.registry.revision == gallery.revision());
    CHECK(Titles(state.registry.sections) == std::vector<std::string>{"Basics", "Receivers", "Demo"});
    REQUIRE(state.registry.flowgraph("radio") != nullptr);
    CHECK(state.registry.flowgraph("radio")->title == "Radio");
    CHECK(state.registry.flowgraph("demo/example") != nullptr);
    CHECK(state.registry.flowgraph("missing") == nullptr);

    actions.reconcile(now + std::chrono::minutes(30));
    CHECK_FALSE(state.registry.fetch.loading);
    actions.reconcile(now + GalleryActions::RefreshInterval + std::chrono::seconds(5));
    CHECK(state.registry.fetch.loading);
    actions.shutdown();
    CHECK_FALSE(state.registry.fetch.loading);
}

TEST_CASE("Sections published outside the compositor mark the registry snapshot stale",
          "[core][compositor][gallery]") {
    TestSupport::SettingsSandbox sandbox("compositor-registry");
    sandbox.require();
    DefaultCompositorState state;
    Home::Gallery gallery;
    GalleryActions actions(state, gallery, [](const Remote::Query&, const std::function<bool()>&,
                                              GalleryActions::Flowgraphs&) {
        return Result::ERROR;
    }, NoCache);
    actions.refreshRegistry();
    REQUIRE_FALSE(state.registry.stale);

    const auto now = GalleryActions::Clock::now();
    REQUIRE(gallery.publish("/plugins/demo.cep", {Section("Demo", {Published("demo/example", "Plugin Demo")})}) ==
            Result::SUCCESS);
    actions.reconcile(now);
    CHECK(state.registry.stale);
    actions.refreshRegistry();
    CHECK(Titles(state.registry.sections) == std::vector<std::string>{"Demo"});
    actions.shutdown();
}

TEST_CASE("Gallery fetches the home gallery rejects count as failed attempts and retry soon",
          "[core][compositor][gallery]") {
    DefaultCompositorState state;
    Home::Gallery gallery;
    REQUIRE(gallery.publish("/plugins/demo.cep",
                            {Section("/plugins/demo.cep", {Published("first", "Plugin Demo")})}) == Result::SUCCESS);
    GalleryActions actions(state, gallery, [](const Remote::Query&, const std::function<bool()>&,
                                              GalleryActions::Flowgraphs& flowgraphs) {
        flowgraphs = {Published("first", "First")};
        return Result::SUCCESS;
    }, NoCache);
    const auto now = GalleryActions::Clock::now();
    Settle(actions, state, now);
    CHECK(state.registry.fetch.failed);
    CHECK(gallery.flowgraph("first")->title == "Plugin Demo");
    actions.reconcile(now + GalleryActions::RetryInterval + std::chrono::seconds(5));
    CHECK(state.registry.fetch.loading);
    actions.shutdown();
}

TEST_CASE("Incomplete gallery fetches that load nothing count as failed attempts and keep the shown sections",
          "[core][compositor][gallery]") {
    DefaultCompositorState state;
    Home::Gallery gallery;
    GalleryActions actions(
        state,
        gallery,
        [](const Remote::Query&, const std::function<bool()>&, GalleryActions::Flowgraphs& flowgraphs) {
            flowgraphs.clear();
            return Result::INCOMPLETE;
        },
        [](GalleryActions::Flowgraphs& flowgraphs) {
            flowgraphs = {Published("cached", "Cached")};
            return Result::SUCCESS;
        });
    const auto now = GalleryActions::Clock::now();
    Settle(actions, state, now);
    CHECK(state.registry.fetch.failed);
    CHECK(gallery.flowgraph("cached").has_value());
    actions.reconcile(now + GalleryActions::RetryInterval + std::chrono::seconds(5));
    CHECK(state.registry.fetch.loading);
    actions.shutdown();
}

TEST_CASE("Gallery fetches retry while offline and publish nothing", "[core][compositor][gallery]") {
    DefaultCompositorState state;
    Home::Gallery gallery;

    U64 attempts = 0;
    GalleryActions actions(state, gallery, [&](const Remote::Query&, const std::function<bool()>&,
                                               GalleryActions::Flowgraphs&) {
        ++attempts;
        return Result::ERROR;
    }, NoCache);
    const auto now = GalleryActions::Clock::now();
    Settle(actions, state, now);
    CHECK(attempts == 1);
    CHECK(state.registry.fetch.failed);
    CHECK(gallery.sections().empty());
    CHECK(gallery.revision() == 0);

    actions.reconcile(now + std::chrono::seconds(10));
    CHECK_FALSE(state.registry.fetch.loading);
    Settle(actions, state, now + GalleryActions::RetryInterval + std::chrono::seconds(5));
    CHECK(attempts == 2);
    CHECK(gallery.sections().empty());
}

TEST_CASE("Cached gallery flowgraphs show at startup before the refresh finishes", "[core][compositor][gallery]") {
    DefaultCompositorState state;
    Home::Gallery gallery;

    U64 loads = 0;
    GalleryActions actions(
        state,
        gallery,
        [&](const Remote::Query&, const std::function<bool()>&, GalleryActions::Flowgraphs& flowgraphs) {
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
    CHECK(state.registry.fetch.loading);
    CHECK(state.registry.fetch.cached);
    CHECK(state.registry.stale);
    actions.refreshRegistry();
    CHECK(state.registry.flowgraph("cached") != nullptr);

    state.registry.fetch.future.wait();
    actions.reconcile(now + GalleryActions::ProgressMinimum);
    CHECK(loads == 1);
    actions.refreshRegistry();
    CHECK(state.registry.flowgraph("fresh") != nullptr);
    CHECK(state.registry.flowgraph("cached") == nullptr);
}

TEST_CASE("Registry snapshot refreshes only while it is marked stale", "[core][compositor][gallery]") {
    TestSupport::SettingsSandbox sandbox("compositor-registry");
    sandbox.require();
    DefaultCompositorState state;
    Home::Gallery gallery;
    REQUIRE(gallery.publish(Home::Gallery::Origin, {Section("Basics", {Published("first", "First")})}) ==
            Result::SUCCESS);
    GalleryActions actions(state, gallery);
    REQUIRE(state.registry.stale);

    actions.refreshRegistry();
    CHECK_FALSE(state.registry.stale);
    CHECK_FALSE(state.registry.blocks.empty());
    CHECK_FALSE(state.registry.sections.empty());

    state.registry.sections.clear();
    actions.refreshRegistry();
    CHECK(state.registry.sections.empty());

    state.registry.stale = true;
    actions.refreshRegistry();
    CHECK_FALSE(state.registry.sections.empty());
}

TEST_CASE("Registry snapshot lists no plugins when none are installed",
          "[core][compositor][gallery]") {
    TestSupport::SettingsSandbox sandbox("compositor-registry");
    sandbox.require();

    DefaultCompositorState state;
    Home::Gallery gallery;
    GalleryActions actions(state, gallery);
    actions.refreshRegistry();
    CHECK_FALSE(state.registry.stale);
    CHECK_FALSE(state.registry.blocks.empty());
    CHECK(state.registry.plugins.empty());
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
    Home::Gallery gallery;
    GalleryActions(state, gallery).refreshRegistry();
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
    Home::Gallery gallery;
    U64 attempts = 0;
    GalleryActions actions(state, gallery, [&](const Remote::Query&, const std::function<bool()>&,
                                               GalleryActions::Flowgraphs& flowgraphs) {
        ++attempts;
        flowgraphs = {Published("fresh", "Fresh")};
        return Result::SUCCESS;
    }, NoCache);
    auto& fetch = state.registry.fetch;
    const auto now = GalleryActions::Clock::now();
    actions.reconcile(now);
    REQUIRE(fetch.loading);
    CHECK_FALSE(fetch.cached);

    fetch.future.wait();
    actions.reconcile(now + std::chrono::milliseconds(500));
    CHECK(fetch.loading);
    CHECK_FALSE(fetch.attempted);
    CHECK_FALSE(gallery.flowgraph("fresh").has_value());
    actions.reconcile(now + GalleryActions::ProgressMinimum);
    CHECK_FALSE(fetch.loading);
    CHECK(fetch.attempted);
    CHECK_FALSE(fetch.failed);
    CHECK(gallery.flowgraph("fresh").has_value());

    const auto later = now + GalleryActions::RefreshInterval + std::chrono::seconds(5);
    actions.reconcile(later);
    REQUIRE(fetch.loading);
    fetch.future.wait();
    actions.reconcile(later);
    CHECK_FALSE(fetch.loading);
    CHECK(attempts == 2);
}

TEST_CASE("Incomplete gallery fetches publish what arrived and retry soon", "[core][compositor][gallery]") {
    DefaultCompositorState state;
    Home::Gallery gallery;
    U64 attempts = 0;
    GalleryActions actions(state, gallery, [&](const Remote::Query&, const std::function<bool()>&,
                                               GalleryActions::Flowgraphs& flowgraphs) {
        ++attempts;
        flowgraphs = {Published("partial", "Partial")};
        return Result::INCOMPLETE;
    }, NoCache);
    const auto now = GalleryActions::Clock::now();
    Settle(actions, state, now);
    CHECK(attempts == 1);
    CHECK_FALSE(state.registry.fetch.failed);
    CHECK(state.registry.stale);
    CHECK(gallery.flowgraph("partial").has_value());

    actions.reconcile(now + std::chrono::seconds(10));
    CHECK_FALSE(state.registry.fetch.loading);
    Settle(actions, state, now + GalleryActions::RetryInterval + std::chrono::seconds(5));
    CHECK(attempts == 2);
}

TEST_CASE("Gallery fetches that throw count as failed attempts", "[core][compositor][gallery]") {
    DefaultCompositorState state;
    Home::Gallery gallery;
    GalleryActions actions(state, gallery, [](const Remote::Query&, const std::function<bool()>&,
                                              GalleryActions::Flowgraphs&) -> Result {
        throw std::runtime_error("malformed listing");
    }, NoCache);
    Settle(actions, state, GalleryActions::Clock::now());
    CHECK(state.registry.fetch.attempted);
    CHECK(state.registry.fetch.failed);
    CHECK(gallery.sections().empty());
}
