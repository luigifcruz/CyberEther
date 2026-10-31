#include <catch2/catch_test_macros.hpp>

#include "compositor/default/actions/workbench.hh"
#include "compositor/default/support/docs.hh"

#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace Jetstream;

namespace {

using Files = std::vector<std::pair<std::string_view, std::string_view>>;

std::vector<std::string> Slugs(const Docs& docs) {
    std::vector<std::string> slugs;
    for (const auto& page : docs.pages()) {
        slugs.push_back(page.slug);
    }
    return slugs;
}

}  // namespace

TEST_CASE("Docs parse frontmatter and order pages like the website navigation", "[core][compositor][docs]") {
    const Files files = {
        {"web", "---\ntitle: Web\norder: 41\ncategory: Usage\nparent: platforms\n---\n\nBrowser."},
        {"overview", "---\ntitle: Overview\ndescription: \"What it is: a framework.\"\norder: 1\n"
                     "category: Getting Started\n---\n\nHello."},
        {"platforms", "---\ntitle: Platforms\norder: 39\ncategory: Usage\n---\n\nThree ways."},
        {"native", "---\ntitle: Native\norder: 40\ncategory: Usage\nparent: platforms\n---\n\nDesktop."},
        {"testing", "---\ntitle: Typography Test\norder: 900\ncategory: Internal\n---\n\nHidden."},
        {"install-guide", "Plain body without frontmatter."},
        {"quick-start", "---\ntitle: Quick Start\norder: 3\ncategory: Getting Started\n---\n\nRun it."},
    };
    const Docs docs = Docs::Parse(files);

    CHECK(Slugs(docs) == std::vector<std::string>{"overview", "quick-start", "platforms", "native", "web",
                                                  "install-guide"});
    REQUIRE(docs.groups().size() == 3);
    CHECK(docs.groups()[0].title == "Getting Started");
    CHECK(docs.groups()[1].title == "Usage");
    CHECK(docs.groups()[1].pages.size() == 3);
    CHECK(docs.groups()[2].title == "Overview");

    const auto& overview = docs.pages()[*docs.find("overview")];
    CHECK(overview.description == "What it is: a framework.");
    CHECK(overview.body == "Hello.");
    CHECK(docs.pages()[*docs.find("native")].depth == 1);
    CHECK(docs.pages()[*docs.find("platforms")].depth == 0);
    CHECK(docs.pages()[*docs.find("install-guide")].title == "Install Guide");
    CHECK_FALSE(docs.find("testing").has_value());
}

TEST_CASE("Docs resolve site links and heading anchors", "[core][compositor][docs]") {
    const auto local = Docs::Resolve("/docs/blocks-and-modules#registration-and-dispatch");
    REQUIRE(local.has_value());
    CHECK(local->slug == "blocks-and-modules");
    CHECK(local->anchor == "registration-and-dispatch");

    const auto web = Docs::Resolve("https://cyberether.org/docs/quick-start/");
    REQUIRE(web.has_value());
    CHECK(web->slug == "quick-start");
    CHECK(web->anchor.empty());

    CHECK_FALSE(Docs::Resolve("https://github.com/luigifcruz/CyberEther").has_value());
    CHECK(Docs::Anchor("Registration And Dispatch") == "registration-and-dispatch");
    CHECK(Docs::Anchor("What is QPSK?") == "what-is-qpsk");
}

TEST_CASE("Bundled docs ship the website pages without the internal test page", "[core][compositor][docs]") {
    const auto& docs = Docs::Bundled();
    REQUIRE(docs.find(Docs::Home).has_value());
    CHECK(docs.find("quick-start").has_value());
    CHECK_FALSE(docs.find("testing").has_value());
    CHECK(docs.pages().front().slug == Docs::Home);
    for (const auto& page : docs.pages()) {
        CHECK_FALSE(page.title.empty());
        CHECK(page.body.rfind("---", 0) != 0);
    }
}

TEST_CASE("Docs tab opens on a page, falls back to the overview, and trades focus", "[core][compositor][docs]") {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    WorkbenchActions actions(state, callbacks);

    actions.handle(MailOpenSettings{});
    actions.handle(MailOpenDocs{.page = "quick-start", .anchor = "next-steps"});
    CHECK(state.interface.docsTab);
    CHECK(state.docsTabActive());
    CHECK_FALSE(state.settingsTabActive());
    CHECK(state.interface.docsPage == "quick-start");
    CHECK(state.interface.docsAnchor == "next-steps");
    CHECK(state.interface.docsNavigations == 1);
    CHECK(state.docsTabActive());
    CHECK_FALSE(state.activeFlowgraph().has_value());

    actions.handle(MailOpenDocs{.page = "missing-page"});
    CHECK(state.interface.docsPage == Docs::Home);
    CHECK(state.interface.docsNavigations == 2);

    actions.handle(MailFocusSettings{});
    CHECK_FALSE(state.docsTabActive());
    actions.handle(MailFocusDocs{});
    CHECK(state.docsTabActive());
    CHECK_FALSE(state.settingsTabActive());

    actions.handle(MailCloseDocs{});
    CHECK_FALSE(state.interface.docsTab);
    CHECK_FALSE(state.docsTabActive());
    actions.handle(MailFocusDocs{});
    CHECK_FALSE(state.docsTabActive());
}

TEST_CASE("Opening docs without a page keeps the page an open docs tab already shows", "[core][compositor][docs]") {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    WorkbenchActions actions(state, callbacks);

    actions.handle(MailOpenDocs{});
    CHECK(state.interface.docsPage == Docs::Home);
    CHECK(state.interface.docsNavigations == 1);

    actions.handle(MailOpenDocs{.page = "quick-start", .anchor = "next-steps"});
    actions.handle(MailOpenSettings{});
    CHECK_FALSE(state.docsTabActive());
    const U64 requests = state.interface.docsTabRequests;

    actions.handle(MailOpenDocs{});
    CHECK(state.docsTabActive());
    CHECK(state.interface.docsTabRequests == requests + 1);
    CHECK(state.interface.docsPage == "quick-start");
    CHECK(state.interface.docsAnchor == "next-steps");
    CHECK(state.interface.docsNavigations == 2);

    actions.handle(MailCloseDocs{});
    actions.handle(MailOpenDocs{});
    CHECK(state.interface.docsPage == Docs::Home);
    CHECK(state.interface.docsAnchor.empty());
    CHECK(state.interface.docsNavigations == 3);
}

TEST_CASE("Docs keep pages whose parents form a cycle at the top level", "[core][compositor][docs]") {
    const Files files = {
        {"alpha", "---\ntitle: Alpha\norder: 1\ncategory: Usage\nparent: beta\n---\n\nA."},
        {"beta", "---\ntitle: Beta\norder: 2\ncategory: Usage\nparent: alpha\n---\n\nB."},
        {"gamma", "---\ntitle: Gamma\norder: 3\ncategory: Usage\nparent: alpha\n---\n\nC."},
        {"self", "---\ntitle: Self\norder: 4\ncategory: Usage\nparent: self\n---\n\nD."},
    };
    const Docs docs = Docs::Parse(files);

    CHECK(Slugs(docs) == std::vector<std::string>{"alpha", "beta", "gamma", "self"});
    for (const auto& page : docs.pages()) {
        CHECK(page.depth == 0);
        CHECK(page.category == "Usage");
    }
    REQUIRE(docs.groups().size() == 1);
    CHECK(docs.groups()[0].pages.size() == 4);
}

TEST_CASE("Docs build website links from a slug and an optional anchor", "[core][compositor][docs]") {
    CHECK(Docs::WebUrl({.slug = "quick-start"}) == "https://cyberether.org/docs/quick-start");
    CHECK(Docs::WebUrl({.slug = "metadata", .anchor = "signal-axes"}) ==
          "https://cyberether.org/docs/metadata#signal-axes");
}

TEST_CASE("Docs search ranks sections and marks every hit in the snippet", "[core][compositor][docs]") {
    const Files files = {
        {"plugins", "---\ntitle: Creating Plugins\norder: 1\ncategory: Usage\n---\n\nIntro text.\n\n"
                    "## Bundling\n\nUse the blueprint's bundler with **blueprint** files.\n"},
        {"blocks", "---\ntitle: Blocks\norder: 2\ncategory: Usage\n---\n\n## File Layout\n\n"
                   "The plugin `blueprint` mirrors it.\n\n## Blueprint\n\nShort.\n"},
    };
    const Docs docs = Docs::Parse(files);

    const auto hits = docs.search("BLUEPRINT", 10);
    REQUIRE(hits.size() == 3);
    CHECK(hits[0].heading == "Blueprint");
    CHECK(hits[0].anchor == "blueprint");
    CHECK(docs.pages()[hits[0].page].slug == "blocks");
    CHECK(hits[1].heading == "Bundling");
    CHECK(hits[1].snippet == "Use the blueprint's bundler with blueprint files.");
    CHECK(hits[1].marks == std::vector<std::pair<U64, U64>>{{8, 19}, {33, 42}});
    CHECK(hits[2].heading == "File Layout");
    CHECK(hits[2].snippet == "The plugin blueprint mirrors it.");

    CHECK(docs.search("blueprint bundler", 10).size() == 1);
    CHECK(docs.search("blueprint", 1).size() == 1);
    CHECK(docs.search("   ", 10).empty());
}

TEST_CASE("Docs search keeps inline code text verbatim", "[core][compositor][docs]") {
    const Docs docs = Docs::Parse({
        {"tensors", "---\ntitle: Tensors\norder: 1\ncategory: Usage\n---\n\n"
                    "## Typed\n\nWrap it in `TypedTensor<T>` or `[a](b)` before <b>use</b>.\n"},
    });

    const auto hits = docs.search("TypedTensor<T>", 10);
    REQUIRE(hits.size() == 1);
    CHECK(hits[0].snippet == "Wrap it in TypedTensor<T> or [a](b) before use.");
    CHECK(docs.search("[a](b)", 10).size() == 1);
}
