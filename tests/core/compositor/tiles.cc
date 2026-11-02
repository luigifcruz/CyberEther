#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "compositor/default/model/meta.hh"
#include "compositor/default/model/tiles.hh"
#include "compositor/default/presenters/flowgraph/labels.hh"
#include "compositor/default/presenters/flowgraph/layout.hh"

#include <any>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

using namespace Jetstream;

namespace {

TileMeta Leaf(const std::string& pane, std::optional<F32> size = std::nullopt) {
    return TileMeta{.size = size, .pane = pane};
}

TileMeta Split(const std::string& split, std::vector<TileMeta> children, std::optional<F32> size = std::nullopt) {
    return TileMeta{.split = split, .size = size, .children = std::move(children)};
}

std::vector<std::string> Panes(const TileMeta& tiles) {
    std::vector<std::string> panes;
    CollectTilePanes(tiles, panes);
    return panes;
}

Sakura::DockTiles DockLeaf(const std::string& pane) {
    return Sakura::DockTiles{.pane = pane};
}

}  // namespace

TEST_CASE("Surface tile keys round trip block and surface names", "[core][compositor][tiles]") {
    const auto key = MakeSurfaceTileKey("fft", "default");
    REQUIRE(key == "surface:fft:default");

    const auto parsed = ParseSurfaceTileKey(key);
    REQUIRE(parsed.has_value());
    REQUIRE(parsed->first == "fft");
    REQUIRE(parsed->second == "default");

    const auto nested = ParseSurfaceTileKey(MakeSurfaceTileKey("fft", "a:b"));
    REQUIRE(nested.has_value());
    REQUIRE(nested->first == "fft");
    REQUIRE(nested->second == "a:b");
}

TEST_CASE("Malformed surface tile keys are rejected", "[core][compositor][tiles]") {
    REQUIRE_FALSE(ParseSurfaceTileKey("fft:default").has_value());
    REQUIRE_FALSE(ParseSurfaceTileKey("config:fft").has_value());
    REQUIRE_FALSE(ParseSurfaceTileKey("Surface:fft:default").has_value());
    REQUIRE_FALSE(ParseSurfaceTileKey("surface:").has_value());
    REQUIRE_FALSE(ParseSurfaceTileKey("surface::default").has_value());
    REQUIRE_FALSE(ParseSurfaceTileKey("surface:fft:").has_value());
    REQUIRE_FALSE(ParseSurfaceTileKey("surface:fft").has_value());
    REQUIRE_FALSE(ParseSurfaceTileKey("").has_value());
}

TEST_CASE("Config tile keys round trip block names and reject malformed keys",
          "[core][compositor][tiles]") {
    const auto key = MakeConfigTileKey("fft");
    REQUIRE(key == "config:fft");
    REQUIRE(ParseConfigTileKey(key) == std::optional<std::string>("fft"));
    REQUIRE(ParseConfigTileKey("config:a:b") == std::optional<std::string>("a:b"));

    REQUIRE_FALSE(ParseConfigTileKey("config:").has_value());
    REQUIRE_FALSE(ParseConfigTileKey("fft").has_value());
    REQUIRE_FALSE(ParseConfigTileKey("surface:fft:default").has_value());
    REQUIRE_FALSE(ParseConfigTileKey("").has_value());
}

TEST_CASE("Tile keys resolve to their owning block", "[core][compositor][tiles]") {
    REQUIRE(TileBlock("surface:fft:default") == std::optional<std::string>("fft"));
    REQUIRE(TileBlock("config:fft") == std::optional<std::string>("fft"));
    REQUIRE_FALSE(TileBlock("flowgraph").has_value());
    REQUIRE_FALSE(TileBlock("surface:fft").has_value());
    REQUIRE_FALSE(TileBlock("config:").has_value());
}

TEST_CASE("Tiles without children are leaves", "[core][compositor][tiles]") {
    REQUIRE(IsTileLeaf(TileMeta{}));
    REQUIRE(IsTileLeaf(Leaf("a")));
    REQUIRE(IsTileLeaf(TileMeta{.split = "row", .children = std::vector<TileMeta>{}}));
    REQUIRE_FALSE(IsTileLeaf(Split("row", {Leaf("a")})));
}

TEST_CASE("Mapping tiles renames leaves and keeps their sizes", "[core][compositor][tiles]") {
    const auto mapped = MapTiles(Leaf("a", 500.0f), [](const std::string& pane) -> std::optional<std::string> {
        return pane + "!";
    });
    REQUIRE(mapped.has_value());
    REQUIRE(IsTileLeaf(*mapped));
    REQUIRE(mapped->pane == "a!");
    REQUIRE(mapped->size == 500.0f);

    const auto tree = Split("column", {Leaf("a", 250.0f), Split("row", {Leaf("b", 1.0f), Leaf("c", 1.0f)}, 750.0f)},
                            2000.0f);
    const auto renamed = MapTiles(tree, [](const std::string& pane) -> std::optional<std::string> {
        return "x" + pane;
    });
    REQUIRE(renamed.has_value());
    REQUIRE(renamed->split == "column");
    REQUIRE(renamed->size == 2000.0f);
    REQUIRE(Panes(*renamed) == std::vector<std::string>{"xa", "xb", "xc"});
    REQUIRE(renamed->children->at(0).size == 250.0f);
    REQUIRE(renamed->children->at(1).size == 750.0f);
    REQUIRE(renamed->children->at(1).split == "row");
}

TEST_CASE("Tile view flags survive serialization, filtering, and dock conversion",
          "[core][compositor][tiles][recovery]") {
    auto collapsed = Leaf("description", 520.0f);
    collapsed.minimized = true;
    auto fullscreen = Leaf("config:a", 310.0f);
    fullscreen.maximized = true;
    const auto original = Split("row", {collapsed, fullscreen, Leaf("terminal:1")});
    Parser::Map encoded;
    REQUIRE(original.serialize(encoded) == Result::SUCCESS);
    TileMeta decoded;
    REQUIRE(decoded.deserialize(encoded) == Result::SUCCESS);
    const auto filtered = FilterTiles(decoded, IsLayoutPane);
    REQUIRE(filtered.has_value());
    const auto dock = ToDockTiles(*filtered, LayoutToDockPanes("graph"));
    REQUIRE(dock.has_value());
    REQUIRE(dock->children.size() == 2);
    CHECK(dock->children[0].minimized);
    CHECK(dock->children[1].maximized);
    const auto restored = ToTileMeta(*dock, DockToLayoutPanes("graph"));
    REQUIRE(restored.has_value());
    CHECK(Parser::Hash(*restored) == Parser::Hash(*filtered));
}

TEST_CASE("Mapping tiles drops empty and rejected leaves", "[core][compositor][tiles]") {
    const auto identity = [](const std::string& pane) -> std::optional<std::string> {
        return pane;
    };
    REQUIRE_FALSE(MapTiles(TileMeta{}, identity).has_value());
    REQUIRE_FALSE(MapTiles(Leaf(""), identity).has_value());
    REQUIRE_FALSE(MapTiles(Leaf("a"), [](const std::string&) -> std::optional<std::string> {
        return std::nullopt;
    }).has_value());
    REQUIRE_FALSE(MapTiles(Leaf("a"), [](const std::string&) -> std::optional<std::string> {
        return std::string();
    }).has_value());

    const auto tree = Split("row", {Leaf("a", 1.0f), Leaf("drop", 2.0f), Leaf("b", 3.0f)});
    const auto mapped = MapTiles(tree, [](const std::string& pane) -> std::optional<std::string> {
        return pane == "drop" ? std::nullopt : std::optional<std::string>(pane);
    });
    REQUIRE(mapped.has_value());
    REQUIRE(Panes(*mapped) == std::vector<std::string>{"a", "b"});
    REQUIRE(mapped->children->at(0).size == 1.0f);
    REQUIRE(mapped->children->at(1).size == 3.0f);
}

TEST_CASE("Mapping tiles collapses single children into the parent size", "[core][compositor][tiles]") {
    const auto keepA = [](const std::string& pane) -> std::optional<std::string> {
        return pane == "a" ? std::optional<std::string>(pane) : std::nullopt;
    };

    const auto collapsed = MapTiles(Split("row", {Leaf("a", 300.0f), Leaf("b", 700.0f)}, 400.0f), keepA);
    REQUIRE(collapsed.has_value());
    REQUIRE(IsTileLeaf(*collapsed));
    REQUIRE(collapsed->pane == "a");
    REQUIRE(collapsed->size == 400.0f);

    const auto unsized = MapTiles(Split("row", {Leaf("a", 300.0f), Leaf("b", 700.0f)}), keepA);
    REQUIRE(unsized.has_value());
    REQUIRE_FALSE(unsized->size.has_value());

    const auto nested = Split("row", {
        Split("column", {Leaf("a", 200.0f), Leaf("b", 800.0f)}, 600.0f),
        Leaf("c", 400.0f),
    });
    const auto mapped = MapTiles(nested, [](const std::string& pane) -> std::optional<std::string> {
        return pane == "b" ? std::nullopt : std::optional<std::string>(pane);
    });
    REQUIRE(mapped.has_value());
    REQUIRE(mapped->split == "row");
    REQUIRE(mapped->children->size() == 2);
    REQUIRE(IsTileLeaf(mapped->children->at(0)));
    REQUIRE(mapped->children->at(0).pane == "a");
    REQUIRE(mapped->children->at(0).size == 600.0f);
    REQUIRE(mapped->children->at(1).size == 400.0f);
}

TEST_CASE("Mapping tiles keeps child sizes as given and normalizes splits", "[core][compositor][tiles]") {
    const auto identity = [](const std::string& pane) -> std::optional<std::string> {
        return pane;
    };

    const auto sized = MapTiles(Split("column", {Leaf("a"), Leaf("b", 200.0f)}), identity);
    REQUIRE(sized.has_value());
    REQUIRE(sized->split == "column");
    REQUIRE_FALSE(sized->children->at(0).size.has_value());
    REQUIRE(sized->children->at(1).size == 200.0f);

    for (const auto& split : {std::string("row"), std::string("diagonal"), std::string()}) {
        const auto mapped = MapTiles(Split(split, {Leaf("a"), Leaf("b")}), identity);
        REQUIRE(mapped.has_value());
        REQUIRE(mapped->split == "row");
    }

    TileMeta unsplit{.children = std::vector<TileMeta>{Leaf("a"), Leaf("b")}};
    const auto mapped = MapTiles(unsplit, identity);
    REQUIRE(mapped.has_value());
    REQUIRE(mapped->split == "row");
}

TEST_CASE("Mapping tiles with no surviving leaves yields nothing", "[core][compositor][tiles]") {
    const auto tree = Split("row", {Leaf("a"), Split("column", {Leaf("b"), Leaf("c")})});
    REQUIRE_FALSE(MapTiles(tree, [](const std::string&) -> std::optional<std::string> {
        return std::nullopt;
    }).has_value());
    REQUIRE_FALSE(FilterTiles(tree, [](const std::string&) { return false; }).has_value());
}

TEST_CASE("Filtering tiles keeps only accepted panes", "[core][compositor][tiles]") {
    const auto tree = Split("row", {Leaf("a", 500.0f), Split("column", {Leaf("b", 300.0f), Leaf("c", 200.0f)}, 400.0f)});

    const auto all = FilterTiles(tree, [](const std::string&) { return true; });
    REQUIRE(all.has_value());
    REQUIRE(Panes(*all) == std::vector<std::string>{"a", "b", "c"});
    REQUIRE(all->children->at(1).children->at(0).size == 300.0f);

    const auto filtered = FilterTiles(tree, [](const std::string& pane) { return pane != "c"; });
    REQUIRE(filtered.has_value());
    REQUIRE(Panes(*filtered) == std::vector<std::string>{"a", "b"});
    REQUIRE(IsTileLeaf(filtered->children->at(1)));
    REQUIRE(filtered->children->at(1).size == 400.0f);
}

TEST_CASE("Collected tile panes follow depth first order", "[core][compositor][tiles]") {
    const auto tree = Split("row", {
        Split("column", {Leaf("a"), Leaf("")}),
        Leaf("b"),
        Split("column", {Split("row", {Leaf("c"), Leaf("d")}), Leaf("e")}),
    });
    REQUIRE(Panes(tree) == std::vector<std::string>{"a", "b", "c", "d", "e"});
    REQUIRE(Panes(TileMeta{}).empty());

    REQUIRE(ContainsTilePane(tree, "d"));
    REQUIRE(ContainsTilePane(Leaf("a"), "a"));
    REQUIRE_FALSE(ContainsTilePane(tree, "f"));
    REQUIRE_FALSE(ContainsTilePane(tree, ""));
}

TEST_CASE("Layout panes are the editor and block tiles only", "[core][compositor][tiles]") {
    REQUIRE(IsLayoutPane("flowgraph"));
    REQUIRE(IsLayoutPane("config:fft"));
    REQUIRE(IsLayoutPane("surface:fft:default"));
    REQUIRE_FALSE(IsLayoutPane("stack_0"));
    REQUIRE_FALSE(IsLayoutPane("config:"));
    REQUIRE_FALSE(IsLayoutPane(""));
}

TEST_CASE("Tile metadata round trips nested trees through YAML", "[core][compositor][tiles][metadata]") {
    const auto tiles = Split("row", {
        Leaf("flowgraph", 640.5f),
        Split("column", {Leaf("config:a", 250.0f), Leaf("config:b", 750.25f)}, 420.0f),
    });

    Parser::Map encoded;
    REQUIRE(tiles.serialize(encoded) == Result::SUCCESS);
    std::string yaml;
    REQUIRE(Parser::YamlEncode(encoded, yaml) == Result::SUCCESS);

    Parser::Map decoded;
    REQUIRE(Parser::YamlDecode(yaml, decoded) == Result::SUCCESS);
    TileMeta restored;
    REQUIRE(restored.deserialize(decoded) == Result::SUCCESS);

    REQUIRE(restored.split == "row");
    REQUIRE_FALSE(restored.size.has_value());
    REQUIRE_FALSE(restored.pane.has_value());
    REQUIRE(restored.children.has_value());
    REQUIRE(restored.children->size() == 2);
    REQUIRE(restored.children->at(0).pane == "flowgraph");
    REQUIRE(*restored.children->at(0).size == Catch::Approx(640.5f));
    REQUIRE_FALSE(restored.children->at(0).children.has_value());

    const auto& column = restored.children->at(1);
    REQUIRE(column.split == "column");
    REQUIRE(*column.size == Catch::Approx(420.0f));
    REQUIRE(Panes(column) == std::vector<std::string>{"config:a", "config:b"});
    REQUIRE(*column.children->at(1).size == Catch::Approx(750.25f));
    REQUIRE(Panes(restored) == Panes(tiles));
    REQUIRE(IsTileLayout(restored));
}

TEST_CASE("Tile metadata decodes hand written YAML", "[core][compositor][tiles][metadata]") {
    Parser::Map decoded;
    REQUIRE(Parser::YamlDecode(R"(
split: column
children:
  - pane: config:fft
    size: 320
  - split: row
    size: 280
    children:
      - pane: surface:fft:default
        size: 500
      - pane: surface:fft:waterfall
        size: 300
)", decoded) == Result::SUCCESS);

    TileMeta tiles;
    REQUIRE(tiles.deserialize(decoded) == Result::SUCCESS);
    REQUIRE(tiles.split == "column");
    REQUIRE(Panes(tiles) == std::vector<std::string>{"config:fft", "surface:fft:default", "surface:fft:waterfall"});
    REQUIRE(*tiles.children->at(0).size == Catch::Approx(320.0f));
    REQUIRE(*tiles.children->at(1).size == Catch::Approx(280.0f));
    REQUIRE(*tiles.children->at(1).children->at(0).size == Catch::Approx(500.0f));
    REQUIRE(IsTileLayout(tiles));
}

TEST_CASE("Layouts saved with weights instead of sizes are not valid layouts", "[core][compositor][tiles][metadata]") {
    Parser::Map decoded;
    REQUIRE(Parser::YamlDecode(R"(
split: row
children:
  - pane: flowgraph
    weight: 640
  - split: column
    weight: 420
    children:
      - pane: config:fft
        weight: 0.5
      - pane: surface:fft:default
        weight: 0.5
)", decoded) == Result::SUCCESS);

    TileMeta tiles;
    REQUIRE(tiles.deserialize(decoded) == Result::SUCCESS);
    CHECK_FALSE(IsTileLayout(tiles));
    CHECK_FALSE(ToDockTiles(tiles).has_value());
}

TEST_CASE("Tile layouts need a known split and a finite size on every child", "[core][compositor][tiles]") {
    CHECK(IsTileLayout(Leaf("a")));
    CHECK(IsTileLayout(Split("row", {Leaf("a", 300.0f), Leaf("b", 0.0f)})));
    CHECK_FALSE(IsTileLayout(Leaf("")));
    CHECK_FALSE(IsTileLayout(TileMeta{}));
    CHECK_FALSE(IsTileLayout(Split("row", {Leaf("a", 300.0f), Leaf("b")})));
    CHECK_FALSE(IsTileLayout(Split("row", {Leaf("a", 300.0f), Leaf("b", -1.0f)})));
    CHECK_FALSE(IsTileLayout(Split("row", {Leaf("a", 300.0f), Leaf("b", std::numeric_limits<F32>::infinity())})));
    auto diagonal = Leaf("a");
    diagonal.split = "diagonal";
    CHECK_FALSE(IsTileLayout(diagonal));
    CHECK_FALSE(IsTileLayout(Split("row", {Leaf("a", 300.0f), diagonal})));
    CHECK_FALSE(IsTileLayout(Leaf("a", -1.0f)));
    CHECK_FALSE(IsTileLayout(Split("row", {Leaf("a", 300.0f)}, std::numeric_limits<F32>::quiet_NaN())));
    CHECK_FALSE(IsTileLayout(Split("diagonal", {Leaf("a", 300.0f), Leaf("b", 300.0f)})));
    CHECK_FALSE(IsTileLayout(Split("row", {Leaf("a", 300.0f), Split("column", {Leaf("b", 1.0f), Leaf("c")}, 200.0f)})));
}

TEST_CASE("Tile metadata converts to dock tiles with sizes and orientation", "[core][compositor][dock][tiles]") {
    const auto meta = Split("row", {
        Leaf("a", 600.0f),
        Split("column", {Leaf("b", 100.0f), Leaf("c", 300.0f)}, 400.0f),
    });

    const auto dock = ToDockTiles(meta);
    REQUIRE(dock.has_value());
    REQUIRE_FALSE(dock->vertical);
    REQUIRE(dock->pane.empty());
    REQUIRE(dock->sizes == std::vector<F32>{600.0f, 400.0f});
    REQUIRE(dock->children.size() == 2);
    REQUIRE(dock->children[0] == DockLeaf("a"));

    const auto& column = dock->children[1];
    REQUIRE(column.vertical);
    REQUIRE(column.sizes == std::vector<F32>{100.0f, 300.0f});
    REQUIRE(column.children == std::vector<Sakura::DockTiles>{DockLeaf("b"), DockLeaf("c")});

    REQUIRE(ToDockTiles(Leaf("a")) == DockLeaf("a"));
    REQUIRE_FALSE(ToDockTiles(TileMeta{}).has_value());
    REQUIRE_FALSE(ToDockTiles(Leaf("")).has_value());
    REQUIRE_FALSE(ToDockTiles(Split("row", {Leaf("a", 600.0f), Leaf("b")})).has_value());
}

TEST_CASE("Tile metadata conversion drops unknown panes and collapses single children",
          "[core][compositor][dock][tiles]") {
    const TilePaneMap map = [](const std::string& pane) -> std::optional<std::string> {
        if (pane == "unknown") {
            return std::nullopt;
        }
        return "pane:" + pane;
    };

    const auto meta = Split("row", {
        Leaf("a", 200.0f),
        Leaf("unknown", 300.0f),
        Split("column", {Leaf("unknown", 100.0f), Leaf("b", 100.0f)}, 500.0f),
    });
    const auto dock = ToDockTiles(meta, map);
    REQUIRE(dock.has_value());
    REQUIRE(dock->sizes == std::vector<F32>{200.0f, 500.0f});
    REQUIRE(dock->children == std::vector<Sakura::DockTiles>{DockLeaf("pane:a"), DockLeaf("pane:b")});

    REQUIRE(ToDockTiles(Split("row", {Leaf("unknown", 1.0f), Leaf("a", 1.0f)}), map) == DockLeaf("pane:a"));
    REQUIRE_FALSE(ToDockTiles(Split("row", {Leaf("unknown", 1.0f), Leaf("unknown", 1.0f)}), map).has_value());
}

TEST_CASE("Dock tiles convert to tile metadata with sizes and orientation", "[core][compositor][dock][tiles]") {
    Sakura::DockTiles dock{
        .sizes = {700.0f},
        .children = {
            DockLeaf("a"),
            Sakura::DockTiles{
                .vertical = true,
                .sizes = {250.0f, 750.0f},
                .children = {DockLeaf("b"), DockLeaf("c")},
            },
        },
    };

    const auto meta = ToTileMeta(dock);
    REQUIRE(meta.has_value());
    REQUIRE(meta->split == "row");
    REQUIRE_FALSE(meta->size.has_value());
    REQUIRE(meta->children->size() == 2);
    REQUIRE(meta->children->at(0).pane == "a");
    REQUIRE(meta->children->at(0).size == 700.0f);
    REQUIRE(meta->children->at(1).size == 0.0f);

    const auto& column = meta->children->at(1);
    REQUIRE(column.split == "column");
    REQUIRE(column.children->at(0).size == 250.0f);
    REQUIRE(column.children->at(1).size == 750.0f);
    REQUIRE(Panes(*meta) == std::vector<std::string>{"a", "b", "c"});

    const auto leaf = ToTileMeta(DockLeaf("a"));
    REQUIRE(leaf.has_value());
    REQUIRE(IsTileLeaf(*leaf));
    REQUIRE(leaf->pane == "a");
    REQUIRE_FALSE(ToTileMeta(DockLeaf("")).has_value());
}

TEST_CASE("Dock tile conversion drops unmapped panes", "[core][compositor][dock][tiles]") {
    const TilePaneMap map = [](const std::string& pane) -> std::optional<std::string> {
        if (pane == "unknown") {
            return std::nullopt;
        }
        if (pane == "blank") {
            return std::string();
        }
        return pane;
    };

    Sakura::DockTiles dock{
        .sizes = {200.0f, 300.0f, 500.0f},
        .children = {DockLeaf("a"), DockLeaf("unknown"), DockLeaf("b")},
    };
    const auto meta = ToTileMeta(dock, map);
    REQUIRE(meta.has_value());
    REQUIRE(Panes(*meta) == std::vector<std::string>{"a", "b"});
    REQUIRE(meta->children->at(0).size == 200.0f);
    REQUIRE(meta->children->at(1).size == 500.0f);

    REQUIRE_FALSE(ToTileMeta(DockLeaf("blank"), map).has_value());
    REQUIRE_FALSE(ToTileMeta(Sakura::DockTiles{
        .children = {DockLeaf("unknown"), DockLeaf("blank")},
    }, map).has_value());
}

TEST_CASE("Tile metadata survives a round trip through dock tiles", "[core][compositor][dock][tiles]") {
    const auto meta = Split("row", {
        Leaf("a", 250.0f),
        Split("column", {Leaf("b", 150.5f), Leaf("c", 50.0f)}, 750.0f),
    });

    const auto dock = ToDockTiles(meta);
    REQUIRE(dock.has_value());
    const auto restored = ToTileMeta(*dock);
    REQUIRE(restored.has_value());
    REQUIRE(Parser::Hash(*restored) == Parser::Hash(meta));

    const auto again = ToDockTiles(*restored);
    REQUIRE(again == dock);
}

TEST_CASE("Flowgraph layouts map between layout panes and dock panes", "[core][compositor][dock][tiles]") {
    const auto toDock = LayoutToDockPanes("graph");
    REQUIRE(toDock(kLayoutEditorPane) == std::optional<std::string>(MakeEditorPaneId("graph")));
    REQUIRE(toDock("config:fft") == std::optional<std::string>("config:fft"));
    REQUIRE(toDock("surface:fft:default") == std::optional<std::string>("surface:fft:default"));
    REQUIRE(toDock(kLayoutDocumentationPane) == std::optional<std::string>("documentation"));
    REQUIRE(toDock(kLayoutDescriptionPane) == std::optional<std::string>("description"));
    REQUIRE(toDock(kLayoutEnvironmentPane) == std::optional<std::string>("environment"));
    REQUIRE_FALSE(toDock("stack_0").has_value());
    REQUIRE_FALSE(toDock("").has_value());

    const auto toLayout = DockToLayoutPanes("graph");
    REQUIRE(toLayout(MakeEditorPaneId("graph")) == std::optional<std::string>("flowgraph"));
    REQUIRE(toLayout("config:fft") == std::optional<std::string>("config:fft"));
    REQUIRE(toLayout("documentation") == std::optional<std::string>(kLayoutDocumentationPane));
    REQUIRE(toLayout("description") == std::optional<std::string>(kLayoutDescriptionPane));
    REQUIRE(toLayout("environment") == std::optional<std::string>(kLayoutEnvironmentPane));
    REQUIRE_FALSE(toLayout(MakeEditorPaneId("other")).has_value());
    REQUIRE_FALSE(toLayout("stack:graph:stack_0").has_value());

    CHECK(IsTransientPane(kLayoutDocumentationPane));
    CHECK_FALSE(IsTransientPane(kLayoutEnvironmentPane));
    CHECK_FALSE(IsTransientPane(kLayoutDescriptionPane));
    CHECK(IsLayoutPane(kLayoutEnvironmentPane));
    CHECK(IsLayoutPane(kLayoutDescriptionPane));
    CHECK_FALSE(IsTransientPane(kLayoutEditorPane));
    CHECK_FALSE(IsTransientPane("config:fft"));
}

TEST_CASE("Flowgraph layouts drop unknown panes on their way to the dock and keep column widths",
          "[core][compositor][dock][tiles]") {
    const auto layout = Split("row", {
        Leaf("flowgraph", 640.0f),
        Split("column", {Leaf("config:fft", 300.0f), Leaf("stack_0", 300.0f)}, 420.0f),
        Leaf("surface:fft:default", 300.0f),
    });

    const auto dock = ToDockTiles(layout, LayoutToDockPanes("graph"));
    REQUIRE(dock.has_value());
    REQUIRE(dock->sizes == std::vector<F32>{640.0f, 420.0f, 300.0f});
    REQUIRE(dock->children == std::vector<Sakura::DockTiles>{
        DockLeaf("editor:graph"),
        DockLeaf("config:fft"),
        DockLeaf("surface:fft:default"),
    });

    const auto restored = ToTileMeta(*dock, DockToLayoutPanes("graph"));
    REQUIRE(restored.has_value());
    REQUIRE(Panes(*restored) == std::vector<std::string>{"flowgraph", "config:fft", "surface:fft:default"});
    REQUIRE(restored->children->at(1).size == 420.0f);
}

TEST_CASE("Tiles describe configs and surfaces by block", "[core][compositor][tiles]") {
    REQUIRE(DescribeTile("config:fft") == "fft Controls");
    REQUIRE(DescribeTile("surface:fft:default") == "fft (default)");
    REQUIRE(DescribeTile("flowgraph") == "flowgraph");
    REQUIRE(DescribeTile("documentation") == "Documentation");
    REQUIRE(DescribeTile("config:") == "config:");
}

TEST_CASE("Flowgraph labels fall back to identifiers", "[core][compositor][tiles]") {
    REQUIRE(MakeFlowgraphTitle("graph", nullptr) == "graph");
    REQUIRE(MakeFlowgraphTitle("graph", std::make_shared<Flowgraph>()) == "graph");

    REQUIRE(MakeSurfaceTileTitle("fft", "Spectrum") == "Spectrum (fft)");
    REQUIRE(MakeSurfaceTileTitle("fft", "") == "fft (fft)");
    REQUIRE(MakeConfigTileTitle("fft", "Spectrum") == "Spectrum Controls (fft)");
    REQUIRE(MakeConfigTileTitle("fft", "") == "fft Controls (fft)");

    REQUIRE(MakeEditorPaneId("graph") == "editor:graph");
    REQUIRE(MakeFlowgraphLayoutGroup("graph") == "layout:graph");
}
