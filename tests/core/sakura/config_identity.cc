#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "component_harness.hh"
#include "compositor/default/views/flowgraph/editor/config/base.hh"

using namespace Jetstream;
using namespace SakuraTest;

namespace {

FlowgraphConfigFieldConfig IdentityField(const std::string& kind, const std::string& id,
                                       const std::string& name, std::vector<Parser::Map>& patches) {
    Parser::Map values;
    if (kind == "text") values[name] = std::string("1");
    else if (kind == "int") values[name] = I32(1);
    else values[name] = F32(1);
    return {.id = id, .name = name, .label = name, .format = {{"type", kind}}, .values = std::move(values),
            .onApply = [&patches](Parser::Map patch, bool) { patches.push_back(std::move(patch)); }};
}

}  // namespace

TEST_CASE("Replacing a config slot with another same-type field discards its focused draft before rebinding",
          "[core][sakura][config-identity]") {
    const std::string kind = GENERATE(std::string("float"), std::string("int"), std::string("text"));
    std::vector<Parser::Map> oldPatches;
    std::vector<Parser::Map> newPatches;
    FlowgraphConfigFieldInstance field;
    field.update(IdentityField(kind, "amplitude", "amplitude", oldPatches));
    Sakura::Stack stack;
    Host host(stack);
    const auto place = [&] {
        stack.update({.items = {{.child = &field.component()}}});
        host.place(MakeContext(), {0, 0, 300, 80});
    };
    place();
    auto grids = Find<Sakura::TextGrid>(stack);
    REQUIRE(grids.size() == 1);
    const auto oldGrid = Sakura::Access::alive(*grids.front());
    grids.front()->focus(true);
    Type(host, "42");
    REQUIRE(oldPatches.empty());
    field.update(IdentityField(kind, "chirp-end-frequency", "chirpEndFreq", newPatches));
    place();
    CHECK(oldGrid.expired());
    CHECK(FocusOwner(host) == nullptr);
    Tap(host, KeyCode::Enter);
    CHECK(newPatches.empty());
    grids = Find<Sakura::TextGrid>(stack);
    REQUIRE(grids.size() == 1);
    grids.front()->focus(true);
    Type(host, "8");
    REQUIRE(Tap(host, KeyCode::Enter));
    REQUIRE(newPatches.size() == 1);
    CHECK(newPatches.front().contains("chirpEndFreq"));
    CHECK_FALSE(newPatches.front().contains("amplitude"));
    if (kind == "text") CHECK(Parser::Get<std::string>(newPatches.front(), "chirpEndFreq") == "8");
    else if (kind == "int") CHECK(Parser::Get<I32>(newPatches.front(), "chirpEndFreq") == 8);
    else CHECK(Parser::Get<F32>(newPatches.front(), "chirpEndFreq") == 8.0f);
}

TEST_CASE("Same-ID config polling preserves a focused draft and submits it through the current callback",
          "[core][sakura][config-identity]") {
    std::vector<Parser::Map> oldPatches;
    std::vector<Parser::Map> currentPatches;
    FlowgraphConfigFieldInstance field;
    field.update(IdentityField("float", "amplitude", "amplitude", oldPatches));
    Sakura::Stack stack;
    Host host(stack);
    const auto place = [&] {
        stack.update({.items = {{.child = &field.component()}}});
        host.place(MakeContext(), {0, 0, 300, 80});
    };
    place();
    const auto grids = Find<Sakura::TextGrid>(stack);
    REQUIRE(grids.size() == 1);
    const auto original = Sakura::Access::alive(*grids.front());
    grids.front()->focus(true);
    Type(host, "42");
    auto polled = IdentityField("float", "amplitude", "amplitude", currentPatches);
    polled.values["amplitude"] = F32(2);
    field.update(polled);
    place();
    CHECK_FALSE(original.expired());
    CHECK(oldPatches.empty());
    CHECK(currentPatches.empty());
    REQUIRE(Tap(host, KeyCode::Enter));
    REQUIRE(currentPatches.size() == 1);
    CHECK(Parser::Get<F32>(currentPatches.front(), "amplitude") == 42.0f);
    CHECK(oldPatches.empty());
}
