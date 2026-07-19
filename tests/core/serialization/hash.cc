#include <catch2/catch_test_macros.hpp>

#include "fixture.hh"

using namespace Jetstream;
using namespace TestSerialization;

TEST_CASE("JST_SERDES hash includes nested and optional fields", "[core][serialization][hash]") {
    SECTION("nested fields affect hash values") {
        OuterConfig lhs;
        OuterConfig rhs;

        REQUIRE(lhs.hash() == rhs.hash());

        rhs.inner.gain = 9;
        REQUIRE(lhs.hash() != rhs.hash());
    }

    SECTION("optional presence and value affect hash values") {
        OptionalConfig lhs;
        OptionalConfig rhs;

        REQUIRE(lhs.hash() == rhs.hash());

        rhs.label = "set";
        REQUIRE(lhs.hash() != rhs.hash());

        lhs.label = "set";
        REQUIRE(lhs.hash() == rhs.hash());

        rhs.steps = std::vector<U64>{7};
        REQUIRE(lhs.hash() != rhs.hash());
    }
}

TEST_CASE("Parser::Hash is order-aware for sequences and order-independent for maps", "[core][serialization][hash]") {
    SECTION("Parser::Map ignores insertion order") {
        Parser::Map lhs;
        lhs["alpha"] = U64{1};
        lhs["beta"] = std::string("two");

        Parser::Map rhs;
        rhs["beta"] = std::string("two");
        rhs["alpha"] = U64{1};

        REQUIRE(Parser::Hash(lhs) == Parser::Hash(rhs));

        rhs["alpha"] = U64{2};
        REQUIRE(Parser::Hash(lhs) != Parser::Hash(rhs));
    }

    SECTION("Parser::Sequence preserves element order") {
        const Parser::Sequence lhs = {U64{1}, std::string("two")};
        const Parser::Sequence rhs = {std::string("two"), U64{1}};
        REQUIRE(Parser::Hash(lhs) != Parser::Hash(rhs));
    }

    SECTION("vectors preserve element order") {
        REQUIRE(Parser::Hash(std::vector<U64>{1, 2, 3}) != Parser::Hash(std::vector<U64>{3, 2, 1}));
    }

    SECTION("optional values include presence and contents") {
        std::optional<U64> lhs;
        std::optional<U64> rhs;
        REQUIRE(Parser::Hash(lhs) == Parser::Hash(rhs));

        rhs = 4;
        REQUIRE(Parser::Hash(lhs) != Parser::Hash(rhs));

        lhs = 4;
        REQUIRE(Parser::Hash(lhs) == Parser::Hash(rhs));
    }
}

TEST_CASE("Parser::Hash handles std::any values", "[core][serialization][hash]") {
    SECTION("empty std::any hashes to zero") {
        REQUIRE(Parser::Hash(std::any{}) == 0);
    }

    SECTION("std::any containing a map hashes like the map") {
        const Parser::Map value = MakeInnerMap(3, true);
        REQUIRE(Parser::Hash(std::any(value)) == Parser::Hash(value));
    }

    SECTION("std::any containing a sequence hashes like the sequence") {
        const Parser::Sequence value = {U64{1}, std::string("two")};
        REQUIRE(Parser::Hash(std::any(value)) == Parser::Hash(value));
    }

    SECTION("std::any containing a scalar is stable") {
        REQUIRE(Parser::Hash(std::any(U64{42})) == Parser::Hash(std::any(U64{42})));
        REQUIRE(Parser::Hash(std::any(U64{42})) != Parser::Hash(std::any(U64{43})));
    }

    SECTION("unsupported std::any values hash to zero") {
        REQUIRE(Parser::Hash(std::any(UnsupportedValue{})) == 0);
    }

    SECTION("scalar types remain distinct") {
        REQUIRE(Parser::Hash(std::any(U64{42})) != Parser::Hash(std::any(std::string("42"))));
    }
}

TEST_CASE("Parser::Hash covers complex and nested sequence values", "[core][serialization][hash]") {
    REQUIRE(Parser::Hash(CF32{1.0f, 2.0f}) == Parser::Hash(CF32{1.0f, 2.0f}));
    REQUIRE(Parser::Hash(CF32{1.0f, 2.0f}) != Parser::Hash(CF32{1.0f, -2.0f}));

    Parser::Sequence lhs;
    lhs.push_back(MakeInnerMap(3, true));
    lhs.push_back(std::vector<U64>{5, 8});

    Parser::Sequence rhs;
    rhs.push_back(MakeInnerMap(3, true));
    rhs.push_back(std::vector<U64>{5, 8});
    REQUIRE(Parser::Hash(lhs) == Parser::Hash(rhs));

    auto& nested = std::any_cast<Parser::Map&>(rhs.at(0));
    nested["gain"] = U64{4};
    REQUIRE(Parser::Hash(lhs) != Parser::Hash(rhs));
}

TEST_CASE("std::hash specializations delegate to Parser::Hash", "[core][serialization][hash]") {
    const Parser::Map data = MakeInnerMap(5, false);
    const Parser::Sequence sequence = {std::string("alpha"), U64{2}};

    REQUIRE(std::hash<Parser::Map>{}(data) == Parser::Hash(data));
    REQUIRE(std::hash<Parser::Sequence>{}(sequence) == Parser::Hash(sequence));
}
