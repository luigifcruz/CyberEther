#include <catch2/catch_test_macros.hpp>

#include <sstream>
#include <vector>

#include "jetstream/memory/token.hh"

namespace {

using namespace Jetstream;

TEST_CASE("Token constructors expose slice components", "[core][memory][token]") {
    SECTION("full slice") {
        const Token token;

        REQUIRE(token.getType() == Token::Type::Colon);
        REQUIRE(token.getA() == 0);
        REQUIRE(token.getB() == 0);
        REQUIRE(token.getC() == 1);
    }

    SECTION("unsigned index") {
        const Token token(U64{42});

        REQUIRE(token.getType() == Token::Type::Number);
        REQUIRE(token.getA() == 42);
        REQUIRE(token.getB() == 0);
        REQUIRE(token.getC() == 1);
    }

    SECTION("unsigned ranges") {
        const Token range(U64{2}, U64{9});
        const Token stepped(U64{2}, U64{9}, U64{3});

        REQUIRE(range.getType() == Token::Type::Colon);
        REQUIRE(range.getA() == 2);
        REQUIRE(range.getB() == 9);
        REQUIRE(range.getC() == 1);
        REQUIRE(stepped.getType() == Token::Type::Colon);
        REQUIRE(stepped.getA() == 2);
        REQUIRE(stepped.getB() == 9);
        REQUIRE(stepped.getC() == 3);
    }

    SECTION("signed overloads") {
        const Token index(I32{5});
        const Token range(I32{1}, I32{7});
        const Token stepped(I32{1}, I32{7}, I32{2});

        REQUIRE(index.getType() == Token::Type::Number);
        REQUIRE(index.getA() == 5);
        REQUIRE(range.getType() == Token::Type::Colon);
        REQUIRE(range.getA() == 1);
        REQUIRE(range.getB() == 7);
        REQUIRE(range.getC() == 1);
        REQUIRE(stepped.getType() == Token::Type::Colon);
        REQUIRE(stepped.getA() == 1);
        REQUIRE(stepped.getB() == 7);
        REQUIRE(stepped.getC() == 2);
    }

    SECTION("ellipsis") {
        const Token token("...");

        REQUIRE(token.getType() == Token::Type::Ellipsis);
        REQUIRE(token.getA() == 0);
        REQUIRE(token.getB() == 0);
        REQUIRE(token.getC() == 1);
    }
}

TEST_CASE("Token streams use slice notation", "[core][memory][token][format]") {
    SECTION("individual tokens") {
        std::ostringstream stream;

        stream << Token(U64{4}) << ' ' << Token(U64{1}, U64{6}) << ' '
               << Token(U64{1}, U64{6}, U64{2}) << ' ' << Token("...");

        REQUIRE(stream.str() == "4 1:6 1:6:2 ...");
    }

    SECTION("default and unit-step ranges omit the step") {
        std::ostringstream stream;

        stream << Token() << ' ' << Token(U64{1}, U64{6}, U64{1});

        REQUIRE(stream.str() == "0:0 1:6");
    }

    SECTION("token lists") {
        std::ostringstream populated;
        std::ostringstream empty;
        const std::vector<Token> tokens = {
            Token(U64{3}), Token("..."), Token(U64{0}, U64{8}, U64{2}), Token()
        };

        populated << tokens;
        empty << std::vector<Token>{};

        REQUIRE(populated.str() == "{3, ..., 0:8:2, 0:0}");
        REQUIRE(empty.str() == "{}");
    }
}

}  // namespace
