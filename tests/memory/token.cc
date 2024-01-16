#include <sstream>

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include "jetstream/memory/token.hh"

using namespace Jetstream;

TEST_CASE("Token Class Tests", "[Token]") {
    SECTION("Default Constructor") {
        Token token;
        REQUIRE(token.get_type() == Token::Type::Colon);
    }

    SECTION("Constructor with Single U64 Parameter") {
        Token token(42);
        REQUIRE(token.get_type() == Token::Type::Number);
        REQUIRE(token.get_a() == 42);
    }

    SECTION("Constructor with Two U64 Parameters") {
        Token token(42, 43);
        REQUIRE(token.get_type() == Token::Type::Colon);
        REQUIRE(token.get_a() == 42);
        REQUIRE(token.get_b() == 43);
        REQUIRE(token.get_c() == 1);
    }

    SECTION("Constructor with Three U64 Parameters") {
        Token token(42, 43, 44);
        REQUIRE(token.get_type() == Token::Type::Colon);
        REQUIRE(token.get_a() == 42);
        REQUIRE(token.get_b() == 43);
        REQUIRE(token.get_c() == 44);
    }

    SECTION("Constructor with Single I32 Parameter") {
        Token token(static_cast<I32>(42));
        REQUIRE(token.get_type() == Token::Type::Number);
        REQUIRE(token.get_a() == 42);
    }

    SECTION("Constructor with Two I32 Parameters") {
        Token token(static_cast<I32>(42), static_cast<I32>(43));
        REQUIRE(token.get_type() == Token::Type::Colon);
        REQUIRE(token.get_a() == 42);
        REQUIRE(token.get_b() == 43);
    }

    SECTION("Constructor with Three I32 Parameters") {
        Token token(static_cast<I32>(42), static_cast<I32>(43), static_cast<I32>(44));
        REQUIRE(token.get_type() == Token::Type::Colon);
        REQUIRE(token.get_a() == 42);
        REQUIRE(token.get_b() == 43);
        REQUIRE(token.get_c() == 44);
    }

    SECTION("Constructor with const char* Parameter") {
        Token token("...");
        REQUIRE(token.get_type() == Token::Type::Ellipsis);
    }

    SECTION("Output Stream Overload") {
        std::ostringstream os;

        SECTION("Number Type Token") {
            Token token(42);
            os << token;
            REQUIRE(os.str() == "42");
        }

        SECTION("Colon Type Token with Two Parameters") {
            Token token(42, 43);
            os.str("");
            os << token;
            REQUIRE(os.str() == "42:43");
        }

        SECTION("Colon Type Token with Three Parameters") {
            Token token(42, 43, 44);
            os.str("");
            os << token;
            REQUIRE(os.str() == "42:43:44");
        }

        SECTION("Ellipsis Type Token") {
            Token token("...");
            os.str("");
            os << token;
            REQUIRE(os.str() == "...");
        }
    }

    SECTION("Output Stream Overload for Token Vector") {
        std::ostringstream os;
        std::vector<Token> tokens{Token(42), Token("..."), Token(42, 43), Token(42, 43, 44)};
        os << tokens;
        REQUIRE(os.str() == "{42, ..., 42:43, 42:43:44}");
    }

    SECTION("std::vector<Token> with Mixed Initializer List") {
        std::vector<Token> tokens = {42, "...", {0, 8, 2}, {24, 48}};

        REQUIRE(tokens[0].get_type() == Token::Type::Number);
        REQUIRE(tokens[0].get_a() == 42);

        REQUIRE(tokens[1].get_type() == Token::Type::Ellipsis);

        REQUIRE(tokens[2].get_type() == Token::Type::Colon);
        REQUIRE(tokens[2].get_a() == 0);
        REQUIRE(tokens[2].get_b() == 8);
        REQUIRE(tokens[2].get_c() == 2);

        REQUIRE(tokens[3].get_type() == Token::Type::Colon);
        REQUIRE(tokens[3].get_a() == 24);
        REQUIRE(tokens[3].get_b() == 48);

        std::ostringstream os;
        os << tokens;
        REQUIRE(os.str() == "{42, ..., 0:8:2, 24:48}");
    }
}

int main(int argc, char* argv[]) {
  return Catch::Session().run(argc, argv);
}