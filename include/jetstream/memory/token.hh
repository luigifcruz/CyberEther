#ifndef JETSTREAM_MEMORY_TOKEN_HH
#define JETSTREAM_MEMORY_TOKEN_HH

#include <vector>
#include <iostream>

#include "jetstream/types.hh"

namespace Jetstream {

struct Token {
 public:
    enum class Type { Number, Colon, Ellipsis };

    Token() : type(Type::Colon) {}
    Token(U64 _a) : a(_a), type(Type::Number) {}
    Token(U64 _a, U64 _b) : a(_a), b(_b), type(Type::Colon) {}
    Token(U64 _a, U64 _b, U64 _c) : a(_a), b(_b), c(_c), type(Type::Colon) {}
    Token(I32 _a) : a(_a), type(Type::Number) {}
    Token(I32 _a, I32 _b) : a(_a), b(_b), type(Type::Colon) {}
    Token(I32 _a, I32 _b, I32 _c) : a(_a), b(_b), c(_c), type(Type::Colon) {}
    Token(const char*) : type(Type::Ellipsis) {}

    constexpr const Type& get_type() const {
        return type;
    }

    constexpr const U64& get_a() const {
        return a;
    }

    constexpr const U64& get_b() const {
        return b;
    }

    constexpr const U64& get_c() const {
        return c;
    }

    friend std::ostream& operator<<(std::ostream& os, const Token& token) {
        switch (token.get_type()) {
            case Token::Type::Number:
                os << token.get_a();
                break;
            case Token::Type::Colon:
                if (token.get_c() == 1) {
                    os << token.get_a() << ":" << token.get_b();
                } else {
                    os << token.get_a() << ":" << token.get_b() << ":" << token.get_c();
                }
                break;
            case Token::Type::Ellipsis:
                os << "...";
                break;
        }
        return os;
    }

    friend std::ostream& operator<<(std::ostream& os, const std::vector<Token>& tokens) {
        os << "{";
        for (const auto& token : tokens) {
            os << token;
            if (&token != &tokens.back()) {
                os << ", ";
            }
        }
        os << "}";
        return os;
    }

 private:
    U64 a = 0;
    U64 b = 0;
    U64 c = 1;
    Type type;
};

}  // namespace Jetstream

template <> struct fmt::formatter<Jetstream::Token> : ostream_formatter {};
template <> struct fmt::formatter<std::vector<Jetstream::Token>> : ostream_formatter {};

#endif
