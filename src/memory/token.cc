#include "jetstream/memory/token.hh"

namespace Jetstream {

Token::Token() : type(Type::Colon) {}
Token::Token(U64 _a) : a(_a), type(Type::Number) {}
Token::Token(U64 _a, U64 _b) : a(_a), b(_b), type(_b == 0 ? Type::ColonZeroEnd : Type::Colon) {}
Token::Token(U64 _a, U64 _b, U64 _c) : a(_a), b(_b), c(_c), type(_b == 0 ? Type::ColonZeroEnd : Type::Colon) {}
Token::Token(U64 _a, U64 _b, U64 _c, bool _hasEnd)
    : a(_a), b(_b), c(_c), type(_hasEnd && _b == 0 ? Type::ColonZeroEnd : Type::Colon) {}
Token::Token(I32 _a) : a(_a), type(Type::Number) {}
Token::Token(I32 _a, I32 _b) : a(_a), b(_b), type(_b == 0 ? Type::ColonZeroEnd : Type::Colon) {}
Token::Token(I32 _a, I32 _b, I32 _c) : a(_a), b(_b), c(_c), type(_b == 0 ? Type::ColonZeroEnd : Type::Colon) {}
Token::Token(const char*) : type(Type::Ellipsis) {}

std::ostream& operator<<(std::ostream& os, const Token& token) {
    switch (token.getType()) {
        case Token::Type::Number:
            os << token.getA();
            break;
        case Token::Type::Colon:
        case Token::Type::ColonZeroEnd:
            if (!token.hasEnd() && token.getC() == 1) {
                os << token.getA() << ":";
            } else if (!token.hasEnd()) {
                os << token.getA() << "::" << token.getC();
            } else if (token.getC() == 1) {
                os << token.getA() << ":" << token.getB();
            } else {
                os << token.getA() << ":" << token.getB() << ":" << token.getC();
            }
            break;
        case Token::Type::Ellipsis:
            os << "...";
            break;
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const std::vector<Token>& tokens) {
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

}  // namespace Jetstream
