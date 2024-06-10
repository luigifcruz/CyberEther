#include "jetstream/memory/token.hh" 

namespace Jetstream {

Token::Token() : type(Type::Colon) {}
Token::Token(U64 _a) : a(_a), type(Type::Number) {}
Token::Token(U64 _a, U64 _b) : a(_a), b(_b), type(Type::Colon) {}
Token::Token(U64 _a, U64 _b, U64 _c) : a(_a), b(_b), c(_c), type(Type::Colon) {}
Token::Token(I32 _a) : a(_a), type(Type::Number) {}
Token::Token(I32 _a, I32 _b) : a(_a), b(_b), type(Type::Colon) {}
Token::Token(I32 _a, I32 _b, I32 _c) : a(_a), b(_b), c(_c), type(Type::Colon) {}
Token::Token(const char*) : type(Type::Ellipsis) {}

std::ostream& operator<<(std::ostream& os, const Token& token) {
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