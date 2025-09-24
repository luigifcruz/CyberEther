#ifndef JETSTREAM_MEMORY_TOKEN_HH
#define JETSTREAM_MEMORY_TOKEN_HH

#include <vector>
#include <iostream>

#include "jetstream/types.hh"

// TODO: Remove testing namespace.
namespace Jetstream::mem2 {

struct Token {
 public:
    enum class Type { Number, Colon, Ellipsis };

    Token();
    Token(U64 _a);
    Token(U64 _a, U64 _b);
    Token(U64 _a, U64 _b, U64 _c);
    Token(I32 _a);
    Token(I32 _a, I32 _b);
    Token(I32 _a, I32 _b, I32 _c);
    Token(const char*);

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

    friend std::ostream& operator<<(std::ostream& os, const Token& token);
    friend std::ostream& operator<<(std::ostream& os, const std::vector<Token>& tokens);

 private:
    U64 a = 0;
    U64 b = 0;
    U64 c = 1;
    Type type;
};

}  // namespace Jetstream

template <> struct jst::fmt::formatter<Jetstream::mem2::Token> : ostream_formatter {};
template <> struct jst::fmt::formatter<std::vector<Jetstream::mem2::Token>> : ostream_formatter {};

#endif  // JETSTREAM_MEMORY_TOKEN_HH
