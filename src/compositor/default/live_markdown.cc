#include "live_markdown.hh"
#include "presenters/flowgraph/key_value.hh"

#include "jetstream/flowgraph_environment.hh"
#include "jetstream/parser.hh"

#include <algorithm>
#include <any>
#include <array>
#include <charconv>
#include <cmath>
#include <exception>
#include <optional>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Jetstream {

namespace {

constexpr U64 MaxRepeat = 500;
constexpr U64 MaxInline = 32;
constexpr U64 MaxDepth = 4;
constexpr U64 MaxColumns = 32;
constexpr U64 MaxFormatWidth = 64;
constexpr U64 MaxCellBytes = 1024;
constexpr U64 MaxOutputBytes = 256 * 1024;
constexpr std::string_view Missing = "--";
constexpr std::string_view StatsTag = "stats";
constexpr std::string_view EscapedCharacters = "\\`*_[]#>|";

struct Segment {
    enum class Kind : U8 {
        Key,
        Index,
        Slice,
    };

    Kind kind = Kind::Key;
    std::string key;
    I64 index = 0;
    std::optional<I64> begin;
    std::optional<I64> end;
};

bool Accepts(const std::string& pattern, const auto& sample) {
    try {
        (void)jst::fmt::format(jst::fmt::runtime(pattern), sample);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

struct Format {
    enum Category : U8 {
        Floating,
        Integer,
        Text,
        Boolean,
        Categories,
    };

    std::string pattern;
    std::array<bool, Categories> accepted = {true, true, true, true};

    bool accepts(const Category category) const {
        return accepted[category];
    }

    bool valid() const {
        return std::any_of(accepted.begin(), accepted.end(), [](const bool value) { return value; });
    }
};

struct Token {
    std::vector<Segment> path;
    Format format;
    std::optional<U64> slice;
    bool lenient = false;
};

struct Piece {
    std::string_view text;
    bool placeholder = false;
    std::optional<Token> token;
};

struct Line {
    enum class Kind : U8 {
        Verbatim,
        Header,
        Body,
        Stats,
    };

    Kind kind = Kind::Verbatim;
    std::string_view text;
    std::vector<Piece> pieces;
    bool table = false;
    bool standalone = false;
    bool spaceAfter = false;
};

struct Value {
    const std::any* pointer = nullptr;
    std::any owned;

    const std::any* get() const {
        return pointer != nullptr ? pointer : &owned;
    }
};

enum class Lookup : U8 {
    Found,
    Missing,
    Invalid,
};

struct Context {
    const Flowgraph::Environment& environment;
    std::unordered_map<std::string, std::optional<std::any>> roots;
    U64 written = 0;
    bool truncated = false;

    const std::any* root(const std::string& key) {
        auto it = roots.find(key);
        if (it == roots.end()) {
            std::optional<std::any> value;
            Parser::Map map;
            if (environment.has(key) && environment.get(key, map) == Result::SUCCESS) {
                value = std::any(std::move(map));
            }
            it = roots.emplace(key, std::move(value)).first;
        }
        return it->second.has_value() ? &it->second.value() : nullptr;
    }

    bool full() const {
        return written >= MaxOutputBytes;
    }

    void emit(std::vector<std::string>& output, std::string line) {
        if (full()) {
            truncated = true;
            return;
        }
        written += line.size() + 1;
        output.push_back(std::move(line));
    }
};

template<typename... T>
struct Vectors {
    static std::optional<U64> size(const std::any& value) {
        std::optional<U64> result;
        const auto probe = [&]<typename V>() {
            if (const auto* vector = std::any_cast<std::vector<V>>(&value)) {
                result = vector->size();
                return true;
            }
            return false;
        };
        (probe.template operator()<T>() || ...);
        return result;
    }

    static bool element(const std::any& value, const U64 index, std::any& out) {
        const auto probe = [&]<typename V>() {
            if (const auto* vector = std::any_cast<std::vector<V>>(&value)) {
                out = (*vector)[index];
                return true;
            }
            return false;
        };
        return (probe.template operator()<T>() || ...);
    }
};

using TypedVectors = Vectors<F32, F64, I8, I16, I32, I64, U8, U16, U32, U64, CF32, CF64, std::string>;

std::string_view TrimView(std::string_view text) {
    const U64 begin = text.find_first_not_of(" \t\r");
    if (begin == std::string_view::npos) {
        return {};
    }
    const U64 end = text.find_last_not_of(" \t\r");
    return text.substr(begin, end - begin + 1);
}

std::vector<std::string_view> SplitLines(std::string_view source) {
    std::vector<std::string_view> lines;
    for (U64 start = 0;;) {
        const U64 end = source.find('\n', start);
        lines.push_back(source.substr(start, end == std::string_view::npos ? std::string_view::npos : end - start));
        if (end == std::string_view::npos) {
            return lines;
        }
        start = end + 1;
    }
}

bool IsFence(std::string_view line) {
    return line.substr(0, 3) == "```";
}

bool IsStatsFence(std::string_view line) {
    const std::string_view info = line.substr(std::min<U64>(line.find_first_not_of("` \t"), line.size()));
    return TrimView(info.substr(0, info.find_first_of(" \t{"))) == StatsTag;
}

bool IsDelimiterRow(std::string_view line) {
    line = TrimView(line);
    if (line.find('-') == std::string_view::npos) {
        return false;
    }
    if (!line.empty() && line.front() == '|') {
        line.remove_prefix(1);
    }
    if (!line.empty() && line.back() == '|') {
        line.remove_suffix(1);
    }
    for (U64 start = 0; start <= line.size();) {
        const U64 bar = std::min(line.find('|', start), line.size());
        std::string_view cell = TrimView(line.substr(start, bar - start));
        if (!cell.empty() && cell.front() == ':') {
            cell.remove_prefix(1);
        }
        if (!cell.empty() && cell.back() == ':') {
            cell.remove_suffix(1);
        }
        if (cell.empty() || cell.find_first_not_of('-') != std::string_view::npos) {
            return false;
        }
        start = bar + 1;
    }
    return true;
}

std::string Escape(std::string_view text) {
    const U64 lead = std::min(text.find_first_not_of(" \t\r\n"), text.size());
    U64 digits = lead;
    while (digits < text.size() && text[digits] >= '0' && text[digits] <= '9') {
        ++digits;
    }
    const U64 marker = digits > lead && digits < text.size() && text[digits] == '.' ? digits : lead;

    std::string escaped;
    escaped.reserve(text.size());
    for (U64 i = 0; i < text.size(); ++i) {
        const char c = text[i];
        if (c == '\n' || c == '\r') {
            escaped += ' ';
            continue;
        }
        if (EscapedCharacters.find(c) != std::string_view::npos ||
            (i == marker && (c == '-' || c == '+' || c == '.'))) {
            escaped += '\\';
        }
        escaped += c;
    }
    return escaped;
}

bool Bounded(std::string_view spec) {
    for (U64 i = 0; i < spec.size();) {
        if (spec[i] < '0' || spec[i] > '9') {
            ++i;
            continue;
        }
        U64 value = 0;
        while (i < spec.size() && spec[i] >= '0' && spec[i] <= '9') {
            value = std::min<U64>(value * 10 + static_cast<U64>(spec[i] - '0'), MaxFormatWidth + 1);
            ++i;
        }
        if (value > MaxFormatWidth) {
            return false;
        }
    }
    return true;
}

Format MakeFormat(std::string_view spec) {
    Format format;
    if (spec.empty()) {
        return format;
    }
    format.pattern = "{:" + std::string(spec) + "}";
    const bool bounded = Bounded(spec);
    format.accepted = {
        bounded && Accepts(format.pattern, F64{0.0}),
        bounded && Accepts(format.pattern, I64{0}),
        bounded && Accepts(format.pattern, std::string()),
        bounded && Accepts(format.pattern, false),
    };
    return format;
}

bool ParseInteger(std::string_view text, I64& value) {
    if (text.empty()) {
        return false;
    }
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    return error == std::errc() && end == text.data() + text.size();
}

bool ParseBound(std::string_view text, std::optional<I64>& bound) {
    if (text.empty()) {
        bound.reset();
        return true;
    }
    I64 value = 0;
    if (!ParseInteger(text, value)) {
        return false;
    }
    bound = value;
    return true;
}

bool ParseBracket(std::string_view inner, Segment& segment) {
    if (inner == "*") {
        segment.kind = Segment::Kind::Slice;
        return true;
    }
    const U64 colon = inner.find(':');
    if (colon == std::string_view::npos) {
        segment.kind = Segment::Kind::Index;
        return ParseInteger(inner, segment.index);
    }
    if (inner.find(':', colon + 1) != std::string_view::npos) {
        return false;
    }
    segment.kind = Segment::Kind::Slice;
    return ParseBound(inner.substr(0, colon), segment.begin) &&
           ParseBound(inner.substr(colon + 1), segment.end);
}

bool ParseToken(std::string_view body, Token& token) {
    constexpr std::string_view root = "env";
    if (body.substr(0, root.size()) != root) {
        return false;
    }

    U64 i = root.size();
    while (i < body.size()) {
        const char c = body[i];
        if (c == ':') {
            token.format = MakeFormat(body.substr(i + 1));
            if (!token.format.valid()) {
                return false;
            }
            break;
        }
        if (c == '.') {
            const U64 start = ++i;
            while (i < body.size() && body[i] != '.' && body[i] != '[' && body[i] != ':') {
                ++i;
            }
            if (i == start) {
                return false;
            }
            token.path.push_back({
                .kind = Segment::Kind::Key,
                .key = std::string(body.substr(start, i - start)),
            });
            continue;
        }
        if (c == '[') {
            const U64 close = body.find(']', i + 1);
            if (close == std::string_view::npos || token.path.empty()) {
                return false;
            }
            Segment segment;
            if (!ParseBracket(body.substr(i + 1, close - i - 1), segment)) {
                return false;
            }
            if (segment.kind == Segment::Kind::Slice) {
                if (token.slice.has_value()) {
                    return false;
                }
                token.slice = token.path.size();
            }
            token.path.push_back(std::move(segment));
            i = close + 1;
            continue;
        }
        return false;
    }

    return !token.path.empty() && token.path.front().kind == Segment::Kind::Key;
}

std::vector<Piece> ScanPieces(std::string_view line) {
    std::vector<Piece> pieces;
    U64 text = 0;
    U64 i = 0;
    const auto flush = [&](const U64 until) {
        if (until > text) {
            pieces.push_back({.text = line.substr(text, until - text)});
        }
    };

    while (i < line.size()) {
        const char c = line[i];
        if (c == '\\') {
            i = std::min<U64>(i + 2, line.size());
            continue;
        }
        if (c == '`') {
            const U64 end = std::min(line.find_first_not_of('`', i), line.size());
            const U64 length = end - i;
            i = end;
            for (U64 close = line.find('`', end); close != std::string_view::npos;) {
                const U64 closeEnd = std::min(line.find_first_not_of('`', close), line.size());
                if (closeEnd - close == length) {
                    i = closeEnd;
                    break;
                }
                close = line.find('`', closeEnd);
            }
            continue;
        }
        if (c == '$' && i + 1 < line.size() && line[i + 1] == '{') {
            const U64 close = line.find('}', i + 2);
            if (close == std::string_view::npos) {
                break;
            }
            flush(i);
            Piece piece{
                .text = line.substr(i, close + 1 - i),
                .placeholder = true,
            };
            Token token;
            if (ParseToken(line.substr(i + 2, close - i - 2), token)) {
                piece.token = std::move(token);
            }
            pieces.push_back(std::move(piece));
            i = close + 1;
            text = i;
            continue;
        }
        ++i;
    }

    flush(line.size());
    return pieces;
}

bool HasTokens(const std::vector<Piece>& pieces) {
    return std::any_of(pieces.begin(), pieces.end(), [](const Piece& piece) { return piece.token.has_value(); });
}

bool Standalone(const std::vector<Piece>& pieces) {
    const Piece* found = nullptr;
    for (const auto& piece : pieces) {
        if (!piece.placeholder) {
            if (!TrimView(piece.text).empty()) {
                return false;
            }
            continue;
        }
        if (found != nullptr || !piece.token.has_value() || piece.token->slice.has_value()) {
            return false;
        }
        found = &piece;
    }
    return found != nullptr;
}

const Token& StandaloneToken(const std::vector<Piece>& pieces) {
    for (const auto& piece : pieces) {
        if (piece.token.has_value()) {
            return piece.token.value();
        }
    }
    return pieces.front().token.value();
}

bool IsContainer(const std::any& value) {
    return value.type() == typeid(Parser::Map) ||
           value.type() == typeid(Parser::Sequence) ||
           TypedVectors::size(value).has_value();
}

std::optional<U64> ContainerSize(const std::any& value) {
    if (value.type() == typeid(Parser::Map)) {
        return std::any_cast<const Parser::Map&>(value).size();
    }
    if (value.type() == typeid(Parser::Sequence)) {
        return std::any_cast<const Parser::Sequence&>(value).size();
    }
    return TypedVectors::size(value);
}

Value ElementAt(const std::any& container, const U64 index, std::string& key) {
    Value element;
    if (container.type() == typeid(Parser::Map)) {
        const auto& entry = *(std::any_cast<const Parser::Map&>(container).begin() + static_cast<std::ptrdiff_t>(index));
        key = entry.key;
        element.pointer = &entry.value;
        return element;
    }
    key = jst::fmt::format("{}", index);
    if (container.type() == typeid(Parser::Sequence)) {
        element.pointer = &std::any_cast<const Parser::Sequence&>(container)[index];
        return element;
    }
    TypedVectors::element(container, index, element.owned);
    return element;
}

std::optional<U64> NormalizeIndex(I64 index, const U64 size) {
    if (index < 0) {
        index += static_cast<I64>(size);
    }
    if (index < 0 || static_cast<U64>(index) >= size) {
        return std::nullopt;
    }
    return static_cast<U64>(index);
}

std::pair<U64, U64> SliceRange(const Segment& segment, const U64 size) {
    const auto clamp = [size](const std::optional<I64>& bound, const U64 fallback) -> U64 {
        if (!bound.has_value()) {
            return fallback;
        }
        I64 value = bound.value();
        if (value < 0) {
            value += static_cast<I64>(size);
        }
        return static_cast<U64>(std::clamp<I64>(value, 0, static_cast<I64>(size)));
    };
    const U64 begin = clamp(segment.begin, 0);
    const U64 end = clamp(segment.end, size);
    return {begin, std::max(begin, end)};
}

Lookup Walk(Context& context, const Token& token, const std::optional<U64> row, const U64 depth, Value& out) {
    const std::any* root = context.root(token.path.front().key);
    if (root == nullptr) {
        return Lookup::Missing;
    }

    Value current{.pointer = root};
    std::string key = token.path.front().key;
    for (U64 s = 1; s < depth; ++s) {
        const auto& segment = token.path[s];
        if (segment.kind == Segment::Kind::Key && segment.key == "_key") {
            current = Value{.owned = key};
            continue;
        }

        const std::any& value = *current.get();
        if (segment.kind == Segment::Kind::Key && value.type() == typeid(Parser::Map)) {
            const auto& map = std::any_cast<const Parser::Map&>(value);
            if (!map.contains(segment.key)) {
                return Lookup::Missing;
            }
            key = segment.key;
            current = Value{.pointer = &map.at(segment.key)};
            continue;
        }

        const auto size = ContainerSize(value);
        if (!size.has_value()) {
            return Lookup::Missing;
        }

        std::optional<U64> position;
        if (segment.kind == Segment::Kind::Slice) {
            if (!row.has_value()) {
                return Lookup::Invalid;
            }
            const auto [begin, end] = SliceRange(segment, size.value());
            if (begin + row.value() < end) {
                position = begin + row.value();
            }
        } else if (segment.kind == Segment::Kind::Index) {
            position = NormalizeIndex(segment.index, size.value());
        } else {
            I64 index = 0;
            if (ParseInteger(segment.key, index)) {
                position = NormalizeIndex(index, size.value());
            }
        }

        if (!position.has_value()) {
            return Lookup::Missing;
        }
        current = ElementAt(value, position.value(), key);
    }

    out = std::move(current);
    return Lookup::Found;
}

U64 SliceLength(Context& context, const Token& token) {
    Value container;
    if (Walk(context, token, std::nullopt, token.slice.value(), container) != Lookup::Found) {
        return 0;
    }
    const auto size = ContainerSize(*container.get());
    if (!size.has_value()) {
        return 0;
    }
    const auto [begin, end] = SliceRange(token.path[token.slice.value()], size.value());
    return end - begin;
}

template<typename T>
bool FormatTyped(const std::any& value, const std::string& pattern, std::string& out) {
    if (value.type() != typeid(T)) {
        return false;
    }
    const auto& typed = std::any_cast<const T&>(value);
    if constexpr (std::is_same_v<T, CF32> || std::is_same_v<T, CF64>) {
        const std::string real = jst::fmt::format(jst::fmt::runtime(pattern), typed.real());
        const std::string imag = jst::fmt::format(jst::fmt::runtime(pattern), std::abs(typed.imag()));
        out = jst::fmt::format("{}{}{}i", real, typed.imag() < 0 ? "-" : "+", imag);
    } else {
        out = jst::fmt::format(jst::fmt::runtime(pattern), typed);
    }
    return true;
}

std::string FormatCell(const std::any& value, const Format& format, U64 depth = 0);

std::string Compact(const std::any& value, const Format& format, const U64 depth) {
    const bool map = value.type() == typeid(Parser::Map);
    if (depth >= MaxDepth) {
        return map ? "{...}" : "[...]";
    }
    const U64 size = ContainerSize(value).value_or(0);
    std::vector<std::string> items;
    items.reserve(std::min(size, MaxInline) + 1);
    U64 bytes = 0;
    U64 shown = 0;
    for (; shown < std::min(size, MaxInline) && bytes < MaxCellBytes; ++shown) {
        std::string key;
        const Value element = ElementAt(value, shown, key);
        const std::string cell = FormatCell(*element.get(), format, depth + 1);
        items.push_back(map ? jst::fmt::format("{}: {}", key, cell) : cell);
        bytes += items.back().size() + 2;
    }
    if (size > shown) {
        items.push_back("...");
    }
    if (map) {
        return jst::fmt::format("{{{}}}", jst::fmt::join(items, ", "));
    }
    return jst::fmt::format("[{}]", jst::fmt::join(items, ", "));
}

bool Fits(const Format& format, const std::any& value) {
    const auto& type = value.type();
    if (type == typeid(F32) || type == typeid(F64) || type == typeid(CF32) || type == typeid(CF64)) {
        return format.accepts(Format::Floating);
    }
    if (type == typeid(I8) || type == typeid(I16) || type == typeid(I32) || type == typeid(I64) ||
        type == typeid(U8) || type == typeid(U16) || type == typeid(U32) || type == typeid(U64)) {
        return format.accepts(Format::Integer);
    }
    if (type == typeid(std::string)) {
        return format.accepts(Format::Text);
    }
    if (type == typeid(bool)) {
        return format.accepts(Format::Boolean);
    }
    return false;
}

void Truncate(std::string& text) {
    if (text.size() <= MaxCellBytes) {
        return;
    }
    U64 end = MaxCellBytes;
    while (end > 0 && (static_cast<U8>(text[end]) & 0xC0) == 0x80) {
        --end;
    }
    text.resize(end);
    text += "...";
}

bool FormatValue(const std::any& value, const Format& format, std::string& out, const U64 depth = 0) {
    if (!value.has_value()) {
        out = Missing;
        return true;
    }
    if (IsContainer(value)) {
        if (!format.valid()) {
            return false;
        }
        out = Compact(value, format, depth);
        Truncate(out);
        return true;
    }
    if (format.pattern.empty()) {
        out = FlowgraphKeyValueDetail::AnyToString(value);
        Truncate(out);
        return true;
    }
    if (!Fits(format, value)) {
        return false;
    }

    const std::string& pattern = format.pattern;
    try {
        if (FormatTyped<F32>(value, pattern, out) || FormatTyped<F64>(value, pattern, out) ||
            FormatTyped<I8>(value, pattern, out) || FormatTyped<I16>(value, pattern, out) ||
            FormatTyped<I32>(value, pattern, out) || FormatTyped<I64>(value, pattern, out) ||
            FormatTyped<U8>(value, pattern, out) || FormatTyped<U16>(value, pattern, out) ||
            FormatTyped<U32>(value, pattern, out) || FormatTyped<U64>(value, pattern, out) ||
            FormatTyped<CF32>(value, pattern, out) || FormatTyped<CF64>(value, pattern, out) ||
            FormatTyped<bool>(value, pattern, out) || FormatTyped<std::string>(value, pattern, out)) {
            Truncate(out);
            return true;
        }
    } catch (const std::exception&) {
        return false;
    }

    out = FlowgraphKeyValueDetail::AnyToString(value);
    Truncate(out);
    return true;
}

std::string FormatCell(const std::any& value, const Format& format, const U64 depth) {
    std::string out;
    if (!FormatValue(value, format, out, depth)) {
        FormatValue(value, Format{}, out, depth);
    }
    return out;
}

std::string RenderLine(Context& context, const std::vector<Piece>& pieces, const std::optional<U64> row) {
    std::string line;
    for (const auto& piece : pieces) {
        if (!piece.placeholder || !piece.token.has_value()) {
            line += piece.text;
            continue;
        }

        Value value;
        const Lookup lookup = Walk(context, piece.token.value(), row, piece.token->path.size(), value);
        if (lookup == Lookup::Missing) {
            line += Missing;
            continue;
        }

        std::string formatted;
        const bool formattedOk = lookup == Lookup::Found &&
                                 (FormatValue(*value.get(), piece.token->format, formatted) ||
                                  (piece.token->lenient && FormatValue(*value.get(), Format{}, formatted)));
        if (!formattedOk) {
            line += piece.text;
            continue;
        }
        line += Escape(formatted);
    }
    return line;
}

std::string ListPrefix(std::string_view line) {
    const U64 indent = line.find_first_not_of(" \t");
    if (indent == std::string_view::npos) {
        return {};
    }
    const std::string_view rest = line.substr(indent);
    if (rest.size() > 1 && (rest[0] == '-' || rest[0] == '*' || rest[0] == '+') && rest[1] == ' ') {
        return std::string(line.substr(0, indent + 2));
    }
    const U64 digits = rest.find_first_not_of("0123456789");
    if (digits != std::string_view::npos && digits > 0 && digits + 1 < rest.size() &&
        rest[digits] == '.' && rest[digits + 1] == ' ') {
        return std::string(line.substr(0, indent + digits + 2));
    }
    return std::string(line.substr(0, indent));
}

std::string MoreLine(std::string_view line, const bool table, const U64 remaining) {
    if (table) {
        return jst::fmt::format("| ... {} more |", remaining);
    }
    return jst::fmt::format("{}... {} more", ListPrefix(line), remaining);
}

void Repeat(Context& context, const Line& line, std::vector<std::string>& output) {
    std::optional<U64> count;
    for (const auto& piece : line.pieces) {
        if (piece.token.has_value() && piece.token->slice.has_value()) {
            count = std::max(count.value_or(0), SliceLength(context, piece.token.value()));
        }
    }

    if (!count.has_value()) {
        context.emit(output, RenderLine(context, line.pieces, std::nullopt));
        return;
    }

    const U64 limit = std::min(count.value(), MaxRepeat);
    U64 shown = 0;
    while (shown < limit && !context.full()) {
        context.emit(output, RenderLine(context, line.pieces, shown++));
    }
    if (count.value() > shown) {
        context.emit(output, MoreLine(line.text, line.table, count.value() - shown));
    }
}

std::string TableRow(const std::vector<std::string>& cells) {
    std::string row = "|";
    for (const auto& cell : cells) {
        row += " " + cell + " |";
    }
    return row;
}

std::vector<std::string> Columns(const std::any& container, const U64 rows, bool& clipped) {
    std::vector<std::string> columns;
    std::unordered_set<std::string> seen;
    clipped = false;
    for (U64 i = 0; i < rows; ++i) {
        std::string key;
        const Value element = ElementAt(container, i, key);
        const auto& map = std::any_cast<const Parser::Map&>(*element.get());
        for (const auto& entry : map) {
            if (seen.contains(entry.key)) {
                continue;
            }
            if (columns.size() == MaxColumns) {
                clipped = true;
                return columns;
            }
            seen.insert(entry.key);
            columns.push_back(entry.key);
        }
    }
    return columns;
}

bool AllMaps(const std::any& container, const U64 rows) {
    if (rows == 0 || TypedVectors::size(container).has_value()) {
        return false;
    }
    for (U64 i = 0; i < rows; ++i) {
        std::string key;
        const Value element = ElementAt(container, i, key);
        if (element.get()->type() != typeid(Parser::Map)) {
            return false;
        }
    }
    return true;
}

Piece Cell(const Token& token, std::optional<std::string> field) {
    Token cell = token;
    cell.lenient = true;
    cell.slice = cell.path.size();
    cell.path.push_back({.kind = Segment::Kind::Slice});
    if (field.has_value()) {
        cell.path.push_back({.kind = Segment::Kind::Key, .key = std::move(field.value())});
    }
    return {.placeholder = true, .token = std::move(cell)};
}

void ExpandContainer(Context& context, const Token& token, const std::any& value, std::vector<std::string>& output) {
    const U64 size = ContainerSize(value).value_or(0);
    if (size == 0) {
        context.emit(output, std::string(Missing));
        return;
    }

    const bool map = value.type() == typeid(Parser::Map);
    const U64 rows = std::min(size, MaxRepeat);
    bool clipped = false;
    std::vector<std::string> columns = AllMaps(value, rows) ? Columns(value, rows, clipped) : std::vector<std::string>{};
    const bool records = !columns.empty();
    Line row{.kind = Line::Kind::Body, .table = map || records};
    if (!row.table) {
        row.text = "- ";
        row.pieces = {{.text = "- "}, Cell(token, std::nullopt)};
        Repeat(context, row, output);
        return;
    }

    std::vector<std::string> header;
    std::vector<Piece> cells;
    if (map) {
        header.emplace_back("Key");
        cells.push_back(Cell(token, "_key"));
    }
    if (records) {
        for (auto& column : columns) {
            header.push_back(Escape(column));
            cells.push_back(Cell(token, std::move(column)));
        }
    } else {
        header.emplace_back("Value");
        cells.push_back(Cell(token, std::nullopt));
    }
    if (clipped) {
        header.emplace_back("...");
        cells.push_back({.text = "..."});
    }

    for (U64 i = 0; i < cells.size(); ++i) {
        row.pieces.push_back({.text = i == 0 ? "| " : " | "});
        row.pieces.push_back(std::move(cells[i]));
    }
    row.pieces.push_back({.text = " |"});

    context.emit(output, TableRow(header));
    context.emit(output, TableRow(std::vector<std::string>(header.size(), "---")));
    Repeat(context, row, output);
}

bool ExpandStandalone(Context& context, const Line& line, std::vector<std::string>& output) {
    const Token& token = StandaloneToken(line.pieces);
    Value value;
    if (Walk(context, token, std::nullopt, token.path.size(), value) != Lookup::Found ||
        !IsContainer(*value.get()) || !token.format.valid()) {
        return false;
    }
    if (!output.empty() && !TrimView(output.back()).empty()) {
        context.emit(output, {});
    }
    ExpandContainer(context, token, *value.get(), output);
    if (line.spaceAfter) {
        context.emit(output, {});
    }
    return true;
}

Line Compile(std::string_view text, const Line::Kind kind, const bool table = false) {
    Line line{.kind = Line::Kind::Verbatim, .text = text, .table = table};
    if (text.find("${") == std::string_view::npos) {
        return line;
    }
    line.pieces = ScanPieces(text);
    if (HasTokens(line.pieces)) {
        line.kind = kind;
        line.standalone = kind == Line::Kind::Body && !table && Standalone(line.pieces);
    } else {
        line.pieces.clear();
    }
    return line;
}

}  // namespace

struct LiveMarkdown::Impl {
    std::string source;
    std::vector<Line> lines;
    bool dynamic = false;

    void compile() {
        const auto views = SplitLines(source);
        lines.reserve(views.size());
        bool fence = false;
        bool stats = false;
        for (U64 i = 0; i < views.size(); ++i) {
            const std::string_view text = views[i];
            if (IsFence(text)) {
                stats = !fence && IsStatsFence(text);
                fence = !fence;
                lines.push_back({.text = text});
            } else if (fence) {
                lines.push_back(stats ? Compile(text, Line::Kind::Stats) : Line{.text = text});
            } else if (text.find('|') != std::string_view::npos && i + 1 < views.size() &&
                       IsDelimiterRow(views[i + 1])) {
                lines.push_back(Compile(text, Line::Kind::Header, true));
                lines.push_back({.text = views[++i]});
            } else {
                lines.push_back(Compile(text, Line::Kind::Body, TrimView(text).starts_with('|')));
            }
        }

        for (U64 i = 0; i < lines.size(); ++i) {
            lines[i].spaceAfter = i + 1 < lines.size() && !TrimView(lines[i + 1].text).empty();
            dynamic = dynamic || lines[i].kind != Line::Kind::Verbatim;
        }
    }
};

LiveMarkdown::LiveMarkdown() : impl(std::make_unique<Impl>()) {}

LiveMarkdown::LiveMarkdown(std::string source) : impl(std::make_unique<Impl>()) {
    impl->source = std::move(source);
    impl->compile();
}

LiveMarkdown::~LiveMarkdown() = default;

LiveMarkdown::LiveMarkdown(LiveMarkdown&&) noexcept = default;

LiveMarkdown& LiveMarkdown::operator=(LiveMarkdown&&) noexcept = default;

const std::string& LiveMarkdown::source() const {
    return impl->source;
}

bool LiveMarkdown::dynamic() const {
    return impl->dynamic;
}

std::string LiveMarkdown::expand(const Flowgraph::Environment& environment) const {
    if (!impl->dynamic) {
        return impl->source;
    }

    Context context{.environment = environment};
    std::vector<std::string> output;
    output.reserve(impl->lines.size());

    for (const auto& line : impl->lines) {
        switch (line.kind) {
            case Line::Kind::Verbatim:
                context.emit(output, std::string(line.text));
                break;
            case Line::Kind::Header:
                context.emit(output, RenderLine(context, line.pieces, std::nullopt));
                break;
            case Line::Kind::Stats:
                Repeat(context, line, output);
                break;
            case Line::Kind::Body:
                if (!line.standalone || !ExpandStandalone(context, line, output)) {
                    Repeat(context, line, output);
                }
                break;
        }
    }

    if (context.truncated) {
        output.emplace_back("... output truncated");
    }

    return jst::fmt::format("{}", jst::fmt::join(output, "\n"));
}

}  // namespace Jetstream
