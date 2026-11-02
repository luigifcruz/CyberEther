#include "base.hh"
#include "tools/text.hh"

#include <algorithm>
#include <array>
#include <iterator>
#include <optional>
#include <regex>

namespace Jetstream::Agent::Detail {
namespace {

using Lines = std::vector<std::string>;
using Normalizer = std::string (*)(const std::string&);

constexpr U64 kHintMatches = 3;

struct Step {
    char kind;
    std::string text;
};

struct Hunk {
    std::string anchor;
    std::vector<Step> steps;
    bool endOfText = false;

    Lines before() const {
        Lines lines;
        for (const auto& step : steps) {
            if (step.kind != '+') {
                lines.push_back(step.text);
            }
        }
        return lines;
    }

    bool changes() const {
        return std::any_of(steps.begin(), steps.end(), [](const Step& step) {
            return step.kind != ' ';
        });
    }
};

struct Replacement {
    U64 start = 0;
    U64 count = 0;
    Lines after;
    U64 removed = 0;
    U64 added = 0;
};

struct Indent {
    std::string add;
    U64 strip = 0;
};

U64 VisibleLineCount(const Lines& lines) {
    if (lines.size() > 1 && lines.back().empty()) {
        return lines.size() - 1;
    }
    return lines.size();
}

std::string Exact(const std::string& line) {
    return line;
}

std::string RightTrim(const std::string& line) {
    const U64 end = line.find_last_not_of(" \t\r");
    if (end == std::string::npos) {
        return {};
    }
    return line.substr(0, end + 1);
}

std::string Trim(const std::string& line) {
    const std::string right = RightTrim(line);
    const U64 start = right.find_first_not_of(" \t");
    if (start == std::string::npos) {
        return {};
    }
    return right.substr(start);
}

std::string FoldPunctuation(const std::string& line) {
    std::string folded;
    folded.reserve(line.size());

    const auto byte = [&](U64 index) {
        return index < line.size() ? static_cast<U8>(line[index]) : U8{0};
    };

    for (U64 i = 0; i < line.size();) {
        const U8 a = byte(i);
        const U8 b = byte(i + 1);
        const U8 c = byte(i + 2);

        const bool generalPunctuation = a == 0xE2 && b == 0x80;
        const bool wideSpace = (generalPunctuation && ((c >= 0x82 && c <= 0x8A) || c == 0xAF)) ||
                               (a == 0xE2 && b == 0x81 && c == 0x9F) ||
                               (a == 0xE3 && b == 0x80 && c == 0x80);

        if (a == 0xC2 && b == 0xA0) {
            folded += ' ';
            i += 2;
        } else if (generalPunctuation && c >= 0x90 && c <= 0x95) {
            folded += '-';
            i += 3;
        } else if (a == 0xE2 && b == 0x88 && c == 0x92) {
            folded += '-';
            i += 3;
        } else if (generalPunctuation && c >= 0x98 && c <= 0x9B) {
            folded += '\'';
            i += 3;
        } else if (generalPunctuation && c >= 0x9C && c <= 0x9F) {
            folded += '"';
            i += 3;
        } else if (wideSpace) {
            folded += ' ';
            i += 3;
        } else {
            folded += line[i];
            i += 1;
        }
    }

    return Trim(folded);
}

constexpr std::array<Normalizer, 4> kPasses = {Exact, RightTrim, Trim, FoldPunctuation};

bool StartsWith(const std::string& line, const char* prefix) {
    return line.rfind(prefix, 0) == 0;
}

std::string HunkAnchor(const std::string& line) {
    std::string anchor = Trim(line.substr(2));

    if (!anchor.empty() && anchor.front() == '-') {
        const U64 close = anchor.find("@@");
        if (close == std::string::npos) {
            anchor.clear();
        } else {
            anchor = Trim(anchor.substr(close + 2));
        }
    }

    if (anchor.size() >= 2 && anchor.compare(anchor.size() - 2, 2, "@@") == 0) {
        anchor = Trim(anchor.substr(0, anchor.size() - 2));
    }

    return anchor;
}

std::string ParsePatch(const std::string& patch, std::vector<Hunk>& hunks) {
    std::string normalized;
    normalized.reserve(patch.size());
    for (const char c : patch) {
        if (c != '\r') {
            normalized += c;
        }
    }

    Lines lines = Text::SplitLines(normalized);
    if (!lines.empty() && lines.back().empty()) {
        lines.pop_back();
    }

    for (U64 index = 0; index < lines.size(); ++index) {
        const auto& line = lines[index];

        const bool envelope = StartsWith(line, "*** Begin Patch") ||
                              StartsWith(line, "*** End Patch") ||
                              StartsWith(line, "*** Update File:");
        if (envelope) {
            continue;
        }

        if (StartsWith(line, "*** End of File")) {
            if (!hunks.empty()) {
                hunks.back().endOfText = true;
            }
            continue;
        }

        const bool fileHeader = StartsWith(line, "--- ") || StartsWith(line, "+++ ");
        if (hunks.empty() && fileHeader) {
            continue;
        }

        if (StartsWith(line, "@@")) {
            hunks.push_back({HunkAnchor(line), {}, false});
            continue;
        }

        if (hunks.empty()) {
            hunks.push_back({});
        }

        auto& steps = hunks.back().steps;
        if (line.empty()) {
            steps.push_back({' ', {}});
        } else if (line.front() == ' ' || line.front() == '-' || line.front() == '+') {
            steps.push_back({line.front(), line.substr(1)});
        } else {
            return jst::fmt::format("Patch line {} does not start with a space, '-', '+', or '@@'. "
                                    "Unchanged lines start with a space.", index + 1);
        }
    }

    if (hunks.empty()) {
        return "Patch has no hunks.";
    }

    for (U64 index = 0; index < hunks.size(); ++index) {
        if (!hunks[index].changes()) {
            return jst::fmt::format("Hunk {} has no '-' or '+' lines, so it changes nothing.", index + 1);
        }
    }

    return {};
}

std::vector<U64> FindSequence(const Lines& normalized, const Lines& pattern, U64 from, bool endOfText,
                              U64 textEnd) {
    std::vector<U64> matches;
    if (pattern.size() > textEnd || from > textEnd - pattern.size()) {
        return matches;
    }

    for (U64 start = from; start + pattern.size() <= textEnd; ++start) {
        if (endOfText && start + pattern.size() != textEnd) {
            continue;
        }

        const auto position = normalized.begin() + static_cast<I64>(start);
        if (std::equal(pattern.begin(), pattern.end(), position)) {
            matches.push_back(start);
        }
    }

    return matches;
}

Lines Normalized(const Lines& lines, Normalizer normalize) {
    Lines result;
    result.reserve(lines.size());
    for (const auto& line : lines) {
        result.push_back(normalize(line));
    }
    return result;
}

std::optional<U64> FindAnchor(const Lines& lines, const std::string& anchor, U64 from) {
    for (const auto normalize : kPasses) {
        const std::string target = normalize(anchor);
        for (U64 index = from; index < lines.size(); ++index) {
            if (normalize(lines[index]) == target) {
                return index;
            }
        }
    }
    return std::nullopt;
}

std::string LineList(const std::vector<U64>& indices) {
    std::string list;
    for (U64 i = 0; i < indices.size(); ++i) {
        jst::fmt::format_to(std::back_inserter(list), "{}{}", i > 0 ? ", " : "", indices[i] + 1);
    }
    return list;
}

std::string MissHint(const Lines& lines, const Lines& before) {
    const auto first = std::find_if(before.begin(), before.end(), [](const std::string& line) {
        return !Trim(line).empty();
    });
    if (first == before.end()) {
        return {};
    }

    const std::string target = FoldPunctuation(*first);
    std::vector<U64> found;
    for (U64 index = 0; index < lines.size() && found.size() < kHintMatches; ++index) {
        if (FoldPunctuation(lines[index]) == target) {
            found.push_back(index);
        }
    }

    if (found.empty()) {
        return jst::fmt::format(" Its first line '{}' does not appear in the text.", Trim(*first));
    }

    return jst::fmt::format(" Its first line '{}' appears at line {}, so a later line in the hunk differs from the text.",
                            Trim(*first), LineList(found));
}

std::string LeadingSpace(const std::string& line) {
    const U64 end = std::min(line.size(), line.find_first_not_of(" \t"));
    return line.substr(0, end);
}

Indent Shift(const std::string& original, const std::string& written) {
    const std::string have = LeadingSpace(original);
    const std::string wrote = LeadingSpace(written);
    if (have.rfind(wrote, 0) == 0) {
        return {have.substr(wrote.size()), 0};
    }
    if (wrote.rfind(have, 0) == 0) {
        return {{}, wrote.size() - have.size()};
    }
    return {};
}

Indent IndentShift(const Lines& lines, U64 start, const Hunk& hunk) {
    U64 index = start;
    for (const auto& step : hunk.steps) {
        if (step.kind == '+') {
            continue;
        }

        const std::string& original = lines[index++];
        if (Trim(step.text).empty() || Trim(original).empty()) {
            continue;
        }
        return Shift(original, step.text);
    }
    return {};
}

std::string Reindent(const std::string& line, const Indent& indent) {
    if (Trim(line).empty()) {
        return line;
    }

    const U64 strip = std::min<U64>(indent.strip, LeadingSpace(line).size());
    return jst::fmt::format("{}{}", indent.add, line.substr(strip));
}

Replacement BuildReplacement(const Lines& lines, U64 start, const Hunk& hunk, bool reindent) {
    const Indent indent = reindent ? IndentShift(lines, start, hunk) : Indent{};
    Replacement replacement{.start = start};

    U64 index = start;
    for (const auto& step : hunk.steps) {
        if (step.kind == ' ') {
            replacement.after.push_back(lines[index++]);
        } else if (step.kind == '-') {
            ++index;
            ++replacement.removed;
        } else {
            replacement.after.push_back(Reindent(step.text, indent));
            ++replacement.added;
        }
    }

    replacement.count = index - start;
    return replacement;
}

bool Blank(const std::string& line) {
    return Trim(line).empty();
}

std::vector<U64> LooseMatches(const Lines& lines, const Hunk& hunk, U64 from, U64 textEnd) {
    Lines needle;
    for (const auto& line : hunk.before()) {
        if (!Blank(line)) {
            needle.push_back(FoldPunctuation(line));
        }
    }

    std::vector<U64> positions;
    Lines haystack;
    for (U64 index = from; index < textEnd; ++index) {
        if (!Blank(lines[index])) {
            positions.push_back(index);
            haystack.push_back(FoldPunctuation(lines[index]));
        }
    }

    std::vector<U64> matches;
    if (needle.empty() || needle.size() > haystack.size()) {
        return matches;
    }

    for (U64 start = 0; start + needle.size() <= haystack.size(); ++start) {
        if (hunk.endOfText && start + needle.size() != haystack.size()) {
            continue;
        }
        if (std::equal(needle.begin(), needle.end(), haystack.begin() + static_cast<I64>(start))) {
            matches.push_back(positions[start]);
        }
    }
    return matches;
}

Replacement LooseReplacement(const Lines& lines, U64 first, const Hunk& hunk, U64 from, U64 textEnd) {
    const auto written = std::find_if(hunk.steps.begin(), hunk.steps.end(), [](const Step& step) {
        return step.kind != '+' && !Blank(step.text);
    });
    const Indent indent = Shift(lines[first], written->text);

    U64 leading = 0;
    for (auto step = hunk.steps.begin(); step != written; ++step) {
        leading += step->kind != '+';
    }

    U64 start = first;
    while (leading > 0 && start > from && Blank(lines[start - 1])) {
        --start;
        --leading;
    }

    Replacement replacement{.start = start};
    U64 index = start;
    for (const auto& step : hunk.steps) {
        if (step.kind == '+') {
            replacement.after.push_back(Reindent(step.text, indent));
            ++replacement.added;
            continue;
        }

        if (Blank(step.text)) {
            if (index < textEnd && Blank(lines[index])) {
                if (step.kind == ' ') {
                    replacement.after.push_back(lines[index]);
                } else {
                    ++replacement.removed;
                }
                ++index;
            } else if (step.kind == ' ') {
                replacement.after.push_back({});
                ++replacement.added;
            }
            continue;
        }

        while (index < textEnd && Blank(lines[index])) {
            replacement.after.push_back(lines[index++]);
        }
        if (step.kind == ' ') {
            replacement.after.push_back(lines[index]);
        } else {
            ++replacement.removed;
        }
        ++index;
    }

    replacement.count = index - start;
    return replacement;
}

bool DropTrailingBlank(Hunk& hunk) {
    if (hunk.steps.empty()) {
        return false;
    }

    const auto& last = hunk.steps.back();
    if (last.kind != ' ' || !last.text.empty()) {
        return false;
    }

    hunk.steps.pop_back();
    return !hunk.before().empty();
}

std::string LocateHunk(const Lines& lines, Hunk hunk, U64 number, U64& cursor, Replacement& replacement) {
    const std::string label = jst::fmt::format("Hunk {}", number);

    U64 from = cursor;
    if (!hunk.anchor.empty()) {
        const auto anchor = FindAnchor(lines, hunk.anchor, cursor);
        if (!anchor) {
            return jst::fmt::format("{}: the @@ line '{}' was not found after the previous hunk.",
                                    label, hunk.anchor);
        }
        from = *anchor + 1;
    }

    const U64 textEnd = VisibleLineCount(lines);
    if (hunk.before().empty()) {
        const bool append = hunk.anchor.empty() || hunk.endOfText;
        const U64 at = append ? textEnd : from;
        replacement = BuildReplacement(lines, at, hunk, false);
        cursor = at;
        return {};
    }

    do {
        const Lines before = hunk.before();
        for (U64 pass = 0; pass < kPasses.size(); ++pass) {
            const auto normalize = kPasses[pass];
            const auto haystack = Normalized(lines, normalize);
            const auto needle = Normalized(before, normalize);
            const auto matches = FindSequence(haystack, needle, from, hunk.endOfText, textEnd);

            if (matches.size() == 1) {
                replacement = BuildReplacement(lines, matches.front(), hunk, pass >= 2);
                cursor = matches.front() + before.size();
                return {};
            }

            if (matches.size() > 1) {
                return jst::fmt::format("{} matches {} places (lines {}). More unchanged context lines, "
                                        "or an @@ line naming the enclosing def or class, would narrow it to one place.",
                                        label, matches.size(), LineList(matches));
            }
        }
    } while (DropTrailingBlank(hunk));

    const auto loose = LooseMatches(lines, hunk, from, textEnd);
    if (loose.size() == 1) {
        replacement = LooseReplacement(lines, loose.front(), hunk, from, textEnd);
        cursor = replacement.start + replacement.count;
        return {};
    }
    if (loose.size() > 1) {
        return jst::fmt::format("{} matches {} places (lines {}). More unchanged context lines, "
                                "or an @@ line naming the enclosing def or class, would narrow it to one place.",
                                label, loose.size(), LineList(loose));
    }

    return jst::fmt::format("{}: its unchanged and '-' lines were not found in order after the previous hunk.{} "
                            "read_block_text shows the current lines.", label, MissHint(lines, hunk.before()));
}

Json LineJson(const Lines& lines, U64 index) {
    return Json({
        {"line", index + 1},
        {"text", lines[index]},
    });
}

Json Excerpt(const Lines& lines, U64 begin, U64 end) {
    Json excerpt = Json::array();
    const U64 last = std::min(end, VisibleLineCount(lines));
    for (U64 index = begin; index < last; ++index) {
        excerpt.push_back(LineJson(lines, index));
    }
    return excerpt;
}

std::vector<std::string> TextFields(const Parser::Map& config) {
    std::vector<std::string> fields;
    for (const auto& entry : config) {
        if (entry.value.type() == typeid(std::string)) {
            fields.push_back(entry.key);
        }
    }
    return fields;
}

Json SearchLines(const Lines& lines, const std::string& pattern, bool regex, U64 context, U64 limit,
                 std::string& error) {
    std::optional<std::regex> expression;
    if (regex) {
        try {
            expression.emplace(pattern);
        } catch (const std::regex_error& failure) {
            error = jst::fmt::format("Invalid regex: {}", failure.what());
            return nullptr;
        }
    }

    const U64 count = VisibleLineCount(lines);
    std::vector<U64> hits;
    for (U64 index = 0; index < count; ++index) {
        bool hit = false;
        if (expression) {
            hit = std::regex_search(lines[index], *expression);
        } else {
            hit = lines[index].find(pattern) != std::string::npos;
        }

        if (hit) {
            hits.push_back(index);
        }
    }

    Json shown = Json::array();
    U64 next = 0;
    U64 hitIndex = 0;
    bool truncated = false;
    for (const U64 hit : hits) {
        const U64 contextStart = hit > context ? hit - context : 0;
        const U64 begin = std::max(next, contextStart);
        const U64 end = std::min(count, hit + context + 1);

        for (U64 index = begin; index < end; ++index) {
            if (shown.size() == limit) {
                truncated = true;
                break;
            }

            auto line = LineJson(lines, index);
            if (hitIndex < hits.size() && hits[hitIndex] == index) {
                line["match"] = true;
                ++hitIndex;
            }
            shown.push_back(std::move(line));
        }

        if (truncated) {
            break;
        }
        next = end;
    }

    Json result = {
        {"matches", hits.size()},
        {"lines", std::move(shown)},
    };
    if (truncated) {
        result["truncated"] = true;
    }
    return result;
}

}  // namespace

Json BlockText::Range(const std::string& text, const Json& args, std::string& error) {
    const Lines lines = Text::SplitLines(text);
    const U64 count = VisibleLineCount(lines);
    const U64 limit = Arguments::Clamp(args, "limit", 200, 1, 500);
    Json result = {{"line_count", count}};

    const std::string pattern = args.value("pattern", std::string{});
    if (!pattern.empty()) {
        const bool regex = args.value("regex", false);
        const U64 context = Arguments::Clamp(args, "context", 2, 0, 20);
        auto found = SearchLines(lines, pattern, regex, context, limit, error);
        if (error.empty()) {
            result.update(found);
        }
        return result;
    }

    const I64 lastLine = std::max<I64>(1, static_cast<I64>(count));
    const U64 offset = Arguments::Clamp(args, "offset", 1, 1, lastLine) - 1;
    const U64 end = std::min(count, offset + limit);

    result["lines"] = Excerpt(lines, offset, end);
    if (end < count) {
        result["next_offset"] = end + 1;
    }
    return result;
}

std::string BlockText::Read(const Graph& flowgraph, const std::string& name, const std::string& field,
                            std::string& text) {
    if (!flowgraph) {
        return "No flowgraph is currently selected.";
    }

    Flowgraph::View::BlockData data;
    if (name.empty() || flowgraph->view().block(name, data) != Result::SUCCESS) {
        return Lookup::Missing(flowgraph, name);
    }

    const auto fields = TextFields(data.config);
    const bool isText = data.config.contains(field) && data.config.at(field).type() == typeid(std::string);
    if (!isText) {
        const std::string list = jst::fmt::format("{}", jst::fmt::join(fields, ", "));
        return jst::fmt::format("Block '{}' has no text field '{}'. Text fields: {}.",
                                name, field, list.empty() ? "none" : list);
    }

    text = std::any_cast<const std::string&>(data.config.at(field));
    return {};
}

BlockText::Patch BlockText::Apply(const std::string& text, const std::string& patch) {
    BlockText::Patch result;

    std::vector<Hunk> hunks;
    result.error = ParsePatch(patch, hunks);
    if (!result.error.empty()) {
        return result;
    }

    Lines lines = text.empty() ? Lines{} : Text::SplitLines(text);
    std::vector<Replacement> replacements(hunks.size());
    U64 cursor = 0;
    for (U64 index = 0; index < hunks.size(); ++index) {
        result.error = LocateHunk(lines, std::move(hunks[index]), index + 1, cursor, replacements[index]);
        if (!result.error.empty()) {
            return result;
        }
    }

    for (auto it = replacements.rbegin(); it != replacements.rend(); ++it) {
        const auto begin = lines.begin() + static_cast<I64>(it->start);
        lines.erase(begin, begin + static_cast<I64>(it->count));
        lines.insert(lines.begin() + static_cast<I64>(it->start), it->after.begin(), it->after.end());
    }

    I64 shift = 0;
    for (const auto& replacement : replacements) {
        const U64 start = static_cast<U64>(static_cast<I64>(replacement.start) + shift);
        const U64 begin = start > 0 ? start - 1 : 0;
        const U64 end = start + replacement.after.size() + 1;

        result.changes.push_back({
            {"line", start + 1},
            {"removed", replacement.removed},
            {"added", replacement.added},
            {"excerpt", Excerpt(lines, begin, end)},
        });

        shift += static_cast<I64>(replacement.after.size()) - static_cast<I64>(replacement.count);
    }

    result.lineCount = VisibleLineCount(lines);
    result.text = Text::JoinLines(lines);
    return result;
}

}  // namespace Jetstream::Agent::Detail
