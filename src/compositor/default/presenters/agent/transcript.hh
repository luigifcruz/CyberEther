#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_AGENT_TRANSCRIPT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_AGENT_TRANSCRIPT_HH

#include "../../model/state.hh"

#include "jetstream/config.hh"
#include "jetstream/render/sakura/components/retained/chat.hh"
#include "tools/text.hh"

#include <algorithm>
#include <cctype>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

class AgentTranscriptFormatter {
 public:
    const Sakura::Retained::Chat::Snapshot& format(
        const DefaultCompositorState::AgentState::Conversation& conversation) {
        if (generation != conversation.generation) {
            entries.clear();
            snapshot.messages = {};
            generation = conversation.generation;
        }

        const auto& messages = conversation.messages;
        const U64 count = messages.size();
        bool historyChanged = entries.size() != count;
        entries.resize(count);

        for (U64 i = 0; i < count; ++i) {
            const bool busy = conversation.streaming && i + 1 == count;
            auto& entry = entries[i];
            if (entry.display && entry.source == messages[i] && entry.busy == busy) {
                continue;
            }

            entry.source = messages[i];
            entry.busy = busy;
            auto display = DisplayMessage(*messages[i], busy);
            entry.display = std::make_shared<const Chat::Message>(std::move(display));
            historyChanged |= i + 1 < count;
        }

        snapshot.generation = conversation.generation;
        snapshot.streaming = conversation.streaming;
        if (historyChanged) {
            auto history = std::make_shared<Chat::Snapshot::Messages::History>();
            history->reserve(count > 0 ? count - 1 : 0);
            for (U64 i = 0; i + 1 < count; ++i) {
                history->push_back(entries[i].display);
            }
            snapshot.messages.history = std::move(history);
        }

        if (count > 0) {
            snapshot.messages.tail = entries.back().display;
        } else {
            snapshot.messages.tail = nullptr;
        }

        return snapshot;
    }

 private:
    using Chat = Sakura::Retained::Chat;
    using Segments = std::vector<Agent::Session::Segment>;
    using SegmentKind = Agent::Session::Segment::Kind;

    struct Entry {
        std::shared_ptr<const Agent::Session::Entry> source;
        bool busy = false;
        std::shared_ptr<const Sakura::Retained::Chat::Message> display;
    };

    static bool IsHeadline(const std::string& line) {
        return line.size() > 4 && line.rfind("**", 0) == 0 && line.find("**", 2) == line.size() - 2;
    }

    static Segments DisplaySegments(const Segments& segments) {
        Segments view;
        const auto add = [&](SegmentKind kind, std::string text, U64 revision, U64 images = 0) {
            const bool merges = kind == SegmentKind::Reasoning || kind == SegmentKind::Summary ||
                                kind == SegmentKind::Note;
            if (merges && !view.empty() && view.back().kind == kind) {
                view.back().text += text;
                view.back().revision += revision;
                return;
            }
            view.push_back({kind, std::move(text), revision, {}, images});
        };

        for (const auto& segment : segments) {
            if (segment.kind == SegmentKind::Summary) {
                std::string::size_type start = 0;
                while (start < segment.text.size()) {
                    const auto end = segment.text.find('\n', start);
                    const bool complete = end != std::string::npos;
                    const auto length = complete ? end - start : std::string::npos;
                    const auto line = segment.text.substr(start, length);
                    const std::string suffix = complete ? "\n" : "";

                    if (IsHeadline(line)) {
                        add(SegmentKind::Summary, line.substr(2, line.size() - 4) + suffix, segment.revision);
                    } else {
                        add(SegmentKind::Reasoning, line + suffix, segment.revision);
                    }

                    if (complete) {
                        start = end + 1;
                    } else {
                        start = segment.text.size();
                    }
                }
            } else if (segment.kind == SegmentKind::Note) {
                add(SegmentKind::Note, segment.label + "\n", segment.revision);
            } else {
                view.push_back(segment);
            }
        }

        return view;
    }

    static std::string QuoteBlock(const std::string& body) {
        auto lines = Text::SplitLines(body);
        for (auto& line : lines) {
            line = "> " + line;
        }
        return Text::JoinLines(lines);
    }

    static std::string ToolDisplayName(const std::string& name) {
        std::string display;
        bool wordStart = true;
        for (const char c : name) {
            if (c == '_' || c == '-' || c == ' ') {
                wordStart = true;
                continue;
            }
            if (wordStart && !display.empty()) {
                display += ' ';
            }

            if (wordStart) {
                display += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            } else {
                display += c;
            }
            wordStart = false;
        }

        return display;
    }

    static std::string NotesMarkdown(const std::string& text) {
        std::string notes;
        for (const auto& line : Text::SplitLines(text)) {
            if (line.empty()) {
                continue;
            }
            if (!notes.empty()) {
                notes += "\n";
            }
            notes += jst::fmt::format("using ***{}***", ToolDisplayName(line));
        }
        return notes;
    }

    static std::string ResponseMarkdown(const Segments& segments, U64 begin, U64 end) {
        std::string markdown;
        for (U64 i = begin; i < end; ++i) {
            const auto& segment = segments[i];
            if (segment.kind == SegmentKind::Reasoning || segment.kind == SegmentKind::Summary) {
                continue;
            }
            if (segment.kind == SegmentKind::Note) {
                Text::AppendParagraph(markdown, NotesMarkdown(segment.text));
                continue;
            }
            if (segment.kind == SegmentKind::Error) {
                Text::AppendParagraph(markdown, "> [!ERROR]\n" + QuoteBlock(segment.text));
                continue;
            }
            Text::AppendParagraph(markdown, segment.text);
        }
        return markdown;
    }

    static void AppendHeadlines(std::string& markdown, const std::string& text) {
        for (const auto& line : Text::SplitLines(text)) {
            if (!line.empty()) {
                Text::AppendParagraph(markdown, jst::fmt::format("**{}**", line));
            }
        }
    }

    static std::string ThoughtMarkdown(const Segments& segments, U64 begin, U64 end) {
        std::string markdown;
        for (U64 i = begin; i < end; ++i) {
            const auto& segment = segments[i];
            if (segment.kind == SegmentKind::Reasoning) {
                const std::string text = Text::TrimBlank(segment.text);
                if (!text.empty()) {
                    Text::AppendParagraph(markdown, QuoteBlock(text));
                }
            } else if (segment.kind == SegmentKind::Note) {
                Text::AppendParagraph(markdown, NotesMarkdown(segment.text));
            } else if (segment.kind == SegmentKind::Summary) {
                AppendHeadlines(markdown, segment.text);
            } else if (segment.kind == SegmentKind::Text) {
                Text::AppendParagraph(markdown, segment.text);
            }
        }
        return markdown;
    }

    static U64 SegmentsRevision(const Segments& segments, U64 begin, U64 end) {
        U64 revision = 1469598103934665603ULL;
        for (U64 i = begin; i < end; ++i) {
            revision ^= segments[i].revision + (static_cast<U64>(segments[i].kind) << 56U);
            revision *= 1099511628211ULL;
        }
        revision ^= end - begin;
        return revision;
    }

    static std::string ThoughtHeadline(const Segments& segments, U64 begin) {
        if (begin >= segments.size() || segments[begin].kind != SegmentKind::Summary) {
            return {};
        }

        const auto& lines = segments[begin].text;
        const U64 lineEnd = lines.find_last_not_of('\n');
        if (lineEnd == std::string::npos) {
            return {};
        }

        const U64 lineStart = lines.rfind('\n', lineEnd);
        const U64 start = lineStart == std::string::npos ? 0 : lineStart + 1;
        return lines.substr(start, lineEnd - start + 1);
    }

    static std::string ThoughtLabel(const std::string& headline, bool busy) {
        if (busy) {
            if (headline.empty()) {
                return "Thinking...";
            }
            return jst::fmt::format("Thinking: {}...", headline);
        }

        if (headline.empty()) {
            return "Thought";
        }
        return jst::fmt::format("Thought: {}", headline);
    }

    static std::vector<std::pair<std::string, U64>> ToolCounts(const Segments& segments, U64 begin, U64 end) {
        std::vector<std::pair<std::string, U64>> counts;
        for (U64 g = begin; g < end; ++g) {
            if (segments[g].kind != SegmentKind::Note) {
                continue;
            }
            for (const auto& line : Text::SplitLines(segments[g].text)) {
                if (line.empty()) {
                    continue;
                }

                const auto it = std::find_if(counts.begin(), counts.end(), [&](const auto& entry) {
                    return entry.first == line;
                });
                if (it == counts.end()) {
                    counts.push_back({line, 1});
                } else {
                    it->second++;
                }
            }
        }

        return counts;
    }

    static std::string ToolSummary(const Segments& segments, U64 begin, U64 end) {
        const auto counts = ToolCounts(segments, begin, end);
        U64 toolCalls = 0;
        for (const auto& [_, calls] : counts) {
            toolCalls += calls;
        }
        if (toolCalls == 0) {
            return {};
        }

        const char* plural = toolCalls == 1 ? "" : "s";
        std::string summary = jst::fmt::format("Called {} tool{}: ", toolCalls, plural);
        for (U64 t = 0; t < counts.size(); ++t) {
            const auto& [name, calls] = counts[t];
            if (t > 0) {
                summary += ", ";
            }

            if (calls > 1) {
                summary += jst::fmt::format("{} ({})", ToolDisplayName(name), calls);
            } else {
                summary += ToolDisplayName(name);
            }
        }

        return summary;
    }

    struct PartBuilder {
        const Segments& segments;
        bool busyTail = false;
        std::vector<Chat::Part> parts;

        static bool isResponse(const Agent::Session::Segment& segment) {
            return segment.kind == SegmentKind::Text ||
                   segment.kind == SegmentKind::Error ||
                   segment.kind == SegmentKind::Status;
        }

        static bool isVisibleText(const Agent::Session::Segment& segment) {
            return isResponse(segment) && !Text::IsBlank(segment.text);
        }

        void addMarkdown(std::string text, U64 begin, U64 end) {
            parts.push_back({
                .kind = Chat::PartKind::Markdown,
                .key = begin,
                .revision = SegmentsRevision(segments, begin, end),
                .text = std::move(text),
            });
        }

        void addThought(U64 begin, U64 end) {
            const bool busy = busyTail && end == segments.size();
            parts.push_back({
                .kind = Chat::PartKind::Disclosure,
                .key = begin,
                .revision = SegmentsRevision(segments, begin, end),
                .text = ThoughtMarkdown(segments, begin, end),
                .label = ThoughtLabel(ThoughtHeadline(segments, begin), busy),
                .summary = ToolSummary(segments, begin, end),
                .busy = busy,
                .expandable = begin < segments.size(),
            });
        }

        U64 addResponseRun(U64 begin) {
            if (!isVisibleText(segments[begin])) {
                return begin + 1;
            }

            U64 end = begin;
            while (end < segments.size() && isResponse(segments[end])) {
                end++;
            }

            std::string text = ResponseMarkdown(segments, begin, end);
            if (!text.empty()) {
                addMarkdown(std::move(text), begin, end);
            }
            return end;
        }

        U64 addThoughtGroup(U64 begin) {
            U64 end = begin;
            while (end < segments.size() && !isVisibleText(segments[end])) {
                end++;
            }

            U64 start = begin;
            for (U64 g = begin + 1; g < end; ++g) {
                if (segments[g].kind == SegmentKind::Summary) {
                    addThought(start, g);
                    start = g;
                }
            }

            addThought(start, end);
            return end;
        }

        std::vector<Chat::Part> build() {
            U64 cursor = 0;
            while (cursor < segments.size()) {
                if (isResponse(segments[cursor])) {
                    cursor = addResponseRun(cursor);
                } else {
                    cursor = addThoughtGroup(cursor);
                }
            }

            if (busyTail && parts.empty()) {
                addThought(segments.size(), segments.size());
            } else if (parts.empty()) {
                addMarkdown("", 0, 0);
            }

            return std::move(parts);
        }
    };

    static Chat::Message DisplayMessage(const Agent::Session::Entry& message, bool busyTail) {
        if (message.user) {
            return {Chat::Role::User, message.text, {}};
        }

        const auto segments = DisplaySegments(message.segments);
        PartBuilder builder{
            .segments = segments,
            .busyTail = busyTail,
        };

        return {Chat::Role::Assistant, {}, builder.build()};
    }

    U64 generation = 0;
    std::vector<Entry> entries;
    Sakura::Retained::Chat::Snapshot snapshot;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_AGENT_TRANSCRIPT_HH
