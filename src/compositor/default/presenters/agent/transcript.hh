#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_AGENT_TRANSCRIPT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_AGENT_TRANSCRIPT_HH

#include "../../model/state.hh"

#include "jetstream/config.hh"
#include "jetstream/render/sakura/components/chat.hh"
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
    const Sakura::Chat::Snapshot& format(const DefaultCompositorState::AgentState::Conversation& conversation);

 private:
    using Chat = Sakura::Chat;
    using Segments = std::vector<Agent::Session::Segment>;
    using SegmentKind = Agent::Session::Segment::Kind;

    struct Entry {
        std::shared_ptr<const Agent::Session::Entry> source;
        bool busy = false;
        std::shared_ptr<const Sakura::Chat::Message> display;
    };

    static bool IsHeadline(const std::string& line);
    static Segments DisplaySegments(const Segments& segments);
    static std::string QuoteBlock(const std::string& body);
    static std::string ToolDisplayName(const std::string& name);
    static std::string NotesMarkdown(const std::string& text);
    static std::string ResponseMarkdown(const Segments& segments, U64 begin, U64 end);
    static void AppendHeadlines(std::string& markdown, const std::string& text);
    static std::string ThoughtMarkdown(const Segments& segments, U64 begin, U64 end);
    static U64 SegmentsRevision(const Segments& segments, U64 begin, U64 end);
    static std::string ThoughtHeadline(const Segments& segments, U64 begin);
    static std::string ThoughtLabel(const std::string& headline, bool busy);
    static std::vector<std::pair<std::string, U64>> ToolCounts(const Segments& segments, U64 begin, U64 end);
    static std::string ToolSummary(const Segments& segments, U64 begin, U64 end);

    struct PartBuilder {
        const Segments& segments;
        bool busyTail = false;
        std::vector<Chat::Part> parts;

        static bool isResponse(const Agent::Session::Segment& segment);
        static bool isVisibleText(const Agent::Session::Segment& segment);
        void addMarkdown(std::string text, U64 begin, U64 end);
        void addThought(U64 begin, U64 end);
        U64 addResponseRun(U64 begin);
        U64 addThoughtGroup(U64 begin);
        std::vector<Chat::Part> build();
    };

    static Chat::Message DisplayMessage(const Agent::Session::Entry& message, bool busyTail);

    U64 generation = 0;
    std::vector<Entry> entries;
    Sakura::Chat::Snapshot snapshot;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_AGENT_TRANSCRIPT_HH
