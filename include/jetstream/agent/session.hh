#ifndef JETSTREAM_AGENT_SESSION_HH
#define JETSTREAM_AGENT_SESSION_HH

#include "jetstream/agent/live.hh"
#include "jetstream/agent/provider.hh"
#include "jetstream/agent/tools.hh"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Agent {

class JETSTREAM_API Session {
 public:
    using ToolExecutor = std::function<Tools::Output(const std::string&, const std::string&, const Cancellation&)>;

    enum class Mode { Text, Voice };

    enum class Phase {
        Idle,
        Connecting,
        Listening,
        Delegating,
    };

    struct Config {
        struct Voice {
            std::string model;
            std::string name;
            std::string instructions;
        };

        std::shared_ptr<const Provider> provider;
        std::string model;
        std::string reasoningEffort;
        std::string reasoningContext;
        std::string systemContext;
        std::vector<Provider::Function> tools;
        ToolExecutor executeTool;
        U64 contextLength = 0;
        U64 maxOutputTokens = 0;
        U64 modelOutputTokens = 0;
        U64 maxToolResultChars = 32768;
        U64 maxImages = 4;
        Voice voice;
    };

    struct Segment {
        enum class Kind {
            Reasoning,
            Note,
            Summary,
            Text,
            Error,
            ToolCall,
            ToolResult,
            Status,
        };

        Kind kind;
        std::string text;
        U64 revision = 0;
        std::string label;
        U64 images = 0;
    };

    struct Entry {
        bool user = false;
        std::string text;
        std::vector<Segment> segments;
    };

    struct Snapshot {
        U64 generation = 0;
        bool streaming = false;
        std::vector<Entry> messages;
        U64 contextTokens = 0;
    };

    struct Status {
        Phase phase = Phase::Idle;
        F32 input = 0.0f;
        F32 output = 0.0f;
    };

    struct Cursor {
        U64 generation = 0;
        U64 revision = 0;
        U64 count = 0;
    };

    struct Update : Snapshot {
        U64 firstMessage = 0;
    };

    class JETSTREAM_API Audio {
     public:
        explicit Audio(U32 sampleRate);
        virtual ~Audio();

        Audio(const Audio&) = delete;
        Audio& operator=(const Audio&) = delete;

        virtual Result start();
        virtual void stop();
        virtual std::vector<I16> capture(U64 minimumSamples);
        virtual void play(const std::string& base64);
        virtual void flush();
        virtual F32 input() const;
        virtual F32 output() const;

     private:
        struct Impl;
        std::unique_ptr<Impl> impl;
    };

    using LiveFactory = std::function<std::unique_ptr<Live>(Live::Config config, Live::Callbacks callbacks)>;
    using AudioFactory = std::function<std::unique_ptr<Audio>(U32 sampleRate)>;

    explicit Session(LiveFactory live = {}, AudioFactory audio = {});
    ~Session();

    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    void setMode(Mode mode);

    bool submit(const std::string& prompt, const Config& config);
    bool listen(const Config& config);
    void hangup();
    void cancel();
    void clear();
    void shutdown();
    std::vector<std::string> poll();

    Snapshot snapshot() const;
    std::optional<Update> updates(Cursor& cursor) const;
    Status status() const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Agent

#endif  // JETSTREAM_AGENT_SESSION_HH
