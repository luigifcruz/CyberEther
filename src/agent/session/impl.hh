#ifndef JETSTREAM_AGENT_SESSION_IMPL_HH
#define JETSTREAM_AGENT_SESSION_IMPL_HH

#include "support/base.hh"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace Jetstream::Agent {

struct Session::Impl {
    class Turn;

    struct Target {
        U64 assistantIndex = 0;
        U64 generation = 0;
    };

    struct CallLog {
        U64 id;
        std::shared_ptr<std::atomic<bool>> cancelled;
        std::optional<U64> openMessage;
        bool openUser = false;
    };

    struct Call {
        U64 id = 0;
        Live::Config config;
        Live::Callbacks callbacks;
        std::shared_ptr<std::atomic<bool>> cancelled;
        Cancellation aborted;
        std::unique_ptr<Audio> audio;
        Live* live = nullptr;
        std::unique_ptr<Detail::ToolRunner> tools;
        U64 resultLimit = 0;

        std::atomic<Live::State> state{Live::State::Connecting};
        std::atomic<bool> delegating{false};
        std::atomic<bool> closing{false};
        std::atomic<F32> input{0.0f};
        std::atomic<F32> output{0.0f};

        std::mutex mutex;
        std::condition_variable changed;
        bool stopping = false;
        bool finished = false;
        std::vector<std::string> errors;
        std::deque<std::vector<Provider::FunctionCall>> batches;

        void close();
    };

    mutable std::mutex mutex;
    std::vector<std::shared_ptr<Entry>> messages;
    U64 revision = 1;
    std::thread worker;
    std::atomic<bool> streaming{false};
    std::optional<CallLog> callLog;
    U64 nextVoiceId = 0;
    std::shared_ptr<std::atomic<bool>> abortRequested = std::make_shared<std::atomic<bool>>(false);
    U64 generation = 0;
    U64 contextTokens = 0;
    std::string sessionId;
    std::optional<Mode> mode;
    std::shared_ptr<const std::vector<Provider::Message>> context;
    std::shared_ptr<Detail::OutputStore> outputs = std::make_shared<Detail::OutputStore>();

    LiveFactory liveFactory;
    AudioFactory audioFactory;
    std::shared_ptr<Call> call;
    std::mutex callMutex;
    std::condition_variable callChanged;
    std::deque<std::shared_ptr<Call>> calls;
    bool quitting = false;
    std::thread connector;

    Impl(LiveFactory live, AudioFactory audio);

    void setMode(Mode next);
    void reset();
    void retire();

    bool owns(Target target) const;
    void append(Target target, Segment::Kind kind, const std::string& chunk);
    void add(Target target, Segment segment);
    void record(Target target, const Provider::Usage& usage);
    void publish(Target target, const Turn& turn);

    std::optional<U64> beginVoice(std::shared_ptr<std::atomic<bool>> cancelled);
    void appendVoiceTranscript(U64 id, bool user, const std::string& delta);
    void finishVoiceTranscript(U64 id, bool user);
    void finishVoiceTool(U64 id);
    void endVoice(U64 id);
    bool voiceActive() const;

    Live::Callbacks callbacks(const std::shared_ptr<Call>& call);
    void pump(Call& call);
    void runTools(Call& call);
    void connect(Call& call);
    void serve();
    void disconnect();
    void stop();

 private:
    CallLog* findLog(U64 id);
    CallLog* activeLog(U64 id);
};

class Session::Impl::Turn {
 public:
    Turn(Impl& session, Target target, Config config, std::shared_ptr<std::atomic<bool>> abort,
         std::shared_ptr<Detail::OutputStore> outputs);

    void run(std::shared_ptr<const std::vector<Provider::Message>> context, const std::string& prompt,
             const std::string& sessionId);

    const std::vector<Provider::Message>& history() const;

 private:
    void loop();
    void fail(Provider::Error::Kind kind, std::string message);
    void finish(Provider::Response& response);
    void cancel();
    void append(Provider::Message message);
    void complete();

    U64 outputLimit() const;
    bool bounded() const;
    U64 resultLimit() const;
    U64 budget() const;

    Provider::Response request(const std::vector<Provider::Message>& messages, bool allowTools, bool summary = false);
    std::optional<std::string> summarize(std::string_view source, U64 targetBudget);
    bool compact(U64 targetBudget);
    Provider::Response stream(bool allowTools = true);
    void execute(const std::vector<Provider::FunctionCall>& calls, bool lastRound);

    void write(Segment::Kind kind, const std::string& text);
    void note(Segment segment);
    void announce(const std::string& name);

    Impl& session;
    Target target;
    Config config;
    std::shared_ptr<std::atomic<bool>> abort;
    Cancellation aborted;
    Detail::ToolRunner tools;
    Provider::Callbacks callbacks;
    U64 announced = 0;
    bool streamed = false;
    bool prepared = false;

    std::vector<Provider::Message> messages;
    std::string sessionId;
    std::string finalText;
    std::shared_ptr<const Provider::State> finalState;
    Result result = Result::SUCCESS;
    Provider::Error error;
};

}  // namespace Jetstream::Agent

#endif  // JETSTREAM_AGENT_SESSION_IMPL_HH
