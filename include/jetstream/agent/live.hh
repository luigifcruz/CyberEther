#ifndef JETSTREAM_AGENT_LIVE_HH
#define JETSTREAM_AGENT_LIVE_HH

#include "jetstream/agent/provider.hh"
#include "jetstream/agent/tools.hh"

#include <functional>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream::Agent {

class JETSTREAM_API Live {
 public:
    enum class State {
        Idle,
        Connecting,
        Listening,
        Closed,
    };

    struct Config {
        std::shared_ptr<const Provider> provider;
        std::string model;
        std::string backendModel;
        std::string voice;
        std::string instructions;
        std::string backendInstructions;
        U32 sampleRate = 24000;
        std::vector<Provider::Function> tools;
    };

    struct Callbacks {
        std::function<void(State state, const std::string& detail)> onState;
        std::function<void(const std::string& message)> onError;
        std::function<void(const std::string& delta)> onInputTranscript;
        std::function<void(const std::string& delta)> onOutputTranscript;
        std::function<void()> onInputTranscriptDone;
        std::function<void()> onOutputTranscriptDone;
        std::function<void(bool active)> onDelegation;
        std::function<void(const std::string& base64)> onAudio;
        std::function<void()> onAudioClear;
        std::function<void(std::vector<Provider::FunctionCall> calls)> onFunctionCalls;
    };

    using ToolOutputs = std::vector<std::pair<Provider::FunctionCall, Tools::Output>>;

    Live(Config config, Callbacks callbacks);
    virtual ~Live();

    Live(const Live&) = delete;
    Live& operator=(const Live&) = delete;

    virtual Result start();
    virtual void stop();
    virtual void send(const std::vector<I16>& samples);
    virtual void reply(const ToolOutputs& outputs, bool disableTools);

    static bool Available();
    static std::span<const std::string> Models();
    static std::span<const std::string> Voices();

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Agent

#endif  // JETSTREAM_AGENT_LIVE_HH
