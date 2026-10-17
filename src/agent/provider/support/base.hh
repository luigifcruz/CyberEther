#ifndef JETSTREAM_AGENT_PROVIDER_SUPPORT_BASE_HH
#define JETSTREAM_AGENT_PROVIDER_SUPPORT_BASE_HH

#include "jetstream/agent/provider.hh"
#include "jetstream/config.hh"
#include "jetstream/platform.hh"

#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace Jetstream::Agent::Detail {

class Diagnostics {
 public:
    static constexpr size_t MaxLength = 2048;

    static std::string Flatten(std::string_view value);
    static std::string Redact(std::string text, std::string_view secret);
    static void Sanitize(Provider::Error& failure, std::string_view secret);
    static Result Report(Provider::Error failure, Provider::Error* output, const std::string& context,
                         std::string_view secret);
    static std::string MissingKey();

    static Provider::Error Payload(const nlohmann::json& payload, I32 status = 0, const std::string& retryAfter = {});
    static Provider::Error Http(I32 status, const std::string& body, const std::string& retryAfter);
    static Provider::Error Transport(const Platform::Http::Error& failure);
    static Provider::Error Response(const Platform::Http::Response& response, nlohmann::json& body);
};

class Retry {
 public:
    static constexpr auto MaxAfter = std::chrono::seconds(120);

    static std::optional<std::chrono::milliseconds> After(std::string_view value,
                                                          std::chrono::system_clock::time_point now);
    static JETSTREAM_API std::optional<std::chrono::milliseconds> Delay(
        const Provider::Error& failure, U32 attempt, U32 maxRetries,
        std::chrono::system_clock::time_point now = std::chrono::system_clock::now());
    static bool Wait(std::chrono::milliseconds delay, const Cancellation& cancelled);
};

class Wire {
 public:
    static std::string Field(const nlohmann::json& object, const char* name);
    static std::string Error(const nlohmann::json& payload);
    static nlohmann::json Replay(const Provider::Message& message, const std::string& host,
                                 const std::string& path, const std::string& model);
    static bool Empty(const Provider::Message& message, const nlohmann::json& state);
    static nlohmann::json Function(const Provider::Function& tool);
    static nlohmann::json FunctionCall(const Provider::FunctionCall& call);
};

class EventStream {
 public:
    struct UsageKeys {
        const char* input;
        const char* output;
        const char* inputDetails;
        const char* outputDetails;
    };

    static constexpr const char* Truncated = "The response reached the output token limit and was cut off.";

    static constexpr UsageKeys ResponsesUsage = {
        "input_tokens", "output_tokens", "input_tokens_details", "output_tokens_details",
    };

    static constexpr UsageKeys ChatCompletionsUsage = {
        "prompt_tokens", "completion_tokens", "prompt_tokens_details", "completion_tokens_details",
    };

    std::function<void(const std::string&)> onToken;
    std::function<void(const std::string&, bool)> onReasoning;
    std::function<void(const std::string&)> onToolCall;
    std::vector<Provider::FunctionCall> toolCalls;
    std::string text;
    Provider::Usage usage;
    Provider::Error error;
    bool outputStarted = false;

    virtual ~EventStream() = default;

    bool append(std::string_view bytes);
    bool finish();
    bool feed(const nlohmann::json& event);
    bool settle();
    virtual std::string payload() const = 0;

 protected:
    virtual bool decode(const nlohmann::json& chunk) = 0;
    virtual bool complete() = 0;

    bool fail(Provider::Error::Kind kind, std::string message);
    void token(const std::string& value);
    void announce(const std::string& name);
    void reasoning(const std::string& value, I64 index, bool summary);
    void readUsage(const nlohmann::json& object, const UsageKeys& keys);

    I64 lastReasoningIndex = -1;
    bool sawChoice = false;
    bool finishedChoice = false;
    bool truncated = false;

 private:
    bool line(std::string value);
    bool dispatch();

    std::string pending;
    std::string data;
    std::string event;
    char lastReasoningChar = '\0';
    bool done = false;
};

class ChatCompletions {
 public:
    static nlohmann::json Encode(const Provider::Config& config, const Provider::Request& request, bool reasoning);
    static std::unique_ptr<EventStream> Decoder();
};

class Responses {
 public:
    static nlohmann::json Encode(const Provider::Config& config, const Provider::Request& request, bool reasoning);
    static std::unique_ptr<EventStream> Decoder();

    static nlohmann::json Function(const Provider::Function& tool);
    static nlohmann::json FunctionOutput(const std::string& callId, const std::string& output);
    static nlohmann::json InputContent(const std::string& text, const std::vector<std::string>& images);
    static std::optional<Provider::FunctionCall> FunctionCall(const nlohmann::json& item);
    static const nlohmann::json* Failed(const nlohmann::json& event);
};

}  // namespace Jetstream::Agent::Detail

#endif  // JETSTREAM_AGENT_PROVIDER_SUPPORT_BASE_HH
