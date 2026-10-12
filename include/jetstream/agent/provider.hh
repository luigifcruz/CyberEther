#ifndef JETSTREAM_AGENT_PROVIDER_HH
#define JETSTREAM_AGENT_PROVIDER_HH

#include "jetstream/types.hh"

#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Jetstream::Agent {

using Cancellation = std::function<bool()>;

class JETSTREAM_API Provider {
 public:
    enum class Kind {
        OpenAI,
        OpenRouter,
        OpenCodeGo,
        Fireworks,
        Custom,
    };

    enum class Api {
        ChatCompletions,
        Responses,
    };

    enum class Scope {
        Models,
        Chat,
    };

    struct Error {
        enum class Kind {
            None,
            Configuration,
            MissingCredentials,
            Authentication,
            Permission,
            NotFound,
            RateLimit,
            Quota,
            InvalidRequest,
            ContextLength,
            Timeout,
            Connection,
            Tls,
            Server,
            Protocol,
            Cancelled,
            Unsupported,
            OutputLimit,
        };

        Kind kind = Kind::None;
        I32 status = 0;
        std::string message;
        std::string code;
        std::string retryAfter;

        explicit operator bool() const {
            return kind != Kind::None;
        }
    };

    struct Usage {
        U64 inputTokens = 0;
        U64 outputTokens = 0;
        U64 cachedTokens = 0;
        U64 cacheWriteTokens = 0;
        U64 reasoningTokens = 0;
        std::optional<F64> cost;
    };

    struct Model {
        std::string id;
        std::string name;
        U64 contextLength = 0;
        U64 outputLength = 0;
        std::vector<std::string> reasoningEfforts;
    };

    struct Function {
        std::string name;
        std::string description;
        std::string parametersJson;
    };

    struct FunctionCall {
        std::string id;
        std::string name;
        std::string arguments;

        std::string caption() const {
            return "Image captured by the '" + name + "' tool call.";
        }
    };

    struct State {
        std::string host;
        std::string path;
        std::string model;
        std::string payload;
    };

    struct Message {
        std::string role;
        std::string content;
        std::vector<FunctionCall> toolCalls;
        std::string toolCallId;
        std::vector<std::string> images;
        std::shared_ptr<const State> state;
        bool error = false;
    };

    struct JETSTREAM_API Profile {
        Kind kind = Kind::Custom;
        std::string name;
        std::string description;
        std::string host;
        Api api = Api::ChatCompletions;
        std::string chatPath = "/v1/chat/completions";
        std::string modelsPath = "/v1/models";
        std::string livePath;
        std::string catalogId;
        bool requestReasoning = false;
        bool anonymousModels = false;
        bool anonymousChat = false;
        bool promptCacheKey = false;
        bool nestedReasoningEffort = false;
        std::vector<std::string> cacheBreakpointModels;
        std::string sessionHeader;
        std::vector<std::pair<std::string, std::string>> headers;

        bool ready(Scope scope, bool hasKey) const;

        static std::span<const Profile> All();
        static const Profile& Find(Kind kind);
        static Kind Identify(const std::string& host, const std::string& chatPath);
    };

    struct Config {
        std::string name;
        Kind kind = Kind::Custom;
        std::string host;
        Api api = Api::ChatCompletions;
        std::string path;
        std::string modelsPath;
        std::function<std::string()> apiKey;
        U64 timeoutSeconds = 120;
        U32 retries = 0;
        bool requestReasoning = false;
        std::string userAgent;
    };

    struct Request {
        const std::vector<Message>& messages;
        std::span<const Function> tools = {};
        bool allowTools = true;
        U64 maxOutputTokens = 0;
        std::string model;
        std::string reasoningEffort;
        std::string reasoningContext;
        std::string sessionId;
    };

    struct Callbacks {
        std::function<void(const std::string& token)> onToken;
        std::function<void(const std::string& reasoning, bool summary)> onReasoning;
        std::function<void(const std::string& name)> onToolCall;
        Cancellation shouldAbort;
    };

    struct Response {
        std::string text;
        std::vector<FunctionCall> toolCalls;
        Usage usage;
        Error error;
        std::shared_ptr<const State> state;
    };

    class JETSTREAM_API Catalog {
     public:
        Result fetch(Error* error = nullptr);
        std::optional<Model> find(Kind kind, const std::string& id) const;

     private:
        std::unordered_map<std::string, Model> entries;
    };

    static const char* Name(Api api);
    static Api Parse(const std::string& name, Api fallback);

    explicit Provider(Config config);
    virtual ~Provider();

    const Config& config() const;
    const Profile& profile() const;

    virtual Result models(std::vector<Model>& models, Error* error = nullptr) const;
    virtual Result stream(const Request& request, const Callbacks& callbacks, Response& response) const;

 private:
    Config configuration;
};

}  // namespace Jetstream::Agent

#endif  // JETSTREAM_AGENT_PROVIDER_HH
