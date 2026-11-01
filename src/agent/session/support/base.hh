#ifndef JETSTREAM_AGENT_SESSION_SUPPORT_BASE_HH
#define JETSTREAM_AGENT_SESSION_SUPPORT_BASE_HH

#include "jetstream/agent/session.hh"

#include <atomic>
#include <deque>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace Jetstream::Agent::Detail {

class OutputStore {
 public:
    OutputStore();

    static Provider::Function Schema();
    std::string bound(std::string content, U64 limit, bool failed);
    Tools::Output read(const std::string& args, U64 maxBytes = 32768) const;

 private:
    std::deque<std::pair<std::string, std::string>> entries;
    U64 bytes = 0;
    U64 next = 0;
    const std::string identity;
};

class ToolRunner {
 public:
    ToolRunner(Session::ToolExecutor execute, std::shared_ptr<OutputStore> outputs);

    Tools::Output run(const Provider::FunctionCall& call, U64 resultLimit, const Cancellation& cancel) const;

    static std::optional<Provider::Message> Attachment(const Provider::FunctionCall& call,
                                                       const Tools::Output& result);

 private:
    Session::ToolExecutor execute;
    std::shared_ptr<OutputStore> outputs;
};

class Context {
 public:
    struct Compaction {
        std::string source;
        std::vector<Provider::Message> retained;
    };

    static std::vector<Provider::Message> Materialize(
        const std::string& systemContext, const std::shared_ptr<const std::vector<Provider::Message>>& context,
        const std::string& prompt, const Cancellation& shouldAbort);
    static std::vector<Provider::Message> Close(std::vector<Provider::Message> messages);
    static U64 Estimate(const Provider::Message& message);
    static U64 Estimate(const std::vector<Provider::Message>& messages, std::span<const Provider::Function> tools = {});
    static void Trim(std::vector<Provider::Message>& messages, U64 keep);
    static std::optional<Compaction> Plan(const std::vector<Provider::Message>& history, U64 budget);
    static std::vector<Provider::Message> Summary(std::string_view source, const std::string& checkpoint,
                                                  std::span<const Provider::Function> tools, U64 budget);
};

}  // namespace Jetstream::Agent::Detail

#endif  // JETSTREAM_AGENT_SESSION_SUPPORT_BASE_HH
