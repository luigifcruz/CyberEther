#include "base.hh"

#include "jetstream/config.hh"

namespace Jetstream::Agent::Detail {

namespace {

Tools::Output Failure(const char* json) {
    return {json, {}, true};
}

}  // namespace

ToolRunner::ToolRunner(Session::ToolExecutor execute, std::shared_ptr<OutputStore> outputs, U64 maxRounds)
    : execute(std::move(execute)), outputs(std::move(outputs)), maxRounds(maxRounds) {}

bool ToolRunner::advance() {
    return rounds.fetch_add(1) + 1 >= maxRounds;
}

bool ToolRunner::exhausted() const {
    return rounds.load() >= maxRounds;
}

void ToolRunner::reset() {
    rounds.store(0);
}

Tools::Output ToolRunner::run(const Provider::FunctionCall& call, U64 resultLimit, bool lastRound,
                              const Cancellation& cancel) const {
    Tools::Output result;
    if (lastRound) {
        result = Failure(R"({"error":"tool-step budget reached; summarize what remains"})");
    } else if (call.name == OutputStore::Schema().name) {
        result = outputs->read(call.arguments, resultLimit);
    } else if (!execute) {
        result = Failure(R"({"error":"tools are unavailable"})");
    } else {
        result = execute(call.name, call.arguments, cancel);
    }

    result.content = outputs->bound(std::move(result.content), resultLimit, result.error);
    return result;
}

std::optional<Provider::Message> ToolRunner::Attachment(const Provider::FunctionCall& call,
                                                        const Tools::Output& result) {
    if (result.images.empty()) {
        return std::nullopt;
    }
    return Provider::Message{"user", call.caption(), {}, "", result.images};
}

}  // namespace Jetstream::Agent::Detail
