#include "impl.hh"

#include <algorithm>
#include <utility>

namespace Jetstream::Agent {

namespace {

constexpr U64 kFallbackOutputTokens = 32768;

Provider::Callbacks Observed(Provider::Callbacks callbacks, const std::function<void()>& mark) {
    const auto wrap = [&mark](auto inner) {
        return [inner = std::move(inner), &mark](const auto&... args) {
            mark();
            if (inner) {
                inner(args...);
            }
        };
    };

    callbacks.onToken = wrap(std::move(callbacks.onToken));
    callbacks.onReasoning = wrap(std::move(callbacks.onReasoning));
    callbacks.onToolCall = wrap(std::move(callbacks.onToolCall));

    return callbacks;
}

std::string ErrorText(const Provider::Error& error) {
    if (error.kind != Provider::Error::Kind::OutputLimit) {
        return error.message;
    }
    return error.message + " Reasoning and the answer share this budget. Increase maxOutputTokens "
           "or set it to 0 for the automatic model allowance, then continue the task.";
}

}  // namespace

Session::Impl::Turn::Turn(Impl& session, Target target, Config config, std::shared_ptr<std::atomic<bool>> abort,
                          std::shared_ptr<Detail::OutputStore> outputs)
    : session(session),
      target(target),
      config(std::move(config)),
      abort(std::move(abort)),
      aborted([flag = this->abort]() {
          return flag->load();
      }),
      tools(this->config.executeTool, std::move(outputs), this->config.maxToolTurns) {
    if (!this->config.tools.empty()) {
        this->config.tools.push_back(Detail::OutputStore::Schema());
    }

    callbacks.onToken = [this](const std::string& token) {
        write(Segment::Kind::Text, token);
    };
    callbacks.onReasoning = [this](const std::string& thought, const bool summary) {
        write(summary ? Segment::Kind::Summary : Segment::Kind::Reasoning, thought);
    };
    callbacks.onToolCall = [this](const std::string& name) {
        announce(name);
        ++announced;
    };
    callbacks.shouldAbort = aborted;
}

void Session::Impl::Turn::run(std::shared_ptr<const std::vector<Provider::Message>> context,
                              const std::string& prompt, const std::string& id) {
    sessionId = id;

    try {
        messages = Detail::Context::Materialize(config.systemContext, context, prompt, aborted);
        prepared = !aborted();
        context.reset();

        if (config.provider) {
            loop();
        } else {
            fail(Provider::Error::Kind::Configuration, "No chat provider is configured.");
        }
    } catch (const std::exception& exception) {
        fail(Provider::Error::Kind::Server, exception.what());
    } catch (...) {
        fail(Provider::Error::Kind::Server, "The agent request failed with an unknown error.");
    }

    if (result != Result::SUCCESS && !aborted()) {
        write(Segment::Kind::Error, ErrorText(error));
    }
    if (aborted()) {
        cancel();
    }

    complete();
    if (prepared) {
        session.publish(target, *this);
    }
}

const std::vector<Provider::Message>& Session::Impl::Turn::history() const {
    return messages;
}

void Session::Impl::Turn::loop() {
    while (!tools.exhausted()) {
        if (aborted()) {
            result = Result::ERROR;
            return;
        }

        auto response = stream();
        if (result != Result::SUCCESS || response.toolCalls.empty()) {
            finish(response);
            return;
        }

        append({
            .role = "assistant",
            .content = std::move(response.text),
            .toolCalls = response.toolCalls,
            .state = std::move(response.state),
        });
        execute(response.toolCalls, tools.advance());
    }

    if (result == Result::SUCCESS && !aborted()) {
        auto response = stream(false);
        finish(response);
    }
}

void Session::Impl::Turn::fail(const Provider::Error::Kind kind, std::string message) {
    result = Result::ERROR;
    error = {kind, 0, std::move(message)};
}

void Session::Impl::Turn::finish(Provider::Response& response) {
    finalText = std::move(response.text);
    finalState = std::move(response.state);

    if (result != Result::SUCCESS) {
        finalState.reset();

        if (!finalText.empty()) {
            finalText = "[Incomplete assistant response: interrupted or failed; "
                        "this is not a completion claim.]\n" + finalText;
        }
    }
}

void Session::Impl::Turn::cancel() {
    if (result != Result::SUCCESS) {
        return;
    }

    result = Result::ERROR;
    finalState.reset();

    if (!finalText.empty()) {
        finalText = "[Incomplete assistant response: cancelled.]\n" + finalText;
    }
}

void Session::Impl::Turn::append(Provider::Message message) {
    messages.push_back(std::move(message));
}

void Session::Impl::Turn::complete() {
    if (!finalText.empty() || finalState) {
        append({
            .role = "assistant",
            .content = finalText,
            .state = std::move(finalState),
        });
    }
}

U64 Session::Impl::Turn::outputLimit() const {
    const U64 modelLimit = config.modelOutputTokens ? config.modelOutputTokens : kFallbackOutputTokens;

    U64 requested = config.maxOutputTokens ? config.maxOutputTokens : modelLimit;
    if (config.modelOutputTokens) {
        requested = std::min(requested, config.modelOutputTokens);
    }

    const U64 window = bounded() ? config.contextLength : kFallbackOutputTokens;
    return std::max<U64>(1, std::min(requested, window / 2));
}

bool Session::Impl::Turn::bounded() const {
    return config.contextLength != 0;
}

U64 Session::Impl::Turn::resultLimit() const {
    if (!bounded()) {
        return std::max<U64>(1024, config.maxToolResultChars);
    }
    return std::max<U64>(1024, std::min(config.maxToolResultChars, budget() * 3 / 8));
}

U64 Session::Impl::Turn::budget() const {
    const auto limit = config.contextLength;
    return limit - std::min(limit, outputLimit() + limit / 10);
}

Provider::Response Session::Impl::Turn::request(const std::vector<Provider::Message>& input, bool allowTools, bool summary) {
    Provider::Response response;
    streamed = false;

    const std::function<void()> mark = [this] {
        streamed = true;
    };

    auto sink = Observed(summary ? Provider::Callbacks{} : callbacks, mark);
    sink.shouldAbort = aborted;

    const Provider::Request payload{
        .messages = input,
        .tools = config.tools,
        .allowTools = allowTools,
        .maxOutputTokens = outputLimit(),
        .model = config.model,
        .reasoningEffort = config.reasoningEffort,
        .reasoningContext = config.reasoningContext,
        .sessionId = sessionId,
    };
    result = config.provider->stream(payload, sink, response);
    error = response.error;

    if (!allowTools && !response.toolCalls.empty()) {
        fail(Provider::Error::Kind::Protocol, "The provider returned tool calls while execution was disabled.");
        response.state.reset();
    }

    if (!summary) {
        session.record(target, response.usage);
    }

    return response;
}

std::optional<std::string> Session::Impl::Turn::summarize(std::string_view source, U64 targetBudget) {
    std::string summary;
    while (!source.empty()) {
        if (aborted()) {
            return std::nullopt;
        }

        const auto input = Detail::Context::Summary(source, summary, config.tools, targetBudget);
        if (input.empty()) {
            fail(Provider::Error::Kind::ContextLength,
                 "The summary cannot fit alongside the remaining conversation. "
                 "History was preserved; use a larger-context model.");
            return std::nullopt;
        }

        auto response = request(input, false, true);
        if (result != Result::SUCCESS || aborted()) {
            return std::nullopt;
        }
        if (response.text.empty()) {
            fail(Provider::Error::Kind::Protocol, "The provider returned an empty conversation summary.");
            return std::nullopt;
        }

        source.remove_prefix(input.back().content.size());
        summary = std::move(response.text);
    }

    return summary;
}

bool Session::Impl::Turn::compact(U64 targetBudget) {
    auto input = Detail::Context::Plan(messages, targetBudget);
    if (!input) {
        fail(Provider::Error::Kind::ContextLength,
             "The current message, tools, or active exchange cannot fit in this "
             "model's context. Use a larger-context model or a smaller request.");
        return false;
    }

    write(Segment::Kind::Status, "Compacting earlier conversation context.\n");

    const auto summary = summarize(input->source, targetBudget);
    if (!summary) {
        return false;
    }

    const U64 summaryIndex = input->retained.front().role == "system" ? 1 : 0;
    input->retained[summaryIndex].content =
        "[Earlier conversation summary; re-inspect live graph state before edits.]\n" + *summary;

    if (Detail::Context::Estimate(input->retained, config.tools) > budget()) {
        fail(Provider::Error::Kind::ContextLength,
             "The compacted context still exceeds the model budget. "
             "History was preserved; reduce the request or use a larger-context model.");
        return false;
    }

    messages = std::move(input->retained);
    return true;
}

Provider::Response Session::Impl::Turn::stream(bool allowTools) {
    announced = 0;
    Detail::Context::Trim(messages, config.maxImages);

    const bool overBudget = bounded() && Detail::Context::Estimate(messages, config.tools) > budget();
    if (overBudget && !compact(budget())) {
        return {};
    }

    auto response = request(messages, allowTools);

    const bool contextRejected = bounded() && error.kind == Provider::Error::Kind::ContextLength;
    const bool nothingStreamed = response.text.empty() && !streamed;
    if (contextRejected && nothingStreamed && !aborted()) {
        if (compact(budget() / 2)) {
            response = request(messages, allowTools);
        }
    }

    return response;
}

void Session::Impl::Turn::execute(const std::vector<Provider::FunctionCall>& calls, bool lastRound) {
    std::vector<Provider::Message> imageMessages;
    for (U64 index = 0; index < calls.size(); ++index) {
        const auto& call = calls[index];
        if (aborted()) {
            result = Result::ERROR;
            break;
        }

        if (index >= announced) {
            announce(call.name);
        }
        note({Segment::Kind::ToolCall, call.arguments, 1, call.name});

        const Tools::Output output = tools.run(call, resultLimit(), lastRound, aborted);
        note({Segment::Kind::ToolResult, output.content, 1, call.name, output.images.size()});

        append({
            .role = "tool",
            .content = output.content,
            .toolCallId = call.id,
            .error = output.error,
        });
        if (auto image = Detail::ToolRunner::Attachment(call, output)) {
            imageMessages.push_back(std::move(*image));
        }
    }

    for (auto& imageMessage : imageMessages) {
        append(std::move(imageMessage));
    }
}

void Session::Impl::Turn::write(Segment::Kind kind, const std::string& text) {
    session.append(target, kind, text);
}

void Session::Impl::Turn::note(Segment segment) {
    session.add(target, std::move(segment));
}

void Session::Impl::Turn::announce(const std::string& name) {
    note({Segment::Kind::Note, {}, 1, name});
}

}  // namespace Jetstream::Agent
