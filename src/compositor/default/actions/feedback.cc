#include "feedback.hh"

#include "jetstream/config.hh"

#include <algorithm>
#include <sstream>

namespace Jetstream {

namespace {

constexpr U64 kMaxDebugInfoLength = 2500;

std::string TrimTrailingWhitespace(std::string value) {
    while (!value.empty() && (value.back() == '\n' || value.back() == '\r' ||
                              value.back() == ' ' || value.back() == '\t')) {
        value.pop_back();
    }
    return value;
}

U64 Utf8Floor(std::string_view value, U64 limit) {
    if (limit >= value.size()) {
        return value.size();
    }
    while (limit > 0 && (static_cast<U8>(value[limit]) & 0xC0) == 0x80) {
        --limit;
    }
    return limit;
}

std::string TruncateDebugInfo(std::string value) {
    if (value.size() <= kMaxDebugInfoLength) {
        return value;
    }
    constexpr std::string_view suffix = "\n... (truncated)";
    const U64 rawLimit = kMaxDebugInfoLength > suffix.size() ? kMaxDebugInfoLength - suffix.size() : 0;
    const U64 limit = Utf8Floor(value, rawLimit);
    value.resize(limit);
    if (const auto newline = value.rfind('\n'); newline != std::string::npos && newline > limit / 2) {
        value.resize(newline);
    }
    value += suffix;
    return value;
}

}  // namespace

std::string FeedbackActions::FormatBlockDebugInfo(const Flowgraph::View::BlockData& block) {
    std::ostringstream out;
    out << "CyberEther: v" << JETSTREAM_VERSION_STR << "-" << JETSTREAM_BUILD_TYPE << "\n";
    if (!block.type.empty()) {
        out << "Block: " << block.name << " (type: " << block.type << ")\n";
    } else {
        out << "Block: " << block.name << "\n";
    }
    out << "Target: " << jst::fmt::format("{} / {} / {}", block.device, block.runtime, block.provider) << "\n";

    const std::string diagnostic = TrimTrailingWhitespace(block.diagnostic);
    if (!diagnostic.empty()) {
        out << "\nError:\n" << diagnostic << "\n";
    }

    std::vector<std::string> inputLines;
    auto appendInput = [&](const std::string& port, const TensorLink& link) {
        if (!link.external.has_value()) {
            return;
        }
        std::string line = jst::fmt::format("- {}: {}.{}", port, link.external->block, link.external->port);
        if (link.resolved()) {
            line += jst::fmt::format(" ({} {} {})", link.tensor.dtype(), link.tensor.shape(), link.tensor.device());
        } else {
            line += " (unresolved)";
        }
        inputLines.push_back(std::move(line));
    };

    std::unordered_set<std::string> seenInputs;
    for (const auto& info : block.interfaceInputs) {
        seenInputs.insert(info.name);
        if (const auto it = block.inputs.find(info.name); it != block.inputs.end()) {
            appendInput(info.name, it->second);
        }
    }
    std::vector<std::string> extraKeys;
    for (const auto& [port, _] : block.inputs) {
        if (!seenInputs.contains(port)) {
            extraKeys.push_back(port);
        }
    }
    std::sort(extraKeys.begin(), extraKeys.end());
    for (const auto& port : extraKeys) {
        appendInput(port, block.inputs.at(port));
    }

    if (!inputLines.empty()) {
        out << "\nInputs:\n";
        for (const auto& line : inputLines) {
            out << line << "\n";
        }
    }

    if (!block.config.empty()) {
        std::string yaml;
        if (Parser::YamlEncode(block.config, yaml) == Result::SUCCESS) {
            yaml = TrimTrailingWhitespace(std::move(yaml));
            if (!yaml.empty()) {
                out << "\nConfig:\n" << yaml << "\n";
            }
        }
    }

    return TruncateDebugInfo(TrimTrailingWhitespace(out.str()));
}

FeedbackActions::FeedbackActions(DefaultCompositorState& state,
                                 DefaultCompositorCallbacks& callbacks) :
    state(state),
    callbacks(callbacks) {}

Result FeedbackActions::handle(const MailSetFeedbackText& msg) {
    state.feedback.text = msg.value;
    return Result::SUCCESS;
}

Result FeedbackActions::handle(const MailSetFeedbackDebugInfo& msg) {
    state.feedback.debugInfo = msg.value;
    return Result::SUCCESS;
}

Result FeedbackActions::handle(const MailReportBlockProblem& msg) {
    state.feedback.text.clear();
    state.feedback.debugInfo.clear();
    state.feedback.blockReport = true;
    ++state.feedback.session;

    bool found = false;
    if (const auto fgIt = state.flowgraph.items.find(msg.flowgraph);
        fgIt != state.flowgraph.items.end() && fgIt->second) {
        Flowgraph::View::BlockData data;
        if (fgIt->second->view().block(msg.block, data) == Result::SUCCESS) {
            state.feedback.debugInfo = FormatBlockDebugInfo(data);
            found = true;
        }
    }
    if (!found) {
        if (const auto fgIt = state.flowgraph.blocks.find(msg.flowgraph);
            fgIt != state.flowgraph.blocks.end()) {
            for (const auto& snapshot : fgIt->second) {
                if (snapshot.name == msg.block) {
                    state.feedback.debugInfo = FormatBlockDebugInfo(snapshot.data);
                    break;
                }
            }
        }
    }

    state.modal = {.content = ModalContent::Feedback};
    return Result::SUCCESS;
}

Result FeedbackActions::handle(const MailSubmitFeedback&) {
    if (state.feedback.status == DefaultCompositorState::FeedbackState::Status::Submitting) {
        return Result::SUCCESS;
    }

    const std::string payload = Feedback::Compose(state.feedback.text, state.feedback.debugInfo);

    if (payload.size() < Feedback::MIN_LENGTH) {
        callbacks.notify(Sakura::ToastType::Error, 5000,
            jst::fmt::format("Feedback must be at least {} characters.", Feedback::MIN_LENGTH));
        return Result::SUCCESS;
    }

    if (payload.size() > Feedback::MAX_LENGTH) {
        callbacks.notify(Sakura::ToastType::Error, 5000,
            jst::fmt::format("Feedback exceeds the {} character limit.", Feedback::MAX_LENGTH));
        return Result::SUCCESS;
    }

    state.feedback.status = DefaultCompositorState::FeedbackState::Status::Submitting;
    state.feedback.text.clear();
    state.feedback.debugInfo.clear();
    state.feedback.blockReport = false;

    callbacks.submitFeedback(payload);
    callbacks.enqueueMail(MailCloseModal{});

    return Result::SUCCESS;
}

}  // namespace Jetstream
