#include <catch2/catch_test_macros.hpp>

#include "compositor/default/actions/feedback.hh"
#include "compositor/default/actions/workbench.hh"

#include <nlohmann/json.hpp>

#include <memory>
#include <string>
#include <vector>

using namespace Jetstream;

namespace {

struct Rig {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    std::vector<std::string> submitted;
    std::vector<std::string> notifications;
    U64 closedModals = 0;
    FeedbackActions feedback{state, callbacks};
    WorkbenchActions workbench{state, callbacks};

    Rig() {
        callbacks.submitFeedback = [this](const std::string& text) {
            submitted.push_back(text);
        };
        callbacks.notify = [this](Sakura::ToastType, U64, const std::string& message) {
            notifications.push_back(message);
        };
        callbacks.enqueueMail = [this](Mail&& mail) {
            if (std::holds_alternative<MailCloseModal>(mail)) {
                ++closedModals;
                workbench.handle(MailCloseModal{});
            }
        };
    }
};

}  // namespace

TEST_CASE("Block error debug info includes block metadata, error, inputs, and YAML config",
          "[core][compositor][feedback]") {
    Flowgraph::View::BlockData block;
    block.name = "fft_1";
    block.type = "fft";
    block.device = DeviceType::CPU;
    block.runtime = RuntimeType::NATIVE;
    block.provider = "generic";
    block.state = Block::State::Errored;
    block.diagnostic = "[MODULE_FFT] Input tensor must be contiguous.\n";
    block.config = {
        {"forward", true},
        {"scale", false},
    };
    block.interfaceInputs = {{.name = "buffer", .label = "Buffer"}};
    TensorLink inputLink;
    inputLink.external = BlockEndpoint{.block = "source_1", .port = "signal"};
    block.inputs["buffer"] = inputLink;
    TensorLink undeclaredLink;
    undeclaredLink.external = BlockEndpoint{.block = "source_2", .port = "out"};
    block.inputs["unexpected"] = undeclaredLink;

    const std::string debugInfo = FeedbackActions::FormatBlockDebugInfo(block);
    CHECK(debugInfo.find("CyberEther: v") != std::string::npos);
    CHECK(debugInfo.find("Block: fft_1 (type: fft)") != std::string::npos);
    CHECK(debugInfo.find(jst::fmt::format("Target: {} / {} / {}", block.device, block.runtime, block.provider)) !=
          std::string::npos);
    CHECK(debugInfo.find("[MODULE_FFT] Input tensor must be contiguous.") != std::string::npos);
    CHECK(debugInfo.find("- buffer: source_1.signal") != std::string::npos);
    CHECK(debugInfo.find("- unexpected: source_2.out") != std::string::npos);
    CHECK(debugInfo.find("Config:") != std::string::npos);
    CHECK(debugInfo.find("forward: true") != std::string::npos);
}

TEST_CASE("Block error debug info truncates oversized UTF-8 configurations cleanly for JSON",
          "[core][compositor][feedback]") {
    Flowgraph::View::BlockData block;
    block.name = "py_1";
    block.type = "python";
    block.diagnostic = "SyntaxError";
    std::string multibyte;
    for (U64 i = 0; i < 1200; ++i) {
        multibyte += "\xF0\x9F\x9A\x80";  // 4-byte UTF-8 rocket emoji
    }
    block.config = {{"code", multibyte}};

    const std::string debugInfo = FeedbackActions::FormatBlockDebugInfo(block);
    CHECK(debugInfo.size() <= 2500);
    CHECK(debugInfo.find("... (truncated)") != std::string::npos);
    REQUIRE_NOTHROW(nlohmann::json({{"feedback", debugInfo}}).dump());
}

TEST_CASE("Reporting a block problem reads from live flowgraph items even when block snapshots are empty",
          "[core][compositor][feedback]") {
    Rig rig;
    auto fg = std::make_shared<Flowgraph>();
    REQUIRE(fg->create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);
    // Create a block with config in the live flowgraph.
    REQUIRE(fg->blockCreate("gain_1", "multiply_constant", {{"constant", 2.5}}, {}) == Result::SUCCESS);
    rig.state.flowgraph.items["fg-live"] = fg;
    REQUIRE(rig.state.flowgraph.blocks.empty());

    REQUIRE(rig.feedback.handle(MailReportBlockProblem{.flowgraph = "fg-live", .block = "gain_1"}) == Result::SUCCESS);
    CHECK(rig.state.modal.content == std::optional<ModalContent>(ModalContent::Feedback));
    CHECK(rig.state.feedback.blockReport);
    CHECK(rig.state.feedback.debugInfo.find("Block: gain_1 (type: multiply_constant)") != std::string::npos);
    CHECK(rig.state.feedback.debugInfo.find("constant: 2.5") != std::string::npos);

    REQUIRE(fg->destroy() == Result::SUCCESS);
}

TEST_CASE("Reporting a block problem submits combined payload and rejects whitespace-only payloads",
          "[core][compositor][feedback]") {
    Rig rig;
    Flowgraph::View::BlockData block;
    block.name = "gain_1";
    block.type = "multiply_constant";
    block.state = Block::State::Errored;
    block.diagnostic = "Invalid factor";
    block.config = {{"factor", 2.0}};
    rig.state.flowgraph.blocks["fg-1"] = {{.name = "gain_1", .data = block}};

    REQUIRE(rig.feedback.handle(MailReportBlockProblem{.flowgraph = "fg-1", .block = "gain_1"}) == Result::SUCCESS);
    REQUIRE(rig.feedback.handle(MailSetFeedbackText{.value = "Tried scaling the IQ stream."}) == Result::SUCCESS);
    REQUIRE(rig.feedback.handle(MailSubmitFeedback{}) == Result::SUCCESS);
    REQUIRE(rig.submitted.size() == 1);
    CHECK(rig.submitted.front().find("Tried scaling the IQ stream.") != std::string::npos);
    CHECK(rig.submitted.front().find("--- Debug Info ---") != std::string::npos);
    CHECK(rig.submitted.front().find("Block: gain_1 (type: multiply_constant)") != std::string::npos);
    CHECK(rig.closedModals == 1);

    // Opening another report while the previous request is still submitting preserves Submitting status.
    REQUIRE(rig.feedback.handle(MailReportBlockProblem{.flowgraph = "fg-1", .block = "gain_1"}) == Result::SUCCESS);
    CHECK(rig.state.feedback.status == DefaultCompositorState::FeedbackState::Status::Submitting);
    CHECK(rig.state.feedback.blockReport);
    CHECK_FALSE(rig.state.feedback.debugInfo.empty());

    // Whitespace-only general feedback is rejected.
    rig.state.feedback.status = DefaultCompositorState::FeedbackState::Status::Idle;
    REQUIRE(rig.workbench.handle(MailOpenModal{.content = ModalContent::Feedback}) == Result::SUCCESS);
    REQUIRE(rig.feedback.handle(MailSetFeedbackText{.value = "            "}) == Result::SUCCESS);
    REQUIRE(rig.feedback.handle(MailSubmitFeedback{}) == Result::SUCCESS);
    CHECK(rig.submitted.size() == 1);
}

TEST_CASE("Opening general feedback modal clears prior block problem narrative and debug info",
          "[core][compositor][feedback]") {
    Rig rig;
    rig.state.feedback.blockReport = true;
    rig.state.feedback.text = "Old block narrative";
    rig.state.feedback.debugInfo = "Block: stale";

    REQUIRE(rig.workbench.handle(MailOpenModal{.content = ModalContent::Feedback}) == Result::SUCCESS);
    CHECK(rig.state.modal.content == std::optional<ModalContent>(ModalContent::Feedback));
    CHECK_FALSE(rig.state.feedback.blockReport);
    CHECK(rig.state.feedback.text.empty());
    CHECK(rig.state.feedback.debugInfo.empty());
}
