#include <jetstream/domains/io/zeromq_receiver/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/io/zeromq_receiver/module.hh>
#include "module_impl.hh"

namespace Jetstream::Blocks {

struct ZeromqReceiverImpl : public Block::Impl,
                            public DynamicConfig<Blocks::ZeromqReceiver> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::ZeromqReceiver> moduleConfig =
        std::make_shared<Modules::ZeromqReceiver>();
    Modules::ZeromqReceiverImpl* moduleImpl = nullptr;
};

Result ZeromqReceiverImpl::configure() {
    moduleConfig->pattern = pattern;
    moduleConfig->mode = mode;
    moduleConfig->address = address;
    moduleConfig->port = port;
    moduleConfig->dataType = dataType;
    moduleConfig->numberOfBatches = numberOfBatches;
    moduleConfig->numberOfTimeSamples = numberOfTimeSamples;
    moduleConfig->bufferMultiplier = bufferMultiplier;

    return Result::SUCCESS;
}

Result ZeromqReceiverImpl::define() {
    const bool bind = candidate()->mode == "bind";
    const bool subscribe = candidate()->pattern == "subscribe";

    JST_CHECK(defineInterfaceOutput("signal",
                                    "Output",
                                    "The output buffer containing samples received over ZeroMQ."));

    JST_CHECK(defineInterfaceConfig("pattern",
                                    "Pattern",
                                    "Subscribe to a publisher or pull from a pusher.",
                                    {{"type", "dropdown"}, {"options", Parser::Sequence{
                                        Parser::Map{{"label", "Subscribe"}, {"value", "subscribe"}},
                                        Parser::Map{{"label", "Pull"}, {"value", "pull"}},
                                    }}}));

    JST_CHECK(defineInterfaceConfig("mode",
                                    "Mode",
                                    "Connect to a sender or listen for senders.",
                                    {{"type", "dropdown"}, {"options", Parser::Sequence{
                                        Parser::Map{{"label", "Connect"}, {"value", "connect"}},
                                        Parser::Map{{"label", "Bind"}, {"value", "bind"}},
                                    }}}));

    JST_CHECK(defineInterfaceConfig("address",
                                    "Address",
                                    bind ? "Local address to listen on." : "Sender address to connect to.",
                                    {{"type", "text"}}));

    JST_CHECK(defineInterfaceConfig("port",
                                    "Port",
                                    bind ? "Local port to listen on." : "Sender port to connect to.",
                                    {{"type", "uint"}}));

    Parser::Sequence dataTypes;
    for (const std::string type : {"CF32", "F32", "CI8", "I8", "CU8", "U8", "CI16", "I16", "CU16", "U16"}) {
        dataTypes.emplace_back(Parser::Map{{"label", type}, {"value", type}});
    }
    JST_CHECK(defineInterfaceConfig("dataType",
                                    "Data Type",
                                    "Sample format of incoming data.",
                                    {{"type", "dropdown"}, {"options", std::move(dataTypes)}}));

    JST_CHECK(defineInterfaceConfig("numberOfBatches",
                                    "Batches",
                                    "Number of batches in output buffer.",
                                    {{"type", "uint"}, {"unit", "batches"}}));

    JST_CHECK(defineInterfaceConfig("numberOfTimeSamples",
                                    "Samples",
                                    "Samples per batch and the FFT size downstream.",
                                    {{"type", "uint"}, {"unit", "samples"}}));

    JST_CHECK(defineInterfaceConfig("bufferMultiplier",
                                    "Buffer Multiplier",
                                    "Internal buffer size multiplier.",
                                    {{"type", "uint"}, {"unit", "x"}}));

    JST_CHECK(defineInterfaceMetric("bufferFill",
                                    "Buffer Fill",
                                    "How full the internal buffer is. Near 100% means the chain is falling behind.",
                                    {{"type", "progressbar"}, {"warning_above", 0.5f}, {"error_above", 0.75f}},
        [this]() -> std::any {
            if (!moduleImpl) {
                return std::pair<std::string, F32>{"0.0%", 0.0f};
            }
            const F32 bufferFill = moduleImpl->getBufferFill();
            return std::pair<std::string, F32>{jst::fmt::format("{:.1f}%", bufferFill * 100.0f),
                                               bufferFill};
        }));

    JST_CHECK(defineInterfaceMetric("bufferLoss",
                                    "Buffer Loss",
                                    "Share of samples overwritten before the chain read them since the stream started.",
                                    {{"type", "progressbar"}, {"warning_above", 0.0f}, {"error_above", 0.01f}},
        [this]() -> std::any {
            const F64 loss = moduleImpl ? moduleImpl->getBufferLoss() : 0.0;
            const auto label = loss > 0.0 && loss < 0.0001
                ? std::string("<0.01%") : jst::fmt::format("{:.2f}%", loss * 100.0);
            return std::pair<std::string, F32>{label, static_cast<F32>(loss)};
        }));

    JST_CHECK(defineInterfaceMetric("throughput",
                                    "Throughput",
                                    "Rate the chain reads.",
                                    {{"type", "label"}},
        [this]() -> std::any {
            if (!moduleImpl) {
                return std::string("N/A");
            }
            return jst::fmt::format("{:.1f} MB/s",
                                    moduleImpl->getThroughput());
        }));

    JST_CHECK(defineInterfaceMetric("peers",
                                    "Peers",
                                    "Senders connected right now.",
                                    {{"type", "label"}},
        [this]() -> std::any {
            if (!moduleImpl) {
                return std::string("N/A");
            }
            return std::to_string(moduleImpl->getPeers());
        }));

    if (subscribe) {
        JST_CHECK(defineInterfaceMetric("lostMessages",
                                        "Lost Messages",
                                        "Buffers a sender with attributes sent that never arrived.",
                                        {{"type", "label"}},
            [this]() -> std::any {
                if (!moduleImpl) {
                    return std::string("N/A");
                }
                return std::to_string(moduleImpl->getLostMessages());
            }));
    }

    return Result::SUCCESS;
}

Result ZeromqReceiverImpl::create() {
    JST_CHECK(moduleCreate("zeromq_receiver", moduleConfig, {}));
    JST_CHECK(moduleExposeOutput("signal", {"zeromq_receiver", "signal"}));

    moduleImpl = moduleHandle("zeromq_receiver")->getImpl<Modules::ZeromqReceiverImpl>();

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(ZeromqReceiverImpl, {"zeromq_receiver"});

}  // namespace Jetstream::Blocks
