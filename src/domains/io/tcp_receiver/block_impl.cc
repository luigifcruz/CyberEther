#include <jetstream/domains/io/tcp_receiver/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/io/tcp_receiver/module.hh>
#include "module_impl.hh"

namespace Jetstream::Blocks {

struct TcpReceiverImpl : public Block::Impl,
                         public DynamicConfig<Blocks::TcpReceiver> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::TcpReceiver> moduleConfig =
        std::make_shared<Modules::TcpReceiver>();
    Modules::TcpReceiverImpl* moduleImpl = nullptr;
};

Result TcpReceiverImpl::configure() {
    moduleConfig->mode = mode;
    moduleConfig->address = address;
    moduleConfig->port = port;
    moduleConfig->dataType = dataType;
    moduleConfig->numberOfBatches = numberOfBatches;
    moduleConfig->numberOfTimeSamples = numberOfTimeSamples;
    moduleConfig->bufferMultiplier = bufferMultiplier;

    return Result::SUCCESS;
}

Result TcpReceiverImpl::define() {
    const bool server = candidate()->mode == "server";

    JST_CHECK(defineInterfaceOutput("signal",
                                    "Output",
                                    "The output buffer containing samples received over TCP."));

    JST_CHECK(defineInterfaceConfig("mode",
                                    "Mode",
                                    "Connect to a sender or listen for one.",
                                    {{"type", "dropdown"}, {"options", Parser::Sequence{
                                        Parser::Map{{"label", "Client"}, {"value", "client"}},
                                        Parser::Map{{"label", "Server"}, {"value", "server"}},
                                    }}}));

    JST_CHECK(defineInterfaceConfig("address",
                                    "Address",
                                    server ? "Local address to listen on." : "Sender address to connect to.",
                                    {{"type", "text"}}));

    JST_CHECK(defineInterfaceConfig("port",
                                    "Port",
                                    server ? "Local port to listen on." : "Sender port to connect to.",
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

    return Result::SUCCESS;
}

Result TcpReceiverImpl::create() {
    JST_CHECK(moduleCreate("tcp_receiver", moduleConfig, {}));
    JST_CHECK(moduleExposeOutput("signal", {"tcp_receiver", "signal"}));

    moduleImpl = moduleHandle("tcp_receiver")->getImpl<Modules::TcpReceiverImpl>();

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(TcpReceiverImpl, {"tcp_receiver"});

}  // namespace Jetstream::Blocks
