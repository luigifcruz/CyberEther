#include <jetstream/domains/io/websocket_receiver/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/io/websocket_receiver/module.hh>
#include "module_impl.hh"

namespace Jetstream::Blocks {

struct WebsocketReceiverImpl : public Block::Impl,
                               public DynamicConfig<Blocks::WebsocketReceiver> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::WebsocketReceiver> moduleConfig =
        std::make_shared<Modules::WebsocketReceiver>();
    Modules::WebsocketReceiverImpl* moduleImpl = nullptr;
};

Result WebsocketReceiverImpl::configure() {
    moduleConfig->url = url;
    moduleConfig->dataType = dataType;
    moduleConfig->numberOfBatches = numberOfBatches;
    moduleConfig->numberOfTimeSamples = numberOfTimeSamples;
    moduleConfig->bufferMultiplier = bufferMultiplier;

    return Result::SUCCESS;
}

Result WebsocketReceiverImpl::define() {
    JST_CHECK(defineInterfaceOutput("signal",
                                    "Output",
                                    "The output buffer containing samples received over WebSocket."));

    JST_CHECK(defineInterfaceConfig("url",
                                    "URL",
                                    "WebSocket server URL.",
                                    {{"type", "text"}}));

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

    JST_CHECK(defineInterfaceMetric("bufferHealth",
                                    "Buffer Health",
                                    "Current buffer occupancy level.",
                                    {{"type", "progressbar"}},
        [this]() -> std::any {
            if (!moduleImpl) {
                return std::pair<std::string, F32>{"0.0%", 0.0f};
            }
            const F32 bufferHealth = moduleImpl->getBufferHealth();
            return std::pair<std::string, F32>{jst::fmt::format("{:.1f}%", bufferHealth * 100.0f),
                                               bufferHealth};
        }));

    JST_CHECK(defineInterfaceMetric("bufferLoss",
                                    "Buffer Loss",
                                    "Share of samples overwritten before the chain read them since the stream started.",
                                    {{"type", "progressbar"}},
        [this]() -> std::any {
            const F64 loss = moduleImpl ? moduleImpl->getBufferLoss() : 0.0;
            const auto label = loss > 0.0 && loss < 0.0001
                ? std::string("<0.01%") : jst::fmt::format("{:.2f}%", loss * 100.0);
            return std::pair<std::string, F32>{label, static_cast<F32>(loss)};
        }));

    JST_CHECK(defineInterfaceMetric("throughput",
                                    "Throughput",
                                    "Current data throughput.",
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

Result WebsocketReceiverImpl::create() {
    JST_CHECK(moduleCreate("websocket_receiver", moduleConfig, {}));
    JST_CHECK(moduleExposeOutput("signal", {"websocket_receiver", "signal"}));

    moduleImpl = moduleHandle("websocket_receiver")->getImpl<Modules::WebsocketReceiverImpl>();

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(WebsocketReceiverImpl, {"websocket_receiver"});

}  // namespace Jetstream::Blocks
