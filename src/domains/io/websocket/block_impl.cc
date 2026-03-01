#include <jetstream/domains/io/websocket/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/io/websocket/module.hh>
#include "module_impl.hh"

namespace Jetstream::Blocks {

struct WebsocketImpl : public Block::Impl,
                       public DynamicConfig<Blocks::Websocket> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Websocket> moduleConfig =
        std::make_shared<Modules::Websocket>();
    Modules::WebsocketImpl* moduleImpl = nullptr;
};

Result WebsocketImpl::configure() {
    moduleConfig->url = url;
    moduleConfig->dataType = dataType;
    moduleConfig->numberOfBatches = numberOfBatches;
    moduleConfig->numberOfTimeSamples = numberOfTimeSamples;
    moduleConfig->bufferMultiplier = bufferMultiplier;

    return Result::SUCCESS;
}

Result WebsocketImpl::define() {
    JST_CHECK(defineInterfaceOutput("signal",
                                    "Output",
                                    "The output buffer containing samples from the WebSocket."));

    JST_CHECK(defineInterfaceConfig("url",
                                    "URL",
                                    "WebSocket server URL.",
                                    "text"));

    JST_CHECK(defineInterfaceConfig("dataType",
                                    "Data Type",
                                    "Sample format of incoming data.",
                                    "dropdown:CF32(CF32),CU8(CU8),CS8(CS8),CI16(CI16),CU16(CU16)"));

    JST_CHECK(defineInterfaceConfig("numberOfBatches",
                                    "Batches",
                                    "Number of batches in output buffer.",
                                    "int:batches"));

    JST_CHECK(defineInterfaceConfig("numberOfTimeSamples",
                                    "Samples",
                                    "Number of samples per batch.",
                                    "int:samples"));

    JST_CHECK(defineInterfaceConfig("bufferMultiplier",
                                    "Buffer Multiplier",
                                    "Internal buffer size multiplier.",
                                    "int:x"));

    JST_CHECK(defineInterfaceMetric("bufferHealth",
                                    "Buffer Health",
                                    "Current buffer occupancy level.",
                                    "progressbar",
        [this]() -> std::any {
            if (!moduleImpl) {
                return F32(0.0f);
            }
            return moduleImpl->getBufferHealth();
        }));

    JST_CHECK(defineInterfaceMetric("throughput",
                                    "Throughput",
                                    "Current data throughput.",
                                    "label",
        [this]() -> std::any {
            if (!moduleImpl) {
                return std::string("N/A");
            }
            return jst::fmt::format("{:.1f} MB/s",
                                    moduleImpl->getThroughput());
        }));

    return Result::SUCCESS;
}

Result WebsocketImpl::create() {
    JST_CHECK(moduleCreate("websocket", moduleConfig, {}));
    JST_CHECK(moduleExposeOutput("signal", {"websocket", "signal"}));

    moduleImpl = moduleHandle("websocket")->getImpl<Modules::WebsocketImpl>();

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(WebsocketImpl);

}  // namespace Jetstream::Blocks
