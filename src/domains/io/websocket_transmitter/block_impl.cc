#include <jetstream/domains/io/websocket_transmitter/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/io/websocket_transmitter/module.hh>
#include "module_impl.hh"

namespace Jetstream::Blocks {

struct WebsocketTransmitterImpl : public Block::Impl,
                                  public DynamicConfig<Blocks::WebsocketTransmitter> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::WebsocketTransmitter> moduleConfig =
        std::make_shared<Modules::WebsocketTransmitter>();
    Modules::WebsocketTransmitterImpl* moduleImpl = nullptr;
};

Result WebsocketTransmitterImpl::configure() {
    moduleConfig->mode = mode;
    moduleConfig->address = address;
    moduleConfig->port = port;
    moduleConfig->url = url;

    return Result::SUCCESS;
}

Result WebsocketTransmitterImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer",
                                   "Input",
                                   "The input buffer containing samples "
                                   "to send over WebSocket."));

    const bool client = candidate()->mode == "client";

    JST_CHECK(defineInterfaceConfig("mode",
                                    "Mode",
                                    "Host a server or connect to one.",
                                    {{"type", "dropdown"}, {"options", Parser::Sequence{
                                        Parser::Map{{"label", "Server"}, {"value", "server"}},
                                        Parser::Map{{"label", "Client"}, {"value", "client"}},
                                    }}}));

    if (client) {
        JST_CHECK(defineInterfaceConfig("url",
                                        "URL",
                                        "WebSocket server URL.",
                                        {{"type", "text"}}));
    } else {
        JST_CHECK(defineInterfaceConfig("address",
                                        "Address",
                                        "Local address to listen on.",
                                        {{"type", "text"}}));

        JST_CHECK(defineInterfaceConfig("port",
                                        "Port",
                                        "Local port to listen on.",
                                        {{"type", "uint"}}));
    }

    JST_CHECK(defineInterfaceMetric("throughput",
                                    "Throughput",
                                    "Smoothed recent send rate.",
                                    {{"type", "label"}},
        [this]() -> std::any {
            if (!moduleImpl) {
                return std::string("N/A");
            }
            return jst::fmt::format("{:.1f} MB/s", moduleImpl->getThroughput());
        }));

    if (client) {
        JST_CHECK(defineInterfaceMetric("connection",
                                        "Connection",
                                        "Whether the server is connected.",
                                        {{"type", "label"}},
            [this]() -> std::any {
                if (!moduleImpl) {
                    return std::string("N/A");
                }
                return std::string(moduleImpl->getConnected() ? "Connected" : "Connecting");
            }));
    } else {
        JST_CHECK(defineInterfaceMetric("clients",
                                        "Clients",
                                        "Clients connected right now.",
                                        {{"type", "label"}},
            [this]() -> std::any {
                if (!moduleImpl) {
                    return std::string("N/A");
                }
                return std::to_string(moduleImpl->getClients());
            }));
    }

    return Result::SUCCESS;
}

Result WebsocketTransmitterImpl::create() {
    JST_CHECK(moduleCreate("websocket_transmitter", moduleConfig, {
        {"buffer", inputs().at("buffer")}
    }));

    moduleImpl = moduleHandle("websocket_transmitter")->getImpl<Modules::WebsocketTransmitterImpl>();

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(WebsocketTransmitterImpl, {"websocket_transmitter"});

}  // namespace Jetstream::Blocks
