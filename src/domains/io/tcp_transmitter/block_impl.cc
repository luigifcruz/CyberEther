#include <jetstream/domains/io/tcp_transmitter/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/io/tcp_transmitter/module.hh>
#include "module_impl.hh"

namespace Jetstream::Blocks {

struct TcpTransmitterImpl : public Block::Impl,
                            public DynamicConfig<Blocks::TcpTransmitter> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::TcpTransmitter> moduleConfig =
        std::make_shared<Modules::TcpTransmitter>();
    Modules::TcpTransmitterImpl* moduleImpl = nullptr;
};

Result TcpTransmitterImpl::configure() {
    moduleConfig->mode = mode;
    moduleConfig->address = address;
    moduleConfig->port = port;

    return Result::SUCCESS;
}

Result TcpTransmitterImpl::define() {
    const bool client = candidate()->mode == "client";

    JST_CHECK(defineInterfaceInput("buffer",
                                   "Input",
                                   "The input buffer containing samples "
                                   "to send over TCP."));

    JST_CHECK(defineInterfaceConfig("mode",
                                    "Mode",
                                    "Serve clients or connect to a receiver.",
                                    {{"type", "dropdown"}, {"options", Parser::Sequence{
                                        Parser::Map{{"label", "Server"}, {"value", "server"}},
                                        Parser::Map{{"label", "Client"}, {"value", "client"}},
                                    }}}));

    JST_CHECK(defineInterfaceConfig("address",
                                    "Address",
                                    client ? "Receiver address to connect to." : "Local address to listen on.",
                                    {{"type", "text"}}));

    JST_CHECK(defineInterfaceConfig("port",
                                    "Port",
                                    client ? "Receiver port to connect to." : "Local port to listen on.",
                                    {{"type", "uint"}}));

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
                                        "Whether the receiver is connected.",
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

Result TcpTransmitterImpl::create() {
    JST_CHECK(moduleCreate("tcp_transmitter", moduleConfig, {
        {"buffer", inputs().at("buffer")}
    }));

    moduleImpl = moduleHandle("tcp_transmitter")->getImpl<Modules::TcpTransmitterImpl>();

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(TcpTransmitterImpl, {"tcp_transmitter"});

}  // namespace Jetstream::Blocks
