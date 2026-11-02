#include <jetstream/domains/io/zeromq_transmitter/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/io/zeromq_transmitter/module.hh>
#include "module_impl.hh"

namespace Jetstream::Blocks {

struct ZeromqTransmitterImpl : public Block::Impl,
                               public DynamicConfig<Blocks::ZeromqTransmitter> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::ZeromqTransmitter> moduleConfig =
        std::make_shared<Modules::ZeromqTransmitter>();
    Modules::ZeromqTransmitterImpl* moduleImpl = nullptr;
};

Result ZeromqTransmitterImpl::configure() {
    moduleConfig->pattern = pattern;
    moduleConfig->mode = mode;
    moduleConfig->address = address;
    moduleConfig->port = port;
    moduleConfig->sendAttributes = sendAttributes;

    return Result::SUCCESS;
}

Result ZeromqTransmitterImpl::define() {
    const bool connect = candidate()->mode == "connect";
    const bool push = candidate()->pattern == "push";

    JST_CHECK(defineInterfaceInput("buffer",
                                   "Input",
                                   "The input buffer containing samples "
                                   "to send over ZeroMQ."));

    JST_CHECK(defineInterfaceConfig("pattern",
                                    "Pattern",
                                    "Publish to every subscriber or push to one puller at a time.",
                                    {{"type", "dropdown"}, {"options", Parser::Sequence{
                                        Parser::Map{{"label", "Publish"}, {"value", "publish"}},
                                        Parser::Map{{"label", "Push"}, {"value", "push"}},
                                    }}}));

    JST_CHECK(defineInterfaceConfig("mode",
                                    "Mode",
                                    "Listen for receivers or connect to one.",
                                    {{"type", "dropdown"}, {"options", Parser::Sequence{
                                        Parser::Map{{"label", "Bind"}, {"value", "bind"}},
                                        Parser::Map{{"label", "Connect"}, {"value", "connect"}},
                                    }}}));

    JST_CHECK(defineInterfaceConfig("address",
                                    "Address",
                                    connect ? "Receiver address to connect to." : "Local address to listen on.",
                                    {{"type", "text"}}));

    JST_CHECK(defineInterfaceConfig("port",
                                    "Port",
                                    connect ? "Receiver port to connect to." : "Local port to listen on.",
                                    {{"type", "uint"}}));

    JST_CHECK(defineInterfaceConfig("sendAttributes",
                                    "Send Attributes",
                                    "Send type, shape, and attributes ahead of each buffer.",
                                    {{"type", "bool"}}));

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

    JST_CHECK(defineInterfaceMetric("peers",
                                    "Peers",
                                    "Receivers connected right now.",
                                    {{"type", "label"}},
        [this]() -> std::any {
            if (!moduleImpl) {
                return std::string("N/A");
            }
            return std::to_string(moduleImpl->getPeers());
        }));

    if (push) {
        JST_CHECK(defineInterfaceMetric("droppedBuffers",
                                        "Dropped Buffers",
                                        "Buffers dropped because no receiver was ready.",
                                        {{"type", "label"}},
            [this]() -> std::any {
                if (!moduleImpl) {
                    return std::string("N/A");
                }
                return std::to_string(moduleImpl->getDroppedBuffers());
            }));
    }

    return Result::SUCCESS;
}

Result ZeromqTransmitterImpl::create() {
    JST_CHECK(moduleCreate("zeromq_transmitter", moduleConfig, {
        {"buffer", inputs().at("buffer")}
    }));

    moduleImpl = moduleHandle("zeromq_transmitter")->getImpl<Modules::ZeromqTransmitterImpl>();

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(ZeromqTransmitterImpl, {"zeromq_transmitter"});

}  // namespace Jetstream::Blocks
