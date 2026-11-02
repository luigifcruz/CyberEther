#include <jetstream/domains/io/udp_transmitter/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/io/udp_transmitter/module.hh>
#include "module_impl.hh"

#include <jetstream/platform.hh>

namespace Jetstream::Blocks {

struct UdpTransmitterImpl : public Block::Impl,
                            public DynamicConfig<Blocks::UdpTransmitter> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::UdpTransmitter> moduleConfig =
        std::make_shared<Modules::UdpTransmitter>();
    Modules::UdpTransmitterImpl* moduleImpl = nullptr;
    Parser::Map interfaceDropdown;
};

Result UdpTransmitterImpl::configure() {
    moduleConfig->address = address;
    moduleConfig->port = port;
    moduleConfig->networkInterface = networkInterface;
    moduleConfig->packetSize = packetSize;

    return Result::SUCCESS;
}

Result UdpTransmitterImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer",
                                   "Input",
                                   "The input buffer containing samples "
                                   "to send over UDP."));

    JST_CHECK(defineInterfaceConfig("address",
                                    "Address",
                                    "Destination address, broadcast address, or multicast group.",
                                    {{"type", "text"}}));

    JST_CHECK(defineInterfaceConfig("port",
                                    "Port",
                                    "Destination port.",
                                    {{"type", "uint"}}));

    if (interfaceDropdown.empty()) {
        Parser::Sequence interfaceOptions;
        interfaceOptions.emplace_back(Parser::Map{{"label", std::string("Automatic")}, {"value", std::string("auto")}});
        std::vector<Platform::UdpSocket::Interface> interfaces;
        Platform::UdpSocket::ListInterfaces(interfaces);
        for (const auto& entry : interfaces) {
            const auto label = jst::fmt::format("{} ({})", entry.name, entry.addresses.front());
            interfaceOptions.emplace_back(Parser::Map{{"label", label}, {"value", entry.name}});
        }
        interfaceDropdown = {{"type", "dropdown"}, {"options", std::move(interfaceOptions)}};
    }

    JST_CHECK(defineInterfaceConfig("networkInterface",
                                    "Interface",
                                    "Network interface for multicast packets.",
                                    interfaceDropdown));

    JST_CHECK(defineInterfaceConfig("packetSize",
                                    "Packet Size",
                                    "Largest payload per packet.",
                                    {{"type", "uint"}, {"unit", "bytes"}}));

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

    JST_CHECK(defineInterfaceMetric("droppedPackets",
                                    "Dropped Packets",
                                    "Packets that failed to send.",
                                    {{"type", "label"}},
        [this]() -> std::any {
            if (!moduleImpl) {
                return std::string("N/A");
            }
            return std::to_string(moduleImpl->getDroppedPackets());
        }));

    return Result::SUCCESS;
}

Result UdpTransmitterImpl::create() {
    JST_CHECK(moduleCreate("udp_transmitter", moduleConfig, {
        {"buffer", inputs().at("buffer")}
    }));

    moduleImpl = moduleHandle("udp_transmitter")->getImpl<Modules::UdpTransmitterImpl>();

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(UdpTransmitterImpl, {"udp_transmitter"});

}  // namespace Jetstream::Blocks
