#include <jetstream/domains/io/udp_receiver/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/io/udp_receiver/module.hh>
#include "module_impl.hh"

#include <jetstream/platform.hh>

namespace Jetstream::Blocks {

struct UdpReceiverImpl : public Block::Impl,
                         public DynamicConfig<Blocks::UdpReceiver> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::UdpReceiver> moduleConfig =
        std::make_shared<Modules::UdpReceiver>();
    Modules::UdpReceiverImpl* moduleImpl = nullptr;
    Parser::Map interfaceDropdown;
};

Result UdpReceiverImpl::configure() {
    moduleConfig->address = address;
    moduleConfig->port = port;
    moduleConfig->networkInterface = networkInterface;
    moduleConfig->filterSource = filterSource;
    moduleConfig->sourceAddress = sourceAddress;
    moduleConfig->dataType = dataType;
    moduleConfig->numberOfBatches = numberOfBatches;
    moduleConfig->numberOfTimeSamples = numberOfTimeSamples;
    moduleConfig->bufferMultiplier = bufferMultiplier;

    return Result::SUCCESS;
}

Result UdpReceiverImpl::define() {
    const auto& config = *candidate();

    JST_CHECK(defineInterfaceOutput("signal",
                                    "Output",
                                    "The output buffer containing samples received over UDP."));

    JST_CHECK(defineInterfaceConfig("address",
                                    "Address",
                                    "Local address or multicast group to listen on.",
                                    {{"type", "text"}}));

    JST_CHECK(defineInterfaceConfig("port",
                                    "Port",
                                    "Local port to listen on.",
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
                                    "Network interface for multicast groups.",
                                    interfaceDropdown));

    JST_CHECK(defineInterfaceConfig("filterSource",
                                    "Filter Source",
                                    "Accept packets from one sender only.",
                                    {{"type", "bool"}}));

    if (config.filterSource) {
        JST_CHECK(defineInterfaceConfig("sourceAddress",
                                        "Source Address",
                                        "Address of the only sender to accept.",
                                        {{"type", "text"}}));
    }

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
                                    {{"type", "progressbar"}},
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
                                    {{"type", "progressbar"}},
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

Result UdpReceiverImpl::create() {
    JST_CHECK(moduleCreate("udp_receiver", moduleConfig, {}));
    JST_CHECK(moduleExposeOutput("signal", {"udp_receiver", "signal"}));

    moduleImpl = moduleHandle("udp_receiver")->getImpl<Modules::UdpReceiverImpl>();

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(UdpReceiverImpl, {"udp_receiver"});

}  // namespace Jetstream::Blocks
